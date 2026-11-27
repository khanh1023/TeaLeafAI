"""Add attributable camera-named source photos without moving any frozen v2 split.

The Datta mirror mixes camera filenames and offline image_* augmentations. Only
camera-named files are eligible, and neighbouring captures in the same minute
are grouped. Hu et al.'s explicit red-leaf-spot folders supply a second source.
All candidate images are checked against the entire frozen v2, including test.
No prediction, accuracy, or model output is used to decide which images to keep.
"""
import argparse
import collections
import hashlib
import io
import json
import random
import csv
import re
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import requests
from PIL import Image, ImageOps
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from build_dataset_v2 import BKTree, fingerprints

ROOT = Path(__file__).resolve().parents[1]
V2_SHA = '47a24012c4f67957c9ec98b997e37404575738d068ec004c415ab7bd03ecf001'
SPLITS = ['train', 'val', 'test']
RATIOS = np.array([.7, .15, .15])
SOURCES = [
    dict(id='datta_camera_named', repo='Project-AgML/tea_leaf_disease_classification',
         revision='d44a9aff2f0f71aae523b22b5f0f821c68bf18ea',
         file='tea-six-classes.parquet',
         url='https://huggingface.co/datasets/Project-AgML/tea_leaf_disease_classification/resolve/d44a9aff2f0f71aae523b22b5f0f821c68bf18ea/data/train-00000-of-00001.parquet',
         sha256='ad72b6e9a2abfb9f157000e3c377162b2a7525e5788d1f57fde066ce627975ba',
         dataset_url='https://www.kaggle.com/datasets/saikatdatta1994/tea-leaf-disease',
         citation='Datta tea-leaf-disease dataset; pinned Project-AgML mirror; used by Balasundaram et al. (2025), DOI 10.1016/j.rineng.2024.103784',
         license='CC BY-NC 4.0 (mirror declaration; academic/non-commercial use only)',
         license_url='https://creativecommons.org/licenses/by-nc/4.0/',
         eligibility='Camera-pattern filenames only; all image_* offline augmentations and helopeltis excluded. Filename is a provenance proxy, not proof of an unmodified camera original.'),
    dict(id='hu_red_spot_2019', file='hu2019-data.zip',
         url='https://data.mendeley.com/public-files/datasets/dbjyfkn6jr/files/a99df65a-78d7-408b-800a-c8bff1ad176d/file_downloaded',
         sha256='ee55070e3f15dbe4ca2ac6ee13664e89a67e327e537fc3bff49d2453cc7a60ea',
         dataset_url='https://data.mendeley.com/datasets/dbjyfkn6jr/1',
         citation='Hu, Wu, Zhang & Wan (2019), Data for: A Low Shot Learning Method for Tea Leaf’s Disease Identification, DOI 10.17632/dbjyfkn6jr.1',
         license='CC BY 4.0', license_url='https://creativecommons.org/licenses/by/4.0/',
         eligibility='Only Tea red leaf spot folders; ambiguous leaf blight and unsupported red scab excluded. Publisher train/test directories are combined before our grouped split.'),
]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def fetch(source, cache):
    path = cache / source['file']
    if not path.exists():
        with requests.get(source['url'], stream=True, timeout=(20, 120)) as response:
            response.raise_for_status()
            with path.open('wb') as out:
                for block in response.iter_content(1024 * 1024):
                    out.write(block)
    if sha(path.read_bytes()) != source['sha256']:
        raise RuntimeError(f'Source checksum mismatch: {path}')
    return path


def group(records):
    parent = list(range(len(records)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i

    def join(i, j):
        a, b = find(i), find(j)
        parent[max(a, b)] = min(a, b)

    identities = {}
    for i, r in enumerate(records):
        for key in ['previous_group', 'capture_group']:
            if r.get(key):
                identity = (key, r[key])
                if identity in identities:
                    join(i, identities[identity])
                identities[identity] = i
    for field in ['phash', 'full_phash']:
        tree = BKTree()
        for i, r in enumerate(records):
            value = int(r[field], 16)
            for j in tree.query(value, 4):
                other = 'full_phash' if field == 'phash' else 'phash'
                if (int(r[other], 16) ^ int(records[j][other], 16)).bit_count() <= 10:
                    join(i, j)
            tree.insert(value, i)
    groups = collections.defaultdict(list)
    for i, r in enumerate(records):
        groups[records[find(i)]['id']].append(r)
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', type=Path, default=ROOT / 'training/dataset-v2')
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'training/dataset-v3')
    parser.add_argument('--reports', type=Path, default=ROOT / 'training/reports/dataset-v3-20261128')
    args = parser.parse_args()
    previous = (args.previous / 'manifest.json').read_bytes()
    if sha(previous) != V2_SHA:
        raise RuntimeError('Frozen v2 manifest changed.')
    if args.output.exists() or args.reports.exists():
        raise RuntimeError('Choose empty output/report directories; no existing version is replaced.')
    args.output.mkdir(parents=True); args.reports.mkdir(parents=True)
    args.cache.mkdir(parents=True, exist_ok=True)
    classes = json.loads((args.previous / 'classes.json').read_text())
    old_summary = json.loads((args.previous / 'split_summary.json').read_text())
    records = json.loads(previous)
    for r in records:
        r['previous_split'] = r['split']; r['previous_group'] = r['group']
        r['_temporary'] = str(args.previous / r['path'])
    previous_ids = {r['id'] for r in records}
    seen = {field: {r[field]: r for r in records} for field in ['raw_sha256', 'pixel_sha256', 'sha256']}
    blacklist = set(old_summary['exact_conflict_ids'])
    exclusions = []
    candidate_counts = collections.Counter()

    def add(raw, label, source, original, capture_group):
        candidate_counts[(source, label)] += 1
        raw_sha = sha(raw)
        reason = None
        if raw_sha[:16] in blacklist:
            reason = 'previous_exact_conflict_quarantine'
        with Image.open(io.BytesIO(raw)) as opened:
            image = ImageOps.exif_transpose(opened).convert('RGB'); image.load()
            size = list(image.size)
            if min(size) < 80:
                reason = 'image_too_small'
            image.thumbnail((512, 512), Image.Resampling.LANCZOS)
            pixel_sha = sha(np.asarray(image).tobytes())
            buffer = io.BytesIO(); image.save(buffer, 'JPEG', quality=92, subsampling=0)
            content = buffer.getvalue()
            hashes = fingerprints(image.copy())
        r = dict(id=raw_sha[:16], raw_sha256=raw_sha, pixel_sha256=pixel_sha,
                 sha256=sha(content), label=label, source=source, original=original,
                 original_size=size, full_phash=f'{hashes[0]:016x}', phash=f'{hashes[1]:016x}',
                 capture_group=capture_group)
        match = None
        for field in seen:
            if r[field] in seen[field]:
                reason = 'duplicate_' + field; match = seen[field][r[field]]; break
        if reason:
            exclusions.append(dict(source=source, original=original, label=label, reason=reason,
                                   matches=match['id'] if match else None,
                                   label_conflict=bool(match and match['label'] != label)))
            return
        path = args.output / 'staging' / label / f'{source}-{r["id"]}.jpg'
        path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(content)
        r['_temporary'] = str(path); records.append(r)
        for field in seen:
            seen[field][r[field]] = r

    parquet = fetch(SOURCES[0], args.cache)
    mappings = {0: 'algal_leaf', 1: 'brown_blight', 2: 'gray_light', 3: 'healthy', 5: 'red_leaf_spot'}
    omitted = collections.Counter()
    for batch in pq.ParquetFile(parquet).iter_batches(batch_size=64):
        for row in batch.to_pylist():
            name = row['image']['path']
            if row['label'] not in mappings:
                omitted['unsupported_helopeltis'] += 1; continue
            match = re.fullmatch(r'(20\d{6}_\d{4})\d{2}(?:\(\d+\))?\.jpg', name, re.I)
            if not match:
                omitted['offline_augmentation_filename'] += 1; continue
            label = mappings[row['label']]
            add(row['image']['bytes'], label, SOURCES[0]['id'],
                f'{SOURCES[0]["repo"]}@{SOURCES[0]["revision"]}:{label}/{name}',
                f'datta:{label}:{match.group(1)}')
    with zipfile.ZipFile(fetch(SOURCES[1], args.cache)) as archive:
        for name in sorted(archive.namelist()):
            if '/Tea red leaf spot/' in name and name.lower().endswith(('.jpg', '.png', '.jpeg')):
                add(archive.read(name), 'red_leaf_spot', SOURCES[1]['id'], name, None)
    print('Eligible camera-named/explicit-label candidates:', dict(candidate_counts), 'omitted:', dict(omitted), flush=True)
    # Recompressed full-frame copies are not additional independent photos.
    # Keep frozen v2 intact and exclude only incoming duplicates/conflicts.
    trees = {field: BKTree() for field in ['phash', 'full_phash']}
    arrays = {}

    def pixels(i):
        if i not in arrays:
            with Image.open(records[i]['_temporary']) as image:
                arrays[i] = np.asarray(image.convert('RGB').resize((128, 128), Image.Resampling.BILINEAR), dtype=np.int16)
        return arrays[i]

    kept = []
    for i, r in enumerate(records):
        duplicate = None
        if r['id'] not in previous_ids:
            candidates = set(j for field, tree in trees.items() for j in tree.query(int(r[field], 16), 4))
            for j in sorted(candidates):
                old = records[j]
                ar = r['original_size'][0] / r['original_size'][1]
                other_ar = old['original_size'][0] / old['original_size'][1]
                if min(abs(ar / other_ar - 1), abs(ar * other_ar - 1)) > .01:
                    continue
                a, b = pixels(i), pixels(j)
                for angle in range(4):
                    rotated = np.rot90(a, angle)
                    for view in (rotated, np.fliplr(rotated)):
                        delta = np.abs(view - b)
                        if float(delta.mean()) <= 1.5 and float(np.percentile(delta, 95)) <= 6:
                            duplicate = old; break
                    if duplicate:
                        break
                if duplicate:
                    break
        if duplicate:
            exclusions.append(dict(source=r['source'], original=r['original'], label=r['label'],
                                   reason='reencoded_or_rotated_full_frame_duplicate', matches=duplicate['id'],
                                   label_conflict=duplicate['label'] != r['label']))
            Path(r['_temporary']).unlink()
        else:
            kept.append(r)
            for field, tree in trees.items():
                tree.insert(int(r[field], 16), i)
    records = kept; del arrays
    groups = group(records)
    bridges = {r['id'] for family in groups.values()
               if len({r['previous_split'] for r in family if 'previous_split' in r}) > 1
               for r in family if r['id'] not in previous_ids}
    if bridges:
        for r in records:
            if r['id'] in bridges:
                exclusions.append(dict(source=r['source'], original=r['original'], label=r['label'],
                                       reason='new_family_bridges_frozen_previous_splits'))
                Path(r['_temporary']).unlink()
        records = [r for r in records if r['id'] not in bridges]
        groups = group(records)
    # A maximum of 19 unused new healthy photos allows integer exact percentages.
    reserve_count = len(records) % 20
    candidates = sorted((r for r in records if r['id'] not in previous_ids and r['label'] == 'healthy'), key=lambda r: r['id'])
    if len(candidates) < reserve_count:
        raise RuntimeError('Not enough new healthy photos for the integer reserve.')
    reserve = candidates[:reserve_count]; reserved_ids = {r['id'] for r in reserve}
    records = [r for r in records if r['id'] not in reserved_ids]; groups = group(records)
    counts = np.array([sum(r['label'] == c for r in records) for c in classes])
    total = len(records); totals = np.array([total * 14 // 20, total * 3 // 20, total * 3 // 20])
    families = list(groups.values()); random.Random(42).shuffle(families)
    # One binary split per family. Preserve all v2 assignments exactly.
    n = len(families); matrix = lil_matrix((n + 24 + 3, n * 3))
    lower = np.zeros(n * 3); upper = np.ones(n * 3)
    for i, family in enumerate(families):
        anchors = {r['previous_split'] for r in family if 'previous_split' in r}
        assert len(anchors) <= 1
        for s, split in enumerate(SPLITS):
            matrix[i, i * 3 + s] = 1
            for c, label in enumerate(classes):
                matrix[n + c * 3 + s, i * 3 + s] = sum(r['label'] == label for r in family)
            matrix[n + 24 + s, i * 3 + s] = len(family)
            if anchors and split not in anchors:
                upper[i * 3 + s] = 0
    desired = counts[:, None] * RATIOS[None, :]
    target_lower = np.concatenate([np.ones(n), np.floor(desired).ravel(), totals])
    target_upper = np.concatenate([np.ones(n), np.ceil(desired).ravel(), totals])
    result = milp(c=np.random.default_rng(42).uniform(0, .001, n * 3),
                  integrality=np.ones(n * 3), bounds=Bounds(lower, upper),
                  constraints=LinearConstraint(matrix.tocsr(), target_lower, target_upper), options={'time_limit': 90})
    if not result.success:
        raise RuntimeError('Grouped exact split infeasible; do not shuffle the old test to make it fit: ' + result.message)
    assignment = np.rint(result.x).reshape(n, 3).argmax(1)
    for i, family in enumerate(families):
        gid = min(r['id'] for r in family)
        for r in family:
            r['split'] = SPLITS[int(assignment[i])]; r['group'] = gid
    quota = np.array([[sum(r['label'] == label and r['split'] == split for r in records)
                       for split in SPLITS] for label in classes])
    for r in reserve:
        r['split'] = 'reserve'; r['group'] = r['id']
    for r in records + reserve:
        original = Path(r.pop('_temporary'))
        target_path = args.output / r['split'] / r['label'] / original.name
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if r['id'] in previous_ids:
            assert r['previous_split'] == r['split']
            target_path.hardlink_to(original)
        else:
            original.rename(target_path)
        r['path'] = str(target_path.relative_to(args.output))
    shutil.rmtree(args.output / 'staging')
    records.sort(key=lambda r: r['id'])
    manifest = json.dumps(records, ensure_ascii=False, indent=2, sort_keys=True)
    summary = dict(version='dataset-v3-20261128', seed=42, total_images=total,
                   previous_total_images=len(previous_ids), added_images=total-len(previous_ids),
                   previous_manifest_sha256=V2_SHA, manifest_sha256=sha(manifest.encode()),
                   counts=dict(zip(classes, counts.tolist())),
                   split_counts={s: dict(zip(classes, quota[:, j].tolist())) for j, s in enumerate(SPLITS)},
                   split_totals=dict(zip(SPLITS, totals.tolist())),
                   actual_ratios=dict(zip(SPLITS, (totals / total).tolist())),
                   target_ratios=dict(zip(SPLITS, RATIOS.tolist())),
                   split_method='All frozen v2 assignments preserved; exact global/floor-ceil per-class quotas solved as grouped integer allocation with seed 42. New capture-minute families stay intact.',
                   previous_assignments_preserved=True, groups=len(groups), largest_group=max(map(len, groups.values())),
                   source_counts=dict(collections.Counter(r['source'] for r in records)),
                   sources=old_summary['sources']+SOURCES, reserve_images=len(reserve),
                   excluded_count=len(exclusions), omitted_source_rows=dict(omitted),
                   exact_conflict_ids=old_summary['exact_conflict_ids'],
                   limitations=old_summary['limitations'][:4]+[
                       'Datta camera-pattern filenames are a proxy for source captures, not proof of original/raw identity; image_* augmentations are excluded.',
                       'New Datta photos with the same label and timestamp minute are grouped; missing plant IDs still prevent proving complete independence.',
                       'The old validation/test have been inspected during prior development; expanded evaluation is internal, not a blind external field test.',
                       'New sources add five matching classes, not new independent samples for anthracnose, bird_eye_spot or white_spot.',
                       'Healthy remains the largest class; report macro and per-class metrics. Additional photos cannot guarantee a 90% accuracy.',
                       'The new Datta source is declared CC BY-NC 4.0: academic/non-commercial use with attribution; do not relicense it as unrestricted commercial data.'])
    for name, value in [('manifest.json', records), ('split_summary.json', summary), ('classes.json', classes),
                        ('reserve.json', reserve), ('exclusions.json', exclusions)]:
        text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
        (args.output / name).write_text(text); (args.reports / name).write_text(text)
    with (args.reports / 'class-counts.csv').open('w', newline='', encoding='utf-8-sig') as out:
        writer = csv.writer(out, lineterminator='\n')
        writer.writerow(['class', 'previous', 'added', 'train', 'val', 'test', 'total'])
        for i, label in enumerate(classes):
            writer.writerow([label, old_summary['counts'][label], int(counts[i])-old_summary['counts'][label],
                             *quota[i].tolist(), int(counts[i])])
        writer.writerow(['TOTAL', len(previous_ids), total-len(previous_ids), *totals.tolist(), total])
    assert sha((args.previous / 'manifest.json').read_bytes()) == V2_SHA
    print(json.dumps({k: summary[k] for k in ['total_images', 'added_images', 'counts', 'split_totals', 'reserve_images', 'excluded_count', 'largest_group', 'manifest_sha256']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
