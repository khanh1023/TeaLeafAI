"""Build a separate, attributed dataset version with exact overall 70/15/15 counts.

No cap on eligible source images. Exact duplicates and contradictory exact labels
are excluded; perceptually related images remain in one split. At most 19 new
healthy singleton images are kept in reserve to make the total divisible by 20.
The frozen v1 dataset and its previously published results are never changed.
"""
import argparse
import collections
import csv
import hashlib
import io
import json
import random
import shutil
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image, ImageOps
import pyarrow.parquet as pq
from scipy.optimize import Bounds, LinearConstraint, milp
from download_expansion_sources import SOURCES as V1_SOURCES, digest_file
from download_dataset_v2_sources import SOURCES as V2_SOURCES

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ['train', 'val', 'test']
RATIOS = np.array([.7, .15, .15])


def sha(data):
    return hashlib.sha256(data).hexdigest()


class BKTree:
    """Hamming-distance lookup for 64-bit perceptual fingerprints."""
    def __init__(self):
        self.root = None

    def insert(self, value, index):
        node = self.root
        if node is None:
            self.root = [value, index, {}]
            return
        while True:
            distance = (value ^ node[0]).bit_count()
            if distance in node[2]:
                node = node[2][distance]
            else:
                node[2][distance] = [value, index, {}]
                return

    def query(self, value, radius):
        pending = [self.root] if self.root else []
        while pending:
            node = pending.pop()
            distance = (value ^ node[0]).bit_count()
            if distance <= radius:
                yield node[1]
            pending.extend(child for edge, child in node[2].items()
                           if distance - radius <= edge <= distance + radius)


def fingerprints(image):
    image.thumbnail((256, 256))
    a = np.asarray(image)
    ys, xs = np.where((a[:, :, 1] > a[:, :, 0] * 1.03) & (a[:, :, 1] > a[:, :, 2] * 1.07))
    crop = image
    if len(xs) > 256:
        crop = image.crop((max(0, int(xs.min()) - 5), max(0, int(ys.min()) - 5),
                           min(image.width, int(xs.max()) + 6), min(image.height, int(ys.max()) + 6)))
    hashes = []
    for view in (image, crop):
        variants = []
        for angle in [0, 90, 180, 270]:
            rotated = view.rotate(angle, expand=True)
            variants.extend(int(str(imagehash.phash(v)), 16) for v in (rotated, ImageOps.mirror(rotated)))
        hashes.append(min(variants))
    return hashes


def class_quotas(counts, totals):
    desired = counts[:, None] * RATIOS[None, :]
    constraints = []
    lower = []
    for i in range(len(counts)):
        a = np.zeros_like(desired); a[i, :] = 1
        constraints.append(a.ravel()); lower.append(counts[i])
    for j in range(3):
        a = np.zeros_like(desired); a[:, j] = 1
        constraints.append(a.ravel()); lower.append(totals[j])
    # Floor/ceiling bounds keep each per-class split within one image of target.
    result = milp(c=(1 - 2 * (desired - np.floor(desired))).ravel(),
                  integrality=np.ones(desired.size),
                  bounds=Bounds(np.floor(desired).ravel(), np.ceil(desired).ravel()),
                  constraints=LinearConstraint(np.array(constraints), lower, lower))
    if not result.success:
        raise RuntimeError('Could not allocate integer class quotas.')
    return np.rint(result.x.reshape(desired.shape)).astype(int)


def remove_reencodings(records, exclusions, conflicts):
    """Discard near-identical full-frame recompressions/rotations, not new captures.

    Require matching aspect ratio and very small pixel error, in addition to a
    pHash candidate match. More varied near neighbours are grouped later instead.
    """
    trees = {field: BKTree() for field in ['phash', 'full_phash']}
    arrays = {}
    kept = []

    def pixels(i):
        if i not in arrays:
            with Image.open(records[i]['_temporary']) as image:
                arrays[i] = np.asarray(image.convert('RGB').resize((128, 128), Image.Resampling.BILINEAR), dtype=np.int16)
        return arrays[i]

    for i, record in enumerate(records):
        candidates = set()
        for field, tree in trees.items():
            candidates.update(tree.query(int(record[field], 16), 4))
        match = None
        for j in sorted(candidates):
            old = records[j]
            ar = record['original_size'][0] / record['original_size'][1]
            other_ar = old['original_size'][0] / old['original_size'][1]
            if min(abs(ar / other_ar - 1), abs(ar * other_ar - 1)) > .01:
                continue
            a, b = pixels(i), pixels(j)
            for k in range(4):
                rotated = np.rot90(a, k)
                for view in (rotated, np.fliplr(rotated)):
                    delta = np.abs(view - b)
                    mae = float(delta.mean())
                    if mae <= 1.5 and float(np.percentile(delta, 95)) <= 6:
                        match = (j, mae)
                        break
                if match:
                    break
            if match:
                break
        if match:
            j, mae = match
            old = records[j]
            if old['label'] != record['label']:
                conflicts.update([old['id'], record['id']])
            exclusions.append(dict(source=record['source'], original=record['original'], label=record['label'],
                                   reason='reencoded_or_rotated_full_frame_duplicate', matches=old['id'],
                                   mean_absolute_pixel_error_128=mae, label_conflict=old['label'] != record['label']))
            Path(record['_temporary']).unlink()
        else:
            kept.append(record)
            for field, tree in trees.items():
                tree.insert(int(record[field], 16), i)
    return kept


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-data', type=Path, default=ROOT / 'training/dataset')
    parser.add_argument('--old-cache', type=Path, required=True)
    parser.add_argument('--new-cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'training/dataset-v2')
    parser.add_argument('--reports', type=Path, default=ROOT / 'training/reports/dataset-v2-20261125')
    parser.add_argument('--reuse-curated', type=Path, help='Reuse verified v2 photos/fingerprints when recomputing a split; never replaces the input.')
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Output exists. Select a new empty version directory; existing data is never replaced.')
    args.output.mkdir(parents=True)
    args.reports.mkdir(parents=True, exist_ok=True)
    classes = json.loads((args.old_data / 'classes.json').read_text())
    old_manifest = (args.old_data / 'manifest.json').read_bytes()
    old_records = json.loads(old_manifest)
    old_summary = json.loads((args.old_data / 'split_summary.json').read_text())
    parquet_source = next(s for s in V1_SOURCES if s['file'] == 'tea-bangladesh.parquet')
    if digest_file(args.old_cache / parquet_source['file']) != parquet_source['sha256']:
        raise RuntimeError('Bangladesh parquet does not match its pinned source checksum.')
    blacklist = set(old_summary['exact_conflict_ids'])
    records = []; exclusions = []; raw_seen = {}; pixel_seen = {}; normalized_seen = {}; conflicts = set()

    def add(record, content):
        if sha(content) != record['sha256']:
            raise RuntimeError('Source receipt/file checksum mismatch.')
        if record['raw_sha256'][:16] in blacklist:
            exclusions.append(dict(source=record['source'], original=record['original'], label=record['label'], reason='v1_exact_label_conflict_quarantine'))
            return
        for key, lookup in [('raw_sha256', raw_seen), ('pixel_sha256', pixel_seen), ('sha256', normalized_seen)]:
            value = record[key]
            if value in lookup:
                old = records[lookup[value]]
                if old['label'] != record['label']:
                    conflicts.add(old['id'])
                exclusions.append(dict(source=record['source'], original=record['original'], label=record['label'], reason=f'duplicate_{key}', matches=old['id'], label_conflict=old['label'] != record['label']))
                return
        record = dict(record)
        record['id'] = record['raw_sha256'][:16]
        for key, lookup in [('raw_sha256', raw_seen), ('pixel_sha256', pixel_seen), ('sha256', normalized_seen)]:
            lookup[record[key]] = len(records)
        with Image.open(io.BytesIO(content)) as opened:
            hashes = fingerprints(opened.convert('RGB'))
        record['full_phash'], record['phash'] = [f'{v:016x}' for v in hashes]
        record.pop('split', None); record.pop('path', None); record.pop('group', None)
        path = args.output / 'staging' / record['label'] / f"{record['source']}-{record['id']}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        record['_temporary'] = str(path)
        records.append(record)

    if args.reuse_curated:
        input_manifest = (args.reuse_curated / 'manifest.json').read_bytes()
        input_summary = json.loads((args.reuse_curated / 'split_summary.json').read_text())
        if input_summary['version'] != 'dataset-v2-20261125' or sha(input_manifest) != input_summary['manifest_sha256']:
            raise RuntimeError('Unrecognized or modified curated input.')
        if sha(old_manifest) != input_summary['previous_manifest_sha256']:
            raise RuntimeError('Curated input used a different v1 seed.')
        exclusions = json.loads((args.reuse_curated / 'exclusions.json').read_text())
        conflicts.update(input_summary['exact_conflict_ids'])
        for record in json.loads(input_manifest) + json.loads((args.reuse_curated / 'reserve.json').read_text()):
            record = dict(record)
            source_path = args.reuse_curated / record['path']
            if digest_file(source_path) != record['sha256']:
                raise RuntimeError('Curated image checksum mismatch.')
            target = args.output / 'staging' / record['label'] / source_path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                target.hardlink_to(source_path)
            except OSError:
                shutil.copyfile(source_path, target)
            for field in ['path', 'split', 'group']:
                record.pop(field, None)
            record['_temporary'] = str(target)
            records.append(record)
        print('Reusing verified curated photos:', len(records), flush=True)
    else:
        for record in old_records:
            add(record, (args.old_data / record['path']).read_bytes())
        print('Seeded from frozen v1:', len(records), flush=True)
    batches = [] if args.reuse_curated else pq.ParquetFile(args.old_cache / 'tea-bangladesh.parquet').iter_batches(batch_size=64)
    for batch in batches:
        for row in batch.to_pylist():
            if row['label'] not in (0, 2):
                continue
            label = {0: 'healthy', 2: 'red_leaf_spot'}[row['label']]
            raw = row['image']['bytes']
            with Image.open(io.BytesIO(raw)) as opened:
                image = ImageOps.exif_transpose(opened).convert('RGB'); image.load()
                size = list(image.size)
                if min(size) < 80:
                    exclusions.append(dict(source='tea_bangladesh_v4', original=row['image']['path'], label=label, reason='image_too_small'))
                    continue
                image.thumbnail((512, 512), Image.Resampling.LANCZOS)
                pixel_sha = sha(np.asarray(image).tobytes())
                buffer = io.BytesIO(); image.save(buffer, 'JPEG', quality=92, subsampling=0)
            content = buffer.getvalue()
            add(dict(label=label, source='tea_bangladesh_v4', original=row['image']['path'], raw_sha256=sha(raw),
                     pixel_sha256=pixel_sha, sha256=sha(content), original_size=size), content)
    print('After all eligible Bangladesh rows:', len(records), flush=True)
    sources = json.loads((args.new_cache / 'sources-v2.json').read_text())
    if {s['id'] for s in sources} != {s['id'] for s in V2_SOURCES}:
        raise RuntimeError('Source download is incomplete. Finish all source downloads first.')
    for source in ([] if args.reuse_curated else sources):
        folder = args.new_cache / source['id']
        exclusions.extend(source.get('invalid_source_images', []))
        receipts = json.loads((folder / 'receipts.json').read_text())
        if len(receipts) != source['eligible_downloaded']:
            raise RuntimeError('Source receipts count differs from source metadata.')
        for record in receipts:
            record = dict(record); filename = record.pop('file')
            add(record, (folder / filename).read_bytes())
        print('After', source['id'], len(records), flush=True)
    if not args.reuse_curated:
        records = remove_reencodings(records, exclusions, conflicts)
    print('After full-frame recompression/rotation deduplication:', len(records), flush=True)
    kept = []
    for r in records:
        if r['id'] in conflicts:
            exclusions.append(dict(source=r['source'], original=r['original'], label=r['label'], reason='exact_label_conflict_quarantine'))
            Path(r['_temporary']).unlink()
        else:
            kept.append(r)
    records = kept
    # Canonical order makes a fresh source rebuild and a cached re-split agree.
    records.sort(key=lambda r: r['id'])
    # A close full-frame hash alone can match different leaves on the same white
    # background. Require support from both full-image and green-region hashes.
    parent = list(range(len(records)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i

    for field in ['phash', 'full_phash']:
        tree = BKTree()
        for i, r in enumerate(records):
            value = int(r[field], 16)
            for j in tree.query(value, 4):
                other_field = 'full_phash' if field == 'phash' else 'phash'
                if (int(r[other_field], 16) ^ int(records[j][other_field], 16)).bit_count() > 10:
                    continue
                a, b = find(i), find(j)
                if a != b:
                    parent[max(a, b)] = min(a, b)
            tree.insert(value, i)
    groups = collections.defaultdict(list)
    for i, r in enumerate(records):
        r['group'] = records[find(i)]['id']; groups[r['group']].append(r)
    # Exact 70/15/15 needs N divisible by 20. Extra photos are retained, never discarded.
    reserve_count = len(records) % 20
    old_ids = {r['id'] for r in old_records}
    candidates = sorted((g[0] for g in groups.values() if len(g) == 1 and g[0]['label'] == 'healthy'
                         and g[0]['id'] not in old_ids), key=lambda r: r['id'])
    if len(candidates) < reserve_count:
        raise RuntimeError('Not enough new healthy singleton photos for integer reserve.')
    reserve = candidates[:reserve_count]; reserved_ids = {r['id'] for r in reserve}
    records = [r for r in records if r['id'] not in reserved_ids]
    groups = {key: values for key, values in groups.items() if key not in reserved_ids}
    counts = np.array([sum(r['label'] == c for r in records) for c in classes])
    total = len(records); totals = np.array([total * 14 // 20, total * 3 // 20, total * 3 // 20])
    quota = class_quotas(counts, totals); remaining = quota.copy()
    index = {c: i for i, c in enumerate(classes)}
    rng = random.Random(42)
    families = list(groups.values()); rng.shuffle(families)
    families.sort(key=lambda g: len(g), reverse=True)
    source_used = collections.Counter()
    source_counts = collections.Counter((r['label'], r['source']) for r in records)
    for family in families:
        vector = np.array([sum(r['label'] == c for r in family) for c in classes])
        feasible = [s for s in range(3) if np.all(remaining[:, s] >= vector)]
        if not feasible:
            raise RuntimeError('Grouped allocation cannot meet quotas; no files have been assigned.')
        def score(s):
            # Spread each source/class across splits while respecting exact class quotas.
            return sum((source_counts[(r['label'], r['source'])] * RATIOS[s]
                        - source_used[(r['label'], r['source'], s)])
                       / max(1, source_counts[(r['label'], r['source'])]) for r in family)
        split_index = max(feasible, key=lambda s: (score(s), remaining[:, s].sum(), -s))
        remaining[:, split_index] -= vector
        for r in family:
            r['split'] = SPLITS[split_index]
            source_used[(r['label'], r['source'], split_index)] += 1
    assert not remaining.any()
    for r in reserve:
        r['split'] = 'reserve'
    for r in records + reserve:
        temporary = Path(r.pop('_temporary'))
        r['path'] = f"{r['split']}/{r['label']}/{temporary.name}"
        target = args.output / r['path']; target.parent.mkdir(parents=True, exist_ok=True)
        temporary.rename(target)
    shutil.rmtree(args.output / 'staging')
    manifest = json.dumps(records, ensure_ascii=False, indent=2, sort_keys=True).encode()
    all_sources = json.loads((args.old_cache / 'sources.json').read_text()) + sources
    for source in all_sources:
        source['id'] = {'rrff4vb35m-v2': 'mendeley_rrff_v2', 'tea-bangladesh-v4': 'tea_bangladesh_v4'}.get(source['id'], source['id'])
    all_sources.insert(0, dict(id='original', dataset_url='https://data.mendeley.com/datasets/j32xdt2ff5/2',
                              citation='User-provided DuLieuGoc; related reference: Kimutai & Forster (2022), DOI 10.17632/j32xdt2ff5.2',
                              license='User-provided originals: retain their original source terms; no new license assigned here.'))
    summary = dict(version='dataset-v2-20261125', seed=42, total_images=total,
                   previous_total_images=len(old_records), source_candidates_before_reserve=total+len(reserve),
                   counts=dict(zip(classes, counts.tolist())),
                   split_counts={s: dict(zip(classes, quota[:, j].tolist())) for j, s in enumerate(SPLITS)},
                   split_totals=dict(zip(SPLITS, totals.tolist())), actual_ratios=dict(zip(SPLITS, (totals/total).tolist())),
                   target_ratios=dict(zip(SPLITS, RATIOS.tolist())),
                   split_method='Exact global and floor/ceil per-class integer quotas; perceptual families assigned intact; deterministic seed 42',
                   grouping='Rotation/mirror-canonical full-image and green-region 64-bit pHash: one distance <=4 AND both distances <=10; transitive union',
                   groups=len(groups), largest_group=max(map(len, groups.values())),
                   groups_by_split={s: len({r['group'] for r in records if r['split'] == s}) for s in SPLITS},
                   source_counts=dict(collections.Counter(r['source'] for r in records)),
                   reserve_images=len(reserve), reserve_reason='New healthy singleton photos retained outside training/evaluation to allow exact integer 70/15/15.',
                   excluded_count=len(exclusions), exact_conflict_ids=sorted(conflicts | blacklist),
                   manifest_sha256=sha(manifest), previous_manifest_sha256=sha(old_manifest),
                   sources=all_sources,
                   limitations=['Source labels inherited from publishers; no new expert diagnosis performed.',
                                'Plant and capture-session identities are incomplete; perceptual groups reduce, but cannot eliminate, all leakage.',
                                'No reliable new independent source was found for anthracnose, bird_eye_spot or white_spot; these classes are not inflated with synthetic photos.',
                                'Offline augmented sources without parent identities, ambiguous blight/rust labels, pests and non-tea leaves excluded.',
                                'Splits are new. Old accuracy values and old checkpoints do not constitute evaluation on this version.',
                                'Healthy is deliberately the largest class; use class weights and report macro metrics to account for imbalance.'])
    assert counts[index['healthy']] > max(v for c, v in zip(classes, counts) if c != 'healthy')
    for filename, data in [('manifest.json', records), ('split_summary.json', summary),
                           ('exclusions.json', exclusions), ('reserve.json', reserve), ('classes.json', classes)]:
        serialized = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True)
        (args.output / filename).write_text(serialized)
        (args.reports / filename).write_text(serialized)
    with (args.reports / 'class-counts.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f, lineterminator='\n'); writer.writerow(['class', 'previous', 'train', 'val', 'test', 'total', 'change'])
        for i, c in enumerate(classes):
            writer.writerow([c, old_summary['counts'][c], *quota[i].tolist(), int(counts[i]), int(counts[i])-old_summary['counts'][c]])
        writer.writerow(['TOTAL', len(old_records), *totals.tolist(), total, total-len(old_records)])
    assert sha((args.old_data / 'manifest.json').read_bytes()) == sha(old_manifest)
    print(json.dumps({k: summary[k] for k in ['total_images', 'counts', 'split_totals', 'actual_ratios', 'groups', 'largest_group', 'reserve_images', 'excluded_count']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
