"""Verify all data hashes, split counts, class quotas and cross-split fingerprints."""
import argparse
import collections
import hashlib
import json
from pathlib import Path

from PIL import Image
from build_dataset_v2 import BKTree


def audit(data):
    manifest_bytes = (data / 'manifest.json').read_bytes()
    records = json.loads(manifest_bytes)
    summary = json.loads((data / 'split_summary.json').read_text())
    classes = json.loads((data / 'classes.json').read_text())
    reserve = json.loads((data / 'reserve.json').read_text())
    assert hashlib.sha256(manifest_bytes).hexdigest() == summary['manifest_sha256']
    assert len(records) == summary['total_images']
    assert len(reserve) == summary['reserve_images'] < 20
    counts = collections.Counter(r['label'] for r in records)
    assert dict(counts) == summary['counts']
    assert counts['healthy'] > max(n for c, n in counts.items() if c != 'healthy')
    paths = set()
    for r in records + reserve:
        assert r['label'] in classes
        assert r['path'].startswith(f"{r['split']}/{r['label']}/")
        assert r['path'] not in paths; paths.add(r['path'])
        content = (data / r['path']).read_bytes()
        assert hashlib.sha256(content).hexdigest() == r['sha256'], r['path']
        with Image.open(data / r['path']) as image:
            image.load()
            assert image.mode == 'RGB' and max(image.size) <= 512 and min(image.size) > 0
    assert {str(p.relative_to(data)) for p in data.rglob('*.jpg')} == paths
    # Every exact identity, perceptual group, and direct pHash neighbour stays in one partition.
    edge_checks = 0
    for key in ['raw_sha256', 'pixel_sha256', 'sha256', 'group']:
        splits = collections.defaultdict(set)
        for r in records:
            splits[r[key]].add(r['split'])
        assert all(len(v) == 1 for v in splits.values()), key
        if key != 'group':
            assert len(splits) == len(records), key
    for field in ['phash', 'full_phash']:
        tree = BKTree()
        for i, r in enumerate(records):
            value = int(r[field], 16)
            for j in tree.query(value, 4):
                other_field = 'full_phash' if field == 'phash' else 'phash'
                if (int(r[other_field], 16) ^ int(records[j][other_field], 16)).bit_count() > 10:
                    continue
                edge_checks += 1
                assert r['group'] == records[j]['group'], (field, r['id'], records[j]['id'])
                assert r['split'] == records[j]['split']
            tree.insert(value, i)
    for split, numerator in [('train', 14), ('val', 3), ('test', 3)]:
        subset = [r for r in records if r['split'] == split]
        assert len(subset) * 20 == len(records) * numerator
        assert len(subset) == summary['split_totals'][split]
        assert dict(collections.Counter(r['label'] for r in subset)) == summary['split_counts'][split]
        for c in classes:
            assert abs(sum(r['label'] == c for r in subset) - counts[c] * numerator / 20) < 1.000001
    result = dict(passed=True, verified_images=len(paths), training_evaluation_images=len(records),
                  reserve_images=len(reserve), checked_neighbour_pairs=edge_checks,
                  manifest_sha256=summary['manifest_sha256'],
                  checks=['All JPEG files decoded; RGB and bounded to 512 px', 'Every file SHA-256 matched manifest',
                          'Exact global 70/15/15 and per-class rounding within one image',
                          'No repeated raw, normalized pixel or stored image SHA-256',
                          'No cross-split perceptual groups or confirmed pHash neighbours (one <=4 bits and both <=10 bits)',
                          'Healthy is larger than every individual disease class'],
                  limitation='Fingerprint checks cannot prove plant/capture-session independence without source identities.')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = audit(args.data)
    if args.output:
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))
