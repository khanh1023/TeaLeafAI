"""Audit v3 fingerprints, exact ratios, frozen assignments and capture families."""
import argparse
import collections
import hashlib
import json
from pathlib import Path

from audit_dataset_v2 import audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--previous', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.data)
    current = json.loads((args.data / 'manifest.json').read_text())
    previous = json.loads((args.previous / 'manifest.json').read_text())
    summary = json.loads((args.data / 'split_summary.json').read_text())
    assert hashlib.sha256((args.previous / 'manifest.json').read_bytes()).hexdigest() == summary['previous_manifest_sha256']
    lookup = {r['id']: r for r in current}
    for old in previous:
        for field in ['label', 'split', 'sha256', 'raw_sha256']:
            assert lookup[old['id']][field] == old[field]
    captures = collections.defaultdict(set)
    for r in current:
        if r.get('capture_group'):
            captures[r['capture_group']].add(r['split'])
        if r['source'] == 'datta_camera_named':
            assert not r['original'].split('/')[-1].startswith('image_')
    assert all(len(splits) == 1 for splits in captures.values())
    result.update(previous_assignments_verified=len(previous), capture_families_verified=len(captures),
                  added_images=len(current)-len(previous))
    result['checks'] += ['Every frozen v2 photo retains its label, bytes and split',
                         'Every selected Datta capture-minute family stays in one split',
                         'Datta image_* offline augmentations excluded']
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
