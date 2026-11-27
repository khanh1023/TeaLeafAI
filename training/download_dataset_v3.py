"""Restore v3 from verified v2 and the bundled additions (standard library only).

Run download_dataset_v2.py first if the previous dataset is not present. The
small additions archive is included in the repository. No output is overwritten.
"""
import argparse
import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'training/reports/dataset-v3-20261128'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def safe_path(value):
    path = PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts or '\\' in value:
        raise RuntimeError('Unsafe archive/manifest path.')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', type=Path, default=ROOT / 'training/dataset-v2')
    parser.add_argument('--output', type=Path, default=ROOT / 'training/dataset-v3')
    parser.add_argument('--archive', type=Path)
    args = parser.parse_args()
    receipt = json.loads((REPORTS / 'package.json').read_text(encoding='utf-8'))
    archive_path = args.archive or ROOT / receipt['repository_path']
    if args.output.exists():
        raise SystemExit('Output exists; select a new empty directory.')
    if not (args.previous / 'manifest.json').exists():
        raise SystemExit('Restore v2 first: python training/download_dataset_v2.py')
    if digest(args.previous / 'manifest.json') != receipt['previous_manifest_sha256']:
        raise RuntimeError('Frozen v2 manifest mismatch.')
    if digest(archive_path) != receipt['sha256']:
        raise RuntimeError('Additions archive checksum mismatch.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.dataset-v3-', dir=args.output.parent) as temporary:
        staging = Path(temporary)
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                path = safe_path(member.filename)
                if not path.parts or path.parts[0] != 'dataset-v3':
                    raise RuntimeError('Wrong archive root.')
            archive.extractall(staging)
        dataset = staging / 'dataset-v3'
        if digest(dataset / 'manifest.json') != receipt['manifest_sha256']:
            raise RuntimeError('Expanded manifest mismatch.')
        previous = {r['id']: r for r in json.loads((args.previous / 'manifest.json').read_text(encoding='utf-8'))}
        records = json.loads((dataset / 'manifest.json').read_text(encoding='utf-8'))
        reserve = json.loads((dataset / 'reserve.json').read_text(encoding='utf-8'))
        if len(records) != receipt['training_evaluation_images'] or len(reserve) != receipt['added_reserve_images']:
            raise RuntimeError('Record count mismatch.')
        for r in records + reserve:
            path = dataset / safe_path(r['path'])
            if r['id'] in previous:
                old = previous[r['id']]
                if any(r[field] != old[field] for field in ['label', 'split', 'raw_sha256', 'sha256']):
                    raise RuntimeError('An old photo moved split or changed identity.')
                source = args.previous / safe_path(old['path'])
                if digest(source) != old['sha256']:
                    raise RuntimeError('Frozen v2 image checksum mismatch.')
                if path.exists():
                    raise RuntimeError('An old image was unexpectedly included in additions.')
                path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(source, path)
                except OSError:
                    shutil.copyfile(source, path)
            if digest(path) != r['sha256']:
                raise RuntimeError('Restored image checksum mismatch: ' + r['path'])
        dataset.rename(args.output)
    print(f'Restored {len(records)} split photos and {len(reserve)} unused reserve photos to {args.output}')


if __name__ == '__main__':
    main()
