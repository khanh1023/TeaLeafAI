"""One-command standard-library restoration and verification of the full v3 data."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify(data, package):
    if digest(data / 'manifest.json') != package['manifest_sha256']:
        raise SystemExit('An existing dataset has a different manifest; select another empty location.')
    entries = json.loads((data / 'manifest.json').read_text(encoding='utf-8'))
    reserve = json.loads((data / 'reserve.json').read_text(encoding='utf-8'))
    for r in entries + reserve:
        path = Path(r['path'])
        if path.is_absolute() or '..' in path.parts or '\\' in r['path']:
            raise SystemExit('Unsafe image path.')
        if digest(data / path) != r['sha256']:
            raise SystemExit('Image checksum mismatch: ' + r['path'])
    return len(entries), len(reserve)


def main():
    for version, report in [('v2', 'dataset-v2-20261125'), ('v3', 'dataset-v3-20261128')]:
        data = ROOT / f'training/dataset-{version}'
        package = json.loads((ROOT / 'training/reports' / report / 'package.json').read_text(encoding='utf-8'))
        if not data.exists():
            subprocess.run([sys.executable, str(ROOT / f'training/download_dataset_{version}.py')], check=True)
        counts = verify(data, package)
        print(f'Verified {version}: {counts[0]} split photos + {counts[1]} unused reserve photos.', flush=True)
    print('V3 ready: 5796 train / 1242 validation / 1242 test (70/15/15).')


if __name__ == '__main__':
    main()
