"""Download and restore the published dataset-v2 archive with SHA-256 verification.

Run from the repository: python training/download_dataset_v2.py
Use --archive FILE to restore an already downloaded ZIP. Never replaces a folder.
Only Python's standard library is needed for restoration.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'training/reports/dataset-v2-20261125'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--output', default=ROOT / 'training/dataset-v2', type=Path)
    args = parser.parse_args()
    package = json.loads((REPORTS / 'package.json').read_text(encoding='utf-8'))
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f'Output already exists; select another empty folder: {output}')
    if args.archive:
        archive_path = args.archive.resolve()
    else:
        cache = ROOT / 'training/source-cache'
        cache.mkdir(parents=True, exist_ok=True)
        archive_path = cache / package['filename']
        if not archive_path.exists():
            partial = archive_path.with_suffix('.zip.part')
            request = urllib.request.Request(package['download_url'], headers={'User-Agent': 'TeaLeafAI-dataset-v2'})
            with urllib.request.urlopen(request, timeout=120) as response, partial.open('wb') as file:
                shutil.copyfileobj(response, file, length=1024 * 1024)
            if digest(partial) != package['sha256']:
                partial.unlink()
                raise SystemExit('Downloaded archive checksum mismatch.')
            partial.rename(archive_path)
    if digest(archive_path) != package['sha256']:
        raise SystemExit('Archive checksum mismatch.')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.dataset-v2-', dir=output.parent) as temp:
        staging = Path(temp)
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                path = PurePosixPath(member.filename)
                if path.is_absolute() or '..' in path.parts or not path.parts or path.parts[0] != 'dataset-v2' or '\\' in member.filename:
                    raise SystemExit('Unsafe archive path.')
            archive.extractall(staging)
        dataset = staging / 'dataset-v2'
        manifest = dataset / 'manifest.json'
        if digest(manifest) != package['manifest_sha256']:
            raise SystemExit('Manifest checksum mismatch.')
        entries = json.loads(manifest.read_text(encoding='utf-8')) + json.loads((dataset / 'reserve.json').read_text(encoding='utf-8'))
        for record in entries:
            relative = PurePosixPath(record['path'])
            if relative.is_absolute() or '..' in relative.parts or '\\' in record['path']:
                raise SystemExit('Unsafe manifest path.')
            if digest(dataset / record['path']) != record['sha256']:
                raise SystemExit(f"Image checksum mismatch: {record['path']}")
        dataset.rename(output)
    print(f'Restored {package["training_evaluation_images"]} split images and {package["reserve_images"]} reserve images to {output}')


if __name__ == '__main__':
    main()
