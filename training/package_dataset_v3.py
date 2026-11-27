"""Package only new v3 photos, with complete manifests and a pinned v2 dependency."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'training/reports/dataset-v3-20261128'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'training/dataset-v3')
    parser.add_argument('--previous', type=Path, default=ROOT / 'training/dataset-v2')
    parser.add_argument('--output', type=Path, default=ROOT / 'training/data/dataset-v3-additions.zip')
    args = parser.parse_args()
    summary = json.loads((args.data / 'split_summary.json').read_text())
    audit = json.loads((REPORTS / 'audit.json').read_text())
    assert audit['passed'] and audit['manifest_sha256'] == summary['manifest_sha256']
    previous = {r['id'] for r in json.loads((args.previous / 'manifest.json').read_text())}
    records = json.loads((args.data / 'manifest.json').read_text()) + json.loads((args.data / 'reserve.json').read_text())
    additions = [r for r in records if r['id'] not in previous]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise RuntimeError('Existing archive is never overwritten.')
    with zipfile.ZipFile(args.output, 'w') as archive:
        def write(name, content, image=False):
            info = zipfile.ZipInfo('dataset-v3/' + name, date_time=(2026, 11, 28, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED if image else zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, content)
        for r in sorted(additions, key=lambda r: r['path']):
            content = (args.data / r['path']).read_bytes()
            assert hashlib.sha256(content).hexdigest() == r['sha256']
            write(r['path'], content, image=True)
        for name in ['manifest.json', 'split_summary.json', 'reserve.json', 'classes.json', 'exclusions.json']:
            write(name, (args.data / name).read_bytes())
        write('audit.json', (REPORTS / 'audit.json').read_bytes())
        write('class-counts.csv', (REPORTS / 'class-counts.csv').read_bytes())
        write('README.md', (ROOT / 'docs/DATASET_V3.md').read_bytes())
    with zipfile.ZipFile(args.output) as archive:
        assert archive.testzip() is None
    package = dict(filename=args.output.name, repository_path=str(args.output.relative_to(ROOT)),
                   sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(), size_bytes=args.output.stat().st_size,
                   manifest_sha256=summary['manifest_sha256'], previous_manifest_sha256=summary['previous_manifest_sha256'],
                   training_evaluation_images=summary['total_images'], added_images=summary['added_images'],
                   added_reserve_images=summary['reserve_images'], depends_on='dataset-v2-20261125',
                   archive_contains='Only new photos plus complete v3 metadata. Restore v2 first with download_dataset_v2.py, then run download_dataset_v3.py.')
    (REPORTS / 'package.json').write_text(json.dumps(package, ensure_ascii=False, indent=2))
    print(json.dumps(package, indent=2))


if __name__ == '__main__':
    main()
