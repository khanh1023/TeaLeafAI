"""Create a deterministic, self-contained dataset ZIP and publishable checksum receipt."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'training/dataset-v2')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reports', type=Path, default=ROOT / 'training/reports/dataset-v2-20261125')
    parser.add_argument('--parts-dir', type=Path, help='Optionally split into GitHub-compatible files under 100 MB.')
    parser.add_argument('--parts-ref', default='dataset-v2-20261125-data')
    args = parser.parse_args()
    records = json.loads((args.data / 'manifest.json').read_text(encoding='utf-8'))
    reserve = json.loads((args.data / 'reserve.json').read_text(encoding='utf-8'))
    summary = json.loads((args.data / 'split_summary.json').read_text(encoding='utf-8'))
    audit = json.loads((args.reports / 'audit.json').read_text(encoding='utf-8'))
    if not audit['passed'] or audit['manifest_sha256'] != summary['manifest_sha256']:
        raise SystemExit('Complete the dataset audit before packaging.')
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def write(archive, name, content, image=False):
        info = zipfile.ZipInfo('dataset-v2/' + name, date_time=(2026, 11, 25, 0, 0, 0))
        info.compress_type = zipfile.ZIP_STORED if image else zipfile.ZIP_DEFLATED
        info.external_attr = 0o100644 << 16
        archive.writestr(info, content)

    with zipfile.ZipFile(args.output, 'w', allowZip64=True) as archive:
        for record in sorted(records + reserve, key=lambda r: r['path']):
            content = (args.data / record['path']).read_bytes()
            if hashlib.sha256(content).hexdigest() != record['sha256']:
                raise RuntimeError('Image changed after audit.')
            write(archive, record['path'], content, image=True)
        for name in ['manifest.json', 'classes.json', 'split_summary.json', 'exclusions.json', 'reserve.json']:
            write(archive, name, (args.data / name).read_bytes())
        write(archive, 'audit.json', (args.reports / 'audit.json').read_bytes())
        write(archive, 'class-counts.csv', (args.reports / 'class-counts.csv').read_bytes())
        write(archive, 'README.md', (ROOT / 'docs/DATASET_V2.md').read_bytes())
    with zipfile.ZipFile(args.output) as archive:
        if archive.testzip() is not None:
            raise RuntimeError('Archive CRC verification failed.')
    h = hashlib.sha256()
    with args.output.open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            h.update(chunk)
    package = dict(filename=args.output.name, size_bytes=args.output.stat().st_size, sha256=h.hexdigest(),
                   release_tag='dataset-v2-20261125',
                   download_url='https://github.com/khanh1023/TeaLeafAI/releases/download/dataset-v2-20261125/' + args.output.name,
                   manifest_sha256=summary['manifest_sha256'],
                   training_evaluation_images=summary['total_images'], reserve_images=summary['reserve_images'])
    if args.parts_dir:
        args.parts_dir.mkdir(parents=True, exist_ok=True)
        parts = []
        with args.output.open('rb') as file:
            index = 1
            while chunk := file.read(90_000_000):
                name = args.output.name + f'.part{index:02}'
                (args.parts_dir / name).write_bytes(chunk)
                parts.append(dict(filename=name, size_bytes=len(chunk), sha256=hashlib.sha256(chunk).hexdigest(),
                                  download_url=f'https://raw.githubusercontent.com/khanh1023/TeaLeafAI/{args.parts_ref}/{name}'))
                index += 1
        package.pop('download_url')
        package['parts'] = parts
        package['data_ref'] = args.parts_ref
    (args.reports / 'package.json').write_text(json.dumps(package, ensure_ascii=False, indent=2), encoding='utf-8')
    (args.reports / 'SHA256SUMS.txt').write_text(f"{h.hexdigest()}  {args.output.name}\n", encoding='utf-8')
    print(json.dumps(package, indent=2))


if __name__ == '__main__':
    main()
