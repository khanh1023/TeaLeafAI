"""Restore the exact published v2 checkpoint as a reproducible v3 warm start."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
COMMIT = '7a18a7e27325cf576846bcc635a4898cd26c7418'
MODEL_SHA = '223a253d6619d19117c0c44e411f86916149a5d7f29851e72d284cff83febb03'
DATA_SHA = '47a24012c4f67957c9ec98b997e37404575738d068ec004c415ab7bd03ecf001'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT / 'training/runs/v2-initial')
    args = p.parse_args()
    if args.output.exists():
        raise SystemExit('Output exists; select another empty directory.')
    contents = {}
    for source, name in [('training/evaluation.json', 'evaluation.json'),
                         ('backend/models/tea_leaf_mobilenetv2.keras', 'tea_leaf_mobilenetv2.keras')]:
        url = f'https://raw.githubusercontent.com/khanh1023/TeaLeafAI/{COMMIT}/{source}'
        request = urllib.request.Request(url, headers={'User-Agent': 'TeaLeafAI-v2-checkpoint'})
        with urllib.request.urlopen(request, timeout=120) as response:
            contents[name] = response.read()
    report = json.loads(contents['evaluation.json'])
    if hashlib.sha256(contents['tea_leaf_mobilenetv2.keras']).hexdigest() != MODEL_SHA:
        raise RuntimeError('Checkpoint SHA-256 mismatch.')
    if report['model_sha256'] != MODEL_SHA or report['dataset_manifest_sha256'] != DATA_SHA:
        raise RuntimeError('Checkpoint evidence mismatch.')
    args.output.mkdir(parents=True)
    for name, content in contents.items():
        (args.output / name).write_bytes(content)
    print('Restored pinned v2 checkpoint and its evaluation to', args.output)


if __name__ == '__main__':
    main()
