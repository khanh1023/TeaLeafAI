"""Resume the audited v2 training/evaluation/export pipeline into staging only.

Live frontend files and GitHub are never modified by this script. Publication
follows separate artifact checks and real browser QA. A completed stage is reused
only when its model checksum and dataset version still match.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DATA_SHA = '47a24012c4f67957c9ec98b997e37404575738d068ec004c415ab7bd03ecf001'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def checkpoint_valid(folder, filename, final=False):
    report_path = folder / 'evaluation.json'
    model = folder / filename
    if not report_path.exists() or not model.exists():
        return False
    report = read(report_path)
    if report.get('dataset_manifest_sha256') != DATA_SHA or report.get('model_sha256') != sha(model):
        raise ValueError('Completed checkpoint/evidence mismatch: ' + str(folder))
    if final and (report.get('test_images') != 1026 or sum(map(sum, report['confusion_matrix'])) != 1026):
        raise ValueError('Final checkpoint is not evaluated on the full v2 test set.')
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'training/dataset-v2')
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--label', default='v2-20261127')
    parser.add_argument('--wait-pid', type=int, help='Wait for an already running fine-tune process before resuming.')
    args = parser.parse_args()
    args.data = args.data.resolve(); args.run_dir = args.run_dir.resolve()
    if sha(args.data / 'manifest.json') != DATA_SHA:
        raise SystemExit('Dataset differs from the audited v2 split.')
    args.run_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.run_dir / 'pipeline-state.json'
    env = {**os.environ, 'KERAS_HOME': str(ROOT.parent / 'keras'),
           'TF_ENABLE_ONEDNN_OPTS': '1', 'TF_NUM_INTRAOP_THREADS': '2',
           'TF_NUM_INTEROP_THREADS': '1', 'OMP_NUM_THREADS': '2',
           'MPLCONFIGDIR': str(ROOT.parent / 'matplotlib')}

    def state(stage, **extra):
        value = dict(stage=stage, dataset_manifest_sha256=DATA_SHA,
                     updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                     runtime={key: env[key] for key in ['TF_ENABLE_ONEDNN_OPTS', 'TF_NUM_INTRAOP_THREADS',
                                                       'TF_NUM_INTEROP_THREADS', 'OMP_NUM_THREADS']}, **extra)
        temp = state_path.with_suffix('.json.tmp')
        temp.write_text(json.dumps(value, indent=2), encoding='utf-8'); temp.replace(state_path)
        print(value['updated_at'], stage, extra, flush=True)

    def run(stage, script, options):
        command = [sys.executable, str(ROOT / 'training' / script), *map(str, options)]
        state(stage, command=command)
        with (args.run_dir / (stage + '.log')).open('a', encoding='utf-8') as log:
            log.write('\nCOMMAND ' + json.dumps(command) + '\n'); log.flush()
            subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)

    try:
        head = args.run_dir / 'mobile-head'; final = args.run_dir / 'mobile-final'
        if args.wait_pid:
            state('waiting-for-current-fine-tune', pid=args.wait_pid)
            while True:
                try:
                    os.kill(args.wait_pid, 0)
                except ProcessLookupError:
                    break
                time.sleep(15)
            if not checkpoint_valid(final, 'tea_leaf_mobilenetv2.keras', final=True):
                raise RuntimeError('Existing fine-tune exited without a complete, valid checkpoint report.')
        if not checkpoint_valid(head, 'tea_leaf_mobilenetv2.keras'):
            run('mobile-head', 'train_expanded.py', ['--data', args.data, '--output', head,
                '--epochs', 30, '--augmented-views', 2, '--defer-test', '--no-feature-cache'])
        if not checkpoint_valid(final, 'tea_leaf_mobilenetv2.keras', final=True):
            run('mobile-final', 'finetune_expanded.py', ['--data', args.data, '--initial', head,
                '--output', final, '--epochs', 10])
        comparison = args.run_dir / 'comparison'
        common = ['--data', args.data, '--manifest', args.data / 'manifest.json',
                  '--expected-manifest-sha', DATA_SHA, '--protocol', 'frozen-imagenet-common-head-' + args.label,
                  '--output', comparison, '--epochs', 30, '--feature-batch', 32,
                  '--seed', 42, '--augmented-views', 2, '--allow-cpu']
        run('comparison-resnet-vgg', 'compare_models.py', common + ['--models', 'ResNet50', 'VGG16']
            + (['--resume'] if (comparison / 'protocol.json').exists() else []))
        run('comparison-mobile', 'compare_models.py', common + ['--models', 'MobileNetV2',
            '--import-mobilenet-head', head, '--resume'])
        exports = args.run_dir / 'exports'; reports = args.run_dir / 'export-reports'
        model_sha = read(final / 'evaluation.json')['model_sha256']
        run('export-default', 'export_browser_model.py', ['--model', final / 'tea_leaf_mobilenetv2.keras',
            '--classes', final / 'classes.json', '--evaluation', final / 'evaluation.json',
            '--data', args.data, '--output', exports / 'default', '--report', reports / 'default',
            '--model-version', f'mobilenetv2-dataset-{args.label}-{model_sha[:10]}', '--evaluate-test'])
        run('export-comparison', 'export_comparison_models.py', ['--run', comparison, '--data', args.data,
            '--output', exports / 'comparison', '--report', reports / 'comparison',
            '--version-label', 'frozen-dataset-' + args.label, '--evaluate-test'])
        state('staging-ready', exports=str(exports), reports=str(reports),
              publication='Not deployed. Promote only after tests and real browser QA.')
    except Exception as error:
        state('failed', error=repr(error)); raise


if __name__ == '__main__':
    main()
