"""Train/select/evaluate/export v3 into staging; never deploy automatically."""
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--data', type=Path, default=ROOT / 'training/dataset-v3')
    p.add_argument('--initial', type=Path, default=ROOT / 'backend/models/tea_leaf_mobilenetv2.keras')
    p.add_argument('--initial-report', type=Path, default=ROOT / 'training/evaluation.json')
    p.add_argument('--wait-pid', type=int, help='Wait for an already running train_dataset_v3.py process, then continue export.')
    args = p.parse_args(); args.run_dir.mkdir(parents=True, exist_ok=True)
    state = dict(started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), status='running', stage='training')

    def save(**updates):
        state.update(updates)
        state['updated_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        (args.run_dir / 'pipeline-state.json').write_text(json.dumps(state, indent=2))

    def run(name, command):
        save(stage=name)
        with (args.run_dir / f'{name}.log').open('a') as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'{name} failed ({result.returncode}); inspect {name}.log')

    try:
        save()
        model_dir = args.run_dir / 'mobile-final'
        evaluation_path = model_dir / 'evaluation.json'
        if args.wait_pid:
            while not evaluation_path.exists():
                try:
                    os.kill(args.wait_pid, 0)
                    stat = Path(f'/proc/{args.wait_pid}/stat')
                    if stat.exists() and stat.read_text().split(') ', 1)[1].startswith('Z '):
                        raise ProcessLookupError()
                except ProcessLookupError:
                    raise RuntimeError('Training process ended without final evaluation; inspect training.log.')
                time.sleep(15)
        elif not evaluation_path.exists():
            run('training', [sys.executable, '-u', str(ROOT / 'training/train_dataset_v3.py'),
                             '--data', str(args.data), '--initial', str(args.initial),
                             '--initial-report', str(args.initial_report), '--output', str(model_dir)])
        evaluation = json.loads(evaluation_path.read_text())
        model_sha = hashlib.sha256((model_dir / 'tea_leaf_mobilenetv2.keras').read_bytes()).hexdigest()
        dataset_sha = hashlib.sha256((args.data / 'manifest.json').read_bytes()).hexdigest()
        assert evaluation['model_sha256'] == model_sha
        assert evaluation['dataset_manifest_sha256'] == dataset_sha
        export = args.run_dir / 'export'
        export_manifest = export / 'manifest.json'
        complete = False
        if export_manifest.exists():
            manifest = json.loads(export_manifest.read_text())
            complete = (manifest.get('keras_sha256') == model_sha and manifest.get('dataset_manifest_sha256') == dataset_sha
                        and manifest.get('parity_samples') == evaluation['test_images']
                        and hashlib.sha256((export / manifest['file']).read_bytes()).hexdigest() == manifest['sha256'])
        if not complete:
            run('export', [sys.executable, '-u', str(ROOT / 'training/export_browser_model.py'),
                           '--model', str(model_dir / 'tea_leaf_mobilenetv2.keras'), '--classes', str(model_dir / 'classes.json'),
                           '--data', str(args.data), '--evaluation', str(evaluation_path), '--output', str(export),
                           '--report', str(args.run_dir / 'export-reports'), '--evaluate-test'])
        save(status='staging-ready', stage='complete', dataset_manifest_sha256=dataset_sha,
             model_sha256=model_sha, test_accuracy=evaluation['accuracy'], macro_f1=evaluation['macro_f1'],
             test_images=evaluation['test_images'], target_met=evaluation['target_met'],
             next_step='Review validation-selected evidence, run actual browser evaluation and publish verified artifacts.')
    except Exception as error:
        save(status='failed', error=str(error))
        raise


if __name__ == '__main__':
    main()
