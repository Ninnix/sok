#!/usr/bin/env python3
"""Prepare a private model dataset in a CPU-only Kaggle job, then reuse it on GPU starts."""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid

import connect

ROOT = Path(__file__).resolve().parent


def dataset_ready(cli, dataset):
    result = subprocess.run([cli, 'datasets', 'status', dataset], capture_output=True, text=True, timeout=60)
    return result.returncode == 0 and result.stdout.strip().splitlines()[-1:] == ['ready']


def wait_cache(cli, dataset, session):
    deadline = time.monotonic() + 9000
    while time.monotonic() < deadline:
        try:
            if dataset_ready(cli, dataset):
                return
            current = connect.control('info', session)
        except (OSError, subprocess.SubprocessError):
            print('Kaggle status unavailable; keeping the CPU worker running and retrying.', flush=True)
            time.sleep(15)
            continue
        if current['version'] != session['version'] or current['status'] not in ('QUEUED', 'RUNNING', 'NEW_SCRIPT'):
            raise RuntimeError('CPU cache job ended before the dataset was ready; inspect its Kaggle log')
        print(f"CPU cache job: {current['status']}; waiting for the model dataset.", flush=True)
        time.sleep(15)
    raise RuntimeError('CPU cache preparation timed out')


def save_defaults(args, dataset):
    path = args.state_dir / 'defaults.json'
    defaults = json.loads(path.read_text()) if path.exists() else {}
    defaults.setdefault('models', {})[connect.model_key(args.model, args.family)] = {'model_dataset': dataset}
    connect.private_write(path, json.dumps(defaults, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', help="any Strata model size; omitted: Strata's default for the T4 target")
    parser.add_argument('--family', help='optional Strata model family')
    parser.add_argument('--dataset', help='private dataset reference, owner/name')
    parser.add_argument('--state-dir', type=Path, default=Path.home() / '.cache/sok')
    args = parser.parse_args()
    if not connect.load_kaggle_token():
        parser.error('Set KAGGLE_API_TOKEN or save your token in ~/.kaggle/access_token')
    args.state_dir = args.state_dir.expanduser().resolve()
    owner = json.loads((ROOT / 'kernel-metadata.json').read_text())['id'].split('/')[0]
    name = re.sub(r'[^a-z0-9-]+', '-', connect.model_key(args.model, args.family))
    dataset = args.dataset or f"{owner}/sok-{name}-weights"
    if not re.fullmatch(r'[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+', dataset):
        parser.error('dataset must be owner/name')
    cli = shutil.which('kaggle') or str(Path(sys.executable).parent / 'kaggle')
    if dataset_ready(cli, dataset):
        save_defaults(args, dataset)
        print('Using existing model cache:', dataset)
        return
    folder = args.state_dir / 'cache-setup'
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    kernel = dataset.split('/')[0] + '/sok-model-cache-cpu'
    try:
        current = connect.control('info', {'kernel': kernel})
    except subprocess.CalledProcessError:
        raise RuntimeError('Cannot inspect the CPU cache job') from None
    stored_path = folder / 'session.json'
    stored = json.loads(stored_path.read_text()) if stored_path.exists() else {}
    active = current['status'] in ('QUEUED', 'RUNNING', 'NEW_SCRIPT')
    if active:
        if (stored.get('version') != current['version'] or stored.get('dataset', dataset) != dataset
                or stored.get('model') != args.model or stored.get('family') != args.family):
            raise RuntimeError('Another CPU cache job is running; wait for it to finish')
        session = stored
    else:
        session = dict(kernel=kernel, version=0, launch_id=uuid.uuid4().hex, dataset=dataset,
                       model=args.model, family=args.family)
        source = 'from pathlib import Path\nimport sys\nPath("/kaggle/temp").mkdir(exist_ok=True)\nsys.path.insert(0,"/kaggle/temp")\n'
        source += 'Path("/kaggle/temp/kaggle_runtime.py").write_text(' + repr((ROOT / 'kaggle_runtime.py').read_text()) + ')\n'
        source += 'Path("/kaggle/temp/strata_models.py").write_text(' + repr((ROOT / 'strata_models.py').read_text()) + ')\n'
        source += (ROOT / 'cache_runtime.py').read_text()
        source += f'\nrun_cache({dataset!r}, {args.model!r}, {session["launch_id"]!r}, family={args.family!r})\n'
        meta = dict(id=kernel, title=kernel.split('/')[1].replace('-', ' ').title(), code_file='cache.py',
                    language='python', kernel_type='script', is_private=True, enable_gpu=False,
                    enable_internet=True, machine_shape='None', dataset_sources=[], competition_sources=[], kernel_sources=[])
        with tempfile.TemporaryDirectory(prefix='sok-cache-') as tmp:
            (Path(tmp) / 'cache.py').write_text(source)
            (Path(tmp) / 'kernel-metadata.json').write_text(json.dumps(meta))
            submitted = connect.control('push', {'kernel': kernel}, folder=Path(tmp), seconds=10000)
        session['version'] = submitted['version']
        print(f"Submitted {kernel}, version {session['version']}", flush=True)
        connect.save_session(folder, session)
    print('Preparing', args.model or "Strata's default", 'on a CPU-only Kaggle worker; no GPU quota is used.', flush=True)
    try:
        if not session.get('session_id'):
            connect.wait_connection(cli, folder, None, session, seconds=600, session_only=True)
        wait_cache(cli, dataset, session)
        save_defaults(args, dataset)
        print('Model cache ready:', dataset, flush=True)
    finally:
        if session.get('session_id'):
            result = connect.control('cancel', session)
            print('CPU cache worker stopped:', result['status'], flush=True)
            stored_path.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
