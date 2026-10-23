"""CPU-only model cache worker, embedded by prepare_cache.py."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import kaggle_runtime as runtime


def run_cache(dataset, model, launch_id):
    runtime.MODEL, runtime.LAUNCH_ID = model, launch_id
    runtime.REPORT['model'] = model
    runtime.announce_session()
    folder = Path('/kaggle/temp/sok-model-cache')
    folder.mkdir(parents=True, exist_ok=True)
    variant = runtime.MODELS[model]
    files = [folder / f"Qwen3.8-Flash-Next-GSQ-RCO-{variant['quant']}-{i:05}-of-00002.gguf" for i in (1, 2)]
    total = sum(size for size, _ in variant['shards'])
    if shutil.disk_usage(folder).free < total + 4 * 2**30:
        raise RuntimeError('Not enough scratch disk for the selected model cache')
    if not os.environ.get('KAGGLE_API_V1_TOKEN') and not os.environ.get('KAGGLE_API_TOKEN'):
        raise RuntimeError('Kaggle notebook API credentials are missing')
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'kaggle==2.2.4'], check=True)
    cli = [sys.executable, '-m', 'kaggle.cli']
    env = dict(os.environ, PYTHONUNBUFFERED='1', TQDM_MININTERVAL='30')
    subprocess.run(cli + ['datasets', 'list', '--mine'], env=env, stdout=subprocess.DEVNULL, check=True)
    print('Cache phase: downloading and verifying', flush=True)
    for path, (size, digest) in zip(files, variant['shards']):
        url = f"https://huggingface.co/{variant['repo']}/resolve/{variant['revision']}/{variant['quant']}/{path.name}"
        runtime.fetch_model_file(url, path, size, digest)
    metadata = {'id': dataset, 'title': dataset.split('/')[1].replace('-', ' ').title(),
        'licenses': [{'name': 'apache-2.0'}],
        'description': f"Unmodified {variant['quant']} shards from https://huggingface.co/{variant['repo']}/tree/{variant['revision']}. Verified against pinned SHA-256 hashes."}
    (folder / 'dataset-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print('Cache phase: uploading', flush=True)
    subprocess.run(cli + ['datasets', 'create', '-p', str(folder), '--keep-tabular'], env=env, check=True)
    print('Cache submitted:', dataset, flush=True)
    for _ in range(240):
        try:
            result = subprocess.run(cli + ['datasets', 'status', dataset], env=env,
                capture_output=True, text=True, timeout=60)
        except subprocess.TimeoutExpired:
            continue
        if result.returncode == 0 and result.stdout.strip() == 'ready':
            print('Cache ready:', dataset, flush=True)
            time.sleep(60)
            return
        if result.returncode == 0 and result.stdout.strip() == 'error':
            raise RuntimeError('Kaggle rejected the model dataset')
        time.sleep(15)
    raise RuntimeError('Dataset processing timed out')
