#!/usr/bin/env python3
"""Submit, inspect and cancel Kaggle notebook sessions through the bundled SDK."""
import argparse
import json
from pathlib import Path
import time


def push(client, kernel, folder, timeout):
    from kagglesdk.kernels.types.kernels_api_service import ApiSaveKernelRequest
    from kagglesdk.kernels.types.kernels_enums import KernelExecutionType
    meta = json.loads((folder / 'kernel-metadata.json').read_text())
    source = (folder / meta['code_file']).read_text()
    if meta['kernel_type'] == 'notebook':
        notebook = json.loads(source)
        for cell in notebook['cells']:
            if isinstance(cell['source'], list):
                cell['source'] = ''.join(cell['source'])
            if cell['cell_type'] == 'code':
                cell['outputs'] = []
                cell['execution_count'] = None
        source = json.dumps(notebook)
    request = ApiSaveKernelRequest()
    request.slug = kernel
    request.new_title = meta['title']
    request.text = source
    request.language = meta['language']
    request.kernel_type = meta['kernel_type']
    request.is_private = True
    request.enable_gpu = meta.get('enable_gpu', False)
    request.enable_internet = meta.get('enable_internet', True)
    request.machine_shape = 'NvidiaTeslaT4' if request.enable_gpu else 'None'
    request.dataset_data_sources = meta.get('dataset_sources', [])
    request.session_timeout_seconds = timeout
    request.kernel_execution_type = KernelExecutionType.SAVE_AND_RUN_ALL
    response = client.save_kernel(request)
    if response.error:
        raise RuntimeError(response.error)
    if response.invalid_dataset_sources:
        raise RuntimeError('Kaggle rejected input datasets: ' + ', '.join(response.invalid_dataset_sources))
    if response.version_number <= 0:
        raise RuntimeError('Kaggle did not report the new notebook version')
    return {'kernel': kernel, 'version': response.version_number}


def status(client, kernel):
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelSessionStatusRequest
    request = ApiGetKernelSessionStatusRequest()
    request.user_name, request.kernel_slug = kernel.split("/")
    return client.get_kernel_session_status(request).status.name


def info(client, kernel):
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
    request = ApiGetKernelRequest()
    request.user_name, request.kernel_slug = kernel.split("/")
    metadata = client.get_kernel(request).metadata
    return {"kernel": kernel, "version": metadata.current_version_number,
            "status": status(client, kernel)}


def cancel(client, kernel, version, session_id):
    from kagglesdk.kernels.types.kernels_api_service import ApiCancelKernelSessionRequest
    if session_id <= 0 or version <= 0:
        raise ValueError("A positive session ID and version are required")
    terminal = ("COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED")
    current = info(client, kernel)
    state = current["status"]
    if state not in terminal or current["version"] != version:
        request = ApiCancelKernelSessionRequest()
        request.kernel_session_id = session_id
        response = client.cancel_kernel_session(request)
        if response.error_message:
            raise RuntimeError(response.error_message)
    if current["version"] != version:
        raise RuntimeError("Cancellation requested for the recorded session, but the notebook version changed; cannot confirm its shutdown")
    deadline = time.monotonic() + 60
    while state not in terminal and time.monotonic() < deadline:
        time.sleep(2)
        current = info(client, kernel)
        if current["version"] != version:
            raise RuntimeError("Cancellation requested, but the notebook version changed; cannot confirm its shutdown")
        state = current["status"]
    if state not in terminal:
        raise RuntimeError(f"Kaggle has not confirmed shutdown: {state}; stop the session in Active Events")
    return {"kernel": kernel, "version": version, "session_id": session_id, "status": state}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("info", "cancel", "push"))
    parser.add_argument("kernel")
    parser.add_argument("--version", type=int)
    parser.add_argument("--session-id", type=int)
    parser.add_argument("--folder", type=Path)
    parser.add_argument("--timeout", type=int, default=14400)
    args = parser.parse_args()
    if args.action == "cancel" and (not args.version or not args.session_id):
        parser.error("cancel requires --version and --session-id")
    if args.action == 'push' and not args.folder:
        parser.error('push requires --folder')
    from kagglesdk import KaggleClient
    from requests.exceptions import ConnectionError, Timeout
    with KaggleClient() as kaggle:
        client = kaggle.kernels.kernels_api_client
        try:
            if args.action == 'push':
                result = push(client, args.kernel, args.folder, args.timeout)
            elif args.action == 'info':
                result = info(client, args.kernel)
            else:
                result = cancel(client, args.kernel, args.version, args.session_id)
        except (ConnectionError, Timeout):
            parser.exit(1, 'Kaggle API unavailable: DNS/network failure or request timeout.\n')
        except Exception as error:
            if args.action != "info" or getattr(getattr(error, "response", None), "status_code", None) not in (403, 404):
                raise
            result = {"kernel": args.kernel, "version": 0, "status": "NOT_FOUND"}
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
