#!/usr/bin/env python3
"""Launch the notebook with Kaggle credentials and collect an encrypted connection key."""
import argparse
import base64
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parent


def model_key(model=None, family=None):
    model = (model or "default").lower()
    return family.lower() + "/" + model if family else model


def load_kaggle_token():
    if os.environ.get('KAGGLE_API_TOKEN'):
        return os.environ['KAGGLE_API_TOKEN']
    from kagglesdk import get_access_token_from_env
    token, _ = get_access_token_from_env()
    if token:
        os.environ['KAGGLE_API_TOKEN'] = token
    return token


def private_write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(path.name + ".tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    os.fchmod(fd, 0o600)
    with os.fdopen(fd, "w") as out:
        out.write(text)
    temporary.replace(path)


def discover(connection):
    request = urllib.request.Request(connection["STRATA_BASE_URL"] + "/models",
        headers={"Authorization": "Bearer " + connection["STRATA_API_KEY"]})
    with urllib.request.urlopen(request, timeout=20) as response:
        models = json.load(response)["data"]
    if (len(models) != 1 or not isinstance(models[0].get("id"), str)
            or not models[0]["id"].strip()):
        raise RuntimeError("Expected one running Strata model in /v1/models")
    return models[0]["id"]


def register_clients(folder, connection, model):
    """Add one provider while preserving the user's other models and settings."""
    record_path = folder / "registration.json"
    if record_path.exists():
        unregister_clients(folder)
    key_path = folder / "api-key"
    private_write(key_path, connection["STRATA_API_KEY"] + "\n")
    pi = json.loads((ROOT / "pi-config-example.json").read_text())["providers"]["sok"]
    pi.update(name="sok (Strata on Kaggle)", baseUrl=connection["STRATA_BASE_URL"],
              apiKey="!cat " + shlex.quote(str(key_path)))
    pi["models"][0].update(id=model, name="Strata " + model.removeprefix("strata-").upper() + " (Kaggle)")
    oc = json.loads((ROOT / "opencode-config-example.json").read_text())["providers"]["sok"]
    oc.pop("env", None)
    oc["settings"].update(baseURL=connection["STRATA_BASE_URL"], apiKey="{file:" + str(key_path) + "}")
    oc["models"] = {model: dict(next(iter(oc["models"].values())), name=pi["models"][0]["name"])}
    config_home = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    pi_home = Path(os.environ.get("PI_CODING_AGENT_DIR", Path.home() / ".pi/agent"))
    records = []
    private_write(record_path, "[]\n")
    try:
        for path, provider in ((pi_home / "models.json", pi), (config_home / "opencode/opencode.json", oc)):
            config = json.loads(path.read_text()) if path.exists() else {}
            providers = config.setdefault("providers", {})
            record = {"path": str(path), "before": providers.get("sok"), "installed": provider}
            providers["sok"] = provider
            if path.name == "opencode.json":
                record["model_before"] = config.get("model")
                record["model_installed"] = "sok/" + model
                config["model"] = record["model_installed"]
            records.append(record)
            private_write(record_path, json.dumps(records, indent=2) + "\n")
            private_write(path, json.dumps(config, indent=2) + "\n")
    except BaseException:
        unregister_clients(folder)
        raise


def unregister_clients(folder):
    record_path = folder / "registration.json"
    if not record_path.exists():
        return
    for record in json.loads(record_path.read_text()):
        path = Path(record["path"])
        if not path.exists():
            continue
        config = json.loads(path.read_text())
        providers = config.get("providers", {})
        if providers.get("sok") == record["installed"]:
            if record["before"] is None:
                providers.pop("sok", None)
            else:
                providers["sok"] = record["before"]
        if "model_installed" in record and config.get("model") == record["model_installed"]:
            if record["model_before"] is None:
                config.pop("model", None)
            else:
                config["model"] = record["model_before"]
        private_write(path, json.dumps(config, indent=2) + "\n")
    record_path.unlink()
    (folder / "api-key").unlink(missing_ok=True)


def client_key(folder):
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    private = folder / "client-key.pem"
    if not private.exists():
        fd = os.open(private, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(fd, "wb") as out:
                subprocess.run(["openssl", "genpkey", "-algorithm", "RSA", "-pkeyopt",
                                "rsa_keygen_bits:3072"], stdout=out, stderr=subprocess.DEVNULL, check=True)
        except BaseException:
            private.unlink(missing_ok=True)
            raise
    public = subprocess.check_output(["openssl", "pkey", "-in", str(private), "-pubout"], text=True)
    return private, public


def log_events(text, marker):
    for line in reversed(text.splitlines()):
        if marker not in line:
            continue
        try:
            event = json.loads(line.lstrip(",[").rstrip(","))
            line = event.get("data", line)
        except ValueError:
            pass
        match = re.search(re.escape(marker) + r"(\{.*\})", line)
        if match:
            yield json.loads(match[1])


def read_session(text, launch_id=None):
    for session in log_events(text, "SOK_SESSION="):
        if (launch_id is None or session.get("launch_id") == launch_id) and \
                isinstance(session.get("session_id"), int) and session["session_id"] > 0:
            return session


def read_connection(text, private, launch_id=None):
    for connection in log_events(text, "SOK_CONNECTION="):
        if launch_id is not None and connection.get("launch_id") != launch_id:
            continue
        base = connection["base_url"]
        if not re.fullmatch(r"https://[a-z0-9.-]+/v1", base):
            raise ValueError("Invalid tunnel URL")
        key = subprocess.check_output([
            "openssl", "pkeyutl", "-decrypt", "-inkey", str(private),
            "-pkeyopt", "rsa_padding_mode:oaep", "-pkeyopt", "rsa_oaep_md:sha256",
            "-pkeyopt", "rsa_mgf1_md:sha256"], input=base64.b64decode(connection["encrypted_api_key"], validate=True))
        return {"STRATA_BASE_URL": base, "STRATA_API_KEY": key.decode()}
    return None



def control(action, session, folder=None, seconds=14400):
    command = [sys.executable, str(ROOT / 'kaggle_session.py'), action, session['kernel']]
    if action == 'cancel':
        command += ['--version', str(session['version']), '--session-id', str(session['session_id'])]
    elif action == 'push':
        command += ['--folder', str(folder), '--timeout', str(seconds)]
    output = subprocess.check_output(command, text=True, timeout=90)
    return json.loads(output.splitlines()[-1])


def save_session(folder, session):
    private_write(folder / 'session.json', json.dumps(session, indent=2) + '\n')


def wait_connection(cli, folder, private, session, seconds=7200, session_only=False):
    deadline = time.monotonic() + seconds
    log_path = folder / 'launch.log'
    while time.monotonic() < deadline:
        private_write(log_path, '')
        with log_path.open('w') as log:
            proc = subprocess.Popen([cli, 'kernels', 'logs', '-f', session['kernel']],
                stdout=log, stderr=log, env=dict(os.environ, PYTHONUNBUFFERED='1'))
            try:
                for _ in range(60):
                    text = log_path.read_text(errors='replace')
                    found = read_session(text, session.get('launch_id'))
                    if found:
                        session.update(found)
                        save_session(folder, session)
                        if session_only:
                            return None
                    connection = read_connection(text, private, session.get('launch_id'))
                    if connection:
                        if not session.get('session_id'):
                            raise RuntimeError('This notebook has no session ID. Restart it with the updated notebook before using automatic shutdown.')
                        event = next(e for e in log_events(text, 'SOK_CONNECTION=')
                            if session.get('launch_id') is None or e.get('launch_id') == session['launch_id'])
                        if event.get('startup'):
                            private_write(folder / 'startup.json', json.dumps(event['startup'], indent=2) + '\n')
                        return connection
                    if proc.poll() is not None or time.monotonic() >= deadline:
                        break
                    time.sleep(2)
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
        try:
            state = control('info', session)
        except (OSError, subprocess.SubprocessError):
            print('Kaggle status unavailable; retrying the same run...', flush=True)
            time.sleep(2)
            continue
        if state['version'] != session['version']:
            raise RuntimeError('The notebook version changed while connecting; refusing to attach to another run')
        if state['status'] not in ('QUEUED', 'RUNNING', 'NEW_SCRIPT'):
            raise RuntimeError(f"Kaggle run ended: {state['status']}; inspect {log_path}")
        print('Kaggle:', state['status'], flush=True)
        time.sleep(2)
    raise RuntimeError(f'Connection timed out; inspect {log_path}')


def launch(args, public, session):
    save_session(args.state_dir, session)
    notebook = json.loads((ROOT / 'sok.ipynb').read_text())
    cell = next(c for c in notebook['cells'] if c['cell_type'] == 'code')
    cell['source'] += [f'\nMODEL = {args.model!r}\nCLIENT_PUBLIC_KEY = {public!r}\n',
        f'FAMILY = {args.family!r}\n',
        f'LAUNCH_ID = {session["launch_id"]!r}\n',
        f'TUNNEL = "localhost.run"\nREMOTE_ACCESS_PERMITTED = True\nSERVE_SECONDS = {args.seconds}\n',
        f'IDLE_SECONDS = {args.idle_seconds}\n',
        f'STAGE_CACHED_MODEL = {args.stage_model!r}\n',
        'RUN_BENCHMARKS = False\n',
        'REPORT["model"] = MODEL\n']
    meta = json.loads((ROOT / 'kernel-metadata.json').read_text())
    meta['id'] = args.kernel
    meta['title'] = args.kernel.split('/')[1].replace('-', ' ').title()
    meta['dataset_sources'] = [args.model_dataset] if args.model_dataset else []
    with tempfile.TemporaryDirectory(prefix='sok-') as tmp:
        folder = Path(tmp)
        (folder / 'kernel-metadata.json').write_text(json.dumps(meta))
        (folder / meta['code_file']).write_text(json.dumps(notebook))
        submitted = control('push', session, folder=folder)
    session['version'] = submitted['version']
    print(f"Submitted {session['kernel']}, version {session['version']}", flush=True)
    save_session(args.state_dir, session)


def stop_run(folder, session):
    result = control('cancel', session)
    print(f"Kaggle shutdown confirmed: {result['status']} (session {session['session_id']})", flush=True)
    unregister_clients(folder)
    for name in ('session.json', 'connection.env', 'api-key'):
        (folder / name).unlink(missing_ok=True)


def run_client(command, connection):
    env = dict(os.environ, **connection)
    for name in ('KAGGLE_API_TOKEN', 'KAGGLE_KEY', 'KG_TOKEN', 'TUNNEL_TOKEN'):
        env.pop(name, None)
    proc = subprocess.Popen(command, env=env)
    old = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        return proc.wait()
    finally:
        signal.signal(signal.SIGINT, old)
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


def install_launchers():
    folder = Path.home() / '.local/bin'
    folder.mkdir(parents=True, exist_ok=True)
    for client in ('opencode', 'pi'):
        path = folder / ('sok-' + client)
        command = shlex.join([sys.executable, str(ROOT / 'connect.py'), '--client', client])
        path.write_text('#!/bin/sh\nexec ' + command + ' "$@"\n')
        path.chmod(0o755)
        print('Installed:', path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kernel', default=json.loads((ROOT / 'kernel-metadata.json').read_text())['id'])
    parser.add_argument('--model', help="any Strata model size; omitted: Strata's default for the Kaggle hardware")
    parser.add_argument('--family', help='optional Strata family, such as qwen, swift, coder, unsloth or qwen36')
    parser.add_argument('--seconds', type=int, default=1800)
    parser.add_argument('--idle-seconds', type=int, default=900, help='finish the notebook after this many idle seconds; default 900, 0 disables it')
    parser.add_argument('--client', choices=['opencode', 'pi'], help='register the provider and stop Kaggle when this client exits')
    parser.add_argument('--attach', action='store_true', help='reuse a running notebook; never launch another run')
    parser.add_argument('--stop', action='store_true', help='cancel the recorded Kaggle session and confirm shutdown')
    parser.add_argument('--keep-running', action='store_true', help='leave Kaggle running after the client exits')
    parser.add_argument('--install-launchers', action='store_true', help='install sok-opencode and sok-pi in ~/.local/bin')
    parser.add_argument('--model-dataset', help='optional Kaggle dataset containing the selected model files')
    parser.add_argument('--stage-model', action=argparse.BooleanOptionalAction, default=None,
                        help='copy attached model files to scratch before loading; --no-stage-model reads them in place')
    parser.add_argument('--state-dir', type=Path, default=Path.home() / '.cache/sok')
    parser.add_argument('command', nargs=argparse.REMAINDER, help='client arguments, or a local command after --')
    args = parser.parse_args()
    if args.install_launchers:
        install_launchers()
        return 0
    defaults_path = args.state_dir.expanduser() / 'defaults.json'
    defaults = json.loads(defaults_path.read_text()) if defaults_path.exists() else {}
    model_defaults = defaults.get('models', {}).get(model_key(args.model, args.family), {})
    args.model_dataset = args.model_dataset or model_defaults.get('model_dataset')
    if args.stage_model is None:
        args.stage_model = model_defaults.get('stage_model', False)
    if not load_kaggle_token():
        parser.error('Set KAGGLE_API_TOKEN or save your token in ~/.kaggle/access_token')
    if not 60 <= args.seconds <= 10800:
        parser.error('seconds must be between 60 and 10800')
    if not 0 <= args.idle_seconds <= 10800:
        parser.error('idle-seconds must be between 0 and 10800')
    args.state_dir = args.state_dir.expanduser().resolve()
    args.state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    session_path = args.state_dir / 'session.json'
    stored = json.loads(session_path.read_text()) if session_path.exists() else {}
    if args.stop:
        if not stored.get('session_id'):
            parser.error('No recorded Kaggle session; attach to the running notebook first')
        stop_run(args.state_dir, stored)
        return 0
    cli = shutil.which('kaggle') or str(Path(sys.executable).parent / 'kaggle')
    if not Path(cli).is_file():
        parser.error('Run this launcher with the Kaggle CLI virtual environment')
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if args.client:
        command = [args.client, *command]
    private, public = client_key(args.state_dir)
    # One local client owns cancellation. A second launcher must not stop its session.
    import fcntl
    with (args.state_dir / 'client.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error('Another sok launcher is already managing this session')
        state = control('info', {'kernel': args.kernel})
        active = state['status'] in ('QUEUED', 'RUNNING', 'NEW_SCRIPT')
        if args.attach and not active:
            parser.error('The notebook is not running; omit --attach to start it')
        reuse = active and stored.get('kernel') == args.kernel and stored.get('version') == state['version']
        session = stored if reuse else dict(kernel=args.kernel, version=state['version'], launch_id=None)
        success = False
        old_signals = {signum: signal.getsignal(signum) for signum in
                       (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
        started = False
        def interrupted(signum, frame):
            raise KeyboardInterrupt
        for signum in (signal.SIGTERM, signal.SIGHUP):
            signal.signal(signum, interrupted)
        try:
            if not active:
                session = dict(kernel=args.kernel, version=0, launch_id=uuid.uuid4().hex)
                started = True
                launch(args, public, session)
            else:
                print('Reusing running notebook:', args.kernel, flush=True)
            connection = None
            if reuse and session.get('base_url') and (args.state_dir / 'api-key').exists():
                connection = {'STRATA_BASE_URL': session['base_url'],
                              'STRATA_API_KEY': (args.state_dir / 'api-key').read_text().strip()}
                try:
                    discover(connection)
                except OSError:
                    connection = None
            if connection is None:
                print('Waiting for the notebook and HTTPS endpoint...', flush=True)
                connection = wait_connection(cli, args.state_dir, private, session)
            model = discover(connection)
            session.update(base_url=connection['STRATA_BASE_URL'], model=model)
            save_session(args.state_dir, session)
            register_clients(args.state_dir, connection, model)
            connection['STRATA_MODEL'] = model
            private_write(args.state_dir / 'connection.env', ''.join(
                f'export {name}={shlex.quote(value)}\n' for name, value in connection.items()))
            print(f"Registered sok (Strata on Kaggle): {model}\nRemote API: {connection['STRATA_BASE_URL']}", flush=True)
            if command and Path(command[0]).name == 'opencode' and '--standalone' not in command:
                command.append('--standalone')
            if command and Path(command[0]).name == 'pi' and not any(
                    arg == '--model' or arg.startswith('--model=') for arg in command[1:]):
                command[1:1] = ['--provider', 'sok', '--model', model]
            success = True
            if command:
                return run_client(command, connection)
            print('Kaggle is running. Stop it with: python connect.py --stop', flush=True)
            return 0
        except KeyboardInterrupt:
            return 130
        finally:
            for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
                signal.signal(signum, signal.SIG_IGN)
            try:
                if not args.keep_running and (command or not success):
                    if started and not session.get('version'):
                        current = control('info', session)
                        if current['version'] > state['version']:
                            session['version'] = current['version']
                    if not session.get('session_id') and session.get('version'):
                        try:
                            wait_connection(cli, args.state_dir, private, session, seconds=90, session_only=True)
                        except (RuntimeError, OSError, subprocess.SubprocessError) as error:
                            raise RuntimeError('Cannot confirm Kaggle shutdown: no session ID was received. Stop the run in Kaggle Active Events or reconnect with --attach.') from error
                    if session.get('session_id'):
                        stop_run(args.state_dir, session)
                    elif session.get('version'):
                        raise RuntimeError('Cannot confirm Kaggle shutdown: no session ID was received. Stop the run in Kaggle Active Events.')
            finally:
                for signum, handler in old_signals.items():
                    signal.signal(signum, handler)


if __name__ == '__main__':
    raise SystemExit(main())
