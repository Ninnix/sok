"""Client registration and shutdown checks; no GPU or model download."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import connect
import cache_runtime
import kaggle_runtime as runtime
import kaggle_session
import prepare_cache


class ClientTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("kagglesdk"), "Install the optional Kaggle CLI to test its SDK")
    def test_saved_kaggle_token_works_in_a_fresh_terminal(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'HOME': tmp}, clear=True):
            connect.private_write(Path(tmp) / '.kaggle/access_token', 'saved-test-token\n')
            self.assertEqual(connect.load_kaggle_token(), 'saved-test-token')
            self.assertEqual(os.environ['KAGGLE_API_TOKEN'], 'saved-test-token')

    def test_environment_token_takes_precedence_over_saved_token(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ,
                {'HOME': tmp, 'KAGGLE_API_TOKEN': 'rotated-test-token'}, clear=True):
            connect.private_write(Path(tmp) / '.kaggle/access_token', 'old-test-token\n')
            self.assertEqual(connect.load_kaggle_token(), 'rotated-test-token')

    def test_interrupted_queued_start_keeps_state_and_reports_unconfirmed_shutdown(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, KAGGLE_API_TOKEN='local-test-token'):
            folder = Path(tmp)
            kernel = 'ninnix/sok'
            session = dict(kernel=kernel, version=9, launch_id='queued-run')
            connect.save_session(folder, session)
            with patch.object(sys, 'argv', ['connect.py', '--attach', '--state-dir', tmp, '--', sys.executable]), \
                    patch.object(connect.shutil, 'which', return_value=sys.executable), \
                    patch.object(connect, 'client_key', return_value=(folder / 'unused-key', 'public-test-key')), \
                    patch.object(connect, 'control', return_value=dict(kernel=kernel, version=9, status='QUEUED')) as control, \
                    patch.object(connect, 'wait_connection', side_effect=[KeyboardInterrupt(), RuntimeError('still queued')]), \
                    patch.object(connect, 'launch') as launch:
                with self.assertRaisesRegex(RuntimeError, 'Cannot confirm Kaggle shutdown'):
                    connect.main()
                launch.assert_not_called()
            self.assertEqual([call.args[0] for call in control.call_args_list], ['info'])
            self.assertEqual(json.loads((folder / 'session.json').read_text()), session)

    def test_connect_retries_an_unavailable_status_request_without_replacing_the_run(self):
        for error in (subprocess.TimeoutExpired('kaggle', 90),
                      subprocess.CalledProcessError(1, ['kaggle_session.py', 'info'])):
            with self.subTest(error=type(error).__name__):
                self.check_status_retry(error)

    def check_status_retry(self, error):
        session = dict(kernel='owner/notebook', version=5, launch_id='same-run')
        count = [0]
        def follower(*args, **kwargs):
            count[0] += 1
            if count[0] == 2:
                kwargs['stdout'].write('SOK_SESSION={"session_id":359710527,"launch_id":"same-run"}\n')
                kwargs['stdout'].flush()
            process = Mock()
            process.poll.return_value = 0
            return process
        with tempfile.TemporaryDirectory() as tmp, patch.object(connect.subprocess, 'Popen', side_effect=follower), \
                patch.object(connect, 'control', side_effect=error) as control, \
                patch.object(connect.time, 'sleep'), patch.object(connect, 'launch') as launch:
            connect.wait_connection('kaggle', Path(tmp), None, session, seconds=10, session_only=True)
            self.assertEqual(json.loads((Path(tmp) / 'session.json').read_text())['session_id'], 359710527)
            control.assert_called_once_with('info', session)
            launch.assert_not_called()
        self.assertEqual(count[0], 2)
        self.assertEqual(session['version'], 5)

    def test_cache_worker_retries_processing_status_without_downloading_again(self):
        variant = dict(quant='IQ2_XS', repo='owner/model', revision='pinned', shards=[(1, 'test-hash')] * 2)
        ok = SimpleNamespace(returncode=0, stdout='')
        pending = SimpleNamespace(returncode=1, stdout='')
        ready = SimpleNamespace(returncode=0, stdout='ready')
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, KAGGLE_API_V1_TOKEN='injected-token-file'), \
                patch.object(cache_runtime, 'Path', return_value=Path(tmp)), \
                patch.object(cache_runtime.shutil, 'disk_usage', return_value=SimpleNamespace(free=8 * 2**30)), \
                patch.object(runtime, 'MODELS', {'iq2_xs': variant}), patch.object(runtime, 'REPORT', {}), \
                patch.object(runtime, 'announce_session'), \
                patch.object(runtime, 'fetch_model_file', side_effect=lambda url, path, size, digest: path.write_bytes(b'x')) as fetch, \
                patch.object(cache_runtime.subprocess, 'run', side_effect=[ok, ok, ok, pending, ready]), \
                patch.object(cache_runtime.time, 'sleep'):
            cache_runtime.run_cache('owner/weights', 'iq2_xs', 'cache-test')
            self.assertEqual(fetch.call_count, 2)
            self.assertEqual(len(list(Path(tmp).glob('*.gguf'))), 2)
            self.assertEqual(json.loads((Path(tmp) / 'dataset-metadata.json').read_text())['id'], 'owner/weights')

    def test_cache_monitor_keeps_the_worker_running_during_a_status_timeout(self):
        session = {"kernel": "owner/cache", "version": 4, "session_id": 359731887}
        with patch.object(prepare_cache, "dataset_ready", side_effect=[False,
                subprocess.TimeoutExpired("kaggle", 60), False, True]), \
                patch.object(connect, "control", return_value={"version": 4, "status": "RUNNING"}) as control, \
                patch.object(prepare_cache.time, "sleep"):
            prepare_cache.wait_cache("kaggle", "owner/weights", session)
        self.assertEqual(control.call_count, 2)
        self.assertTrue(all(call.args == ('info', session) for call in control.call_args_list))

    def test_idle_shutdown_waits_for_prefill_decode_and_short_requests(self):
        clock = [0.0]
        seen = []
        def activity(path, key):
            self.assertEqual((path, key), ("/v1/status", "inference-test-key"))
            now = clock[0]
            busy = 2 <= now < 12
            requests = 0 if now < 2 else 1 if now < 15 else 2
            last = 0 if now < 2 else 2 if now < 12 else 12 if now < 15 else 15
            seen.append((now, busy))
            return {"activity": {"in_flight": int(busy), "requests": requests, "last_request_at": last}}
        with patch.object(runtime.time, "monotonic", side_effect=lambda: clock[0]), \
                patch.object(runtime.time, "sleep", side_effect=lambda seconds: clock.__setitem__(0, clock[0] + seconds)), \
                patch.object(runtime, "PROCESSES", {}), patch.object(runtime, "api_json", side_effect=activity), \
                patch.object(runtime, "save_report") as save:
            runtime.serve_for(60, "inference-test-key", idle_seconds=5)
        self.assertGreaterEqual(clock[0], 20)
        self.assertLess(clock[0], 25)
        self.assertTrue(any(now > 7 and busy for now, busy in seen))
        save.assert_called_once()

    def test_disabled_idle_shutdown_does_not_poll_inference(self):
        clock = [0.0]
        with patch.object(runtime.time, "monotonic", side_effect=lambda: clock[0]), \
                patch.object(runtime.time, "sleep", side_effect=lambda seconds: clock.__setitem__(0, clock[0] + seconds)), \
                patch.object(runtime, "PROCESSES", {}), patch.object(runtime, "api_json") as api:
            runtime.serve_for(3, idle_seconds=0)
        api.assert_not_called()

    def test_registration_preserves_other_providers_and_restores_defaults(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {
                "HOME": tmp, "XDG_CONFIG_HOME": tmp + "/config", "PI_CODING_AGENT_DIR": tmp + "/pi"}):
            folder = Path(tmp) / "state"
            original = {"model": "other/model", "providers": {"other": {"name": "Other"}}}
            oc = Path(tmp) / "config/opencode/opencode.json"
            connect.private_write(oc, json.dumps(original))
            key = "private-inference-test-key"
            connect.register_clients(folder, {"STRATA_BASE_URL": "https://example.lhr.life/v1",
                "STRATA_API_KEY": key}, "strata-q2")
            for path in (oc, Path(tmp) / "pi/models.json"):
                self.assertNotIn(key, path.read_text())
                provider = json.loads(path.read_text())["providers"]["sok"]
                self.assertIn("strata-q2", json.dumps(provider))
            self.assertEqual((folder / "api-key").stat().st_mode & 0o777, 0o600)
            changed = json.loads(oc.read_text())
            changed["username"] = "added while the client was running"
            oc.write_text(json.dumps(changed))
            connect.unregister_clients(folder)
            restored = json.loads(oc.read_text())
            self.assertEqual(restored["providers"], original["providers"])
            self.assertEqual(restored["model"], original["model"])
            self.assertIn("username", restored)
            self.assertFalse((folder / "api-key").exists())

    def test_client_gets_inference_credentials_without_kaggle_credentials(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {
                "KAGGLE_API_TOKEN": "local-admin-token", "KAGGLE_KEY": "legacy-admin-token"}):
            path = Path(tmp) / "environment.json"
            code = ("import json,os,sys; from pathlib import Path; "
                    "Path(sys.argv[1]).write_text(json.dumps(dict(os.environ))); sys.exit(7)")
            result = connect.run_client([sys.executable, "-c", code, str(path)],
                {"STRATA_BASE_URL": "https://example.lhr.life/v1", "STRATA_API_KEY": "inference-test-key"})
            self.assertEqual(result, 7)
            env = json.loads(path.read_text())
            self.assertEqual(env["STRATA_API_KEY"], "inference-test-key")
            self.assertNotIn("KAGGLE_API_TOKEN", env)
            self.assertNotIn("KAGGLE_KEY", env)

    def test_failed_shutdown_keeps_the_session_record_for_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            session = dict(kernel="owner/notebook", version=3, session_id=359710527)
            connect.save_session(folder, session)
            connect.private_write(folder / "api-key", "inference-test-key")
            with patch.object(connect, "control", side_effect=RuntimeError("shutdown not confirmed")):
                with self.assertRaisesRegex(RuntimeError, "not confirmed"):
                    connect.stop_run(folder, session)
            self.assertEqual(json.loads((folder / "session.json").read_text()), session)
            self.assertTrue((folder / "api-key").exists())

    def test_notebook_reports_the_runtime_session_not_the_version_number(self):
        with patch.dict(os.environ, {"KAGGLE_CONTAINER_NAME": "kaggle_example-359710527-webtier"}), \
                patch.object(runtime, "MODEL", "invalid"), patch.object(runtime, "LAUNCH_ID", "test-run"), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            with self.assertRaises(ValueError):
                runtime.hardware()
        found = connect.read_session(output.getvalue(), "test-run")
        self.assertEqual(found["session_id"], 359710527)
        self.assertIsNone(connect.read_session(output.getvalue(), "another-run"))
        event = json.dumps({"data": output.getvalue()})
        self.assertEqual(connect.read_session(event, "test-run"), found)

    def test_reused_run_is_cancelled_after_client_failure_or_interrupt(self):
        for failure in (7, KeyboardInterrupt(), OSError("client could not start")):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmp, \
                    patch.dict(os.environ, {"HOME": tmp, "KAGGLE_API_TOKEN": "local-test-token",
                        "XDG_CONFIG_HOME": tmp + "/config", "PI_CODING_AGENT_DIR": tmp + "/pi"}):
                folder = Path(tmp) / "state"
                kernel = "ninnix/sok"
                session = dict(kernel=kernel, version=9, session_id=359710527,
                               base_url="https://example.lhr.life/v1", launch_id="test-run")
                connect.save_session(folder, session)
                connect.private_write(folder / "api-key", "inference-test-key\n")
                calls = []
                def control(action, value):
                    calls.append((action, dict(value)))
                    return dict(kernel=kernel, version=9,
                                status="RUNNING" if action == "info" else "CANCEL_ACKNOWLEDGED")
                args = ["connect.py", "--attach", "--state-dir", str(folder), "--", sys.executable]
                child = Mock(side_effect=failure) if isinstance(failure, BaseException) else Mock(return_value=failure)
                with patch.object(sys, "argv", args), patch.object(connect, "control", side_effect=control), \
                        patch.object(connect.shutil, "which", return_value=sys.executable), \
                        patch.object(connect, "discover", return_value="strata-iq2_xs"), \
                        patch.object(connect, "run_client", child), patch.object(connect, "launch") as launch:
                    if isinstance(failure, OSError):
                        with self.assertRaises(OSError):
                            connect.main()
                    else:
                        self.assertEqual(connect.main(), 130 if isinstance(failure, KeyboardInterrupt) else 7)
                    launch.assert_not_called()
                self.assertEqual([action for action, _ in calls], ["info", "cancel"])
                self.assertEqual(calls[-1][1]["session_id"], 359710527)
                self.assertFalse((folder / "session.json").exists())
                self.assertFalse((folder / "api-key").exists())

    @unittest.skipUnless(hasattr(signal, "SIGHUP"), "terminal-close signals require POSIX")
    def test_closing_the_terminal_stops_the_child_and_cancels_the_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            script = '''
import json,os,sys
from pathlib import Path
from unittest.mock import patch
import connect
folder=Path(sys.argv[1])
kernel='ninnix/sok'
connect.save_session(folder,dict(kernel=kernel,version=9,session_id=359710527,base_url='https://example.lhr.life/v1'))
connect.private_write(folder/'api-key','inference-test-key')
def control(action,session):
    if action=='cancel': (folder/'cancelled').write_text(str(session['session_id']))
    return dict(kernel=kernel,version=9,status='RUNNING' if action=='info' else 'CANCEL_ACKNOWLEDGED')
code="import os,sys,time; from pathlib import Path; Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(60)"
sys.argv=['connect.py','--attach','--state-dir',str(folder),'--',sys.executable,'-c',code,str(folder/'ready')]
with patch.object(connect,'control',side_effect=control),patch.object(connect,'discover',return_value='strata-iq2_xs'),patch.object(connect.shutil,'which',return_value=sys.executable):
    raise SystemExit(connect.main())
'''
            env = dict(os.environ, HOME=tmp, XDG_CONFIG_HOME=tmp + "/config",
                       PI_CODING_AGENT_DIR=tmp + "/pi", KAGGLE_API_TOKEN="local-test-token")
            with tempfile.TemporaryFile() as log:
                proc = subprocess.Popen([sys.executable, "-c", script, tmp], cwd=ROOT, env=env, stdout=log, stderr=log)
                try:
                    for _ in range(300):
                        if (folder / "ready").exists() or proc.poll() is not None:
                            break
                        time.sleep(0.05)
                    if not (folder / "ready").exists():
                        log.seek(0)
                        self.fail(log.read().decode())
                    child_pid = int((folder / "ready").read_text())
                    proc.send_signal(signal.SIGHUP)
                    self.assertEqual(proc.wait(timeout=15), 130)
                    self.assertEqual((folder / "cancelled").read_text(), "359710527")
                    with self.assertRaises(ProcessLookupError):
                        os.kill(child_pid, 0)
                finally:
                    if proc.poll() is None:
                        proc.kill()
                    proc.wait()


@unittest.skipUnless(importlib.util.find_spec("kagglesdk"), "Install the optional Kaggle CLI to test its SDK")
class SubmissionTests(unittest.TestCase):
    def test_network_failures_print_a_short_message_without_exception_details(self):
        from requests.exceptions import ConnectionError, Timeout
        for error in (ConnectionError('temporary DNS failure: private-test-token'), Timeout('private-test-token')):
            with self.subTest(error=type(error).__name__), \
                    patch.object(sys, 'argv', ['kaggle_session.py', 'info', 'owner/notebook']), \
                    patch('kagglesdk.KaggleClient') as client, \
                    contextlib.redirect_stderr(io.StringIO()) as output:
                client.return_value.__enter__.return_value.kernels.kernels_api_client.get_kernel.side_effect = error
                with self.assertRaises(SystemExit) as caught:
                    kaggle_session.main()
                self.assertEqual(caught.exception.code, 1)
                self.assertEqual(output.getvalue(),
                    'Kaggle API unavailable: DNS/network failure or request timeout.\n')

    def test_push_explicitly_executes_private_cpu_and_gpu_jobs(self):
        from kagglesdk.kernels.types.kernels_enums import KernelExecutionType
        for gpu in (False, True):
            with self.subTest(gpu=gpu), tempfile.TemporaryDirectory() as tmp:
                folder = Path(tmp)
                meta = dict(title='Test', code_file='run.ipynb', kernel_type='notebook',
                    language='python', enable_gpu=gpu, dataset_sources=['owner/weights'])
                (folder / 'kernel-metadata.json').write_text(json.dumps(meta))
                (folder / 'run.ipynb').write_text(json.dumps({'cells': [dict(cell_type='code',
                    source=['print(1)\n'], execution_count=7, outputs=[{'old': 'output'}])]}))
                client = Mock()
                client.save_kernel.return_value = SimpleNamespace(error='', invalid_dataset_sources=[], version_number=9)
                result = kaggle_session.push(client, 'owner/notebook', folder, 600)
                request = client.save_kernel.call_args.args[0]
                self.assertEqual(request.kernel_execution_type, KernelExecutionType.SAVE_AND_RUN_ALL)
                self.assertTrue(request.is_private)
                self.assertEqual(request.machine_shape, 'NvidiaTeslaT4' if gpu else 'None')
                self.assertEqual(request.dataset_data_sources, ['owner/weights'])
                self.assertEqual(request.session_timeout_seconds, 600)
                cell = json.loads(request.text)['cells'][0]
                self.assertEqual(cell['source'], 'print(1)\n')
                self.assertEqual(cell['outputs'], [])
                self.assertIsNone(cell['execution_count'])
                self.assertEqual(result, dict(kernel='owner/notebook', version=9))


@unittest.skipUnless(importlib.util.find_spec("kagglesdk"), "Install the optional Kaggle CLI to test its SDK")
class CancellationTests(unittest.TestCase):
    def test_cancellation_waits_for_acknowledgement_and_checks_the_version(self):
        states = iter(("RUNNING", "CANCEL_REQUESTED", "CANCEL_ACKNOWLEDGED"))
        client = Mock()
        client.get_kernel.return_value.metadata.current_version_number = 3
        client.get_kernel_session_status.side_effect = lambda q: SimpleNamespace(status=SimpleNamespace(name=next(states)))
        client.cancel_kernel_session.return_value.error_message = ""
        with patch.object(kaggle_session.time, "sleep"):
            result = kaggle_session.cancel(client, "owner/notebook", 3, 359710527)
        self.assertEqual(result["status"], "CANCEL_ACKNOWLEDGED")
        self.assertEqual(client.cancel_kernel_session.call_args.args[0].kernel_session_id, 359710527)
        self.assertTrue(all(c.args[0].version_label == "" for c in client.get_kernel_session_status.call_args_list))

    def test_backend_rejection_is_reported(self):
        client = Mock()
        client.get_kernel.return_value.metadata.current_version_number = 3
        client.get_kernel_session_status.return_value.status.name = "RUNNING"
        client.cancel_kernel_session.return_value.error_message = "Permission denied"
        with self.assertRaisesRegex(RuntimeError, "Permission denied"):
            kaggle_session.cancel(client, "owner/notebook", 3, 359710527)

    def test_cancellation_request_alone_does_not_claim_shutdown(self):
        client = Mock()
        client.get_kernel.return_value.metadata.current_version_number = 3
        client.get_kernel_session_status.return_value.status.name = "RUNNING"
        client.cancel_kernel_session.return_value.error_message = ""
        with patch.object(kaggle_session.time, "monotonic", side_effect=[0, 61]):
            with self.assertRaisesRegex(RuntimeError, "not confirmed shutdown"):
                kaggle_session.cancel(client, "owner/notebook", 3, 359710527)

    def test_new_notebook_version_cannot_confirm_an_older_sessions_shutdown(self):
        client = Mock()
        client.get_kernel.return_value.metadata.current_version_number = 4
        client.get_kernel_session_status.return_value.status.name = "COMPLETE"
        client.cancel_kernel_session.return_value.error_message = ""
        with self.assertRaisesRegex(RuntimeError, "version changed"):
            kaggle_session.cancel(client, "owner/notebook", 3, 359710527)
        self.assertEqual(client.cancel_kernel_session.call_args.args[0].kernel_session_id, 359710527)


if __name__ == "__main__":
    unittest.main()
