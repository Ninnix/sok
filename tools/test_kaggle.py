"""Offline checks for sok; STRATA_SOURCE enables an optional upstream HTTP check."""
import contextlib
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import benchmark
import agent_smoke
import connect
import kaggle_runtime as runtime
from tools.build_kaggle_notebook import build


@contextlib.contextmanager
def http_server(handler):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


class DownloadTests(unittest.TestCase):
    def test_resume_and_hash(self):
        data = b"model bytes" * 100

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                offset = int(self.headers["Range"].split("=")[1].split("-")[0])
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {offset}-{len(data)-1}/{len(data)}")
                self.end_headers()
                self.wfile.write(data[offset:])

            def log_message(self, *args):
                pass

        with tempfile.TemporaryDirectory() as tmp, http_server(Handler) as base:
            path = Path(tmp) / "weights.gguf"
            path.with_suffix(".gguf.part").write_bytes(data[:17])
            digest = hashlib.sha256(data).hexdigest()
            runtime.fetch_model_file(base, path, len(data), digest)
            self.assertEqual(path.read_bytes(), data)
            self.assertFalse(path.with_suffix(".gguf.part").exists())
            with patch.object(urllib.request, "urlopen", side_effect=AssertionError("must reuse")):
                runtime.fetch_model_file(base, path, len(data), digest)
            path.write_bytes(b"x" * len(data))
            with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
                runtime.fetch_model_file(base, path, len(data), digest)

    def test_ignored_range_does_not_append(self):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"abcdef")

            def log_message(self, *args):
                pass

        with tempfile.TemporaryDirectory() as tmp, http_server(Handler) as base:
            path = Path(tmp) / "weights.gguf"
            part = path.with_suffix(".gguf.part")
            part.write_bytes(b"abc")
            with self.assertRaisesRegex(RuntimeError, "resume range"):
                runtime.fetch_model_file(base, path, 6, hashlib.sha256(b"abcdef").hexdigest())
            self.assertEqual(part.read_bytes(), b"abc")

    def test_staged_cache_resumes_and_rejects_corruption_before_publication(self):
        files = [b'weights' * 200, b'ngram' * 200]
        variant = dict(id='iq2_xs', family='qwen', model='IQ2_XS', files=[
            dict(path=f'weights-{i}.gguf', url='https://example.invalid/model', bytes=len(data),
                 sha256=hashlib.sha256(data).hexdigest()) for i, data in enumerate(files)])
        for corrupt_source in (False, True):
            with self.subTest(corrupt_source=corrupt_source), tempfile.TemporaryDirectory() as tmp, \
                    patch.object(runtime, 'SELECTED_MODEL', variant), patch.object(runtime, 'select_model', return_value=variant), \
                    patch.object(runtime, 'MODEL_DIR', Path(tmp) / 'input'), \
                    patch.object(runtime, 'MODEL_DOWNLOAD_DIR', Path(tmp) / 'scratch'), \
                    patch.object(runtime, 'OUTPUT_DIR', Path(tmp) / 'results'), \
                    patch.object(runtime, 'STAGE_CACHED_MODEL', True), patch.object(runtime, 'REPORT', {'phases_s': {}}), \
                    patch.object(runtime.shutil, 'disk_usage', return_value=SimpleNamespace(free=8 * 2**30)), \
                    patch.object(urllib.request, 'urlopen', side_effect=AssertionError('cached files must not download')):
                sources, targets = runtime.model_paths(source=True), runtime.model_paths()
                sources[0].parent.mkdir()
                targets[0].parent.mkdir(parents=True)
                for path, data in zip(sources, files):
                    path.write_bytes(data)
                if corrupt_source:
                    sources[0].write_bytes(b'x' * len(files[0]))
                    with self.assertRaisesRegex(RuntimeError, 'SHA-256 mismatch'):
                        runtime.download()
                    self.assertFalse(targets[0].exists())
                    continue
                targets[0].with_suffix('.gguf.part').write_bytes(files[0][:17])
                self.assertEqual(runtime.download(), targets)
                self.assertEqual([path.read_bytes() for path in targets], files)
                self.assertTrue(all(not path.with_suffix('.gguf.part').exists() for path in targets))
                self.assertEqual(runtime.REPORT['model_staged_from'], list(map(str, sources)))
                targets[0].write_bytes(b'x' * len(files[0]))
                with self.assertRaisesRegex(RuntimeError, 'SHA-256 mismatch'):
                    runtime.download()

    def test_model_selection_and_persistence_defaults(self):
        self.assertIsNone(runtime.MODEL)
        self.assertIsNone(runtime.FAMILY)
        self.assertTrue(runtime.PERSISTENT_DISK)
        self.assertTrue(str(runtime.CACHE_DIR).startswith("/kaggle/working/"))
        for count in (1, 2, 3, 6):
            variant = dict(id="any-model", files=[dict(path=f"weights-{i}.gguf") for i in range(count)])
            with self.subTest(count=count), patch.object(runtime, "SELECTED_MODEL", variant):
                self.assertEqual(len(runtime.model_paths()), count)


class NotebookTests(unittest.TestCase):
    def test_notebook_is_current_and_executable(self):
        actual = json.loads((ROOT / "sok.ipynb").read_text())
        self.assertEqual(actual, build())
        cells = [c for c in actual["cells"] if c["cell_type"] == "code"]
        self.assertEqual(len(cells), 8)
        for cell in cells:
            self.assertIsNone(cell["execution_count"])
            self.assertEqual(cell["outputs"], [])
            compile("".join(cell["source"]), "notebook", "exec")

    def test_tunnel_requires_explicit_configuration(self):
        with patch.object(runtime, "TUNNEL", "localhost.run"), \
                patch.object(runtime, "REMOTE_ACCESS_PERMITTED", False):
            with self.assertRaisesRegex(RuntimeError, "AUP"):
                runtime.start_tunnel("test-key")

    def test_tunnel_url_and_encrypted_connection(self):
        event = {"type": "v1.tcpip_forward.register.accepted", "id": "localhost/test.lhr.life"}
        base = runtime.tunnel_url("banner\n" + json.dumps(event))
        self.assertEqual(base, "https://test.lhr.life/v1")
        event["id"] = "localhost/untrusted.example"
        self.assertEqual(runtime.tunnel_url(json.dumps(event)), "")
        with tempfile.TemporaryDirectory() as tmp:
            private, public = connect.client_key(Path(tmp))
            key = "random-test-secret-not-for-source-or-logs"
            output = io.StringIO()
            with patch.object(runtime, "CACHE_DIR", Path(tmp)), \
                    patch.object(runtime, "CLIENT_PUBLIC_KEY", public), \
                    patch.object(runtime, "PUBLIC_BASE_URL", base), contextlib.redirect_stdout(output):
                runtime.publish_connection(key)
            text = output.getvalue()
            self.assertNotIn(key, text)
            self.assertEqual(private.stat().st_mode & 0o777, 0o600)
            for log in (text, "," + json.dumps({"data": text})):
                connection = connect.read_connection(log, private)
                self.assertEqual(connection, {"STRATA_BASE_URL": base, "STRATA_API_KEY": key})

    def test_upstream_configuration_preserves_each_models_engine_options(self):
        for arch, gpu, args in (("qwen35moe", 0, ["--native", "qwen36.gguf", "--mtp", "2"]),
                               ("qwen4exp", [0, 1], ["--pack", "pack", "--native", "first-of-six.gguf", "--kv", "int8"])):
            with self.subTest(arch=arch), tempfile.TemporaryDirectory() as tmp, \
                    patch.object(runtime, "CACHE_DIR", Path(tmp)), \
                    patch.object(runtime, "OUTPUT_DIR", Path(tmp) / "results"), \
                    patch.object(runtime, "MODEL_DIR", Path(tmp) / "model"), \
                    patch.object(runtime, "MODEL_DOWNLOAD_DIR", Path(tmp) / "scratch"), \
                    patch.object(runtime, "SELECTED_MODEL", dict(id="upstream", files=[dict(path="weights.gguf", sha256="a" * 64)])), \
                    patch.object(runtime, "REPORT", dict(strata_revision="source-commit", phases_s={})), \
                    patch.object(runtime.subprocess, "check_output", return_value="strata 0.1.48\n"), \
                    patch.object(runtime, "strata_request", return_value=dict(
                        model_name="upstream-model", arch=arch, gpu=gpu, args=args.copy(), host="0.0.0.0")):
                engine = Path(tmp) / "strata/build/strata"
                engine.parent.mkdir(parents=True)
                engine.write_bytes(b"built-engine")
                runtime.MODEL_DIR.mkdir()
                (runtime.MODEL_DIR / "weights.gguf").write_bytes(b"verified-weights")
                cfg = runtime.configure()
                self.assertEqual(cfg["args"], args)
                self.assertEqual(cfg["gpu"], gpu)
                self.assertEqual(cfg["arch"], arch)
                self.assertEqual(cfg["model_name"], "upstream-model")
                self.assertEqual(cfg["exe"], str(engine))
                self.assertEqual(cfg["host"], "127.0.0.1")
                self.assertNotIn("api_key", cfg)
                verified = Path(tmp) / "scratch/verified/upstream/weights.gguf"
                self.assertTrue(verified.is_symlink())
                self.assertEqual(verified.with_suffix(".gguf.done").read_text(), "sha256 " + "a" * 64)
                self.assertFalse((runtime.MODEL_DIR / "weights.gguf.done").exists())


class StreamingTests(unittest.TestCase):
    def test_failed_benchmark_keeps_measurements_and_error(self):
        key = "test-key-that-must-stay-private"

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{"data":[{"id":"strata-iq2_xs"}]}')

            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                self.send_response(400 if self.path == "/overflow" else 200)
                self.end_headers()
                body = ({"error": {"message": "context exceeded: " + key}} if self.path == "/overflow" else
                        {"choices": [{"finish_reason": "stop", "message": {"content": "OK"}}]})
                self.wfile.write(json.dumps(body).encode())

            def log_message(self, *args):
                pass

        with tempfile.TemporaryDirectory() as tmp, http_server(Handler) as base:
            output = Path(tmp) / "benchmark.json"
            calls = 0

            def chat(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    benchmark.request(base, "/overflow", key, {})
                return {"ttft_s": 1, "elapsed_s": 2, "timings": {}}

            with patch.object(benchmark, "chat", side_effect=chat), \
                    patch.object(benchmark, "tool_test", return_value={"ok": True}), \
                    patch.object(benchmark, "resources", return_value={}), \
                    self.assertRaises(urllib.error.HTTPError):
                benchmark.benchmark(base, key, words=[128, 512], output=output)
            saved = output.read_text()
            report = json.loads(saved)
            self.assertFalse(report["ok"])
            self.assertEqual(len(report["runs"]), 1)
            self.assertTrue(report["tool_test"]["ok"])
            self.assertIn("context exceeded", report["error"])
            self.assertNotIn(key, saved)

    def test_fragmented_tool_arguments(self):
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                events = [
                    {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_1",
                         "function": {"name": "echo", "arguments": '{"text":'}}]}}]},
                    {"choices": [{"delta": {"tool_calls": [{"index": 0,
                         "function": {"arguments": '"ok"}'}}]}, "finish_reason": "tool_calls"}]},
                    {"choices": [], "usage": {"completion_tokens": 9},
                     "timings": {"predicted_per_second": 12}},
                ]
                for event in events:
                    self.wfile.write(("data: " + json.dumps(event) + "\n\n").encode())
                self.wfile.write(b"data: [DONE]\n\n")

            def log_message(self, *args):
                pass

        with http_server(Handler) as base:
            result = benchmark.chat(base, "test-key", [])
        call = result["message"]["tool_calls"][0]
        self.assertEqual(call["id"], "call_1")
        self.assertEqual(json.loads(call["function"]["arguments"]), {"text": "ok"})
        self.assertEqual(result["usage"]["completion_tokens"], 9)
        self.assertEqual(result["timings"]["predicted_per_second"], 12)

    def test_real_strata_http_adapter(self):
        source = os.environ.get("STRATA_SOURCE")
        if not source:
            self.skipTest("Set STRATA_SOURCE to an official Strata checkout for its HTTP adapter check")
        if not all(importlib.util.find_spec(name) for name in ("jinja2", "regex")):
            self.skipTest("Run with Strata's Python dependencies to test its HTTP server")
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        key = "offline-test-authentication-key"
        answer = '<parameter=text>strata-tool-ok</parameter>\n</function>\n</tool_call>'
        scripts = [answer, "strata-tool-ok",
                   "<tool_call>\n<function=read_file>\n</function>\n</tool_call>",
                   "<tool_call>\n<function=write_file>\n<parameter=content>def answer():\n    return 42\n"
                   "</parameter>\n</function>\n</tool_call>",
                   "<tool_call>\n<function=run_test>\n</function>\n</tool_call>",
                   "Fixed and tested."]
        with tempfile.TemporaryFile() as log:
            proc = subprocess.Popen([sys.executable, str(Path(source) / "serve/server.py"), "--engine", "mock",
                                     "--port", str(port), *[x for s in scripts for x in ("--script", s)]],
                                    env=dict(os.environ, STRATA_API_KEY=key), stdout=log, stderr=log)
            base = f"http://127.0.0.1:{port}/v1"
            try:
                for _ in range(100):
                    if proc.poll() is not None:
                        log.seek(0)
                        self.fail(log.read().decode())
                    try:
                        with benchmark.request(base, "/models", key, timeout=1) as response:
                            model = json.load(response)["data"][0]["id"]
                        break
                    except OSError:
                        time.sleep(0.05)
                else:
                    self.fail("Mock server never became ready")
                with self.assertRaises(urllib.error.HTTPError) as error:
                    urllib.request.urlopen(base + "/models")
                self.assertEqual(error.exception.code, 401)
                error.exception.close()
                self.assertTrue(benchmark.tool_test(base, key, model)["ok"])
                result = agent_smoke.smoke(base, key, model)
                self.assertEqual(result["local_tools"], ["read_file", "write_file", "run_test"])
                self.assertTrue(result["ok"])
            finally:
                proc.terminate()
                proc.wait(timeout=10)


if __name__ == "__main__":
    unittest.main()
