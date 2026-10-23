"""Release updates and persistent builds, using local Git repositories and no GPU."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import urllib.error

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import kaggle_runtime as runtime


class InstallTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.upstream = self.folder / "upstream"
        self.upstream.mkdir()
        self.git("init", "-q")
        self.git("config", "user.name", "sok test")
        self.git("config", "user.email", "sok@example.invalid")
        for name in ("requirements.txt", "requirements-build.txt", "CMakeLists.txt"):
            (self.upstream / name).write_text("")
        self.add_release("v1.0.0")
        self.cache = self.folder / "cache"
        self.nvcc = self.folder / "nvcc"
        self.nvcc.write_text("#!/bin/sh\necho 'test CUDA compiler'\n")
        self.nvcc.chmod(0o755)
        self.installs = 0
        self.builds = 0
        self.fail_build = False
        for name, value in (("CACHE_DIR", self.cache), ("STRATA_REPO", self.upstream.as_uri()),
                            ("CUDA_COMPILER", str(self.nvcc)), ("OUTPUT_DIR", self.folder / "results"),
                            ("REPORT", {"phases_s": {}})):
            self.enterContext(patch.object(runtime, name, value))
        self.latest = self.enterContext(patch.object(runtime, "latest_strata_release", return_value="v1.0.0"))
        self.original_run = runtime.run
        self.enterContext(patch.object(runtime, "run", side_effect=self.run_command))
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))

    def git(self, *args):
        return subprocess.check_output(["git", "-C", str(self.upstream), *args], text=True,
                                       stderr=subprocess.DEVNULL).strip()

    def add_release(self, tag):
        (self.upstream / "version").write_text(tag)
        self.git("add", ".")
        self.git("commit", "-qm", tag)
        self.git("tag", "-a", tag, "-m", tag)

    def run_command(self, args, **kwargs):
        args = list(map(str, args))
        if args[1:3] == ["-m", "pip"]:
            self.installs += 1
            self.assertIn(str(self.cache / "strata/requirements.txt"), args)
            self.assertIn(str(self.cache / "strata/requirements-build.txt"), args)
        elif Path(args[0]).name == "cmake":
            if "--build" in args:
                self.builds += 1
                if self.fail_build:
                    raise subprocess.CalledProcessError(1, args)
                engine = self.cache / "strata/build/strata"
                engine.parent.mkdir(parents=True, exist_ok=True)
                engine.write_text("#!/bin/sh\nexit 0\n")
                engine.chmod(0o755)
            else:
                self.assertIn("-DCMAKE_CUDA_ARCHITECTURES=75", args)
        else:
            self.original_run(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **kwargs)

    def test_current_release_skips_dependencies_and_build(self):
        runtime.install()
        self.assertEqual((self.installs, self.builds), (1, 1))
        self.assertFalse(runtime.REPORT["strata_build_reused"])
        runtime.install()
        self.assertEqual((self.installs, self.builds), (1, 1))
        self.assertTrue(runtime.REPORT["strata_build_reused"])
        self.assertEqual(self.latest.call_count, 2)
        stamp = json.loads((self.cache / "strata-build.json").read_text())
        self.assertEqual(stamp["revision"], self.git("rev-parse", "v1.0.0^{commit}"))
        self.assertEqual(runtime.REPORT["strata_release"], "v1.0.0")
        subprocess.run(["git", "-C", str(self.cache / "strata"), "diff", "--exit-code", "HEAD"], check=True)

    def test_new_release_updates_checkout_and_rebuilds(self):
        runtime.install()
        self.add_release("v1.1.0")
        self.latest.return_value = "v1.1.0"
        runtime.install()
        self.assertEqual((self.installs, self.builds), (2, 2))
        self.assertEqual((self.cache / "strata/version").read_text(), "v1.1.0")
        self.assertEqual(runtime.REPORT["strata_revision"], self.git("rev-parse", "HEAD"))
        runtime.install()
        self.assertEqual((self.installs, self.builds), (2, 2))

    def test_failed_update_is_retried(self):
        runtime.install()
        self.add_release("v1.1.0")
        self.latest.return_value = "v1.1.0"
        self.fail_build = True
        with self.assertRaises(subprocess.CalledProcessError):
            runtime.install()
        self.assertFalse((self.cache / "strata-build.json").exists())
        self.fail_build = False
        runtime.install()
        self.assertEqual((self.installs, self.builds), (3, 3))
        self.assertEqual(json.loads((self.cache / "strata-build.json").read_text())["release"], "v1.1.0")

    def test_incomplete_or_damaged_cache_is_rebuilt(self):
        runtime.install()
        for missing in ("strata/build/strata", "venv/bin/python", "strata-build.json"):
            with self.subTest(missing=missing):
                (self.cache / missing).unlink()
                runtime.install()
                self.assertFalse(runtime.REPORT["strata_build_reused"])
        (self.cache / "strata-build.json").write_text("interrupted write")
        runtime.install()
        self.assertEqual((self.installs, self.builds), (5, 5))

    def test_release_lookup_failure_does_not_claim_cache_is_current(self):
        runtime.install()
        stamp = (self.cache / "strata-build.json").read_text()
        self.latest.side_effect = urllib.error.URLError("offline")
        with self.assertRaises(urllib.error.URLError):
            runtime.install()
        self.assertEqual((self.installs, self.builds), (1, 1))
        self.assertEqual((self.cache / "strata-build.json").read_text(), stamp)


class ReleaseTests(unittest.TestCase):
    def test_latest_stable_release_metadata(self):
        for release in ({"tag_name": "v2.0.0"}, {"tag_name": "v2.0.0-rc1", "prerelease": True},
                        {"tag_name": "v2.0.0", "draft": True}, {}, {"tag_name": ""}):
            with self.subTest(release=release), patch.object(runtime.urllib.request, "urlopen",
                    return_value=io.BytesIO(json.dumps(release).encode())) as request:
                if release == {"tag_name": "v2.0.0"}:
                    self.assertEqual(runtime.latest_strata_release(), "v2.0.0")
                else:
                    with self.assertRaisesRegex(RuntimeError, "stable Strata release"):
                        runtime.latest_strata_release()
                self.assertEqual(request.call_args.args[0].full_url,
                                 "https://api.github.com/repos/Niko1221/Strata/releases/latest")


if __name__ == "__main__":
    unittest.main()
