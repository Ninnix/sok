"""Model defaults, family selection and forwarding to the unmodified Strata installer."""
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import connect
import prepare_cache
import strata_models


def catalog():
    families = {name: dict(title=name, tag="" if name == "qwen" else name + "-",
        hf="https://example.invalid/" + name + "/{q}/", file="{q}-{i:05}.gguf")
        for name in ("qwen", "swift", "coder", "unsloth", "future", "packed")}
    families["packed"]["ready"] = True
    models = {"IQ2_XS": {}, "IQ1_M": dict(families=["coder"]),
              "Q8_0": dict(families=["unsloth"], shards=6),
              "New-Quant": dict(families=["future"], shards=3),
              "Packed": dict(families=["packed"])}
    own = dict(qwen36=dict(title="qwen36", tag="qwen36-", hf="https://example.invalid/qwen36/",
        file="{q}.gguf", models={"Q8_0": {}}, sha256={"Q8_0.gguf": (1, "a" * 64)}))
    return SimpleNamespace(FAMILIES=families, MODELS=models, OWN_PATH_FAMILIES=own,
        model_file=lambda f, m, i: f["file"].format(q=m, i=i),
        model_shards=lambda f, m: models[m].get("shards", 2),
        file_pin=lambda url: (1, "a" * 64),
        ready_files=lambda m: [("pack", "owner/model", "pinned", "pack/index.txt", 1, "b" * 64),
                              ("dense", "owner/model", "pinned", "dense.gguf", 2, "c" * 64)],
        ram_gb=lambda: 30, tier_pick=Mock(return_value=("swift", "IQ2_XS")),
        menu_families=lambda hip: list(families),
        recommended_size=Mock(side_effect=lambda f, names, ram, vram: names[0]),
        own_path_size=lambda info, ram: next(iter(info["models"])))


class ModelTests(unittest.TestCase):
    def test_omitted_model_uses_upstream_recommendation(self):
        setup = catalog()
        result = strata_models.select(setup, ram_gb=30, vram_gb=15)
        self.assertEqual((result["family"], result["model"]), ("swift", "IQ2_XS"))
        setup.tier_pick.assert_called_once_with(30, 15)
        setup.recommended_size.assert_called_once_with("swift", ["IQ2_XS"], 30, 15)

    def test_any_catalog_size_and_shard_count_is_accepted(self):
        setup = catalog()
        for model, family, count in (("iq2_xs", "swift", 2), ("IQ1_M", None, 2),
                                     ("q8_0", "unsloth", 6), ("new-quant", "future", 3),
                                     ("q8_0", "qwen36", 1)):
            with self.subTest(model=model, family=family):
                result = strata_models.select(setup, model, family)
                self.assertEqual(len(result["files"]), count)
                self.assertTrue(all(len(item["sha256"]) == 64 for item in result["files"]))
                if family:
                    self.assertEqual(result["family"], family)

    def test_legacy_coder_alias_and_family_specific_default(self):
        setup = catalog()
        self.assertEqual(strata_models.select(setup, "coder")["id"], "coder-iq1_m")
        self.assertEqual(strata_models.select(setup, family="qwen36")["id"], "qwen36-q8_0")

    def test_unknown_model_and_incompatible_family_are_rejected(self):
        for model, family in (("missing", None), ("IQ1_M", "swift"), ("IQ2_XS", "missing")):
            with self.subTest(model=model, family=family), self.assertRaisesRegex(ValueError, "Choose from"):
                strata_models.select(catalog(), model, family)

    def test_ready_packs_preserve_relative_layout_and_pinned_sources(self):
        result = strata_models.select(catalog(), "Packed")
        self.assertTrue(result["ready"])
        self.assertEqual(result["files"][0]["path"], "owner--model/pack/index.txt")
        self.assertEqual(result["files"][0]["url"],
                         "https://huggingface.co/owner/model/resolve/pinned/pack/index.txt")

    def test_prepare_uses_official_installer_and_preserves_its_configuration(self):
        for own_path in (False, True):
            with self.subTest(own_path=own_path), tempfile.TemporaryDirectory() as tmp:
                folder = Path(tmp)
                repo = folder / "strata"
                repo.mkdir()
                (repo / "setup.py").write_text('''import json,os,sys
from pathlib import Path
root=Path(os.environ["STRATA_INSTALL_DIR"])
assert os.environ["STRATA_CONTROL_INSTALL"]=="1"
assert "STRATA_API_KEY" not in os.environ
config=dict(model_name="from-upstream",arch="from-upstream",args=sys.argv[1:])
(root/"strata-selected.json").write_text(json.dumps(config))
''')
                request = dict(install_dir=str(folder / "install"), components_dir=str(folder / "components"),
                    selection=dict(family="future", model="New-Quant", ready=False, own_path=own_path),
                    context=8192, port=8080, kv="q4_0", data_dir=str(folder / "data"), models_dir=str(folder / "models"))
                with patch.dict(os.environ, STRATA_API_KEY="private-test-key"):
                    result = strata_models.prepare(repo, request)
                    result = strata_models.prepare(repo, request)
                self.assertEqual(result["model_name"], "from-upstream")
                self.assertEqual(result["arch"], "from-upstream")
                self.assertIn("--no-start", result["args"])
                self.assertIn("New-Quant", result["args"])
                self.assertEqual("--gpus" in result["args"], not own_path)

    @unittest.skipUnless(os.environ.get("STRATA_SOURCE"), "Set STRATA_SOURCE to check the actual upstream catalog")
    def test_every_model_in_the_installed_upstream_catalog(self):
        setup = strata_models.load_setup(Path(os.environ["STRATA_SOURCE"]))
        for family, info in {**setup.FAMILIES, **setup.OWN_PATH_FAMILIES}.items():
            names = list(info["models"]) if family in setup.OWN_PATH_FAMILIES else [
                m for m, d in setup.MODELS.items() if family in d.get("families", ("qwen", "swift"))]
            for model in names:
                with self.subTest(family=family, model=model):
                    result = strata_models.select(setup, model.lower(), family)
                    self.assertEqual(result["model"], model)
                    self.assertTrue(result["files"])
                    self.assertTrue(all(item["bytes"] > 0 and len(item["sha256"]) == 64 for item in result["files"]))


class LauncherModelTests(unittest.TestCase):
    def test_launch_forwards_default_and_arbitrary_model_family(self):
        for model, family in ((None, None), ("Q8_0", "qwen36")):
            with self.subTest(model=model, family=family), tempfile.TemporaryDirectory() as tmp:
                args = SimpleNamespace(model=model, family=family, state_dir=Path(tmp),
                    kernel="owner/sok", model_dataset=None, stage_model=False, seconds=60, idle_seconds=900)
                session = dict(kernel=args.kernel, launch_id="test-run")
                def push(action, value, folder):
                    meta = json.loads((folder / "kernel-metadata.json").read_text())
                    notebook = json.loads((folder / meta["code_file"]).read_text())
                    code = next(cell for cell in notebook["cells"] if cell["cell_type"] == "code")
                    namespace = {}
                    exec("".join(code["source"]), namespace)
                    self.assertEqual(namespace["MODEL"], model)
                    self.assertEqual(namespace["FAMILY"], family)
                    self.assertTrue(namespace["STRATA_MODELS_SOURCE"])
                    return dict(version=1)
                with patch.object(connect, "control", side_effect=push):
                    connect.launch(args, "public-test-key", session)

    def test_discovery_accepts_new_upstream_model_ids(self):
        for model in ("qwen3.8-flash-next-swift-iq3_s", "qwen3.6-35b-a3b-q8_0", "future-model"):
            with self.subTest(model=model), patch.object(connect.urllib.request, "urlopen",
                    return_value=io.BytesIO(json.dumps(dict(data=[dict(id=model)])).encode())):
                self.assertEqual(connect.discover(dict(STRATA_BASE_URL="https://example.invalid/v1",
                                                      STRATA_API_KEY="test-key")), model)

    def test_family_cache_defaults_do_not_overwrite_other_families(self):
        with tempfile.TemporaryDirectory() as tmp:
            for family in ("qwen", "swift"):
                prepare_cache.save_defaults(SimpleNamespace(model="IQ2_XS", family=family, state_dir=Path(tmp)),
                                            "owner/" + family)
            models = json.loads((Path(tmp) / "defaults.json").read_text())["models"]
            self.assertEqual(models[connect.model_key("iq2_xs", "qwen")]["model_dataset"], "owner/qwen")
            self.assertEqual(models[connect.model_key("IQ2_XS", "swift")]["model_dataset"], "owner/swift")


if __name__ == "__main__":
    unittest.main()
