"""Use the installed Strata release's catalog and model preparation code."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys


def load_setup(repo):
    sys.path.insert(0, str(repo))
    spec = importlib.util.spec_from_file_location("sok_upstream_setup", repo / "setup.py")
    setup = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(setup)
    return setup


def select(setup, model=None, family=None, ram_gb=None, vram_gb=15):
    families = {**setup.FAMILIES, **setup.OWN_PATH_FAMILIES}
    family = family.lower() if family else None
    if model and model.lower() == "coder":
        model, family = "IQ1_M", family or "coder"
    elif model and model.lower() == "q2":
        model = "Q2_0"
    if family and family not in families:
        raise ValueError("Unknown Strata family. Choose from: " + ", ".join(families))
    sizes = {name: list(data["models"]) if name in setup.OWN_PATH_FAMILIES else
             sorted((m for m, info in setup.MODELS.items() if name in info.get("families", ("qwen", "swift"))),
                    key=lambda m: bool(setup.MODELS[m].get("experimental")))
             for name, data in families.items()}
    if model:
        offered = [(f, m) for f, names in sizes.items() for m in names
                   if m.lower() == model.lower() and (not family or family == f)]
        if not offered:
            names = sizes[family] if family else sorted({m for names in sizes.values() for m in names})
            raise ValueError("Unknown Strata model. Choose from: " + ", ".join(names))
        family, model = next((pair for pair in offered if pair[0] == "qwen"), offered[0])
    else:
        ram_gb = setup.ram_gb() if ram_gb is None else ram_gb
        tier = setup.tier_pick(ram_gb, vram_gb)
        family = family or (tier[0] if tier else setup.menu_families(False)[0])
        model = (setup.own_path_size(families[family], ram_gb) if family in setup.OWN_PATH_FAMILIES else
                 setup.recommended_size(family, sizes[family], ram_gb, vram_gb))
    info = families[family]
    files = []
    if info.get("ready"):
        for role, repo, revision, path, size, digest in setup.ready_files(model):
            files.append(dict(path=repo.replace("/", "--") + "/" + path,
                url=f"https://huggingface.co/{repo}/resolve/{revision}/{path}", bytes=size, sha256=digest))
    else:
        names = ([info["file"].format(q=model)] if family in setup.OWN_PATH_FAMILIES else
                 [setup.model_file(info, model, i) for i in range(1, setup.model_shards(info, model) + 1)])
        for name in names:
            url = info["hf"].format(q=model) + name
            pin = setup.file_pin(url)
            if pin is None:
                raise ValueError("Strata has no pinned size and hash for " + name)
            files.append(dict(path=name, url=url, bytes=pin[0], sha256=pin[1]))
    details = setup.MODELS.get(model, {}) if family not in setup.OWN_PATH_FAMILIES else {}
    return dict(id=(info.get("tag", "") + model).lower(), model=model, family=family,
                title=info["title"], files=files, ready=bool(info.get("ready")),
                own_path=family in setup.OWN_PATH_FAMILIES,
                requirements=["requirements-nvfp4-local.txt"] if details.get("prepare") else [],
                license=details.get("license", info.get("license", "See the model repository")))


def prepare(repo, request):
    install = Path(request["install_dir"])
    install.mkdir(parents=True, exist_ok=True)
    before = {p: p.stat().st_mtime_ns for p in install.glob("strata-*.json")}
    selection = request["selection"]
    command = [sys.executable, str(repo / "setup.py"), "--setup", "--yes", "--no-start",
               "--family", selection["family"], "--model", selection["model"],
               "--backend", "cuda", "--vision", "no", "--mtp", "off",
               "--context", str(request["context"]), "--port", str(request["port"]),
               "--data-dir", request["data_dir"], "--models-dir", request["models_dir"]]
    if not selection["own_path"]:
        command += ["--gpus", "all", "--kv", request["kv"]]
    if not selection["ready"]:
        command += ["--gguf-dir", request["models_dir"]]
    env = dict(os.environ, STRATA_CONTROL_INSTALL="1", STRATA_INSTALL_DIR=str(install),
               STRATA_COMPONENTS_DIR=request["components_dir"], STRATA_ENGINE_BACKEND="cuda",
               PYTHONUNBUFFERED="1")
    for name in ("PYTHONPATH", "STRATA_API_KEY", "KAGGLE_API_TOKEN", "KAGGLE_KEY", "KG_TOKEN"):
        env.pop(name, None)
    subprocess.run(command, cwd=repo, env=env, stdin=subprocess.DEVNULL, check=True)
    changed = [p for p in install.glob("strata-*.json") if p.stat().st_mtime_ns != before.get(p)]
    if len(changed) != 1:
        raise RuntimeError("Expected one model configuration from Strata setup")
    return json.loads(changed[0].read_text())


def main():
    repo, request_path, reply_path = map(Path, sys.argv[1:])
    request = json.loads(request_path.read_text())
    if request["action"] == "select":
        result = select(load_setup(repo), request.get("model"), request.get("family"),
                        request.get("ram_gb"), request.get("vram_gb", 15))
    elif request["action"] == "prepare":
        result = prepare(repo, request)
    else:
        raise ValueError("Unknown Strata model request")
    reply_path.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
