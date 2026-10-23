"""Notebook implementation. tools/build_kaggle_notebook.py copies these sections into the notebook."""

# %% Configuration
if "stop" in globals():
    stop()

import atexit
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

MODEL = "iq2_xs"                # "iq2_xs", "coder" or "q2"; only this model is downloaded
PERSISTENT_DISK = True           # also select Session options > Persistence > Files only in Kaggle
CACHE_DIR = Path("/kaggle/working/sok" if PERSISTENT_DISK else "/kaggle/temp/sok")
MODEL_DIR = None                 # optional attached dataset directory; matching attached datasets are also found automatically
MODEL_DOWNLOAD_DIR = Path("/kaggle/temp/strata-models")  # the selected model exceeds Kaggle's 20 GB working disk
STAGE_CACHED_MODEL = False       # copy attached shards to scratch before verification and inference
OUTPUT_DIR = Path("/kaggle/working/strata-results")
STRATA_REPO = "https://github.com/Niko1221/Strata.git"
STRATA_RELEASE_API = "https://api.github.com/repos/Niko1221/Strata/releases/latest"
CONTEXT = 8192
PREFILL = 1024
PORT = 8080
KV = "q4_0"
EXPERT_MODE = "resident"          # "resident" keeps the non-GPU experts in RAM; "mmap" uses the file cache
CUDA_COMPILER = "/usr/local/cuda/bin/nvcc"
BUILD_JOBS = 2
TUNNEL = "none"                  # "localhost.run" gives an account-free HTTPS endpoint
PUBLIC_BASE_URL = ""             # assigned automatically by the tunnel
CLIENT_PUBLIC_KEY = ""           # public RSA key supplied by connect.py; never put a private key here
LAUNCH_ID = ""                   # public nonce supplied by connect.py to distinguish runs
REMOTE_ACCESS_PERMITTED = False  # enable only after checking Kaggle's rules for your intended use
SERVE_SECONDS = 0                # 0: benchmark and stop; bounded interactive serving, e.g. 1800
IDLE_SECONDS = 900               # finish the notebook after 15 idle minutes; 0 disables it
RUN_BENCHMARKS = True            # interactive launchers skip the benchmark; run benchmark.py separately
BENCHMARK_WORDS = [128, 512, 2048]  # sizes in words, not claimed token counts
BENCHMARK_REPEATS = 1

STARTED = time.monotonic()
PROCESSES = {}
REPORT = {"model": MODEL, "context": CONTEXT, "phases_s": {}}
MODELS = {
    "coder": {
        "repo": "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF",
        "revision": "5348543e0147355ac9cbcb031184a3546350988e",
        "quant": "IQ1_M", "profile": "expert-profile-coder.bin",
        "shards": [(29608446496, "e11083ba855e7666b48ea3f2db6a9c3a20c18751a012cc24f948de91b7087fad"),
                   (28800138432, "316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113")],
    },
    "q2": {
        "repo": "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF",
        "revision": "ed59f92082b1e93c0e96d60a8b11aab089b52f09",
        "quant": "Q2_0", "profile": "expert-profile.bin",
        "shards": [(37623740192, "69820c02ec7d0b45ef2ebb19d6620299db749fe2aded7f39f93c6b88b199b720"),
                   (28800138432, "316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113")],
    },
    "iq2_xs": {
        "repo": "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF",
        "revision": "ed59f92082b1e93c0e96d60a8b11aab089b52f09",
        "quant": "IQ2_XS", "profile": "expert-profile.bin",
        "shards": [(39225954592, "92cee27ae5bbadcd732416a0f7a7f0acc092399dbbe8f5a5efa707c2ec0a49d7"),
                   (28800138432, "316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113")],
    },
}


def secret(name):
    value = os.environ.get(name, "").strip()
    if not value and Path("/kaggle").exists():
        try:
            from kaggle_secrets import UserSecretsClient
            value = UserSecretsClient().get_secret(name).strip()
        except Exception:
            pass
    return value


def run(args, **kwargs):
    if "env" not in kwargs:
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        kwargs["env"] = env
    subprocess.run([str(x) for x in args], check=True, **kwargs)


def model_paths(source=False):
    model = MODELS[MODEL]
    names = [f"Qwen3.8-Flash-Next-GSQ-RCO-{model['quant']}-{i:05}-of-00002.gguf" for i in (1, 2)]
    if STAGE_CACHED_MODEL and not source:
        folder = MODEL_DOWNLOAD_DIR / MODEL
    elif MODEL_DIR:
        folder = Path(MODEL_DIR)
    else:
        attached = [p.parent for p in Path("/kaggle/input").rglob(names[0]) if (p.parent / names[1]).is_file()]
        folder = sorted(attached)[0] if attached else MODEL_DOWNLOAD_DIR / MODEL
    return [folder / name for name in names]


def save_report():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "startup.json").write_text(json.dumps(REPORT, indent=2) + "\n")


def stop_process(name):
    proc = PROCESSES.pop(name, None)
    if proc is None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()


def stop():
    stop_process("tunnel")
    stop_process("server")


atexit.register(stop)

# %% Hardware and environment detection
def announce_session():
    match = re.search(r"-(\d+)-webtier$", os.environ.get("KAGGLE_CONTAINER_NAME", ""))
    if match:
        session = {"session_id": int(match[1]), "launch_id": LAUNCH_ID}
        REPORT.update(session)
        print("SOK_SESSION=" + json.dumps(session), flush=True)


def hardware():
    announce_session()
    if MODEL not in MODELS:
        raise ValueError(f"MODEL must be one of {', '.join(MODELS)}")
    if CONTEXT < 1024 or PREFILL < 128 or not 1 <= PORT <= 65535:
        raise ValueError("Invalid context, prefill size, or port")
    if EXPERT_MODE not in ("resident", "mmap") or KV not in ("int8", "q4_0", "k8v4"):
        raise ValueError("Invalid expert mode or KV format")
    if not 0 <= SERVE_SECONDS <= 10800:
        raise ValueError("SERVE_SECONDS must be between 0 and 10800; Kaggle's own limit still applies")
    if not 0 <= IDLE_SECONDS <= 10800:
        raise ValueError("IDLE_SECONDS must be between 0 and 10800")
    if TUNNEL not in ("none", "localhost.run"):
        raise ValueError('TUNNEL must be "none" or "localhost.run"')
    if TUNNEL != "none" and not REMOTE_ACCESS_PERMITTED:
        raise ValueError("Review Kaggle's AUP before enabling REMOTE_ACCESS_PERMITTED")
    if TUNNEL != "none" and (not shutil.which("ssh") or (CLIENT_PUBLIC_KEY and not shutil.which("openssl"))):
        raise RuntimeError("Remote access needs the image's OpenSSH and OpenSSL clients")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,name,memory.total,compute_cap,driver_version",
        "--format=csv,noheader,nounits"], text=True)
    gpus = [line.split(", ") for line in raw.strip().splitlines()]
    if len(gpus) != 2 or any(float(g[3]) < 7.5 or int(g[2]) < 14000 for g in gpus):
        raise RuntimeError("Select Kaggle's GPU T4 x2 accelerator (two GPUs with at least 14 GB each)")
    mem = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    ram = int(mem["MemTotal"].split()[0]) * 1024
    free = shutil.disk_usage(CACHE_DIR).free
    missing = sum(max(0, size - (p.stat().st_size if p.exists() else
                                p.with_suffix('.gguf.part').stat().st_size
                                if p.with_suffix('.gguf.part').exists() else 0))
                  for p, (size, _) in zip(model_paths(), MODELS[MODEL]["shards"]))
    if MODEL_DIR and any(not p.is_file() or p.stat().st_size != size
                        for p, (size, _) in zip(model_paths(source=True), MODELS[MODEL]['shards'])):
        raise RuntimeError("MODEL_DIR must contain both complete shards of the selected model")
    model_folder = model_paths()[0].parent
    probe = model_folder
    while not probe.exists():
        probe = probe.parent
    model_free = shutil.disk_usage(probe).free
    same_disk = probe.stat().st_dev == CACHE_DIR.stat().st_dev
    if free < 4 * 2**30 or model_free < missing + (4 * 2**30 if same_disk else 0):
        raise RuntimeError(f"Need {missing / 2**30 + 4:.1f} GiB free for model, build and pack; "
                           f"cache has {free / 2**30:.1f} GiB, model disk has {model_free / 2**30:.1f} GiB. "
                           "Attach a model dataset via MODEL_DIR.")
    REPORT.update(gpus=gpus, ram_bytes=ram, cpu_count=os.cpu_count(), disk_free_bytes=free,
                  persistent_disk=PERSISTENT_DISK, cache_dir=str(CACHE_DIR),
                  model_dir=str(model_folder), model_from_attached_dataset=str(model_folder).startswith('/kaggle/input/'))
    print(raw.strip())
    print(f"RAM: {ram / 2**30:.1f} GiB; CPUs: {os.cpu_count()}; free disk: {free / 2**30:.1f} GiB")
    print("Disk persistence: Files only must be selected in Kaggle Session options." if PERSISTENT_DISK
          else "Disk persistence is disabled.")
    print(f"Model directory: {model_folder}. Attach a Kaggle dataset to retain model weights across sessions.")
    save_report()

# %% Dependency installation
def latest_strata_release():
    request = urllib.request.Request(STRATA_RELEASE_API, headers={
        "Accept": "application/vnd.github+json", "User-Agent": "sok"})
    with urllib.request.urlopen(request, timeout=30) as response:
        release = json.load(response)
    tag = release.get("tag_name")
    if not isinstance(tag, str) or not tag or release.get("draft") or release.get("prerelease"):
        raise RuntimeError("GitHub did not return a stable Strata release")
    return tag


def install():
    begin = time.monotonic()
    release = latest_strata_release()
    print(f"Latest Strata release: {release}", flush=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    repo = CACHE_DIR / "strata"
    if not (repo / ".git").exists():
        run(["git", "clone", "--depth=1", "--no-checkout", "--branch", release, STRATA_REPO, repo])
    run(["git", "-C", repo, "fetch", "--depth=1", "origin", f"refs/tags/{release}"])
    revision = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "FETCH_HEAD^{commit}"], text=True).strip()
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    python = CACHE_DIR / "venv/bin/python"
    build = repo / "build"
    options = ["-DSTRATA_ENABLE_CUDA=ON", "-DCMAKE_BUILD_TYPE=Release",
               "-DCMAKE_CUDA_ARCHITECTURES=75", "-DSTRATA_BUILD_TESTS=OFF", "-DSTRATA_MMQ_KQUANTS=OFF"]
    expected = dict(release=release, revision=revision, python=list(sys.version_info[:2]), options=options)
    stamp = CACHE_DIR / "strata-build.json"
    try:
        installed = json.loads(stamp.read_text())
    except (OSError, ValueError):
        installed = None
    reused = (installed == expected and head == revision and (repo / "CMakeLists.txt").is_file()
              and python.is_file() and os.access(python, os.X_OK)
              and (build / "strata").is_file() and os.access(build / "strata", os.X_OK))
    if reused:
        print(f"Strata {release} is already built; skipping installation and build.", flush=True)
    else:
        # A failed update must never leave a success record for a partial build.
        stamp.unlink(missing_ok=True)
        run(["git", "-C", repo, "checkout", "--detach", revision])
        run([sys.executable, "-m", "venv", "--without-pip", "--upgrade", CACHE_DIR / "venv"])
        requirements = [repo / "requirements.txt", repo / "requirements-build.txt"]
        run([sys.executable, "-m", "pip", "--python", python, "install",
             *[arg for path in requirements if path.is_file() for arg in ("-r", path)]])
        nvcc = Path(CUDA_COMPILER) if Path(CUDA_COMPILER).is_file() else shutil.which("nvcc")
        if not nvcc:
            raise RuntimeError("CUDA compiler missing: select a Kaggle GPU image with a CUDA toolkit")
        version = subprocess.check_output([str(nvcc), "--version"], text=True).strip()
        print(version)
        REPORT["cuda_compiler"] = version
        cmake = CACHE_DIR / "venv/bin/cmake"
        run([cmake, "-S", repo, "-B", build, *options, f"-DCMAKE_CUDA_COMPILER={nvcc}"])
        run([cmake, "--build", build, "--target", "strata", "--parallel", str(BUILD_JOBS)])
        if not (build / "strata").is_file() or not os.access(build / "strata", os.X_OK):
            raise RuntimeError("Strata build did not produce an executable")
        temporary = stamp.with_suffix(".tmp")
        temporary.write_text(json.dumps(expected, indent=2) + "\n")
        temporary.replace(stamp)
    REPORT.update(strata_release=release, strata_revision=revision, strata_build_reused=reused)
    REPORT["phases_s"]["install_and_build"] = time.monotonic() - begin
    save_report()

# %% Model download and verification
def stage_model_file(source, path, size):
    """Put cached shards on scratch for the engine's random reads; verification still happens before publication."""
    if source.stat().st_size != size:
        raise RuntimeError(f"Wrong size for cached model: {source}")
    part = path.with_suffix(path.suffix + '.part')
    offset = part.stat().st_size if part.exists() else 0
    if offset > size:
        raise RuntimeError(f"Oversized partial file: {part}")
    print(f"Staging {path.name}: {offset / 1e9:.1f} / {size / 1e9:.1f} GB", flush=True)
    with source.open('rb') as inp, part.open('ab' if offset else 'wb') as out:
        inp.seek(offset)
        last = time.monotonic()
        while chunk := inp.read(8 * 1024**2):
            out.write(chunk)
            if time.monotonic() - last >= 30:
                print(f"Staging {path.name}: {out.tell() / 1e9:.1f} / {size / 1e9:.1f} GB", flush=True)
                last = time.monotonic()
    if part.stat().st_size != size:
        raise RuntimeError(f"Incomplete cached model copy: {part}")


def fetch_model_file(url, path, size, sha256):
    """Resume only byte ranges the server confirms, then verify the complete file before publishing it."""
    part = path.with_suffix(path.suffix + ".part")
    if not path.exists():
        for attempt in range(5):
            offset = part.stat().st_size if part.exists() else 0
            if offset == size:
                break
            if offset > size:
                raise RuntimeError(f"Oversized partial file: {part}")
            try:
                req = urllib.request.Request(url, headers={"Range": f"bytes={offset}-", "Accept-Encoding": "identity"})
                with urllib.request.urlopen(req, timeout=90) as response:
                    if offset and (response.status != 206 or
                                   not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-")):
                        raise RuntimeError("Download server ignored the requested resume range")
                    with part.open("ab" if offset else "wb") as out:
                        last = time.monotonic()
                        while chunk := response.read(8 * 1024**2):
                            out.write(chunk)
                            if time.monotonic() - last > 30:
                                print(f"{path.name}: {out.tell() / 1e9:.1f} / {size / 1e9:.1f} GB", flush=True)
                                last = time.monotonic()
                if part.stat().st_size == size:
                    break
            except (OSError, urllib.error.URLError):
                if attempt == 4:
                    raise RuntimeError(f"Download interrupted: rerun to resume {path.name}") from None
                time.sleep(2)
        candidate = part
    else:
        candidate = path
    if candidate.stat().st_size != size:
        raise RuntimeError(f"Wrong size for {candidate}; rerun to resume an incomplete download")
    h = hashlib.sha256()
    checked, last = 0, time.monotonic()
    print(f"Verifying {path.name}: {size / 1e9:.1f} GB", flush=True)
    with candidate.open("rb") as f:
        while chunk := f.read(16 * 1024**2):
            h.update(chunk)
            checked += len(chunk)
            if time.monotonic() - last >= 30:
                print(f"Verifying {path.name}: {checked / 1e9:.1f} / {size / 1e9:.1f} GB", flush=True)
                last = time.monotonic()
    if h.hexdigest() != sha256:
        raise RuntimeError(f"SHA-256 mismatch: remove {candidate} and rerun the download cell")
    if candidate != path:
        part.replace(path)
    print(f"Verified {path.name}: {size} bytes, SHA-256 {sha256}", flush=True)


def download():
    begin = time.monotonic()
    model = MODELS[MODEL]
    paths = model_paths()
    folder = paths[0].parent
    folder.mkdir(parents=True, exist_ok=True)
    missing = sum(max(0, size - max((p.stat().st_size if p.exists() else 0),
                                  (p.with_suffix('.gguf.part').stat().st_size
                                   if p.with_suffix('.gguf.part').exists() else 0)))
                  for p, (size, _) in zip(paths, model["shards"]))
    if missing and shutil.disk_usage(folder).free < missing + 4 * 2**30:
        raise RuntimeError(f"Need {missing / 2**30 + 4:.1f} GiB free for the selected model and pack. "
                           "Use an attached model dataset or a larger disk; /kaggle/temp is ephemeral.")
    sources = model_paths(source=True)
    for source, path, (size, digest) in zip(sources, paths, model["shards"]):
        if STAGE_CACHED_MODEL and source != path and source.is_file() and not path.exists():
            stage_model_file(source, path, size)
        url = f"https://huggingface.co/{model['repo']}/resolve/{model['revision']}/{model['quant']}/{path.name}"
        fetch_model_file(url, path, size, digest)
    if STAGE_CACHED_MODEL:
        REPORT['model_staged_from'] = [str(source) for source, path in zip(sources, paths) if source != path]
    REPORT["model_files"] = [dict(path=str(p), bytes=s, sha256=h) for p, (s, h) in zip(paths, model["shards"])]
    REPORT["phases_s"]["model_download_and_verify"] = time.monotonic() - begin
    save_report()
    return paths

# %% Strata configuration
def configure():
    begin = time.monotonic()
    repo = CACHE_DIR / "strata"
    python = CACHE_DIR / "venv/bin/python"
    model = MODELS[MODEL]
    shards = model_paths()
    if not all(p.is_file() for p in shards):
        raise RuntimeError("Download and verify both shards first")
    pack = CACHE_DIR / "packs" / MODEL
    run([python, repo / "tools/iq_pack.py", "--gguf", shards[0], "--out", pack])
    cfg = {
        "exe": str(repo / "build/strata"), "cwd": str(repo), "backend": "cuda",
        "gpu": [0, 1], "gpu_order": "as_given", "layer_split": "24",
        "tokenizer": str(pack / "tokenizer"), "model_name": "strata-" + MODEL,
        "host": "127.0.0.1", "port": PORT, "open_browser": False,
        "log": str(OUTPUT_DIR / "engine.log"), "engine_silence_s": 900,
        "args": ["--pack", str(pack), "--native", str(shards[0]), "--ple-gguf", str(shards[1]),
                 "--expert-profile", str(repo / "data" / model["profile"]), "--expert-cache", "auto",
                 "--resident-experts" if EXPERT_MODE == "resident" else "--mmap-experts",
                 "--trim-stage-weights", "--max-context", str(CONTEXT), "--kv", KV,
                 "--prefill", str(PREFILL), "--spec", "2", "--ple-io", "direct", "--pool-workers", "3"],
    }
    path = CACHE_DIR / "server.json"
    path.write_text(json.dumps(cfg, indent=2) + "\n")
    REPORT["configuration"] = cfg
    REPORT["phases_s"]["pack"] = time.monotonic() - begin
    save_report()
    print(f"Configured {cfg['model_name']}: layers 0–23 on GPU 0, 24–47 on GPU 1; context {CONTEXT}")
    return cfg

# %% Inference server startup
def api_json(path, key="", base=None):
    req = urllib.request.Request((base or f"http://127.0.0.1:{PORT}") + path,
                                 headers={"Authorization": "Bearer " + key} if key else {})
    with urllib.request.urlopen(req, timeout=10) as response:
        return json.load(response)


def start_server():
    stop_process("tunnel")
    stop_process("server")
    key = secret("STRATA_API_KEY")
    if not key:
        if TUNNEL != "none" and not CLIENT_PUBLIC_KEY:
            raise RuntimeError("Launch with connect.py or attach a STRATA_API_KEY secret")
        import secrets
        key = secrets.token_urlsafe(32)
    if len(key) < 24 or "," in key or any(c.isspace() for c in key):
        raise ValueError("STRATA_API_KEY must be a single key of at least 24 characters without spaces")
    env = dict(os.environ, STRATA_API_KEY=key, PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1")
    env.pop("PYTHONPATH", None)
    for name in ("KAGGLE_API_TOKEN", "KG_TOKEN", "KAGGLE_KEY", "TUNNEL_TOKEN"):
        env.pop(name, None)
    repo = CACHE_DIR / "strata"
    begin = time.monotonic()
    with (OUTPUT_DIR / "server.log").open("a") as log:
        proc = subprocess.Popen([str(CACHE_DIR / "venv/bin/python"), str(repo / "serve/server.py"),
                                 "--engine", "strata", "--config", str(CACHE_DIR / "server.json"),
                                 "--host", "127.0.0.1", "--port", str(PORT)],
                                stdout=log, stderr=log, env=env, start_new_session=True)
    PROCESSES["server"] = proc
    try:
        while time.monotonic() - begin < 1800:
            if proc.poll() is not None:
                for name in ("server.log", "engine.log"):
                    path = OUTPUT_DIR / name
                    if path.exists():
                        print(name + ":\n" + path.read_text(errors="replace")[-8000:].replace(key, "<redacted>"))
                raise RuntimeError("Strata exited: inspect strata-results/server.log and engine.log")
            try:
                health = api_json("/health", key)
                if health.get("loaded") and health.get("status") == "ok":
                    break
            except (OSError, urllib.error.URLError):
                pass
            print("Waiting for the model to load...", flush=True)
            time.sleep(10)
        else:
            raise RuntimeError("Model load exceeded 30 minutes; inspect the logs")
        try:
            api_json("/v1/models")
        except urllib.error.HTTPError as e:
            if e.code != 401:
                raise RuntimeError(f"Authentication check returned HTTP {e.code}") from None
        else:
            raise RuntimeError("Server accepted an unauthenticated API request")
        api_json("/v1/models", key)
    except BaseException:
        stop()
        raise
    REPORT["phases_s"]["server_start_to_ready"] = time.monotonic() - begin
    REPORT["cold_start_to_ready_s"] = time.monotonic() - STARTED
    save_report()
    print(f"Authenticated API ready at http://127.0.0.1:{PORT}/v1; key kept in memory")
    return key

# %% Secure tunnel startup
def tunnel_url(text):
    for line in text.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.get("type") != "v1.tcpip_forward.register.accepted":
            continue
        host = event.get("id", "").split("/")[-1]
        if re.fullmatch(r"[a-z0-9.-]+", host) and host.endswith((".lhr.life", ".lhr.rocks", ".localhost.run")):
            return "https://" + host + "/v1"
    return ""


def publish_connection(key):
    if not CLIENT_PUBLIC_KEY:
        return
    if not CLIENT_PUBLIC_KEY.startswith("-----BEGIN PUBLIC KEY-----"):
        raise ValueError("CLIENT_PUBLIC_KEY must contain a public RSA key")
    path = CACHE_DIR / "client-public.pem"
    path.write_text(CLIENT_PUBLIC_KEY)
    encrypted = subprocess.check_output([
        "openssl", "pkeyutl", "-encrypt", "-pubin", "-inkey", str(path),
        "-pkeyopt", "rsa_padding_mode:oaep", "-pkeyopt", "rsa_oaep_md:sha256",
        "-pkeyopt", "rsa_mgf1_md:sha256"], input=key.encode())
    connection = {"base_url": PUBLIC_BASE_URL, "encrypted_api_key": base64.b64encode(encrypted).decode(),
                  "launch_id": LAUNCH_ID, "startup": REPORT}
    print("SOK_CONNECTION=" + json.dumps(connection), flush=True)


def start_tunnel(key):
    global PUBLIC_BASE_URL
    stop_process("tunnel")
    if TUNNEL == "none":
        print("Tunnel disabled: testing inference inside the notebook.")
        return
    if TUNNEL != "localhost.run" or not REMOTE_ACCESS_PERMITTED:
        raise RuntimeError("Review Kaggle's AUP before enabling REMOTE_ACCESS_PERMITTED and localhost.run")
    if not shutil.which("ssh"):
        raise RuntimeError("OpenSSH client missing from this Kaggle image")
    env = dict(os.environ)
    for name in ("STRATA_API_KEY", "KAGGLE_API_TOKEN", "KG_TOKEN", "KAGGLE_KEY", "TUNNEL_TOKEN"):
        env.pop(name, None)
    path = OUTPUT_DIR / "tunnel.log"
    with path.open("w") as log:
        proc = subprocess.Popen([
            "ssh", "-F", "/dev/null", "-T", "-o", "BatchMode=yes", "-o", "IdentityAgent=none",
            "-o", "IdentitiesOnly=yes", "-o", "IdentityFile=none", "-o", "StrictHostKeyChecking=accept-new",
            "-o", f"UserKnownHostsFile={CACHE_DIR / 'tunnel-known-hosts'}",
            "-o", "ConnectTimeout=15", "-o", "ExitOnForwardFailure=yes",
            "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=3",
            "-R", f"80:127.0.0.1:{PORT}", "nokey@localhost.run", "--", "--output", "json"],
            env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
    PROCESSES["tunnel"] = proc
    try:
        for _ in range(60):
            if proc.poll() is not None:
                raise RuntimeError("Tunnel exited; inspect strata-results/tunnel.log")
            PUBLIC_BASE_URL = tunnel_url(path.read_text(errors="replace"))
            if PUBLIC_BASE_URL:
                break
            time.sleep(1)
        else:
            raise RuntimeError("Tunnel did not assign an HTTPS URL; inspect strata-results/tunnel.log")
        for _ in range(30):
            if proc.poll() is not None:
                raise RuntimeError("Tunnel exited; inspect strata-results/tunnel.log")
            try:
                api_json("/v1/models", key, PUBLIC_BASE_URL.removesuffix("/v1"))
                break
            except (OSError, urllib.error.URLError):
                time.sleep(2)
        else:
            raise RuntimeError("Account-free tunnel is not reachable; inspect strata-results/tunnel.log")
        try:
            api_json("/v1/models", base=PUBLIC_BASE_URL.removesuffix("/v1"))
        except urllib.error.HTTPError as e:
            if e.code != 401:
                raise RuntimeError("Public endpoint did not return Strata's authentication response") from None
            e.close()
        else:
            raise RuntimeError("Public endpoint accepted an unauthenticated request")
        publish_connection(key)
    except BaseException:
        stop_process("tunnel")
        raise
    REPORT["public_base_url"] = PUBLIC_BASE_URL
    save_report()
    print(f"Remote API: {PUBLIC_BASE_URL}; use your STRATA_API_KEY locally", flush=True)

# %% Endpoint verification and benchmark
def serve_for(seconds, key="", idle_seconds=None):
    """Wait for a bounded user session; never restart Kaggle sessions or fake activity."""
    idle_seconds = IDLE_SECONDS if idle_seconds is None else idle_seconds
    if idle_seconds and not key:
        raise ValueError("Idle shutdown needs the inference API key")
    deadline = time.monotonic() + seconds
    idle_since, previous = time.monotonic(), None
    while time.monotonic() < deadline:
        for name, proc in PROCESSES.items():
            if proc.poll() is not None:
                raise RuntimeError(f"{name} exited; inspect strata-results logs")
        if idle_seconds:
            activity = api_json("/v1/status", key)["activity"]
            signature = (activity["requests"], activity["last_request_at"])
            now = time.monotonic()
            if activity["in_flight"] or signature != previous:
                idle_since = now
            elif now - idle_since >= idle_seconds:
                REPORT["shutdown_reason"] = "idle_timeout"
                save_report()
                print(f"Idle for {idle_seconds} seconds; finishing the notebook to release its GPU session.", flush=True)
                return
            previous = signature
        interval = min(5, max(1, idle_seconds / 4)) if idle_seconds else 10
        time.sleep(min(interval, max(0, deadline - time.monotonic())))
