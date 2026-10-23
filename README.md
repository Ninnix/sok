# sok — Strata on Kaggle

<p align="center">
  <img src="logo.png" alt="Strata on Kaggle logo" width="220">
</p>

**Experimental:** Development is active; Kaggle runs can fail.

sok installs and builds the latest stable release of
[Strata](https://github.com/Niko1221/Strata) on Kaggle's two T4 GPUs.
It uses official Strata without patches. This repository contains the notebook,
Kaggle setup and local client tools; Strata's engine and server are downloaded
from upstream.

Pi and OpenCode run on your computer and execute tools there. Kaggle runs
inference through Strata's authenticated OpenAI-compatible API.

## Run the notebook

1. Import [sok.ipynb](sok.ipynb) into Kaggle.
2. Select **GPU T4 ×2**, **Internet on**, and
   **Session options → Persistence → Files only**.
3. Leave `MODEL = None` and `FAMILY = None` for Strata's default, or set a model
   size and optional family from its catalog in the first cell.
4. Run the eight sections in order. The default run benchmarks the localhost
   API and stops. Reports go to `/kaggle/working/strata-results`.

Setup checks GitHub's latest stable Strata release on every run, fetches its tag
and builds the official source for compute capability 7.5. Strata's own
requirements files install its Python and build dependencies; its CMake
configuration fetches its engine dependencies. The image must already provide
Git, a C++ compiler and the CUDA toolkit. sok does not install drivers.

The checkout, virtual environment and build live in `/kaggle/working/sok`.
A successful build records the release, commit, Python version and build options
in `strata-build.json`. When that record matches the latest release and the
Python environment and engine exist, setup skips dependency installation and
compilation. A new release or incomplete build triggers installation and build.
Failed builds leave no success record and are retried on the next run.
Internet access is needed to check the latest release, including on cached runs.

Kaggle must have **Files only** persistence enabled to retain the installation
between sessions. `PERSISTENT_DISK = True` selects the persistent path; it cannot
change Kaggle's UI setting. Set `CACHE_DIR` for another persistent volume.

## Models and storage

Omitting the model lets the installed Strata release choose its default for the
notebook's hardware. To choose explicitly, use any model size in that release's
catalog and, when needed, its family. Names are case insensitive. For example:

```python
MODEL = None                    # Strata's default
FAMILY = None
```

```python
MODEL = "IQ3_S"
FAMILY = "swift"
```

The catalog comes from official Strata, including its Qwen, Swift, Coder,
Unsloth, Gyro, NVFP4 and Qwen3.6 families. See
[Strata's models documentation](https://github.com/Niko1221/Strata/blob/main/docs/MODELS.md)
for its current sizes. The old `coder` and `q2` model aliases still work.
An explicit model remains subject to the hardware and disk requirements checked
by Strata's installer.

Only the selected model's files are downloaded. Strata supplies the filenames,
shard count, pinned URLs, sizes and SHA-256 hashes. Downloads resume from `.part`
files, and every selected file is verified before preparation or inference.
This also covers single-file models, models with more than two shards, and
ready-made packs. The chosen family, size and files are recorded in `startup.json`.

Weights download to `/kaggle/temp/strata-models`, which is ephemeral. For durable
weights, attach a Kaggle dataset containing the selected model's files, preserving
its folder layout. Matching files under `/kaggle/input` are reused automatically;
`MODEL_DIR` can select their directory explicitly. On a larger persistent volume,
set `MODEL_DOWNLOAD_DIR` to that volume.

`STAGE_CACHED_MODEL = True` copies attached files to scratch before verification
and loading. Copies resume, and hashes must pass before the files are used.
This needs space for the selected model plus 4 GiB. The attached dataset remains
the durable copy.

Strata's own `setup.py` prepares the model and generates its engine configuration.
sok supplies the locally compiled engine through Strata's component interface.
The installer selects the pack format, GPU layout and memory options for that
family; sok keeps the server on localhost. Context starts at 8K, and images are
disabled. Packs and any expert conversions use `MODEL_DOWNLOAD_DIR`, so allow
space for those files in addition to the download. Strata may create a separate
expert file for its low-RAM mode. See its
[multi-GPU documentation](https://github.com/Niko1221/Strata/blob/main/docs/MULTI_GPU.md).

## Local clients

The optional Kaggle CLI runs locally. Notebook setup installs Strata's own
requirements, independently of this file.

```bash
python -m venv .venv-kaggle
. .venv-kaggle/bin/activate
python -m pip install -r requirements.txt
python connect.py --install-launchers
```

Set `KAGGLE_API_TOKEN` or save your Kaggle token in `~/.kaggle/access_token`
with mode 0600. Then run `sok-pi` in your project, or:

```bash
sok-opencode -- /path/to/project
sok-pi                                  # Strata's default model
sok-pi --family coder --model IQ1_M
sok-pi --family swift --model IQ3_S
sok-opencode --family qwen36 --model UD-Q4_K_XL -- /path/to/project
python connect.py -- python agent_smoke.py
```

Change the owner in [kernel-metadata.json](kernel-metadata.json), or pass
`--kernel your-name/sok`. The launcher submits a private notebook, reuses an
active run, waits for loaded inference and HTTPS, discovers `/v1/models`, and
registers **sok (Strata on Kaggle)** in the clients. `Kaggle: RUNNING` means the
notebook has started; the model can still be loading.
Model and family choices apply to new runs; attaching reuses the model already loaded.

The launcher temporarily merges the `sok` provider into the clients' JSON
settings, preserving other providers. OpenCode starts with `--standalone` so it
reads the current connection. After confirmed shutdown, the launcher restores
the previous provider and default model, preserving unrelated settings.

```bash
sok-opencode --attach -- /path/to/project  # reuse a running notebook
sok-pi --idle-seconds 600                 # ten idle minutes on a new run
python connect.py --stop                 # stop the recorded session
```

Closing a managed client cancels its recorded Kaggle session and waits for
acknowledgement. The default idle timeout is 15 minutes; `--idle-seconds 0`
disables it. `--seconds 1800` sets the separate serving limit. These settings
apply to new runs. `--keep-running` leaves the notebook running on client exit.
State and connection files live under `~/.cache/sok`, with credentials private
to your user. A failed cancellation retains the session record for retry.

## Durable model cache

Download and upload the model once using a CPU-only Kaggle worker:

```bash
python prepare_cache.py --dataset your-name/sok-default-weights
python prepare_cache.py --family swift --model IQ3_S --dataset your-name/sok-swift-iq3-s
```

This verifies every selected file and creates a private dataset without GPU allocation.
With no model specified, the worker uses Strata's default for a 30 GiB RAM /
15 GiB T4 target, rather than the CPU worker's hardware. Explicit model and family
selection works the same way as in the launcher.
It uses Kaggle's injected notebook upload credentials. Dataset storage and CPU
notebook limits apply. The launcher remembers the dataset in
`~/.cache/sok/defaults.json`, separately for each model/family selection, and
attaches it on subsequent starts. To select a
dataset explicitly, pass `--model-dataset owner/dataset` to `connect.py`.
`--stage-model` stages attached weights to scratch; `--no-stage-model` reads
in place. Compiled engines are kept in the notebook's persistent cache and
checked against the latest Strata release, rather than loaded from datasets.

## Remote access

For permitted remote inference, the launcher opens an outbound HTTPS tunnel
through [localhost.run](https://localhost.run/docs/), using the image's OpenSSH
and OpenSSL clients. No tunnel account, domain or token is needed. The hostname
changes each session. Strata binds to `127.0.0.1` and requires an API key.

The notebook generates its inference key in memory. The launcher sends a public
RSA key; the notebook returns the inference key encrypted with RSA-OAEP/SHA-256.
The private key stays on your computer. `KAGGLE_API_TOKEN` stays local during
inference launches and is removed from the coding client's environment.

For manual remote setup, attach a separate `STRATA_API_KEY` of at least 24
characters in Kaggle Secrets, then set:

```python
TUNNEL = "localhost.run"
REMOTE_ACCESS_PERMITTED = True
SERVE_SECONDS = 1800
```

Remote access remains opt-in. Check [Kaggle's rules](https://www.kaggle.com/aup)
for your intended use. GPU quota includes setup and idle time; only ending the
session releases the GPU allocation. The notebook has no automatic restarts.

For manual client setup, use [pi-config-example.json](pi-config-example.json)
or [opencode-config-example.json](opencode-config-example.json), replacing the
example model ID with the value returned by `/v1/models`. Managed runs
save shell variables for the current endpoint:

```bash
. ~/.cache/sok/connection.env
python benchmark.py --base-url "$STRATA_BASE_URL"
python agent_smoke.py --base-url "$STRATA_BASE_URL"
```

## Development and validation

sok's runtime and benchmark use Python's standard library. Regenerate the
self-contained notebook after changing the runtime, `strata_models.py` or benchmark:

```bash
python tools/build_kaggle_notebook.py
python tools/build_kaggle_notebook.py --check
python -m unittest discover -s tools -p 'test_*.py'
```

Offline tests cover release updates, skipped builds, failed-build recovery,
model defaults and families, resumable downloads, hashes, notebook generation,
streaming, client registration
and session cleanup. With the optional local requirements installed, they also
cover the Kaggle SDK. Set `STRATA_SOURCE` to an official checkout to run the
upstream catalog check and optional HTTP adapter check with its dependencies installed.

[Earlier Kaggle measurements](docs/KAGGLE.md) describe the engine revisions
actually tested. They do not validate the latest automatically selected release.
New release builds and inference need a Kaggle T4 run to validate on that image.
