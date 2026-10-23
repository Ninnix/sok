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
3. Keep `MODEL = "iq2_xs"`, or select `"coder"` or `"q2"` in the first cell.
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

Only the selected model is downloaded. Both GGUF shards are required. Downloads
resume from `.part` files; every shard is verified against its pinned byte size
and SHA-256 hash before inference.

| `MODEL` | Model | Total bytes |
| --- | --- | ---: |
| `iq2_xs` | Qwen3.8 Flash Next IQ2_XS (default) | 68,026,093,024 |
| `coder` | Qwen3.8 Flash Next Coder IQ1_M | 58,408,584,928 |
| `q2` | Qwen3.8 Flash Next Q2_0 | 66,423,878,624 |

These models exceed the 20 GB working disk observed in the original Kaggle run.
Weights therefore download to `/kaggle/temp/strata-models`, which is ephemeral.
For durable weights, attach a Kaggle dataset containing the selected pair.
Matching files under `/kaggle/input` are reused automatically; `MODEL_DIR` can
select their directory explicitly. On a larger persistent volume, set
`MODEL_DOWNLOAD_DIR` to that volume.

`STAGE_CACHED_MODEL = True` copies attached shards to scratch before verification
and loading. Copies resume, and hashes must pass before the files are used.
This needs space for the selected model plus 4 GiB. The attached dataset remains
the durable copy.

The starting configuration uses upstream options for two GPU stages split at
layer 24, resident CPU experts, 8K context, q4_0 KV and a 1,024-token prefill
chunk. Native packs avoid duplicating routed experts. MTP is omitted. See
[Strata's multi-GPU documentation](https://github.com/Niko1221/Strata/blob/main/docs/MULTI_GPU.md)
for its supported options.

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
sok-pi --model coder
python connect.py -- python agent_smoke.py
```

Change the owner in [kernel-metadata.json](kernel-metadata.json), or pass
`--kernel your-name/sok`. The launcher submits a private notebook, reuses an
active run, waits for loaded inference and HTTPS, discovers `/v1/models`, and
registers **sok (Strata on Kaggle)** in the clients. `Kaggle: RUNNING` means the
notebook has started; the model can still be loading.

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
python prepare_cache.py --dataset your-name/sok-iq2-xs-weights
```

This verifies the shards and creates a private dataset without GPU allocation.
It uses Kaggle's injected notebook upload credentials. Dataset storage and CPU
notebook limits apply. The launcher remembers the dataset in
`~/.cache/sok/defaults.json` and attaches it on subsequent starts. To select a
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
or [opencode-config-example.json](opencode-config-example.json). Managed runs
save shell variables for the current endpoint:

```bash
. ~/.cache/sok/connection.env
python benchmark.py --base-url "$STRATA_BASE_URL"
python agent_smoke.py --base-url "$STRATA_BASE_URL"
```

## Development and validation

sok's runtime and benchmark use Python's standard library. Regenerate the
self-contained notebook after changing the runtime or benchmark:

```bash
python tools/build_kaggle_notebook.py
python tools/build_kaggle_notebook.py --check
python -m unittest discover -s tools -p 'test_*.py'
```

Offline tests cover release updates, skipped builds, failed-build recovery,
resumable downloads, hashes, notebook generation, streaming, client registration
and session cleanup. With the optional local requirements installed, they also
cover the Kaggle SDK. Set `STRATA_SOURCE` to an official checkout to run the
optional upstream HTTP adapter check with its dependencies installed.

[Earlier Kaggle measurements](docs/KAGGLE.md) describe the engine revisions
actually tested. They do not validate the latest automatically selected release.
New release builds and inference need a Kaggle T4 run to validate on that image.
