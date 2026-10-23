# sok — Strata on Kaggle

sok sets up official Strata on Kaggle so coding agents can use remote inference
while executing their tools locally.

- Install the latest stable release from `Niko1221/Strata`, without patches.
- Build the CUDA engine for Kaggle's two T4 GPUs using upstream CMake and requirements.
- Retain the checkout, virtual environment and build on Kaggle's persistent disk.
- Check the latest release each run. Skip dependency installation and compilation
  when that release is already built; rebuild after an update or incomplete build.
- Record success only after a completed build. Retry failures on the next run.
- Use Strata's default when no model is specified, and accept every model/family
  in the installed release's catalog. Download only its files and verify their pinned hashes.
- Reuse attached model datasets; retain resumable downloads and optional staging.
- Use Strata's own server and configuration options, authenticated on localhost.
- Keep the self-contained notebook, local Pi/OpenCode launchers, bounded sessions,
  encrypted connection delivery, diagnostics and measured benchmarks.

Strata owns inference, its APIs and GPU support. sok owns Kaggle installation,
persistence and client connection. Engine changes belong upstream.
