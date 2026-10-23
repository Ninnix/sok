# AGENTS.md

sok means Strata on Kaggle. It installs and builds official Strata releases;
this repository contains no Strata engine or server source and applies no patches.

- `kaggle_runtime.py` implements the notebook. `tools/build_kaggle_notebook.py`
  generates `sok.ipynb`; regenerate it after runtime, `strata_models.py` or benchmark changes.
- `install()` checks the latest stable upstream release each run. The checkout,
  virtual environment and successful build persist in `/kaggle/working/sok`.
  Skip installation and build only when the recorded version and build match
  and the environment and executable exist. Never record a failed build as current.
- Use upstream requirements and CMake. Do not vendor the engine, patch it, or
  substitute a fixed engine dataset for the latest release.
- `strata_models.py` reads the installed release's catalog and calls its unmodified
  `setup.py` to prepare models. Omitted model/family means Strata's default; never
  maintain a second model allowlist. Preserve the upstream family-specific engine options.
- `connect.py` manages local clients and Kaggle sessions. `prepare_cache.py`
  prepares durable model datasets on a CPU worker. Models remain hash-pinned.
- Keep API keys out of notebook source, logs and command arguments. Keep Strata
  bound to localhost and require authentication for tunnels.
- Keep dependencies minimal. The optional local dependency is the Kaggle CLI;
  the runtime uses Python's standard library.
- Run `python tools/build_kaggle_notebook.py --check` and
  `python -m unittest discover -s tools -p 'test_*.py'` before finishing.
  No GPU or downloads are needed. Kaggle SDK checks need `requirements.txt`.
- Report real measurements with the hardware and revision tested. Historical
  reports in `bench/results/` do not establish current-release performance.
