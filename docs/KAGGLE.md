# Kaggle validation

sok now builds the latest stable release from
[official Strata](https://github.com/Niko1221/Strata) without source changes.
The notebook records its release and exact commit in `startup.json`, together
with whether the persistent build was reused. Future measurements should include
that report, the actual hardware and compiler, and API and tool checks.

The reports below predate the switch to automatic upstream releases. They retain
the source revision, configuration, measurements and raw output from each run.
They are historical evidence, not validation of the latest Strata release.

| Report | What was tested |
| --- | --- |
| [Coder, 2026-10-17](../bench/results/2026-10-17-kaggle-t4-coder/README.md) | Two Tesla T4s, CUDA 12.8.93, GCC 13.3, chat, SSE, authentication, tool calling and local Pi/OpenCode tasks. Nine short responses measured 22.7–30.2 generated tokens/s. |
| [Q2_0, 2026-10-17](../bench/results/2026-10-17-kaggle-t4-q2/README.md) | API and tool checks, nine synthetic requests, memory samples and comparison with Coder. |
| [IQ2_XS, 2026-10-17](../bench/results/2026-10-17-kaggle-t4-iq2-xs/README.md) | API and tool checks, nine synthetic requests and comparison with Q2_0 and Coder. |
| [Client lifecycle, 2026-10-18](../bench/results/2026-10-18-kaggle-client-lifecycle/README.md) | Local client registration and confirmed Kaggle cancellation. |
| [Cached start, 2026-10-18](../bench/results/2026-10-18-kaggle-cached-start/README.md) | Model-dataset preparation and startup behavior. |

Offline tests exercise sok's release selection, first installation, cached rerun,
new-release update and failed-build retry with local Git repositories. They do
not build CUDA or allocate Kaggle hardware. The optional upstream HTTP adapter
check uses an external official checkout selected by `STRATA_SOURCE`.

On Kaggle, validate a release by building on T4 ×2, verifying the selected model,
reaching loaded health, rejecting unauthenticated API requests, and checking
chat, SSE and a tool-result follow-up. Compare startup reports from a first run
and a rerun with Files only persistence enabled to verify build reuse.
