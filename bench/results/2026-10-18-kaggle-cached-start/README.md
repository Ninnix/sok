# Cached IQ2_XS startup, 2026-10-18 UTC

The private dataset `ninnix/strata-kaggle-iq2-xs-weights`, version 1, contains
the two selected IQ2_XS shards: 68,026,093,024 bytes. The CPU worker verified
both pinned SHA-256 hashes before upload. Kaggle subsequently reported the
dataset ready. The [setup record](cache-setup.json) retains the verification,
upload timing and dataset status.

This was a one-time CPU preparation, with no GPU allocation. The worker reached
verified files after 24 minutes 49 seconds and spent another 47 minutes
36 seconds uploading and submitting the dataset. Backend processing followed.
An initial HTTP 403 during processing ended that worker's status check; the
uploaded dataset continued processing and became ready. The worker now retries
nonzero processing-status responses without downloading the shards again.

A separate 64 MiB private upload succeeded in 7.76 seconds using only Kaggle's
injected notebook credentials. Its temporary dataset was deleted. The cache
helper therefore uses those credentials directly, without a CPU tunnel or
copying the local user's API key into the worker.

The GPU rerun attaches that weights dataset and the existing sm_75 engine
dataset. It selects IQ2_XS, skips benchmarks, and requests one short response
through Pi 1.3.0 before cancelling the session.

Batch versions 8, 9 and 10 remained queued without executing notebook cells.
The user stopped those runs in Active Events. A separate interactive GPU
attempt also failed to allocate within its 15-minute deadline. Its cancellation
was confirmed through the operation API, with reserved GPU time at zero.
No cached GPU startup speed or new Pi inference result is claimed.

A CPU-only interactive session using the same two datasets reached Jupyter
readiness in 20.66 seconds. Its executed script listed both GGUF shards with
their expected byte sizes and the 89,038,744-byte engine. The Node wrapper
returned a nonzero exit status after printing that list, so this is evidence
of cache availability, not a successful end-to-end client test. The CPU session
was cancelled. Full SHA-256 verification attempts did not yield a saved timing;
controller elapsed times are not verification measurements.

The [attempt records](attempts.json) retain the CPU cache check and the failed
interactive GPU allocation. Cache readiness on a CPU session does not measure
GPU provisioning or loaded model readiness. The reduction from the earlier
startup time remains unknown.

For comparison, the earlier [IQ2_XS run](../2026-10-17-kaggle-t4-iq2-xs/README.md)
took 1,097.24 seconds from its first configuration cell to loaded health,
including 840.46 seconds downloading/verifying weights and 230.04 seconds
starting the loaded server. Those figures exclude Kaggle's allocation queue.
