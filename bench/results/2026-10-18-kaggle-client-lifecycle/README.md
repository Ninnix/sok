# Managed clients and session cancellation, 2026-10-18 UTC

OpenCode 2.0.28 and Pi 1.3.0 both listed `strata-iq2_xs` through the new
automatic provider registration. Each command attached to an already running
private Kaggle CPU notebook through authenticated HTTPS. After the client
command exited, the launcher cancelled that notebook's runtime session and
waited for Kaggle's `CANCEL_ACKNOWLEDGED` status.

The [results](results.json) retain the actual model-list lines and session IDs.
OpenCode used `models --standalone`; Pi used `--offline --list-models strata`.
The endpoint implemented authenticated model discovery for these checks; it
did not run model inference. These probes used CPU sessions, not GPU quota.
The earlier [IQ2_XS dual-T4 report](../2026-10-17-kaggle-t4-iq2-xs/README.md)
contains the real model's chat, streaming and tool-call measurements.

The cancellation check used the session ID reported by Kaggle's runtime
container name, rather than the notebook's sequential version number. The
public status endpoint rejected numeric `version_label` selectors in the live
probe; the implementation therefore checks current status and notebook metadata
before confirming shutdown. It reports a verification failure if the version
changes instead of treating another run's completion as evidence.

Local tests cover normal client exit, nonzero exit, failed client startup,
interruption, terminal `SIGHUP`, credential filtering, settings restoration,
backend rejection and cancellation without acknowledgement. Idle tests include
a request longer than the timeout and a short completed request between polls.
The actual Strata HTTP lifecycle and queue tests also passed. No inference-only
GPU-hour accounting is claimed: Kaggle charges allocated GPU-session time.
Cache-monitor tests also cover a temporary Kaggle status timeout, retaining the
running CPU worker until its dataset is ready. Cache preparation uses Kaggle's
injected notebook API credentials and does not need a tunnel.
The cache worker also retries a nonzero processing-status response after upload
without downloading the shards again.
Connection tests retain the same run through a timed-out status request. An
interrupted queued startup without a runtime ID retains its state and reports
shutdown as unconfirmed rather than guessing a cancellation ID.
