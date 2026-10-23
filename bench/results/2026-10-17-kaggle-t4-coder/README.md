# Coder on Kaggle dual T4, 2026-10-17

Measured with the private [notebook run, version 3](https://www.kaggle.com/code/ninnix/strata-kaggle-coder-server).
The local notebook sources and raw reports accompany this implementation.

Hardware: two Tesla T4s, 15,360 MiB each, driver 580.178.04, four vCPUs,
31.3 GiB visible host RAM and a 30 GiB container memory limit. CUDA 12.8.93.
Engine revision `477ac13cbe8e582c20484871ce459d20d82df09d`.
Coder IQ1_M, layer split 24, resident expert complement, q4_0 KV,
8,192-token context, 1,024-token prefill, `--spec 2`, no MTP model.

Three streamed requests per input size, temperature zero, reasoning disabled,
at most 128 generated tokens. Each prompt starts with a fresh request nonce
and repeats the ordinary word `item`. These synthetic prompts and short answers
do not establish coding-agent throughput or performance on varied source code.
Token counts and rates below come from server usage and engine timings.

| Input words | Prompt tokens | Prompt tokens/s | Generated tokens/s | TTFT seconds |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 177 | 110.90–125.80 | 22.70–30.10 | 1.44–1.63 |
| 512 | 561 | 266.90–270.50 | 27.70–30.20 | 2.11–2.14 |
| 2048 | 2097 | 430.40–443.00 | 27.30–29.50 | 4.77–4.91 |

Authenticated models, non-streamed chat, SSE termination and finish reasons,
a structured echo call and its tool-result follow-up all passed. This run used
localhost; remote agent validation is recorded separately.

Peak sampled memory during the benchmark (two-second sampling):

- GPU 0: 13,737 MiB; GPU 1: 14,397 MiB.
- Server process tree RSS: 9.56 GiB, including mapped pages.
- Container current memory: 20.88 GiB, including file cache and other notebook processes.
- Container lifetime peak: 28.57 GiB, including earlier setup/download activity;
  this is not the inference process's RSS.

Startup: 14.86 s installation using the verified prebuilt T4 engine,
701.24 s model download and hash verification, 8.24 s pack creation,
140.03 s server start to loaded health. Total 864.60 s (14.41 minutes).
This cold-start total reuses a compiled engine, not downloaded weights.
A preceding source build on Kaggle took 722.29 s; it is not included in this total.

[Raw benchmark](benchmark.json) includes actual model outputs, token counts,
timings and memory samples. [Startup report](startup.json) contains the exact
launch configuration, model hashes and engine-cache hash.
Q2, larger configured context windows and long agent sessions were not tested
in this run.

## Remote coding tasks

A separate Kaggle run opened localhost.run through OpenSSH without a tunnel
account or token. `connect.py` retrieved the generated inference key through
RSA-OAEP ciphertext in the private notebook logs. No Kaggle credential or
private key was sent to the notebook.

[Three repeated tasks](remote-agent.json) passed over HTTPS/SSE. Each task
required four model turns and completed the local `read_file`, `write_file`
and `run_test` tools in order. Elapsed times were 27.45, 21.72 and 8.07 seconds;
the first requests overlapped notebook benchmarking and later requests could
reuse prompt state, so these times are not a controlled throughput comparison.

[Actual client checks](client-validation.json) also passed:

| Client | Completed local tools | Command result | Elapsed seconds |
| --- | --- | --- | ---: |
| Pi 1.3.0 | read, edit, bash | assertion passed, exit 0 | 29.54 |
| OpenCode 2.0.28 | read, edit, shell | assertion passed, exit 0 | 92.66 |

Each client read `answer.py`, changed its return value from 41 to 42 and ran
`python -c` with an assertion on the edited function. Tool events confirmed the
read, edit and successful command, and a separate local assertion checked the
resulting file. Pi used the generated isolated models configuration with
thinking off. OpenCode used the example configuration with `run --standalone
--auto`; its process used the test directory as both working directory and PWD.
These short tasks do not establish stability over long coding sessions.
