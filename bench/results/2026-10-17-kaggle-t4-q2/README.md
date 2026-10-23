# Q2_0 versus Coder on Kaggle dual T4, 2026-10-17 UTC

Q2_0 generated tokens about 14–15% faster on the two larger synthetic prompts.
It processed those prompts about 4–6% slower and used more CPU RAM.
These are three-request means per prompt size, not a general coding-quality
or long-session comparison.

| Prompt tokens | Coder generated tokens/s | Q2_0 generated tokens/s | Q2_0 change |
| ---: | ---: | ---: | ---: |
| 177 | 26.50 | 28.53 | +7.7% |
| 561 | 29.03 | 33.23 | +14.5% |
| 2097 | 28.20 | 32.40 | +14.9% |

| Prompt tokens | Coder prompt tokens/s | Q2_0 prompt tokens/s | Coder TTFT seconds | Q2_0 TTFT seconds |
| ---: | ---: | ---: | ---: | ---: |
| 177 | 120.60 | 122.50 | 1.51 | 1.48 |
| 561 | 268.83 | 257.53 | 2.12 | 2.21 |
| 2097 | 437.10 | 409.97 | 4.84 | 5.15 |

The same prompt text construction was used: a fresh request nonce followed by
128, 512 or 2,048 repetitions of `item`, then a request for a short Python
function. Actual token counts matched across models. Each request allowed at
most 128 generated tokens, with temperature zero and reasoning disabled.
Answers and output lengths differ; per-token speed does not guarantee a shorter
end-to-end answer. See the complete model outputs and finish reasons in the raw
reports. The shortest group had substantial generation-rate variation: Coder
22.7–30.1 and Q2_0 23.3–34.0 tokens/s.

## Hardware and configuration

Separate Kaggle sessions on the same hardware class: two Tesla T4s, 15,360 MiB
each, driver 580.178.04, four vCPUs and a 30 GiB container RAM limit. CUDA 12.8.93.
Both reused engine SHA-256 `0f6029cab4fa28308b3e16c7879c639f899e8b92e8a659afd94a56854ac2308d`,
built from revision `477ac13cbe8e582c20484871ce459d20d82df09d`.

Both used an 8,192-token context, 1,024-token prefill, q4_0 KV, layer split 24,
resident experts, direct PLE I/O, three CPU pool workers and `--spec 2`, without
an MTP model. Each used its corresponding upstream expert profile. Each notebook
downloaded only its selected model's two shards.

## Memory and startup

Peak samples during the benchmark, at two-second intervals:

| Measurement | Coder | Q2_0 |
| --- | ---: | ---: |
| Server process tree RSS, including mapped pages | 9.56 GiB | 21.72 GiB |
| Container current memory, including file cache and other notebook processes | 20.88 GiB | 27.66 GiB |
| GPU 0 memory | 13,737 MiB | 14,481 MiB |
| GPU 1 memory | 14,397 MiB | 14,387 MiB |

Engine logs reported 2.69 GiB of pinned expert complement for Coder and 9.16 GiB
for Q2_0. About 95% and 75% of experts respectively occupied the two GPU caches;
these are expert-count fractions, not the request-weighted hit rates.
Q2_0 fit the 30 GiB limit at this configuration. Larger contexts were not tested.

| Startup phase | Coder seconds | Q2_0 seconds |
| --- | ---: | ---: |
| Install using the verified prebuilt engine | 14.86 | 18.59 |
| Download and verify selected model | 701.24 | 815.39 |
| Pack preparation | 8.24 | 8.61 |
| Server start to loaded health | 140.03 | 220.04 |
| First configuration cell to loaded health | 864.60 | 1062.91 |

These startup totals exclude Kaggle's allocation/image startup before the first
cell and include downloaded weights, not a source compilation. Network and host
variation affect them. Q2_0 contains 66,423,878,624 bytes versus Coder's
58,408,584,928 bytes, including their common 28,800,138,432-byte PLE table.

## Evidence

The [Q2_0 notebook](https://www.kaggle.com/code/ninnix/strata-kaggle-q2-benchmark),
version 1, completed successfully. Authenticated models, non-streamed chat, SSE
termination, finish reasons, a structured echo call and its tool-result follow-up
all passed. Pi/OpenCode coding tasks were not repeated with Q2_0.

- [Q2_0 benchmark](benchmark.json), [startup configuration and hashes](startup.json), [engine log](engine.txt).
- [Computed comparison](comparison.json).
- [Coder baseline and raw reports](../2026-10-17-kaggle-t4-coder/README.md), [Coder engine log](../2026-10-17-kaggle-t4-coder/engine.txt).
