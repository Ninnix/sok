# IQ2_XS, Q2_0 and Coder on Kaggle dual T4, 2026-10-17–18 UTC

IQ2_XS loaded and passed authenticated models, chat, SSE termination, finish
reasons, a structured echo call and its tool-result follow-up. It generated
tokens about 7–10% slower than Q2_0 in these short synthetic tests. It is now
the notebook and launcher default at the user's request; Coder and Q2_0 remain
selectable. Coding quality was not compared.

Three-request means per prompt size:

| Prompt tokens | Coder generated tokens/s | Q2_0 generated tokens/s | IQ2_XS generated tokens/s | IQ2_XS versus Q2_0 |
| ---: | ---: | ---: | ---: | ---: |
| 177 | 26.50 | 28.53 | 26.60 | −6.8% |
| 561 | 29.03 | 33.23 | 30.07 | −9.5% |
| 2097 | 28.20 | 32.40 | 29.17 | −10.0% |

| Prompt tokens | Coder prompt tokens/s | Q2_0 prompt tokens/s | IQ2_XS prompt tokens/s | IQ2_XS TTFT seconds |
| ---: | ---: | ---: | ---: | ---: |
| 177 | 120.60 | 122.50 | 108.27 | 1.67 |
| 561 | 268.83 | 257.53 | 221.40 | 2.57 |
| 2097 | 437.10 | 409.97 | 354.27 | 5.96 |

IQ2_XS prompt processing was about 12–14% slower than Q2_0. Q2_0's TTFT means
were 1.48, 2.21 and 5.15 seconds respectively.

The prompt construction matched the earlier runs: a fresh request nonce,
128, 512 or 2,048 repetitions of `item`, then a request for a short Python
function. Actual prompt token counts matched across all three models.
Temperature was zero, reasoning disabled and maximum output 128 tokens.
Answers and output lengths differ, so per-token speed does not establish
end-to-end task speed. IQ2_XS samples were 22.7/26.6/30.5, 29.5/28.8/31.9 and
27.6/28.6/31.3 generated tokens/s. These small samples are not a sustained
agent or quality benchmark.

## Hardware and configuration

Separate Kaggle sessions used the same hardware class: two Tesla T4s with
15,360 MiB each, driver 580.178.04, four vCPUs and a 30 GiB container RAM limit.
All three reused engine SHA-256
`0f6029cab4fa28308b3e16c7879c639f899e8b92e8a659afd94a56854ac2308d`,
built from revision `477ac13cbe8e582c20484871ce459d20d82df09d` with CUDA 12.8.93.

The settings were unchanged: 8,192-token context, 1,024-token prefill, q4_0 KV,
layer split 24, resident experts, direct PLE I/O, three CPU pool workers and
`--spec 2`, without an MTP model. IQ2_XS and Q2_0 used `expert-profile.bin`;
Coder used its pruned-model profile. Only the selected model's two shards
were downloaded in each notebook.

## Memory and startup

Peak samples during the benchmark, at two-second intervals:

| Measurement | Coder | Q2_0 | IQ2_XS |
| --- | ---: | ---: | ---: |
| Server process tree RSS, including mapped pages | 9.56 GiB | 21.72 GiB | 21.62 GiB |
| Container current memory, including file cache and notebook processes | 20.88 GiB | 27.66 GiB | 28.16 GiB |
| GPU 0 memory | 13,737 MiB | 14,481 MiB | 14,487 MiB |
| GPU 1 memory | 14,397 MiB | 14,387 MiB | 14,397 MiB |

IQ2_XS's engine log reported a 10.70 GiB page-locked expert complement,
versus Q2_0's 9.16 GiB and Coder's 2.69 GiB. Its two GPU caches held 17,702 of
24,576 experts, about 72% by count; this is not a request-weighted hit rate.
The container's lifetime memory peak was 28.56 GiB, including setup and download
cache. Sampled RSS and container memory measure different things and must not
be added together.

The engine reported 474/570 MiB of free VRAM after loading and logged its low
VRAM warning. All short requests completed. Larger contexts, longer agent
sessions and actual Pi/OpenCode tasks were not tested with IQ2_XS. Earlier
Pi/OpenCode task checks used Coder.

| Startup phase | Coder seconds | Q2_0 seconds | IQ2_XS seconds |
| --- | ---: | ---: | ---: |
| Install using the verified prebuilt engine | 14.86 | 18.59 | 17.70 |
| Download and verify selected model | 701.24 | 815.39 | 840.46 |
| Pack preparation | 8.24 | 8.61 | 8.81 |
| Server start to loaded health | 140.03 | 220.04 | 230.04 |
| First configuration cell to loaded health | 864.60 | 1062.91 | 1097.24 |

These totals exclude Kaggle's allocation/image startup and source compilation.
Network and host variation affect setup. IQ2_XS's two files total
68,026,093,024 bytes, including the common 28,800,138,432-byte PLE shard.
Remote GGUF header inspection matched Strata's supported IQ2_XS layout:
35.45 GB of routed experts at 2.35 bits/weight, mixing several quant formats.

## Evidence

The private [IQ2_XS notebook](https://www.kaggle.com/code/ninnix/strata-kaggle-iq2-xs-benchmark),
version 1, completed successfully. The run began on October 17 UTC and its
outputs were collected on October 18 UTC.

- [Benchmark responses, timings and memory samples](benchmark.json).
- [Startup phases, configuration and verified model hashes](startup.json).
- [Engine log](engine.txt), [GGUF header inspection](header-inspection.txt).
- [Computed comparison of all three models](comparison.json).
- [Coder baseline](../2026-10-17-kaggle-t4-coder/README.md) and [Q2_0 comparison](../2026-10-17-kaggle-t4-q2/README.md).
