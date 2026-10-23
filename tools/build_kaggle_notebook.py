#!/usr/bin/env python3
"""Build the self-contained Kaggle notebook from reviewable Python sources."""
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]


def build():
    sections = re.split(r"^# %% (.+)\n", (ROOT / "kaggle_runtime.py").read_text(), flags=re.M)[1:]
    calls = ["", "hardware()", "install()", "download()", "configure()",
             "API_KEY = start_server()", "try:\n    start_tunnel(API_KEY)\nexcept BaseException:\n    stop()\n    raise", ""]
    notes = [
        "Only the selected model is downloaded. IQ2_XS is the default; set MODEL = 'coder' or 'q2' for an alternative. "
        "Enable **Session options → Persistence → Files only** before running. The Kaggle CLI cannot set this UI option. "
        "sok (Strata on Kaggle) installs official Strata without patches. "
        "The default cache is /kaggle/working/sok. Kaggle's saved-output size limit may be smaller than the model; "
        "the model downloads to /kaggle/temp and is ephemeral unless supplied as an attached Kaggle dataset. "
        "Matching attached model shards are reused automatically. Use connect.py for a Kaggle-only remote launch, "
        "or attach STRATA_API_KEY in Add-ons → Secrets for manual remote setup.",
        "Select GPU T4 ×2 and Internet on. Report actual RAM and disk capacity before downloading.",
        "Check the latest stable release of official Strata and build it for sm_75. "
        "Strata's CMake downloads its own dependencies. The checkout, virtual environment and successful build "
        "persist in CACHE_DIR. If that release is already built, installation and compilation are skipped.",
        "Both shards of one model are required: weights and the IQ4_NL n-gram table. "
        "Downloads resume, and every shard is checked against its pinned size and SHA-256. "
        "STAGE_CACHED_MODEL optionally copies attached shards to scratch before checking and loading them.",
        "Use a native pack without duplicating experts.bin, split at layer 24, trim each stage's dense weights, "
        "and start with 8K context and q4_0 KV. MTP is omitted for the initial memory budget.",
        "Bind to localhost, require an API key, wait for loaded health, and verify that unauthenticated requests fail. "
        "Secrets stay in process memory and are never written into the config or notebook output.",
        "Disabled by default while Kaggle permission is unconfirmed. Its AUP prohibits server farming and resource abuse. "
        "If permitted for your use, set TUNNEL = 'localhost.run' and REMOTE_ACCESS_PERMITTED above. "
        "OpenSSH creates an HTTPS endpoint without a tunnel account, domain or token. The hostname changes per session. "
        "connect.py supplies a public key so the generated inference credential can be returned encrypted.",
        "Test models, non-streaming, SSE, finish reasons, a structured tool call and the next conversation turn. "
        "Measure real token counts and engine timings at increasing prompt sizes. "
        "SERVE_SECONDS is bounded; stopping or interrupting this cell shuts down the tunnel and server. "
        "IDLE_SECONDS optionally ends the notebook after no active or queued inference requests. "
        "Kaggle counts allocated GPU-session time, including setup and idle time; unloading weights alone does not save quota. "
        "Run agent_smoke.py on your local computer to test local file and command tools.",
    ]
    benchmark = (ROOT / "benchmark.py").read_text()
    ending = f'''benchmark_path = CACHE_DIR / "benchmark.py"
benchmark_path.write_text({benchmark!r})
import importlib.util
spec = importlib.util.spec_from_file_location("strata_benchmark", benchmark_path)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)
try:
    local_base = f"http://127.0.0.1:{{PORT}}/v1"
    model_id = "strata-" + MODEL
    if RUN_BENCHMARKS:
        result = bench.benchmark(local_base, API_KEY, model_id, BENCHMARK_WORDS,
                                 BENCHMARK_REPEATS, PROCESSES["server"].pid,
                                 OUTPUT_DIR / "benchmark.json")
        if TUNNEL != "none":
            bench.tool_test(PUBLIC_BASE_URL, API_KEY, model_id)
            print("Public HTTPS streaming and tool-call round trip passed.")
    print("Results:", OUTPUT_DIR)
    if SERVE_SECONDS:
        print(f"Serving for at most {{SERVE_SECONDS}} seconds; interrupt this cell to stop.")
        serve_for(SERVE_SECONDS, API_KEY)
finally:
    stop()
    API_KEY = None
'''
    cells = []
    for i, (title, source) in enumerate(zip(sections[::2], sections[1::2])):
        cells.append({"cell_type": "markdown", "metadata": {}, "id": f"section-{i+1}",
                      "source": [f"## {i+1}. {title}\n\n", notes[i] + "\n"]})
        code = source.strip() + "\n\n" + (ending if i == 7 else calls[i]) + "\n"
        cells.append({"cell_type": "code", "metadata": {}, "id": f"code-{i+1}",
                      "execution_count": None, "outputs": [], "source": code.splitlines(keepends=True)})
    return {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}}}


if __name__ == "__main__":
    path = ROOT / "sok.ipynb"
    content = json.dumps(build(), indent=1, ensure_ascii=False) + "\n"
    if "--check" in sys.argv:
        if not path.exists() or path.read_text() != content:
            raise SystemExit("Regenerate with python tools/build_kaggle_notebook.py")
    else:
        path.write_text(content)
