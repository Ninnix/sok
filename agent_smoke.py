#!/usr/bin/env python3
"""Let the remote model read, edit, and test one temporary local file through tools."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from benchmark import chat

TOOLS = [
    {"type": "function", "function": {
        "name": "read_file", "description": "Read the local file answer.py.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "write_file", "description": "Replace the local file answer.py with the supplied Python source.",
        "parameters": {"type": "object", "properties": {"content": {"type": "string"}},
                       "required": ["content"], "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "run_test", "description": "Run the local command that checks answer.py.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}},
]


def smoke(base, key, model, rounds=12):
    messages = [
        {"role": "system", "content": "You are a coding agent. Use the available local tools to complete the task. "
         "Read the file before editing it, then run the test. Do not claim success until the test passes."},
        {"role": "user", "content": "Fix answer.py so that answer() returns 42. Read it, edit it, run the test, "
         "then briefly report the result."},
    ]
    seen, passed = [], False
    with tempfile.TemporaryDirectory(prefix="strata-agent-") as tmp:
        path = Path(tmp) / "answer.py"
        path.write_text("def answer():\n    return 41\n")
        for turn in range(rounds):
            result = chat(base, key, messages, model, max_tokens=512, tools=TOOLS)
            messages.append(result["message"])
            calls = result["message"].get("tool_calls") or []
            if not calls:
                if passed and all(name in seen for name in ("read_file", "write_file", "run_test")):
                    return {"ok": True, "turns": turn + 1, "local_tools": seen}
                messages.append({"role": "user", "content": "Use read_file, write_file and run_test before finishing."})
                continue
            for call in calls:
                name = call["function"]["name"]
                args = json.loads(call["function"]["arguments"])
                if name == "read_file" and not args:
                    output = path.read_text()
                elif name == "write_file" and set(args) == {"content"} and "read_file" in seen:
                    path.write_text(args["content"])
                    passed = False
                    output = "answer.py updated"
                elif name == "run_test" and not args:
                    # The model edits a file; the fixed test command runs on this machine with credentials removed.
                    env = {k: v for k, v in os.environ.items()
                           if not any(s in k.upper() for s in ("KEY", "TOKEN", "SECRET", "PASSWORD"))}
                    proc = subprocess.run([sys.executable, "-I", "-c",
                                           "import runpy; assert runpy.run_path('answer.py')['answer']() == 42; print('PASS')"],
                                          cwd=tmp, env=env, capture_output=True, text=True, timeout=10)
                    passed = proc.returncode == 0 and "write_file" in seen
                    output = json.dumps({"exit_code": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr})
                else:
                    raise RuntimeError("Invalid tool call or an edit attempted before reading the file")
                seen.append(name)
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": output})
    raise RuntimeError(f"Agent did not finish successfully within {rounds} turns")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.environ.get("STRATA_BASE_URL", "http://127.0.0.1:8080/v1"))
    parser.add_argument("--model", default=os.environ.get("STRATA_MODEL", "strata-iq2_xs"))
    parser.add_argument("--rounds", type=int, default=12, help="maximum turns per task before giving up")
    parser.add_argument("--repeats", type=int, default=1, help="repeat the complete local coding task")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.rounds < 1 or args.repeats < 1:
        parser.error("rounds and repeats must be positive")
    key = os.environ.get("STRATA_API_KEY", "")
    if not key:
        parser.error("Set STRATA_API_KEY in the environment")
    runs = []
    for _ in range(args.repeats):
        begin = time.monotonic()
        result = smoke(args.base_url, key, args.model, args.rounds)
        result["elapsed_s"] = time.monotonic() - begin
        runs.append(result)
        print(json.dumps(result), flush=True)
    if args.output:
        args.output.write_text(json.dumps({"ok": True, "model": args.model, "runs": runs}, indent=2) + "\n")
