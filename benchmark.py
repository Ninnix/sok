#!/usr/bin/env python3
"""Measure the real chat API and check streamed tool calls. Standard library only."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import urllib.error
import urllib.request


def request(base, path, key, body=None, timeout=900):
    headers = {"Authorization": "Bearer " + key, "Content-Type": "application/json"}
    try:
        return urllib.request.urlopen(urllib.request.Request(
            base.rstrip("/") + path, data=json.dumps(body).encode() if body is not None else None,
            headers=headers), timeout=timeout)
    except urllib.error.HTTPError as e:
        detail = e.read(8192).decode("utf-8", errors="replace")
        e.close()
        e.msg = str(e.reason) + ": " + (detail.replace(key, "<redacted>") if key else detail)
        raise


def discover_model(base, key):
    with request(base, "/models", key) as response:
        models = json.load(response)["data"]
    if len(models) != 1 or not models[0].get("id"):
        raise RuntimeError("Expected one running model in /v1/models; specify --model explicitly")
    return models[0]["id"]


def chat(base, key, messages, model="strata", max_tokens=128, tools=None, tool_choice=None):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens,
            "temperature": 0, "reasoning_effort": "none", "stream": True,
            "stream_options": {"include_usage": True}}
    if tools:
        body["tools"] = tools
    if tool_choice:
        body["tool_choice"] = tool_choice
    begin = time.monotonic()
    first = None
    content, calls, usage, timings, finish, done = [], {}, {}, {}, None, False
    with request(base, "/chat/completions", key, body) as response:
        if "text/event-stream" not in response.headers.get("Content-Type", ""):
            raise RuntimeError("Streaming request did not return text/event-stream")
        for raw in response:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                done = True
                break
            event = json.loads(data)
            if event.get("error"):
                raise RuntimeError("Server returned an error during streaming")
            usage = event.get("usage") or usage
            timings = event.get("timings") or timings
            for choice in event.get("choices", []):
                delta = choice.get("delta") or {}
                if first is None and any(delta.get(k) for k in ("content", "reasoning_content", "tool_calls")):
                    first = time.monotonic() - begin
                content.append(delta.get("content") or "")
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
                for part in delta.get("tool_calls") or []:
                    call = calls.setdefault(part["index"], {"id": "", "type": "function",
                                                            "function": {"name": "", "arguments": ""}})
                    call["id"] += part.get("id") or ""
                    for field in ("name", "arguments"):
                        call["function"][field] += (part.get("function") or {}).get(field) or ""
    if not done or finish not in ("stop", "length", "tool_calls") or first is None:
        raise RuntimeError("Incomplete SSE response, missing output, or invalid finish_reason")
    message = {"role": "assistant", "content": "".join(content) or None}
    if calls:
        message["tool_calls"] = [calls[i] for i in sorted(calls)]
        if finish != "tool_calls":
            raise RuntimeError("Tool calls must finish with tool_calls")
    return {"ttft_s": first, "elapsed_s": time.monotonic() - begin, "usage": usage,
            "timings": timings, "finish_reason": finish, "message": message}


def resources(server_pid=None):
    result = {}
    try:
        mem = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
        result["system_available_bytes"] = int(mem["MemAvailable"].split()[0]) * 1024
        for name in ("current", "peak", "max"):
            path = Path(f"/sys/fs/cgroup/memory.{name}")
            if path.exists() and (value := path.read_text().strip()).isdigit():
                result[f"cgroup_memory_{name}_bytes"] = int(value)
        if server_pid:
            pending, total = [int(server_pid)], 0
            while pending:
                pid = pending.pop()
                try:
                    status = Path(f"/proc/{pid}/status").read_text().splitlines()
                    total += next(int(x.split()[1]) * 1024 for x in status if x.startswith("VmRSS:"))
                    pending.extend(map(int, Path(f"/proc/{pid}/task/{pid}/children").read_text().split()))
                except (OSError, StopIteration):
                    pass
            result["server_tree_rss_bytes"] = total
        raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
                                       "--format=csv,noheader,nounits"], text=True, timeout=5)
        result["gpu_memory_mib"] = [list(map(int, line.split(","))) for line in raw.strip().splitlines()]
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return result


def tool_test(base, key, model):
    tools = [{"type": "function", "function": {
        "name": "echo", "description": "Return the supplied text unchanged.",
        "parameters": {"type": "object", "properties": {"text": {"type": "string"}},
                       "required": ["text"], "additionalProperties": False}}}]
    messages = [{"role": "user", "content": "Call echo with text exactly strata-tool-ok."}]
    first = chat(base, key, messages, model, 256, tools,
                 {"type": "function", "function": {"name": "echo"}})
    calls = first["message"].get("tool_calls") or []
    if len(calls) != 1 or not calls[0]["id"] or calls[0]["function"]["name"] != "echo" or \
            json.loads(calls[0]["function"]["arguments"]) != {"text": "strata-tool-ok"}:
        raise RuntimeError("Structured echo tool call was incorrect")
    messages.extend([first["message"], {"role": "tool", "tool_call_id": calls[0]["id"],
                                       "content": "strata-tool-ok"},
                     {"role": "user", "content": "Reply with the tool result only."}])
    second = chat(base, key, messages, model, 128)
    if second["finish_reason"] != "stop" or "strata-tool-ok" not in (second["message"]["content"] or ""):
        raise RuntimeError("Model did not consume the tool result on the next turn")
    return {"ok": True, "call": first, "followup": second}


def benchmark(base, key, model=None, words=(128, 512, 2048), repeats=1, server_pid=None,
              output=None):
    if not key:
        raise ValueError("Set STRATA_API_KEY in the environment")
    with request(base, "/models", key) as response:
        models = json.load(response)
    if model is None:
        if len(models["data"]) != 1:
            raise RuntimeError("Specify --model when the endpoint lists several models")
        model = models["data"][0]["id"]
    if model not in [m["id"] for m in models["data"]]:
        raise RuntimeError(f"Model {model} is absent from /v1/models")
    samples, finished = [], threading.Event()

    def sample():
        while not finished.is_set():
            samples.append(resources(server_pid))
            finished.wait(2)

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    report = {"model": model, "runs": [], "resources_before": resources(server_pid)}
    try:
        body = {"model": model, "messages": [{"role": "user", "content": "Reply with OK."}],
                "max_tokens": 32, "reasoning_effort": "none", "stream": False}
        with request(base, "/chat/completions", key, body) as response:
            answer = json.load(response)
        if answer["choices"][0]["finish_reason"] != "stop" or not answer["choices"][0]["message"]["content"]:
            raise RuntimeError("Non-streamed chat completion failed")
        report["non_streaming_ok"] = True
        report["tool_test"] = tool_test(base, key, model)
        for size in words:
            for repeat in range(repeats):
                text = f"Request {time.time_ns()}.\n" + " ".join(["item"] * size)
                result = chat(base, key, [{"role": "user", "content":
                              text + "\nWrite a Python function that counts the listed items. Explain briefly."}], model)
                result.update(prompt_words=size, repeat=repeat,
                              prompt_tokens_per_s=result["timings"].get("prompt_per_second"),
                              generation_tokens_per_s=result["timings"].get("predicted_per_second"))
                report["runs"].append(result)
                print(json.dumps({k: result[k] for k in ("prompt_words", "ttft_s", "elapsed_s",
                                                       "prompt_tokens_per_s", "generation_tokens_per_s")}))
        report["ok"] = True
    except Exception as e:
        report["ok"] = False
        report["error"] = str(e).replace(key, "<redacted>")
        raise
    finally:
        finished.set()
        thread.join(timeout=6)
        report["resource_samples"] = samples
        report["resources_after"] = resources(server_pid)
        if output:
            Path(output).write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.environ.get("STRATA_BASE_URL", "http://127.0.0.1:8080/v1"))
    parser.add_argument("--model", default=os.environ.get("STRATA_MODEL"), help="default: discover the running model")
    parser.add_argument("--words", type=int, nargs="+", default=[128, 512, 2048])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--server-pid", type=int)
    parser.add_argument("--output", type=Path, default=Path("benchmark-results.json"))
    args = parser.parse_args()
    if args.repeats < 1 or any(n < 1 for n in args.words):
        parser.error("words and repeats must be positive")
    result = benchmark(args.base_url, os.environ.get("STRATA_API_KEY", ""), args.model,
                       args.words, args.repeats, args.server_pid, args.output)
    print(f"Saved {args.output}; rates are null when the engine supplies no timings.")


if __name__ == "__main__":
    main()
