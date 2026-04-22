"""perf_test.py — Concurrency sweep against a deployed vLLM endpoint.

Sends N requests at concurrency levels c=1, c=8, c=32 (configurable) and
measures TTFT (when streaming), per-stream tokens/sec, aggregate
tokens/sec, and end-to-end latency.

The prompt is intentionally simple and fixed — we want to compare
serving variants under the same load, not benchmark prompt complexity.
For the build123d-shaped workload, use evaluate_build123d.py with
--concurrency instead.

Usage:
  python perf_test.py \\
      --endpoint http://192.168.44.36:8000 \\
      --served-name Qwen/Qwen3.6-35B-A3B-FP8 \\
      --concurrencies 1,8,32 \\
      --requests-per-c 16 \\
      --max-tokens 256 \\
      --output /workspace/results/perf-fp8-solo.json
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

PROMPT = (
    "Write a short Python function that computes the n-th Fibonacci number "
    "using memoization. Include a one-line docstring. Just the code, no "
    "explanation."
)


def one_request(endpoint: str, served_name: str, max_tokens: int,
                api_key: str | None, stream: bool):
    url = endpoint.rstrip("/") + "/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": served_name,
        "messages": [{"role": "user", "content": PROMPT}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": stream,
    }
    t0 = time.time()
    if not stream:
        r = requests.post(url, json=payload, headers=headers, timeout=600)
        dt = time.time() - t0
        r.raise_for_status()
        data = r.json()
        usage = data.get("usage") or {}
        return {
            "ok": True, "ttft_s": None, "total_s": dt,
            "completion_tokens": usage.get("completion_tokens"),
            "prompt_tokens": usage.get("prompt_tokens"),
        }
    # Streaming path — measure TTFT (time to first content delta)
    with requests.post(url, json=payload, headers=headers, stream=True, timeout=600) as r:
        r.raise_for_status()
        ttft = None
        completion_tokens = 0
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            chunk = line[len("data: "):]
            if chunk.strip() == "[DONE]":
                break
            try:
                obj = json.loads(chunk)
            except Exception:
                continue
            choice = (obj.get("choices") or [{}])[0]
            delta = choice.get("delta") or {}
            content = delta.get("content")
            if content:
                if ttft is None:
                    ttft = time.time() - t0
                # Rough token count: vLLM doesn't always emit per-chunk usage,
                # so we count chunks with content as a lower bound. The total
                # comes from final usage if present.
                completion_tokens += 1
            usage = obj.get("usage")
            if usage:
                completion_tokens = usage.get("completion_tokens", completion_tokens)
        dt = time.time() - t0
        return {"ok": True, "ttft_s": ttft, "total_s": dt,
                "completion_tokens": completion_tokens, "prompt_tokens": None}


def run_sweep(endpoint: str, served_name: str, concurrencies: list[int],
              requests_per_c: int, max_tokens: int, api_key: str | None
              ) -> list[dict]:
    results = []
    for c in concurrencies:
        # c=1 uses streaming so we get TTFT; higher c uses non-streaming
        # because aggregate throughput is what we care about there.
        stream = (c == 1)
        n = requests_per_c
        print(f"\n--- c={c}  n={n}  stream={stream}  max_tokens={max_tokens} ---")
        per: list[dict] = []
        sweep_t0 = time.time()
        with ThreadPoolExecutor(max_workers=c) as pool:
            futures = [
                pool.submit(one_request, endpoint, served_name, max_tokens, api_key, stream)
                for _ in range(n)
            ]
            for fut in as_completed(futures):
                try:
                    per.append(fut.result())
                except Exception as e:
                    per.append({"ok": False, "error": f"{type(e).__name__}: {e}"})
        sweep_dt = time.time() - sweep_t0
        ok = [r for r in per if r.get("ok")]
        n_err = len(per) - len(ok)
        completion_tokens = sum((r.get("completion_tokens") or 0) for r in ok)
        per_req_total = [r["total_s"] for r in ok if r.get("total_s")]
        per_req_total.sort()

        agg_tps = (completion_tokens / sweep_dt) if sweep_dt > 0 else 0.0
        per_stream_tps_mean = (
            sum((r.get("completion_tokens") or 0) / r["total_s"] for r in ok if r.get("total_s")) / len(ok)
            if ok else 0.0
        )
        ttfts = [r["ttft_s"] for r in ok if r.get("ttft_s") is not None]
        ttft_mean = (sum(ttfts) / len(ttfts)) if ttfts else None

        def _p(arr, q):
            if not arr:
                return None
            i = min(len(arr) - 1, int(q * len(arr)))
            return arr[i]

        summary = {
            "concurrency": c,
            "requests": n,
            "stream": stream,
            "max_tokens": max_tokens,
            "errors": n_err,
            "wall_s": round(sweep_dt, 3),
            "completion_tokens_total": completion_tokens,
            "agg_tokens_per_s": round(agg_tps, 2),
            "per_stream_tokens_per_s_mean": round(per_stream_tps_mean, 2),
            "latency_s_mean": round(sum(per_req_total) / len(per_req_total), 3) if per_req_total else None,
            "latency_s_p50": round(_p(per_req_total, 0.50), 3) if per_req_total else None,
            "latency_s_p95": round(_p(per_req_total, 0.95), 3) if per_req_total else None,
            "ttft_s_mean": round(ttft_mean, 3) if ttft_mean is not None else None,
        }
        print(json.dumps(summary, indent=2))
        results.append(summary)
    return results


def main():
    p = argparse.ArgumentParser(description="Concurrency sweep against vLLM /v1/chat/completions")
    p.add_argument("--endpoint", required=True, help="e.g. http://192.168.44.36:8000")
    p.add_argument("--served-name", required=True,
                   help="model name vLLM is serving (e.g. Qwen/Qwen3.6-35B-A3B-FP8)")
    p.add_argument("--concurrencies", default="1,8,32",
                   help="comma-separated concurrency levels (default 1,8,32)")
    p.add_argument("--requests-per-c", type=int, default=16,
                   help="requests per concurrency level (default 16)")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--api-key", default=None)
    p.add_argument("--output", required=True, help="JSON file to write results to")
    p.add_argument("--label", default=None,
                   help="optional label saved into the output (e.g. 'fp8-solo')")
    args = p.parse_args()

    cs = [int(x) for x in args.concurrencies.split(",") if x.strip()]
    print(f"Endpoint: {args.endpoint}  served_name={args.served_name}")
    print(f"Concurrencies: {cs}  requests_per_c={args.requests_per_c}  max_tokens={args.max_tokens}")

    sweep_t0 = time.time()
    results = run_sweep(
        args.endpoint, args.served_name, cs, args.requests_per_c,
        args.max_tokens, args.api_key)
    total_wall = time.time() - sweep_t0

    payload = {
        "endpoint": args.endpoint,
        "served_name": args.served_name,
        "label": args.label,
        "concurrencies": cs,
        "requests_per_c": args.requests_per_c,
        "max_tokens": args.max_tokens,
        "total_wall_s": round(total_wall, 3),
        "sweep": results,
    }
    import os
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
