
# ===================== replay_driver.py =====================
"""Synchronous replay with policy flag and token-aware backend.

Run examples:
  # Baseline LRU on repetitive short prompts
  python replay_driver.py --policy lru --trace data/repetitive_short.jsonl

  # Extended CA-LRU (fixed w=0.5)
  python replay_driver.py --policy calru --trace data/repetitive_short.jsonl

  # Extended CA-LRU + bandit (online tuning)
  python replay_driver.py --policy calru_bandit --trace data/repetitive_short.jsonl

Token-aware backend parameters:
  --base-lat 0.002        (2 ms base)
  --per-token 0.001       (1 ms per token, using whitespace tokens)

Outputs land in results/<policy>/
"""
from __future__ import annotations
import argparse, json, os, csv, time
from pathlib import Path
import numpy as np

try:
    import psutil
except ImportError:
    psutil = None

from policy import LRUCache, CALRUCache


def estimate_tokens(prompt: str) -> int:
    # simple whitespace token approximation; replace with a real tokenizer later
    return max(1, len(prompt.split()))


def fake_llm_token_aware(prompt: str, base_s: float, per_tok_s: float) -> str:
    sleep_s = base_s + per_tok_s * estimate_tokens(prompt)
    time.sleep(sleep_s)
    return prompt[::-1]


class ResourceSampler:
    def __init__(self, interval: float, out_path: Path):
        self.interval = interval
        self.out_path = out_path
        self.rows = []
        self.t0 = time.perf_counter()
        self.proc = psutil.Process() if psutil else None
        if self.proc:
            self.proc.cpu_percent(interval=None)
        self._last = 0.0

    def maybe_tick(self):
        if not self.proc:
            return
        now = time.perf_counter()
        if now - self._last >= self.interval:
            t = now - self.t0
            rss_mb = self.proc.memory_info().rss / (1024 * 1024)
            cpu = self.proc.cpu_percent(interval=None)
            self.rows.append((t, rss_mb, cpu))
            self._last = now

    def flush(self):
        if not self.rows:
            return
        with open(self.out_path, "w", newline="") as f:
            cw = csv.writer(f)
            cw.writerow(["timestamp_s", "rss_mb", "cpu_percent"])
            cw.writerows(self.rows)


def build_cache(policy: str, capacity: int) -> object:
    if policy == "lru":
        return LRUCache(capacity=capacity)
    if policy == "calru":
        return CALRUCache(capacity=capacity, fixed_w=0.5)
    if policy == "calru_bandit":
        return CALRUCache(capacity=capacity, arms=(0.0, 0.25, 0.5, 0.75, 1.0), eps=0.05)
    raise ValueError(f"Unknown policy: {policy}")


def run(args: argparse.Namespace) -> None:
    out_dir = Path("results") / args.policy
    out_dir.mkdir(parents=True, exist_ok=True)

    cache = build_cache(args.policy, args.capacity)
    sampler = ResourceSampler(interval=0.5, out_path=out_dir / "resources.csv")

    lat = []
    hits = total = 0

    t_wall0 = time.perf_counter()
    with open(args.trace, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            prompt = obj.get("prompt", "")
            if not prompt:
                continue

            t0 = time.perf_counter()
            val, hit = cache.get(
                prompt,
                lambda k: fake_llm_token_aware(k, args.base_lat, args.per_token),
            )
            dt = time.perf_counter() - t0

            lat.append(dt)
            total += 1
            if hit:
                hits += 1

            sampler.maybe_tick()

    wall = time.perf_counter() - t_wall0
    sampler.flush()

    arr = np.asarray(lat, dtype=float)
    mean_lat = float(arr.mean()) if arr.size else 0.0
    p95_lat = float(np.percentile(arr, 95)) if arr.size else 0.0
    p99_lat = float(np.percentile(arr, 99)) if arr.size else 0.0
    hit_rate = (hits / total) if total else 0.0
    qps = (total / wall) if wall > 0 else 0.0

    # write outputs under results/<policy>/
    with open(out_dir / "latencies.csv", "w", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(["latency_ms"])
        cw.writerows([[x * 1000.0] for x in arr])

    summary = (
        f"Policy: {args.policy}\n"
        f"Trace: {args.trace}\n"
        f"Total Requests: {total}\n"
        f"Hit Rate: {hit_rate:.2%}\n"
        f"Mean: {mean_lat*1000:.2f} ms | p95: {p95_lat*1000:.2f} ms | p99: {p99_lat*1000:.2f} ms\n"
        f"QPS: {qps:.2f} | Wall Time: {wall:.2f} s\n"
    )
    with open(out_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write(summary)
    print(summary, end="")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Replay JSONL trace with selectable cache policy")
    p.add_argument("--trace", required=True, help="Path to JSONL trace file")
    p.add_argument("--policy", choices=["lru", "calru", "calru_bandit"], default="calru_bandit")
    p.add_argument("--capacity", type=int, default=512)
    p.add_argument("--base-lat", dest="base_lat", type=float, default=0.002, help="Base backend latency in seconds")
    p.add_argument("--per-token", dest="per_token", type=float, default=0.001, help="Latency per whitespace token in seconds")
    args = p.parse_args()
    run(args)
