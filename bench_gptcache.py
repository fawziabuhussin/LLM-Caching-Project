# bench_gptcache.py
import argparse, json, math, statistics, time, shutil
from collections import Counter
from pathlib import Path
from collections import Counter, OrderedDict
from gptcache import cache, Config
from gptcache.manager import manager_factory
from gptcache.similarity_evaluation.exact_match import ExactMatchEvaluation

# ---------- utils ----------
def now_ms():
    return time.perf_counter() * 1000.0

def pctl(values, pct):
    if not values:
        return 0.0
    k = max(0, min(len(values) - 1, int(math.ceil(pct / 100.0 * len(values)) - 1)))
    return sorted(values)[k]

def load_prompts(trace_path):
    prompts = []
    with open(trace_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
                if isinstance(obj, dict):
                    if "prompt" in obj:
                        prompts.append(obj["prompt"])
                    elif "question" in obj:
                        prompts.append(obj["question"])
                    elif "messages" in obj and obj["messages"]:
                        q = " ".join(
                            m.get("content", "")
                            for m in obj["messages"]
                            if m.get("role") == "user"
                        ).strip()
                        prompts.append(q or s)
                    else:
                        prompts.append(s)
                else:
                    prompts.append(s)
            except Exception:
                prompts.append(s)
    return prompts

# ---------- simple, robust embedding (Python list) ----------
def hashed_embedding(text: str, dim: int = 256):
    # bag-of-words hashed into dim slots, L2-normalized; returns a PYTHON LIST
    v = [0.0] * dim
    for tok in text.lower().split():
        h = (hash(tok) & 0x7FFFFFFF) % dim
        v[h] += 1.0
    n2 = sum(x * x for x in v)
    if n2 > 0:
        n = n2 ** 0.5
        v = [x / n for x in v]
    return v

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--capacity", type=int, default=512)
    ap.add_argument("--base-lat", type=float, default=0.0005, help="seconds")
    ap.add_argument("--per-token", type=float, default=0.0002, help="seconds/token")
    ap.add_argument("--mode", choices=["baseline", "tinylfu_admit", "tinylfu_k3", "window2q", "size_admit"], default="baseline")
    ap.add_argument("--admit-k", type=int, default=2, help="TinyLFU doorkeeper: admit after k-th sighting")
    ap.add_argument("--admit-tok", type=int, default=200, help="Always admit if tokens >= this")
    ap.add_argument("--window-cap", type=int, default=4096, help="2Q window size (keys)")
    ap.add_argument("--dim", type=int, default=256, help="embedding dimension")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    # ----- init GPTCache: SQLite + FAISS; capacity-limited; clean data dir to avoid init-eviction -----
    data_dir = Path(".gptcache_run")
    shutil.rmtree(data_dir, ignore_errors=True)  # fresh start for CI; prevents buggy eviction on init
    data_dir.mkdir(exist_ok=True)

    dm = manager_factory(
        "sqlite,faiss",
        data_dir=str(data_dir),
        vector_params={"dimension": args.dim, "top_k": 1},
        max_size=args.capacity,
    )

    cache.init(
        embedding_func=lambda q: hashed_embedding(q, args.dim),  # returns list -> FAISS sees (1, dim)
        data_manager=dm,
        similarity_evaluation=ExactMatchEvaluation(),            # exact string equality
        config=Config(similarity_threshold=1.0),
    )

    # TinyLFU doorkeeper: admit on 2nd sighting
    from collections import Counter
    freq = Counter()
    window = OrderedDict()   # for window2q

    prompts = load_prompts(args.trace)
    N = len(prompts)
    hits = 0
    lat_ms = []
    t0 = time.perf_counter()

    for q in prompts:
        t_start = now_ms()

        # 1) embed + search top-1
        emb = cache.embedding_func(q)  # list length == args.dim
        candidates = cache.data_manager.search(emb)

        # 2) evaluate hit via QUESTION only (schema-safe)
        hit = False
        if candidates:
            res = candidates[0]
            try:
                scalar = cache.data_manager.get_scalar_data(res)
                sim = cache.similarity_evaluation.evaluation(
                    {"question": q},
                    {"question": getattr(scalar, "question", "")},
                )
                if sim >= cache.config.similarity_threshold:
                    hit = True
                    cache.data_manager.hit_cache_callback(res)
            except Exception:
                hit = False  # treat as miss on any API mismatch

        if hit:
            hits += 1
            lat_ms.append(now_ms() - t_start)
            continue

        # MISS: simulate backend latency and optionally insert
        tok = max(1, len(q.split()))
        sleep_s = args.base_lat + args.per_token * tok
        time.sleep(sleep_s)
        ans = f"ANS({tok})"

        allow_insert = True
        if args.mode == "tinylfu_admit":
            freq[q] += 1
            allow_insert = (freq[q] >= args.admit_k) or (tok >= args.admit_tok)

        elif args.mode == "tinylfu_k3":
            freq[q] += 1
            allow_insert = (freq[q] >= 3) or (tok >= args.admit_tok)

        elif args.mode == "size_admit":
            allow_insert = tok >= args.admit_tok

        elif args.mode == "window2q":
            # 2Q: first sighting goes to window; second sighting within window admits
            seen = window.pop(q, 0)
            if seen >= 1:
                allow_insert = True
            else:
                allow_insert = False
                window[q] = 1
            # trim window to cap
            while len(window) > args.window_cap:
                window.popitem(last=False)

        if allow_insert:
            try:
                cache.data_manager.save(q, ans, emb)
            except Exception:
                pass

        elapsed = now_ms() - t_start + (sleep_s * 1000.0)
        lat_ms.append(elapsed)

    wall = time.perf_counter() - t0
    hit_rate = 100.0 * hits / max(1, N)
    mean_ms = statistics.fmean(lat_ms) if lat_ms else 0.0
    p95_ms = pctl(lat_ms, 95.0)
    p99_ms = pctl(lat_ms, 99.0)
    qps = N / wall if wall > 0 else float("inf")

    outdir = Path(args.outdir) / f"gptcache_{args.mode}_cap{args.capacity}"
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "summary.txt", "w") as s:
        s.write(f"Policy: gptcache_{args.mode}\n")
        s.write(f"Trace: {args.trace}\n")
        s.write(f"Total Requests: {N}\n")
        s.write(f"Hit Rate: {hit_rate:.2f}%\n")
        s.write(f"Mean: {mean_ms:.2f} ms | p95: {p95_ms:.2f} ms | p99: {p99_ms:.2f} ms\n")
        s.write(f"QPS: {qps:.2f} | Wall Time: {wall:.2f} s\n")

    print((outdir / "summary.txt").read_text())

if __name__ == "__main__":
    main()
