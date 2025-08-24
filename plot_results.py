# # plot_results.py
# import os, re, csv, math
# from pathlib import Path
# import pandas as pd
# import matplotlib.pyplot as plt

# RESULTS_DIR = Path("results")
# OUT_CSV = RESULTS_DIR / "benchmark_summary.csv"
# PLOTS_DIR = RESULTS_DIR / "plots"
# PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# # If you want MB estimates on the CSV, set the embedding dim you used in bench_gptcache.py
# EMBED_DIM = 512  # 512-dim -> ~2KB per vector (4 bytes * 512)
# BYTES_PER_ITEM = 4 * EMBED_DIM
# MB_PER_ITEM = BYTES_PER_ITEM / (1024 * 1024)

# num_pat = r"([0-9]+(?:\.[0-9]+)?)"

# def parse_summary(path: Path):
#     row = {
#         "policy": None,
#         "trace": None,
#         "total": None,
#         "hit_rate_pct": None,
#         "mean_ms": None,
#         "p95_ms": None,
#         "p99_ms": None,
#         "qps": None,
#         "wall_s": None,
#         "capacity_items": None,
#         "capacity_mb_est": None,
#         "dir": str(path.parent),
#         "file": str(path),
#     }
#     # Pull capacity (items) from directory name if present: *_capNNN
#     m_cap = re.search(r"_cap(\d+)", path.parent.name)
#     if m_cap:
#         cap_items = int(m_cap.group(1))
#         row["capacity_items"] = cap_items
#         row["capacity_mb_est"] = round(cap_items * MB_PER_ITEM, 2)

#     try:
#         txt = path.read_text()
#     except FileNotFoundError:
#         return None

#     for line in txt.splitlines():
#         s = line.strip()
#         if s.startswith("Policy:"):
#             row["policy"] = s.split(":", 1)[1].strip()
#         elif s.startswith("Trace:"):
#             row["trace"] = s.split(":", 1)[1].strip()
#         elif s.startswith("Total Requests"):
#             r = re.search(r"\d+", s)
#             if r: row["total"] = int(r.group())
#         elif s.startswith("Hit Rate"):
#             r = re.search(num_pat, s)
#             if r: row["hit_rate_pct"] = float(r.group(1))
#         elif s.startswith("Mean"):
#             m = re.search(r"Mean:\s*([\d.]+)\s*ms\s*\|\s*p95:\s*([\d.]+)\s*ms\s*\|\s*p99:\s*([\d.]+)\s*ms", s)
#             if m:
#                 row["mean_ms"], row["p95_ms"], row["p99_ms"] = map(float, m.groups())
#         elif s.startswith("QPS"):
#             m = re.search(r"QPS:\s*([\d.]+)\s*\|\s*Wall Time:\s*([\d.]+)\s*s", s)
#             if m:
#                 row["qps"], row["wall_s"] = map(float, m.groups())
#     return row

# def load_all():
#     rows = []
#     if not RESULTS_DIR.exists():
#         print("No results/ directory found.")
#         return pd.DataFrame()
#     for sub in RESULTS_DIR.iterdir():
#         if not sub.is_dir(): 
#             continue
#         # focus on GPTCache runs, but include others if you want
#         if not (sub.name.startswith("gptcache_") or sub.name in ("lru","calru","calru_bandit")):
#             continue
#         summary = sub / "summary.txt"
#         if summary.exists():
#             r = parse_summary(summary)
#             if r: rows.append(r)
#     return pd.DataFrame(rows)

# def tidy_labels(df):
#     # Normalize policy labels for nice legends
#     def short(p):
#         p = (p or "").lower()
#         if p.startswith("gptcache_"):
#             return p.replace("gptcache_", "")
#         return p
#     df["policy_label"] = df["policy"].apply(short)
#     return df

# def plot_metric(df, metric, ylabel, outfile):
#     # one chart per metric; x = capacity_items
#     # we don’t set any explicit colors (matplotlib will choose)
#     fig, ax = plt.subplots(figsize=(6.8, 4.2))
#     for policy, g in df.groupby("policy_label"):
#         g2 = g.dropna(subset=["capacity_items", metric]).sort_values("capacity_items")
#         if g2.empty: 
#             continue
#         ax.plot(g2["capacity_items"], g2[metric], marker="o", label=policy)
#     ax.set_xlabel("Cache capacity (items)")
#     ax.set_ylabel(ylabel)
#     ax.set_title(ylabel + " vs capacity")
#     ax.grid(True, linestyle="--", alpha=0.4)
#     ax.legend(title="policy")
#     fig.tight_layout()
#     fig.savefig(outfile, dpi=160)
#     plt.close(fig)

# def main():
#     df = load_all()
#     if df.empty:
#         print("No summaries found in results/. Run your benchmarks first.")
#         return

#     df = tidy_labels(df)

#     # Sort and write CSV
#     df = df.sort_values(["policy_label","capacity_items"])
#     df.to_csv(OUT_CSV, index=False)
#     print(f"Wrote {OUT_CSV}")

#     # Make plots (only where capacity is known)
#     dfx = df.dropna(subset=["capacity_items"])

#     plot_metric(dfx, "hit_rate_pct", "Hit rate (%)", PLOTS_DIR / "hit_rate_vs_capacity.png")
#     plot_metric(dfx, "mean_ms", "Mean latency (ms)", PLOTS_DIR / "mean_vs_capacity.png")
#     plot_metric(dfx, "p95_ms", "p95 latency (ms)", PLOTS_DIR / "p95_vs_capacity.png")
#     plot_metric(dfx, "p99_ms", "p99 latency (ms)", PLOTS_DIR / "p99_vs_capacity.png")
#     plot_metric(dfx, "qps", "QPS", PLOTS_DIR / "qps_vs_capacity.png")

#     print(f"Saved plots to {PLOTS_DIR.resolve()}")

# if __name__ == "__main__":
#     main()
# plot_results.py  (baseline vs tinylfu_admit only)
import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = Path("results")
OUT_CSV = RESULTS_DIR / "benchmark_summary.csv"
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# If you want MB estimates on the CSV, set the embedding dim used in bench_gptcache.py
EMBED_DIM = 512                      # 512-dim -> ~2KB per vector (4 bytes * 512)
BYTES_PER_ITEM = 4 * EMBED_DIM
MB_PER_ITEM = BYTES_PER_ITEM / (1024 * 1024)

NUM_PAT = r"([0-9]+(?:\.[0-9]+)?)"
KEEP_POLICIES = {"baseline", "tinylfu_admit"}   # <-- only these two appear in plots

def parse_summary(path: Path):
    row = {
        "policy": None,
        "trace": None,
        "total": None,
        "hit_rate_pct": None,
        "mean_ms": None,
        "p95_ms": None,
        "p99_ms": None,
        "qps": None,
        "wall_s": None,
        "capacity_items": None,
        "capacity_mb_est": None,
        "dir": str(path.parent),
        "file": str(path),
    }
    # capacity from folder name *_capNNN
    m_cap = re.search(r"_cap(\d+)", path.parent.name)
    if m_cap:
        cap_items = int(m_cap.group(1))
        row["capacity_items"] = cap_items
        row["capacity_mb_est"] = round(cap_items * MB_PER_ITEM, 2)

    txt = path.read_text()
    for line in txt.splitlines():
        s = line.strip()
        if s.startswith("Policy:"):
            row["policy"] = s.split(":", 1)[1].strip()
        elif s.startswith("Trace:"):
            row["trace"] = s.split(":", 1)[1].strip()
        elif s.startswith("Total Requests"):
            r = re.search(r"\d+", s)
            if r: row["total"] = int(r.group())
        elif s.startswith("Hit Rate"):
            r = re.search(NUM_PAT, s)
            if r: row["hit_rate_pct"] = float(r.group(1))
        elif s.startswith("Mean"):
            m = re.search(r"Mean:\s*([\d.]+)\s*ms\s*\|\s*p95:\s*([\d.]+)\s*ms\s*\|\s*p99:\s*([\d.]+)\s*ms", s)
            if m:
                row["mean_ms"], row["p95_ms"], row["p99_ms"] = map(float, m.groups())
        elif s.startswith("QPS"):
            m = re.search(r"QPS:\s*([\d.]+)\s*\|\s*Wall Time:\s*([\d.]+)\s*s", s)
            if m:
                row["qps"], row["wall_s"] = map(float, m.groups())
    return row

def load_all():
    rows = []
    if not RESULTS_DIR.exists():
        print("No results/ directory found.")
        return pd.DataFrame()
    for sub in RESULTS_DIR.iterdir():
        if not sub.is_dir():
            continue
        # only GPTCache runs
        if not sub.name.startswith("gptcache_"):
            continue
        summary = sub / "summary.txt"
        if summary.exists():
            rows.append(parse_summary(summary))
    return pd.DataFrame(rows)

def tidy_and_filter(df: pd.DataFrame) -> pd.DataFrame:
    # normalize to short labels and filter to baseline/tinylfu_admit
    def short(p):
        p = (p or "").lower()
        return p.replace("gptcache_", "")
    df["policy_label"] = df["policy"].apply(short)
    df = df[df["policy_label"].isin(KEEP_POLICIES)].copy()
    return df

def plot_metric(df, metric, ylabel, outfile):
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    for policy in ["baseline", "tinylfu_admit"]:
        g = df[df["policy_label"] == policy].dropna(subset=["capacity_items", metric])
        g = g.sort_values("capacity_items")
        if g.empty:
            continue
        ax.plot(g["capacity_items"], g[metric], marker="o", label=policy)
    ax.set_xlabel("Cache capacity (items)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs capacity")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(title="policy")
    fig.tight_layout()
    fig.savefig(outfile, dpi=160)
    plt.close(fig)

def main():
    df = load_all()
    if df.empty:
        print("No summaries found in results/. Run your benchmarks first.")
        return

    df = tidy_and_filter(df)

    # write CSV (filtered to the two policies)
    df = df.sort_values(["policy_label", "capacity_items"])
    df.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV}")

    dfx = df.dropna(subset=["capacity_items"])
    plot_metric(dfx, "hit_rate_pct", "Hit rate (%)", PLOTS_DIR / "hit_rate_vs_capacity.png")
    plot_metric(dfx, "mean_ms", "Mean latency (ms)", PLOTS_DIR / "mean_vs_capacity.png")
    plot_metric(dfx, "p95_ms", "p95 latency (ms)", PLOTS_DIR / "p95_vs_capacity.png")
    plot_metric(dfx, "p99_ms", "p99 latency (ms)", PLOTS_DIR / "p99_vs_capacity.png")
    plot_metric(dfx, "qps", "QPS", PLOTS_DIR / "qps_vs_capacity.png")

    print(f"Saved plots to {PLOTS_DIR.resolve()}")

if __name__ == "__main__":
    main()