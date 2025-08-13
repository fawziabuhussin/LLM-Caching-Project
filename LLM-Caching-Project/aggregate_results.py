# aggregate_results.py
import re, os, csv
BASE = "results"
POLICIES = ["lru", "calru", "calru_bandit"]
FIELDS = ["Hit Rate", "Mean", "p95", "p99", "QPS"]

def parse_summary(path):
    m = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            for fld in FIELDS:
                if line.startswith(fld):
                    val = re.findall(r"([0-9]+\\.?[0-9]*)", line)
                    m[fld] = float(val[0]) if "QPS" in fld else float(val[0])
    return m

rows = []
for p in POLICIES:
    s = parse_summary(os.path.join(BASE, p, "summary.txt"))
    rows.append({"policy": p, **s})

# compute deltas vs lru
base = rows[0]
for r in rows[1:]:
    r["ΔHit(%)"] = (r["Hit Rate"] - base["Hit Rate"]) / base["Hit Rate"] * 100 if base["Hit Rate"] else 0.0
    r["Δp95(%)"] = (base["p95"] - r["p95"]) / base["p95"] * 100 if base["p95"] else 0.0

print("policy".ljust(14), "HitRate%".rjust(9), "p95(ms)".rjust(9), "p99(ms)".rjust(9), "QPS".rjust(8), "ΔHit%".rjust(8), "Δp95%".rjust(8))
for r in rows:
    print(f"{r['policy']:<14}{r['Hit Rate']:>9.2f}{r['p95']:>9.2f}{r['p99']:>9.2f}{r['QPS']:>8.2f}{r.get('ΔHit(%)',0):>8.2f}{r.get('Δp95(%)',0):>8.2f}")

os.makedirs(BASE, exist_ok=True)
with open(os.path.join(BASE, "compare.csv"), "w", newline="") as f:
    cw = csv.writer(f)
    cw.writerow(["policy", "hit_rate_pct", "mean_ms", "p95_ms", "p99_ms", "qps", "delta_hit_pct_vs_lru", "delta_p95_pct_vs_lru"])
    for r in rows:
        cw.writerow([r["policy"], r["Hit Rate"], r["Mean"], r["p95"], r["p99"], r["QPS"], r.get("ΔHit(%)",0), r.get("Δp95(%)",0)])
