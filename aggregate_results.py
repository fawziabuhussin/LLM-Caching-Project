#!/usr/bin/env python3
import os, re, csv

BASE = "results"
NUM = r"([0-9]+(?:\.[0-9]+)?)"

def parse_summary(path):
    m = {
        "policy": None, "trace": None, "total": None,
        "hit_rate_pct": None, "mean_ms": None, "p95_ms": None, "p99_ms": None,
        "qps": None, "wall_s": None,
    }
    try:
        with open(path, "r") as f:
            for line in f:
                s = line.strip()
                if s.startswith("Policy:"):
                    m["policy"] = s.split(":", 1)[1].strip()
                elif s.startswith("Trace:"):
                    m["trace"] = s.split(":", 1)[1].strip()
                elif s.startswith("Total Requests"):
                    r = re.search(r"\d+", s)
                    if r: m["total"] = int(r.group())
                elif s.startswith("Hit Rate"):
                    r = re.search(NUM, s)
                    if r: m["hit_rate_pct"] = float(r.group(1))
                elif s.startswith("Mean"):
                    # "Mean: X ms | p95: Y ms | p99: Z ms"
                    vals = re.findall(NUM, s)
                    if len(vals) >= 3:
                        m["mean_ms"], m["p95_ms"], m["p99_ms"] = map(float, vals[:3])
                elif s.startswith("QPS"):
                    # "QPS: A | Wall Time: B s"
                    vals = re.findall(NUM, s)
                    if len(vals) >= 2:
                        m["qps"], m["wall_s"] = map(float, vals[:2])
    except FileNotFoundError:
        pass
    return m

def main():
    rows = []
    if not os.path.isdir(BASE):
        print("No results/ directory found.")
        return

    for d in sorted(os.listdir(BASE)):
        path = os.path.join(BASE, d, "summary.txt")
        if not os.path.isfile(path):  # skip non-policy dirs
            continue
        s = parse_summary(path)
        if not s.get("policy"):  # fallback to folder name
            s["policy"] = d
        rows.append(s)

    # Find LRU for deltas
    lru = next((r for r in rows if r.get("policy","").lower() == "lru"), None)
    for r in rows:
        if lru and lru.get("p95_ms") not in (None, 0):
            r["delta_hit_pp_vs_lru"] = (r.get("hit_rate_pct") or 0.0) - (lru.get("hit_rate_pct") or 0.0)
            r["delta_p95_pct_vs_lru"] = 100.0 * (((r.get("p95_ms") or 0.0) - lru["p95_ms"]) / lru["p95_ms"])
        else:
            r["delta_hit_pp_vs_lru"] = None
            r["delta_p95_pct_vs_lru"] = None

    hdr = ["policy","trace","total","hit_rate_pct","mean_ms","p95_ms","p99_ms","qps","wall_s","delta_hit_pp_vs_lru","delta_p95_pct_vs_lru"]
    print("\t".join(hdr))
    for r in rows:
        print("\t".join(str(r.get(k, "")) for k in hdr))

    out_csv = os.path.join(BASE, "compare.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=hdr)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {out_csv}")

if __name__ == "__main__":
    main()