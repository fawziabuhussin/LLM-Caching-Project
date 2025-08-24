# !/usr/bin/env bash
# Example benchmark script — warm‑up then eval.
# Usage: ./run_benchmarks.sh sharegpt.jsonl

set -euo pipefail
TRACE=$1
mkdir -p results
python replay_driver.py "${TRACE}" > results/summary.txt
