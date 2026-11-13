#!/bin/bash
# Experiment after "Go", in protocol order. Safe to re-run: finished work is skipped.
#   ./run_all.sh prep   small sample for the judge decision, before the blind ratings
#   ./run_all.sh main   everything after the judge decision, runs unattended (systemd/install.sh)
set -euo pipefail
cd "$(dirname "$0")"
R() { .venv/bin/python -m ragbench "$@"; }
exec > >(tee -a ../results/run.log) 2>&1  # Shown on the dashboard.

case "${1:-}" in
prep)
  R prepare
  ./serve.sh 1; R generate-dev --limit-configs 20
  R calibration-export --limit-configs 20
  ./serve.sh 8; R judge-calibration
  ;;
main)
  R prepare
  R judge-gate                   # Stops here if neither judge agrees with the raters
  ./serve.sh 1; R generate-dev   # 1 slot: latency without queueing
  ./serve.sh 8; R judge-dev
  ./serve.sh 1; R generate-noise # Needs dev scores: includes the top configurations
  ./serve.sh 8; R judge-noise; R controls
  R optimize; R freeze           # Finalists are fixed before any test question runs
  ./serve.sh 1; R generate-test; R baselines
  ./serve.sh 8; R judge-test; R second-judge
  ./serve.sh 1; R latency-seq; R latency-cpu
  ./serve.sh 4; R latency-load
  R rating-export-test; R analyze
  R cleanup-models               # Saves disk space; the digests stay in the database
  ;;
*) echo "usage: $0 prep|main"; exit 1 ;;
esac
