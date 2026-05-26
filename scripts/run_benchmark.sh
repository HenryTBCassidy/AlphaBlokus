#!/bin/bash
# Launch wrapper for scripts/benchmark_phases.py.
#
# Why this exists:
# - Detached runs under systemd-run/nohup/etc default to block-buffered
#   stdout, hiding per-game progress until exit. PYTHONUNBUFFERED=1 plus
#   ``sys.stdout.reconfigure(line_buffering=True)`` in the script itself
#   force line buffering both ways.
# - Tees the combined stdout/stderr to a stable log path so a second
#   shell session can ``tail -f`` the log while the benchmark runs,
#   independent of however the original session was launched.
#
# Usage::
#
#     # Default config + log path:
#     ./scripts/run_benchmark.sh
#
#     # Custom config, log auto-derived from run_name:
#     ./scripts/run_benchmark.sh --config run_configurations/profile_baseline.json
#
#     # Override log path:
#     BENCHMARK_LOG=/tmp/my-benchmark.log ./scripts/run_benchmark.sh
#
# Args are passed straight through to benchmark_phases.py.

set -euo pipefail

cd "$(dirname "$0")/.."

# Default config — same one the benchmark script's own default uses.
CONFIG=""
for ((i=1; i<=$#; i++)); do
  if [[ "${!i}" == "--config" ]]; then
    j=$((i+1))
    CONFIG="${!j}"
    break
  fi
done
CONFIG="${CONFIG:-run_configurations/profile_baseline.json}"

# Derive a default log path from the config's run_name. Falls back to
# /tmp if the config doesn't parse cleanly — we never want to fail the
# whole run because we couldn't pick a log location.
RUN_NAME=$(uv run python -c "
import json, sys
try:
    print(json.load(open('${CONFIG}'))['run_name'])
except Exception:
    print('benchmark')
" 2>/dev/null || echo benchmark)

DEFAULT_LOG="temp/${RUN_NAME}_benchmark/run.log"
LOG_PATH="${BENCHMARK_LOG:-$DEFAULT_LOG}"
mkdir -p "$(dirname "$LOG_PATH")"

echo "[run_benchmark] config: $CONFIG"
echo "[run_benchmark] log:    $LOG_PATH"
echo "[run_benchmark] tail with: tail -f $LOG_PATH"
echo

export PYTHONUNBUFFERED=1

# stdbuf on Linux forces line buffering for child processes too — harmless
# on macOS (won't be found, falls through). The pipe to tee on its own is
# fine because Python is line-buffered via the script + env var.
exec uv run python -m scripts.benchmark_phases "$@" 2>&1 | tee "$LOG_PATH"
