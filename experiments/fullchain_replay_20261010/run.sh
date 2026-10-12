#!/usr/bin/env bash
# Full-chain dual-criterion replay. Usage: bash run.sh plan | smoke | full | status | analyze [max tier] | stop
# Order: plan → smoke (4 jobs, one process on each of GPUs 2–5, foreground; automatic gate) → full (starts only if the gate passes; background, 3 processes per GPU)
#       → while running, 'analyze 1' shows interim tier-1 results → once ALL DONE appears in the log, analyze
set -euo pipefail
export HANDOFF_RUN_ROOT=${HANDOFF_RUN_ROOT:-$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd -P)}   # path configuration, see README
D=$HANDOFF_RUN_ROOT/fullchain_replay_20261010
PY=${HANDOFF_PYTHON:-python}
SUP=$HANDOFF_RUN_ROOT/newtasks_20261004/supervise_multi.py
export PYTHONDONTWRITEBYTECODE=1
cd "$D"
[ "$(pwd -P)" = "$(readlink -f "$D")" ] || { echo "working directory does not match $D"; exit 2; }
mkdir -p logs
case "${1:-}" in
  plan)
    "$PY" -B plan_jobs.py 2>&1 | tee logs/plan.log ;;
  smoke)
    "$PY" -u -B "$SUP" --run "$D" --jobs "$D/jobs_smoke.json" --worker "$D/fc_worker.py" --out results_smoke \
      --per-gpu 1 --order 2,3,4,5 --max-util 100 --min-free-mib 12000 2>&1 | tee logs/smoke_supervise.log
    "$PY" -B analyze.py --jobs jobs_smoke.json --res results_smoke --out analysis_smoke --gate 2>&1 | tee logs/smoke_analysis.log ;;
  full)
    if ! "$PY" -B analyze.py --jobs jobs_smoke.json --res results_smoke --out analysis_smoke --gate > logs/gate_check.log 2>&1; then
      echo "smoke-test gate failed (see logs/gate_check.log and analysis_smoke/mismatches.jsonl); full run not started"; exit 3
    fi
    if [ -f logs/supervise_main.pid ] && kill -0 "$(cat logs/supervise_main.pid)" 2>/dev/null; then
      echo "scheduler already running: pid $(cat logs/supervise_main.pid)"; exit 4
    fi
    nohup "$PY" -u -B "$SUP" --run "$D" --jobs "$D/jobs.json" --worker "$D/fc_worker.py" \
      --per-gpu 3 --order 2,3,4,5 --max-util 100 --min-free-mib 12000 \
      > logs/supervise_main.log 2>&1 < /dev/null &
    echo $! > logs/supervise_main.pid; echo "supervisor pid $(cat logs/supervise_main.pid)" ;;
  status)
    cat logs/status_jobs.json 2>/dev/null || true; echo
    tail -n 5 logs/supervise_main.log 2>/dev/null || true
    "$PY" -B - <<'EOF'
import json, os, pathlib
from collections import Counter
D = pathlib.Path(os.environ["HANDOFF_RUN_ROOT"]) / "fullchain_replay_20261010"
jobs = json.loads((D / "jobs.json").read_text())["jobs"]
tot, done = Counter(), Counter()
for j in jobs:
    tot[j["tier"]] += 1; done[j["tier"]] += (D / "results" / f"{j['id']}.json").exists()
print("progress by tier:", {t: f"{done[t]}/{tot[t]}" for t in sorted(tot)})
EOF
    echo "errors: $(ls errors 2>/dev/null | wc -l)"
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader -i 2,3,4,5 ;;
  analyze)
    T="${2:-3}"
    "$PY" -B analyze.py --jobs jobs.json --res results --out "analysis_t${T}" --max-tier "$T" --cross results_smoke \
      2>&1 | tee "logs/analysis_t${T}.log" ;;
  stop)
    [ -f logs/supervise_main.pid ] && kill "$(cat logs/supervise_main.pid)" 2>/dev/null || true
    pkill -f "fc_worker.py --run $D" || true
    echo "stopped; completed results are kept, and rerunning full skips completed jobs" ;;
  *)
    echo "usage: bash run.sh plan|smoke|full|status|analyze [tier]|stop"; exit 1 ;;
esac
