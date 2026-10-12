#!/usr/bin/env bash
# Net-displacement replay. Usage: bash run.sh plan | smoke | full | status | analyze | retry
# Order: plan → smoke (3 chains, GPU 2, automatic check) → full (starts only if the check passes; background) → once ALL DONE appears in the log, analyze
#       → optional retry (second pass, retry arm only) → analyze again
set -euo pipefail
export HANDOFF_RUN_ROOT=${HANDOFF_RUN_ROOT:-$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd -P)}   # path configuration, see README
D=$HANDOFF_RUN_ROOT/netdisp_replay_20261009
PY=${HANDOFF_PYTHON:-python}
SUP=$HANDOFF_RUN_ROOT/newtasks_20261004/supervise_multi.py
ARMS=none,inverse,prior,benefit
export PYTHONDONTWRITEBYTECODE=1
cd "$D"; mkdir -p logs
case "${1:-}" in
  plan)
    "$PY" -B plan_jobs.py 2>&1 | tee logs/plan.log ;;
  smoke)
    "$PY" -u -B replay_worker.py --run "$D" --jobs "$D/jobs_smoke.json" --gpu 2 --arms "$ARMS" 2>&1 | tee logs/smoke_gpu2.log
    "$PY" -B analyze.py --jobs jobs_smoke.json --out analysis_smoke --gate 2>&1 | tee logs/smoke_analysis.log ;;
  full)
    if ! "$PY" -B analyze.py --jobs jobs_smoke.json --out analysis_smoke --gate > logs/gate_check.log 2>&1; then
      echo "smoke-test check failed (see logs/gate_check.log); full run not started"; exit 3
    fi
    nohup "$PY" -u -B "$SUP" --run "$D" --jobs "$D/jobs.json" --worker "$D/replay_worker.py" \
      --per-gpu 2 --order 2,3,4,5 --max-util 100 --min-free-mib 20000 --extra "--arms $ARMS" \
      > logs/supervise_main.log 2>&1 < /dev/null &
    echo $! > logs/supervise_main.pid; echo "supervisor pid $(cat logs/supervise_main.pid)" ;;
  status)
    cat logs/status_jobs.json 2>/dev/null || true; echo
    tail -n 5 logs/supervise_main.log 2>/dev/null || true
    echo "results: $(ls results 2>/dev/null | grep -c '\.json$' || true)  errors: $(ls errors 2>/dev/null | wc -l)"
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader -i 2,3,4,5 ;;
  analyze)
    "$PY" -B analyze.py --jobs jobs.json --out analysis 2>&1 | tee logs/analysis.log ;;
  retry)
    nohup "$PY" -u -B "$SUP" --run "$D" --jobs "$D/jobs.json" --worker "$D/replay_worker.py" --out results_retry \
      --per-gpu 2 --order 2,3,4,5 --max-util 100 --min-free-mib 20000 --extra "--arms retry" \
      > logs/supervise_retry.log 2>&1 < /dev/null &
    echo $! > logs/supervise_retry.pid; echo "supervisor pid $(cat logs/supervise_retry.pid)" ;;
  *)
    echo "usage: bash run.sh plan|smoke|full|status|analyze|retry"; exit 1 ;;
esac
