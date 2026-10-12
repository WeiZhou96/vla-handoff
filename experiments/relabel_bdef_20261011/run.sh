#!/usr/bin/env bash
# relabel_bdef_20261011 entry point: plan | inventory [gpu] [n] | smoke | full | status | analyze | escalate <t1,t2> | stop
set -euo pipefail
export HANDOFF_RUN_ROOT=${HANDOFF_RUN_ROOT:-$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd -P)}   # path configuration, see README
LINK=$HANDOFF_RUN_ROOT/relabel_bdef_20261011
R="$(readlink -f "$LINK")"           # symlink: always use the resolved path
CVR="$(readlink -f "$HANDOFF_RUN_ROOT")"
cd "$R"
PY=${HANDOFF_PYTHON:-python}
mkdir -p logs

get_sched() {   # copy the existing scheduler into this directory (the source file is not modified)
  if [ ! -f "$R/supervise_multi.py" ]; then
    local src=""
    for c in "$CVR/fullchain_replay_20261010/supervise_multi.py" "$CVR/closedloop_20261004/supervise_multi.py"; do
      if [ -f "$c" ]; then src="$c"; break; fi
    done
    if [ -z "$src" ]; then
      src="$(find "$CVR" -maxdepth 2 -name supervise_multi.py -not -path "$R/*" 2>/dev/null | sort | head -n 1 || true)"
    fi
    [ -n "$src" ] || { echo "supervise_multi.py not found"; exit 2; }
    grep -q -- '"--order"' "$src" || { echo "$src is not the expected scheduler"; exit 2; }
    grep -q 'assert set(order) <= {2, 3, 4, 5}' "$src" || { echo "$src lacks the GPU 2-5 restriction"; exit 2; }
    cp "$src" "$R/supervise_multi.py"
    echo "$(date '+%F %T') scheduler copied from $src sha256=$(sha256sum "$R/supervise_multi.py" | cut -d' ' -f1)" | tee -a logs/scheduler_source.txt
  fi
}

do_plan() { $PY -B plan_jobs.py "$@" 2>&1 | tee -a logs/plan.log; }

do_smoke() {
  [ -f jobs_smoke.json ] || do_plan
  get_sched
  if [ -f smoke_FAILED.json ] && [ -d results_smoke ]; then
    ts=$(date +%s); mv results_smoke "results_smoke_failed_$ts"
    [ -d claims_results_smoke ] && mv claims_results_smoke "claims_results_smoke_failed_$ts"
    [ -d errors_results_smoke ] && mv errors_results_smoke "errors_results_smoke_failed_$ts"
  fi
  echo "$(date '+%F %T') smoke test started"
  $PY -u "$R/supervise_multi.py" --run "$R" --jobs "$R/jobs_smoke.json" --worker "$R/relabel_worker.py" \
      --out results_smoke --order 2,3,4,5 --per-gpu 1 > logs/smoke_sched.log 2>&1
  $PY -B analyze.py --smoke-check 2>&1 | tee logs/smoke_check.log
  [ -f smoke_PASSED.json ]
}

case "${1:-}" in
  plan) shift; do_plan "$@" ;;
  inventory)
    G="${2:-2}"; N="${3:-10}"
    case "$G" in 2|3|4|5) ;; *) echo "only GPUs 2-5 are allowed"; exit 2 ;; esac
    $PY -u -B inventory.py --gpu "$G" --sample "$N" 2>&1 | tee logs/inventory.log ;;
  smoke) do_smoke && echo "smoke test passed" || { echo "smoke test failed, see smoke_FAILED.json"; exit 1; } ;;
  full)
    [ -f jobs.json ] || do_plan
    if [ ! -f smoke_PASSED.json ]; then
      do_smoke || { echo "smoke test failed; full run not started"; exit 1; }
    fi
    get_sched
    if pgrep -f "supervise_multi.py --run $R --jobs $R/jobs.json" > /dev/null; then echo "full-run scheduler is already running"; exit 0; fi
    nohup $PY -u "$R/supervise_multi.py" --run "$R" --jobs "$R/jobs.json" --worker "$R/relabel_worker.py" \
        --order 2,3,4,5 --per-gpu 3 > logs/full_sched.log 2>&1 &
    echo "full run started, scheduler pid $!; progress: bash run.sh status" ;;
  escalate)
    [ -n "${2:-}" ] || { echo "usage: run.sh escalate task1,task2"; exit 2; }
    [ -f smoke_PASSED.json ] || { echo "smoke test has not passed yet"; exit 1; }
    if pgrep -f "supervise_multi.py --run $R --jobs $R/jobs_escalate.json" > /dev/null; then echo "escalation scheduler is already running"; exit 0; fi
    do_plan --escalate "$2"
    get_sched
    nohup $PY -u "$R/supervise_multi.py" --run "$R" --jobs "$R/jobs_escalate.json" --worker "$R/relabel_worker.py" \
        --order 2,3,4,5 --per-gpu 3 > logs/escalate_sched.log 2>&1 &
    echo "escalation jobs started, pid $!" ;;
  status) $PY -B analyze.py --status ;;
  analyze) $PY -B analyze.py 2>&1 | tee logs/analyze.log ;;
  stop)
    pkill -f "supervise_multi.py --run $R" || true
    sleep 2
    pkill -f "$R/relabel_worker.py" || true
    echo "stopped (claims of unfinished jobs are reclaimed by the scheduler on the next start)" ;;
  *) echo "usage: bash run.sh plan | inventory [gpu] [n] | smoke | full | status | analyze | escalate <tasks> | stop"; exit 2 ;;
esac
