#!/usr/bin/env bash
# Run script for round-3 stage 3 (B_def). Usage: bash run.sh selftest|addendum|smoke|full|status [results|results_smoke]|analyze|stop
set -euo pipefail
export HANDOFF_RUN_ROOT=${HANDOFF_RUN_ROOT:-$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd -P)}   # path configuration, see README
NEW=$HANDOFF_RUN_ROOT/closedloop3_bdef_20261012
LOCK=$HANDOFF_RUN_ROOT/benefit_model_bdef_20261011
JOBS=$LOCK/jobs_cl3.json
PY=${HANDOFF_PYTHON:-python}
SCHED_SRC=$HANDOFF_RUN_ROOT/newtasks_20261004/supervise_multi.py
GPUS="2 3 4 5"
HERE=$(cd "$(dirname "$(readlink -f "$0")")" && pwd -P)
[ "$HERE" = "$(readlink -f "$NEW")" ] || { echo "run.sh must be located in $(readlink -f "$NEW") (actual $HERE)"; exit 2; }
cd "$HERE"
mkdir -p logs
die(){ echo "ERROR: $*" >&2; exit 2; }
nonempty(){ [ -d "$1" ] && [ -n "$(ls -A "$1" 2>/dev/null)" ]; }
alive(){ [ -f "$1" ] && kill -0 "$(cat "$1")" 2>/dev/null; }

selftest(){
  for f in worker_cl3.py analyze_cl3.py; do "$PY" -c "import ast,sys; ast.parse(open(sys.argv[1]).read())" "$f"; done
  bash -n run.sh
  CUDA_VISIBLE_DEVICES="" "$PY" -B worker_cl3.py --run "$NEW" --jobs "$JOBS" --selftest | tee logs/selftest.log
  grep -q "^SELFTEST OK" logs/selftest.log || die "selftest failed"
  echo "smoke chains (fixed rule): $("$PY" -B analyze_cl3.py --pick-smoke)"
}

addendum(){
  [ -e lock_addendum.json ] && die "lock_addendum.json already exists: refusing to overwrite"
  for d in results results_smoke; do nonempty "$d" && die "$d/ is not empty: stage-3 data appear to exist already"; done
  selftest
  if [ -e supervise_multi.py ]; then cmp -s supervise_multi.py "$SCHED_SRC" || die "supervise_multi.py differs from $SCHED_SRC"
  else cp "$SCHED_SRC" supervise_multi.py; fi
  SMOKE=$("$PY" -B analyze_cl3.py --pick-smoke)
  [ -n "$SMOKE" ] || die "no smoke chains selected"
  "$PY" - "$SMOKE" "$SCHED_SRC" <<'EOF'
import hashlib, json, os, socket, sys, time
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
L = os.environ["HANDOFF_RUN_ROOT"] + "/benefit_model_bdef_20261011/"
A = dict(time=time.strftime("%F %T"), host=socket.gethostname(),
         note="prereg_round3.md §7/§12: stage-3 worker and analysis hashed before the first stage-3 job; "
              "smoke chains are unchanged jobs of jobs_cl3.json, not part of the analysed sample (§10: all 627 chains run "
              "with the full design); the full run re-runs them and only bitwise identity is reported",
         files={f: dict(sha256=sha(f)) for f in ("worker_cl3.py", "analyze_cl3.py", "run.sh", "supervise_multi.py")},
         scheduler_source=dict(path=sys.argv[2], sha256=sha(sys.argv[2])),
         locked_config_sha256=sha(L + "locked_config.json"), prereg_sha256=sha(L + "prereg_round3.md"),
         smoke_ids=[x for x in sys.argv[1].split(",") if x], smoke_counts_in_analysis=False)
assert A["locked_config_sha256"] == "7a0c297dfdd7a2869d3ebc1a4d94fdc6714589e238665d11ab08ee8bf32dfa2e", "locked_config sha"
assert A["prereg_sha256"] == "4c1eaba310383dfba9505062c0a17c045b836259b605b13848e43f5001cf90d5", "prereg sha"
fd = os.open("lock_addendum.json", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
with os.fdopen(fd, "w") as f:
    json.dump(A, f, indent=1, ensure_ascii=False)
print(json.dumps(A, indent=1, ensure_ascii=False))
EOF
  chmod 444 lock_addendum.json
  echo "lock_addendum.json written (read-only)"
}

smoke(){
  [ -f lock_addendum.json ] || die "run bash run.sh addendum first"
  nonempty refused && die "markers present in refused/: identify the cause before running"
  IDS=$("$PY" -c 'import json;print(",".join(json.load(open("lock_addendum.json"))["smoke_ids"]))')
  "$PY" - <<'EOF'
import json, os
from pathlib import Path
for c in Path("claims_results_smoke").glob("*.json"):
    if (Path("results_smoke") / c.name).exists():
        continue
    try:
        os.kill(int(json.loads(c.read_text())["pid"]), 0)
    except (OSError, KeyError, ValueError):
        c.unlink(); print("releasing stale claim", c.name)
EOF
  pids=()
  for g in $GPUS; do
    free=$(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits -i "$g" | awk -F, '{print $1-$2}')
    if [ "$free" -lt 14000 ]; then echo "gpu$g free memory ${free}MiB < 14000: skipped"; continue; fi
    "$PY" -u -B worker_cl3.py --run "$NEW" --jobs "$JOBS" --gpu "$g" --out results_smoke --ids "$IDS" > "logs/smoke_gpu$g.log" 2>&1 &
    pids+=($!); echo "smoke worker gpu$g pid $!"; sleep 30
  done
  [ ${#pids[@]} -gt 0 ] || die "none of GPUs 2–5 has enough free memory"
  for p in "${pids[@]}"; do wait "$p" || true; done
  "$PY" -B analyze_cl3.py --smoke-gate
}

full(){
  "$PY" -B analyze_cl3.py --require-gate || die "smoke gate not passed or inconsistent with the addendum: full run not started"
  nonempty refused && die "markers present in refused/"
  alive logs/scheduler.pid && die "scheduler already running (pid $(cat logs/scheduler.pid))"
  nohup "$PY" -u supervise_multi.py --run "$NEW" --jobs "$JOBS" --worker "$NEW/worker_cl3.py" --per-gpu 3 --order 2,3,4,5 \
    >> logs/scheduler_full.log 2>&1 &
  echo $! > logs/scheduler.pid
  nohup bash -c 'while kill -0 "$1" 2>/dev/null; do
      if [ -n "$(ls -A refused 2>/dev/null)" ]; then echo "$(date "+%F %T") refused marker appeared: stopping scheduler"; kill "$1"; fi
      sleep 60; done' _ "$(cat logs/scheduler.pid)" >> logs/watchdog.log 2>&1 &
  echo $! > logs/watchdog.pid
  echo "scheduler pid $(cat logs/scheduler.pid), watchdog pid $(cat logs/watchdog.pid); log logs/scheduler_full.log"
}

status(){
  "$PY" -B analyze_cl3.py --status --out "${1:-results}"
  if alive logs/scheduler.pid; then echo "scheduler running, pid $(cat logs/scheduler.pid)"; else echo "scheduler not running"; fi
  [ -f logs/status_jobs_cl3.json ] && { cat logs/status_jobs_cl3.json; echo; } || true
}

stop(){
  for f in logs/watchdog.pid logs/scheduler.pid; do if alive "$f"; then kill "$(cat "$f")"; echo "stopped $f"; fi; done
  pkill -f "worker_cl3.py --run .*closedloop3_bdef_20261012" || true
  echo "worker processes stopped; unfinished claims are reclaimed by the scheduler on the next full run (their processes no longer exist)"
}

case "${1:-}" in
  selftest) selftest ;;
  addendum) addendum ;;
  smoke) smoke ;;
  full) full ;;
  status) status "${2:-results}" ;;
  analyze) "$PY" -B analyze_cl3.py --analyze ;;
  stop) stop ;;
  *) echo "Usage: bash run.sh selftest|addendum|smoke|full|status [results|results_smoke]|analyze|stop"; exit 1 ;;
esac
