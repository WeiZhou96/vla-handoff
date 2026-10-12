#!/usr/bin/env bash
# Stage 2 (retrain, re-tune, lock) and the stage-3 chain plan. Usage: bash run.sh train|tune|plan|lock|status
# Order: train -> tune -> plan -> (confirm prereg_round3.md is final) -> lock. After lock, train/tune/plan are always refused.
set -euo pipefail
export HANDOFF_RUN_ROOT=${HANDOFF_RUN_ROOT:-$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd -P)}   # path configuration, see README
D=$HANDOFF_RUN_ROOT/benefit_model_bdef_20261011
PY=${PY:-${HANDOFF_PYTHON:-python}}   # same environment as the closed-loop worker (same sklearn version)
cd "$D"
mkdir -p logs analysis
locked() { [[ -e locked_config.json ]]; }
refuse_if_locked() { if locked; then echo "locked_config.json exists: refusing '$1'"; exit 1; fi; }

case "${1:-status}" in
  train)
    refuse_if_locked train
    CUDA_VISIBLE_DEVICES= "$PY" -B train_benefit_v4.py --rp keep 2>&1 | tee logs/train_keep.log
    CUDA_VISIBLE_DEVICES= "$PY" -B train_benefit_v4.py --rp exclude --no-save --no-loto 2>&1 | tee logs/train_rpexclude.log
    ;;
  tune)
    refuse_if_locked tune
    "$PY" -B lock_config3.py --tune 2>&1 | tee logs/tune.log
    ;;
  plan)
    refuse_if_locked plan
    "$PY" -B plan_cl3.py 2>&1 | tee logs/plan.log
    ;;
  lock)
    refuse_if_locked lock
    FC=${FC_WORKER:-}
    if [[ -z "$FC" ]]; then
      mapfile -t C < <(find -H "$HANDOFF_RUN_ROOT" -maxdepth 3 -name fc_worker.py -not -path "*benefit_model_bdef_20261011*" 2>/dev/null | sort)
      if [[ ${#C[@]} -ne 1 ]]; then
        echo "fc_worker.py candidates (${#C[@]}): ${C[*]:-none}. Set FC_WORKER=/path/to/fc_worker.py and rerun."; exit 1
      fi
      FC=${C[0]}
    fi
    echo "fc_worker: $FC"
    "$PY" -B lock_config3.py --lock --fc-worker "$FC" 2>&1 | tee logs/lock.log
    ;;
  status)
    "$PY" -B - <<'EOF'
import json, hashlib, os
from pathlib import Path
D = Path(".")
def j(p):
    try: return json.loads((D / p).read_text())
    except Exception as e: return None
def sha(p):
    h = hashlib.sha256(); h.update((D / p).read_bytes()); return h.hexdigest()
print("model v4:", (D / "models/benefit_models_v4_all.joblib").exists())
lc = j("analysis/label_check.json")
print("label_check ok:", None if lc is None else lc.get("ok"), "| k changes:", None if lc is None else lc.get("k_change_by_task"),
      "| composition equal v3:", None if lc is None else lc.get("data_composition", {}).get("equal"))
rp = j("analysis/v4_report_all.json")
if rp: print("offline all|ALL n:", rp["selection"]["all|ALL"]["n"], "(see logs/train_keep.log for tables)")
t = j("analysis/tune_report.json")
if t: print("tune: dev_rule", t["dev_rule"], "| delta_B", t["delta_benefit"], "| delta_round1", t["delta_round1_carried_in_round2"],
            "| dev sets", {k: (v["n"], v["delta"]) for k, v in t["dev_sets"].items()}, "| cbar check", t["cbar_check"])
p = j("analysis/plan_report.json")
if p: print("plan ok:", p["ok"], "| n", p["n_total"], "| bases", p["bases"], "| strata", p["strata"], "| shortfall", p["shortfall"],
            "| dev linked", p["catalog"]["dev_linked"], "| unseen-base chains", p["unseen_base_chains"])
L = j("locked_config.json")
if L:
    rec = (D / "prereg_round3.sha256").read_text().split()[0]
    print("LOCKED at", L["created"], "| delta", L["delta_benefit"], "| arms", L["arms"])
    print("prereg unchanged since lock:", rec == sha("prereg_round3.md"))
else:
    print("not locked")
res = D / "results"
print("stage-3 result files:", len(list(res.glob('*.json'))) if res.exists() else 0)
EOF
    ;;
  *) echo "usage: bash run.sh train|tune|plan|lock|status"; exit 2 ;;
esac
