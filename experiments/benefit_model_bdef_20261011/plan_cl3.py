# -*- coding: utf-8 -*-
"""Closed-loop chain plan for round 3 (stage 3). The selection does not use any execution results.

Definition of "used" (chain level; a chain is excluded if any condition holds):
  (1) its catalog index appears in round-1 jobs_cl.json or round-2 jobs_cl2.json (the full-chain replays used exactly these chains);
  (2) its catalog index can be reliably linked to a labelled dev handoff: candidate indices are taken from explicit fields of the dev
      records (index/sequence_index/...) or from prefixed numbers in parent_id/sample_id/observation; a source x method is accepted only
      if its verification rate on verifiable records is >= 99% with at least 20 verified records (verification: the record's task_b is in
      the chain's task sequence, or tasks are identical);
  (3) any other index whose (initial configuration, task sequence) is identical to a chain in (1)(2).
Base configurations (discrete initial configurations in the catalog) may coincide with earlier stages: the same base configuration with a
new perturbation seed counts as a new chain (as in rounds 1 and 2); base configurations never seen in (1)(2) are reported as a subset.
Stratification: 11 target tasks; each stratum takes unused chains in which the task is at positions 1-4, grouped by base configuration
with hash round-robin across groups (as in plan_cl2.py); tasks with fewer available chains are selected first. Sample size: see SAMPLE SIZE.
Perturbation seed int(sha256("cl3-perturb|i")[:8],16); the perturbation function is worker_cl.perturb.
"""
import os  # path configuration for the public release; see README, "Environment variables"
_EXPERIMENTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_ROOT = os.environ.get("HANDOFF_RUN_ROOT") or _EXPERIMENTS
RHO_ROOT = os.environ.get("RHO_ROOT") or os.path.join(_EXPERIMENTS, "rho")
import collections, hashlib, json, math, re, sys
from pathlib import Path
from statistics import NormalDist

HERE = Path(__file__).resolve().parent
NEWDIR = Path(f"{RUN_ROOT}/benefit_model_bdef_20261011").resolve()
if HERE != NEWDIR:
    sys.exit(f"must run from {NEWDIR}")
if (HERE / "locked_config.json").exists():
    sys.exit("locked: refusing to re-plan")
CV = Path(f"{RUN_ROOT}")
CATALOG = f"{RHO_ROOT}/outputs/experiments/two_task_expansion_20260913_222701/logs/sequence_catalog.json"
RAW = Path(f"{RHO_ROOT}/outputs/experiments/two_task_expansion_20260913_222701/raw")
TARGETS = sorted(["lift_blue_block_slider", "lift_pink_block_drawer", "lift_pink_block_slider", "lift_red_block_slider",
                  "push_blue_block_right", "push_into_drawer", "push_pink_block_right", "push_red_block_right",
                  "rotate_pink_block_right", "rotate_red_block_left", "stack_block"])
PUSH_ROT = {"push_red_block_right", "push_blue_block_right", "rotate_pink_block_right", "push_pink_block_right", "rotate_red_block_left"}

# ---------------- SAMPLE SIZE (primary hypothesis H1: benefit_B − retry, paired difference in full-chain success rate) ----------------
ALPHA1, POWER, EFFECT = 0.025, 0.80, 0.05          # one-sided α (= lower bound of the 95% two-sided interval > 0), power, smallest effect of interest
SD, DEFF, EXCL = 0.411, 1.11, 0.05                 # SD of benefit−retry in the r1 full-chain replay (larger of the two rounds); largest deff over comparisons; exclusion allowance
z = NormalDist().inv_cdf
ZZ = z(1 - ALPHA1) + z(POWER)
n_req = math.ceil(ZZ ** 2 * SD ** 2 * DEFF / EFFECT ** 2)
n_infl = math.ceil(n_req / (1 - EXCL) - 1e-9)
N_TOTAL = math.ceil(n_infl / len(TARGETS)) * len(TARGETS)
PER_TASK = N_TOTAL // len(TARGETS)
assert (n_req, N_TOTAL, PER_TASK) == (589, 627, 57), (n_req, n_infl, N_TOTAL)
mde = lambda sd, deff, n: ZZ * sd * math.sqrt(deff / n)
REPLAY = {"r1 benefit-retry": (0.411, 1.09), "r2 benefit-retry": (0.327, 1.00), "r1 benefit-none": (0.430, 1.11),
          "r2 benefit-none": (0.384, 1.01), "r1 benefit-prior": (0.348, 1.09), "r2 benefit-prior": (0.384, 0.99),
          "r1 benefit_noretry-retry": (0.467, 1.09), "r2 benefit_noretry-retry": (0.359, 0.96)}
n_an = math.floor(N_TOTAL * (1 - EXCL))
mde_tab = {k: round(mde(sd, de, n_an), 4) for k, (sd, de) in REPLAY.items()}

h = lambda *a: hashlib.sha256("|".join(map(str, a)).encode()).hexdigest()
cat = json.load(open(CATALOG))
bkey = lambda i: json.dumps(cat[i][0], sort_keys=True)
ckey = lambda i: json.dumps([cat[i][0], cat[i][1]], sort_keys=True)
bid = lambda i: h("base", bkey(i))[:16]

# ---------------- (1) rounds 1 and 2 ----------------
prev = {}
for name, p in (("r1", CV / "closedloop_20261004/jobs_cl.json"), ("r2", CV / "closedloop2_20261005/jobs_cl2.json")):
    js = json.loads(p.read_text())["jobs"]
    prev[name] = dict(n=len(js), idx={int(j["index"]) for j in js}, bases={j["base"] for j in js},
                      base_formula_match=sum(bid(int(j["index"])) == j["base"] for j in js),
                      tasks_match=sum(list(cat[int(j["index"])][1]) == list(j["tasks"]) for j in js))
used_idx = prev["r1"]["idx"] | prev["r2"]["idx"]

# ---------------- (2) linkage to dev data ----------------
INDEX_KEYS = ("index", "sequence_index", "seq_index", "catalog_index", "chain_index", "seq_idx", "sequence_id")
RX = [re.compile(r"(?:seq|sequence|idx|index|chain)[_\-]?(\d+)", re.I), re.compile(r"^(\d+)$")]


def cand_index(rec):
    for k in INDEX_KEYS:
        v = rec.get(k)
        if isinstance(v, int) or (isinstance(v, str) and v.isdigit()):
            return int(v), "field:" + k
    for k in ("parent_id", "sample_id", "observation"):
        v = rec.get(k)
        if isinstance(v, str):
            for j, rx in enumerate(RX):
                m = rx.search(v)
                if m:
                    return int(m.group(1)), f"regex{j}:{k}"
    return None, None


def validate(rec, i):
    if not (0 <= i < len(cat)):
        return False
    if isinstance(rec.get("tasks"), list):
        return list(rec["tasks"]) == list(cat[i][1])
    t = rec.get("task_b") or rec.get("task")
    return None if t is None else (t in cat[i][1])


def dev_records():
    for f in sorted(RAW.glob("gpu*/handoffs.jsonl")):
        for line in f.read_text().splitlines():
            yield "raw_two_task", json.loads(line)
    for f in sorted((CV / "fix_20260928/runs/j8_two_task").glob("gpu*/labels.jsonl")):
        for line in f.read_text().splitlines():
            yield "j8_two_task", json.loads(line)
    for d in ("multitask_vocab_20261002", "gating_vocab_20261003", "newtasks_20261004"):
        for f in sorted((CV / d / "results").glob("j8_*.json")):
            yield d + ":j8", json.loads(f.read_text())
        for f in sorted((CV / d).glob("jobs*.json")):
            try:
                for j in json.loads(f.read_text()).get("jobs", []):
                    yield d + ":jobs", j
            except Exception:
                pass
    pf = CV / "repair_20261001/plan.json"
    if pf.exists():
        for s in json.loads(pf.read_text()).get("states", []):
            yield "repair_20261001:plan", s


stat = collections.defaultdict(lambda: dict(records=0, candidates=0, valid=0, invalid=0, unknown=0, idx=set()))
nrec = collections.Counter()
for src, rec in dev_records():
    nrec[src] += 1
    i, how = cand_index(rec)
    if i is None:
        continue
    s = stat[(src, how)]; s["candidates"] += 1
    v = validate(rec, i)
    if v is None:
        s["unknown"] += 1
    elif v:
        s["valid"] += 1; s["idx"].add(i)
    else:
        s["invalid"] += 1
dev_idx, link = set(), {}
for (src, how), s in stat.items():
    tested = s["valid"] + s["invalid"]
    acc = tested >= 20 and s["valid"] >= 20 and s["valid"] / tested >= 0.99
    if acc:
        dev_idx |= s["idx"]
    link[f"{src}|{how}"] = dict(candidates=s["candidates"], valid=s["valid"], invalid=s["invalid"], unknown=s["unknown"],
                                accepted=acc, n_indices=len(s["idx"]))
excl_idx = used_idx | dev_idx
excl_ck = {ckey(i) for i in excl_idx}
excluded = {i for i in range(len(cat)) if i in excl_idx or ckey(i) in excl_ck}
seen_bases = prev["r1"]["bases"] | prev["r2"]["bases"] | {bid(i) for i in dev_idx}

# ---------------- stratified chain sampling ----------------
eligible = {t: [i for i in range(len(cat)) if i not in excluded and t in cat[i][1][1:5]] for t in TARGETS}
avail = {t: dict(chains=len(v), bases=len({bkey(i) for i in v})) for t, v in eligible.items()}
order = sorted(TARGETS, key=lambda t: (avail[t]["chains"], t))
chosen, take_by, fill = set(), {}, []


def rr(t):
    by = collections.defaultdict(list)
    for i in eligible[t]:
        if i not in chosen:
            by[bkey(i)].append(i)
    bases = sorted(by, key=lambda k: h("cl3-base", t, k))
    for k in bases:
        by[k].sort(key=lambda i: h("cl3-chain", i))
    seq, rnd = [], 0
    while any(len(by[k]) > rnd for k in bases):
        seq += [by[k][rnd] for k in bases if len(by[k]) > rnd]
        rnd += 1
    return seq


for t in order:
    take = rr(t)[:PER_TASK]
    chosen.update(take); take_by[t] = take
short = {t: PER_TASK - len(v) for t, v in take_by.items() if len(v) < PER_TASK}
need = N_TOTAL - len(chosen)
progress = True
while need > 0 and progress:          # fixed in advance: if a stratum falls short, the total is filled from the other strata in turn, in the same order
    progress = False
    for t in order:
        if need <= 0:
            break
        s = rr(t)
        if s:
            chosen.add(s[0]); take_by[t].append(s[0]); fill.append(s[0]); need -= 1; progress = True
jobs = []
for t in order:
    for i in take_by[t]:
        tk = cat[i][1]
        tpos = [p for p in range(1, 5) if tk[p] in TARGETS]
        jobs.append(dict(id=f"cl3_{i}", index=i, base=bid(i), stratum=t, fill=i in fill, tasks=tk,
                         perturb_seed=int(h("cl3-perturb", i)[:8], 16), base_seen_before=bid(i) in seen_bases,
                         push_rotate_target=any(tk[p] in PUSH_ROT for p in tpos), first_target_pos=min(tpos)))
by_s = collections.defaultdict(list)
for j in jobs:
    by_s[j["stratum"]].append(j)
inter = [by_s[t][r] for r in range(max(len(v) for v in by_s.values())) for t in TARGETS if r < len(by_s[t])]
assert not ({j["index"] for j in inter} & excluded) and len({j["index"] for j in inter}) == len(inter)
ok = len(inter) == N_TOTAL
cnt = collections.Counter(t for j in inter for t in j["tasks"][1:5] if t in TARGETS)
per_base = collections.Counter(j["base"] for j in inter)
report = dict(
    ok=ok, targets=TARGETS, n_total=len(inter), n_target_planned=N_TOTAL, per_task=PER_TASK,
    sample_size=dict(alpha_one_sided=ALPHA1, power=POWER, effect=EFFECT, sd=SD, deff=DEFF, exclusion_allowance=EXCL,
                     n_required=n_req, n_inflated=n_infl, n_total=N_TOTAL, n_analysed_assumed=n_an, mde_at_n_analysed=mde_tab),
    catalog=dict(size=len(cat), excluded=len(excluded), excluded_r1r2=len(used_idx), dev_linked=len(dev_idx),
                 dev_linked_not_r1r2=len(dev_idx - used_idx), excluded_duplicates_only=len(excluded - excl_idx)),
    previous_rounds={k: {kk: (len(vv) if isinstance(vv, set) else vv) for kk, vv in v.items()} for k, v in prev.items()},
    dev_linkage=dict(records=dict(nrec), methods=link), availability=avail, selection_order=order, shortfall=short, filled=len(fill),
    strata=dict(collections.Counter(j["stratum"] for j in inter)), bases=len(per_base), max_chains_per_base=max(per_base.values()) if per_base else 0,
    bases_per_stratum={t: len({j["base"] for j in by_s[t]}) for t in TARGETS},
    target_handoffs_pos1_4=dict(cnt), first_target_task=dict(collections.Counter(j["tasks"][j["first_target_pos"]] for j in inter)),
    push_rotate_chains=sum(j["push_rotate_target"] for j in inter), unseen_base_chains=sum(not j["base_seen_before"] for j in inter),
    unseen_bases=len({j["base"] for j in inter if not j["base_seen_before"]}))
(HERE / "analysis").mkdir(exist_ok=True)
(HERE / "analysis/plan_report.json").write_text(json.dumps(report, indent=1))
print(json.dumps({k: v for k, v in report.items() if k != "dev_linkage"}, indent=1))
print("dev linkage:", json.dumps(report["dev_linkage"], indent=1))
if not ok:
    sys.exit(f"PLAN FAILED: only {len(inter)} of {N_TOTAL} chains available; jobs file not written")
out = dict(purpose="round-3 closed-loop confirmation under B_def; stratified over 11 targets; perturbed initial states",
           n_total=N_TOTAL, per_task=PER_TASK, jobs=inter)
txt = json.dumps(out, indent=1)
jf = HERE / "jobs_cl3.json"
if jf.exists() and jf.read_text() != txt:
    sys.exit("jobs_cl3.json exists with different content: refusing to overwrite")
jf.write_text(txt)
print("wrote", jf, "| jobs", len(inter), "| bases", len(per_base))
