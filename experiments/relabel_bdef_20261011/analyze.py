# -*- coding: utf-8 -*-
"""T3 analysis and smoke-test gate.

  python -B analyze.py --status        progress, failures and estimated remaining time
  python -B analyze.py --smoke-check   smoke gate: all checks pass -> smoke_PASSED.json, exit code 0; otherwise smoke_FAILED.json, exit code 1
  python -B analyze.py                 full analysis: report.md, analysis/*.json|csv, labels_bdef/<dir>/ (same format as the original
                                       label_<id>.json, usable directly as the stage-2 data directory), mapping.json
Checks: mode A vs. stored labels field by field (success, steps, k, n, restore_max_error); mode B agrees with mode A before the
     divergence point; "B inferred from A" agrees with the actual B run (verify sample); none identity; repeatability of smoke vs. full runs on the same branch.
"""
import argparse, collections, copy, csv, json, os, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import inventory as inv  # noqa: E402

R = inv.rroot()
TOL = 1e-6
INVERSE = {"push_red_block_right": "push_red_block_left", "push_blue_block_right": "push_blue_block_left",
           "rotate_pink_block_right": "rotate_pink_block_left", "push_pink_block_right": "push_pink_block_left",
           "rotate_red_block_left": "rotate_red_block_right"}


def jload(p):
    return json.loads(Path(p).read_text())


def load_jobs():
    jobs = {}
    for name in ("jobs.json", "jobs_escalate.json"):
        if (R / name).exists():
            for j in jload(R / name)["jobs"]:
                jobs[j["id"]] = j
    return jobs


def load_results(sub):
    out, d = {}, R / sub
    if d.exists():
        for f in d.glob("*.json"):
            try:
                out[f.stem] = jload(f)
            except Exception as e:
                out[f.stem] = dict(error=f"unreadable: {e!r}")
    return out


def judge(r, job):
    if r is None:
        return False, ["missing"]
    if "error" in r:
        return False, ["error"]
    if "invalid" in r:
        return False, ["invalid: " + r["invalid"]]
    c = r.get("checks")
    if not c:
        return False, ["no checks"]
    why = []
    if not c["match_stored"]:
        why.append("A≠stored")
    if c["a_disagree"]:
        why.append("a_disagree")
    if not c["consistency"]:
        why.append("A/B divergence check")
    if not c["pred_ok"]:
        why.append("B predicted≠B run")
    if not c["identity"]:
        why.append("none identity")
    if r["arm_A"]["k"] != job["stored_k"]:
        why.append("k_A≠stored")
    if job["role"] == "none_identity" and not (r.get("post_repair") or {}).get("equals_handoff_poses", False):
        why.append("s≠h for none")
    if (r.get("code") or {}).get("copied_identical") is not True:
        why.append("copied code")
    return not why, why


def rates(res):
    sec = sum(r.get("seconds", 0) - r.get("seconds_restore", 0) for r in res.values() if "checks" in r)
    steps = sum(r["checks"]["steps_run"] for r in res.values() if "checks" in r)
    rs = sum(r.get("seconds_restore", 0) for r in res.values() if "checks" in r)
    acts = sum(r["checks"]["n_restores"] * (r.get("prefix_len", 0) + r.get("repair_len", 0)) for r in res.values() if "checks" in r)
    return (sec / steps if steps else None), (rs / acts if acts else None)


# ---------------- status ----------------
def status():
    jobs = load_jobs()
    res = load_results("results")
    tot = collections.Counter(j["role"] for j in jobs.values())
    done = collections.Counter(jobs[i]["role"] for i in res if i in jobs)
    bad = [(i, judge(res[i], jobs[i])[1]) for i in res if i in jobs and not judge(res[i], jobs[i])[0]]
    print(time.strftime("%F %T"), f"done {sum(done.values())}/{len(jobs)}")
    for k in sorted(tot):
        print(f"  {k:24s} {done[k]:5d}/{tot[k]:5d}")
    nerr = len(list((R / "errors").glob("*.json"))) if (R / "errors").exists() else 0
    print(f"  failed (check not passed or error) {len(bad)}; errors/ {nerr}")
    for i, w in bad[:10]:
        print("   ", i, jobs[i]["task_b"], jobs[i]["cand"], w)
    dn = [i for i in res if i in jobs and "seconds" in res[i]]
    if dn:
        rate = sum(res[i]["seconds"] for i in dn) / max(1, sum(jobs[i]["est"] for i in dn))
        rem = sum(j["est"] for i, j in jobs.items() if i not in res)
        print(f"  estimated remaining (12 processes) ≈ {rem * rate / 12 / 3600:.1f} h")
    st = R / "logs" / "status_jobs.json"
    if st.exists():
        print("  scheduler:", st.read_text()[:300])


# ---------------- smoke gate ----------------
def smoke_check():
    sj = {j["id"]: j for j in jload(R / "jobs_smoke.json")["jobs"]}
    res = load_results("results_smoke")
    rows, ok_all = [], bool(sj)
    for i, j in sj.items():
        ok, why = judge(res.get(i), j)
        ok_all = ok_all and ok
        r = res.get(i) or {}
        rows.append(dict(id=i, role=j["role"], task=j["task_b"], cand=j["cand"], ok=ok, why=why, stored_k=j["stored_k"],
                         kA=(r.get("arm_A") or {}).get("k"), kB=(r.get("arm_B") or {}).get("k"),
                         b_runs=(r.get("checks") or {}).get("n_b_runs"), seconds=r.get("seconds")))
    nb = sum(x["b_runs"] or 0 for x in rows)
    if nb == 0:
        ok_all = False
    spp, c1 = rates({i: r for i, r in res.items() if i in sj})
    proj = inv.estimate(list(load_jobs().values()), spp, 12, 0.0, c1) if (spp and c1) else None
    out = dict(time=time.strftime("%F %T"), passed=ok_all, rows=rows, b_runs=nb, measured_sec_per_step=spp,
               measured_restore_sec_per_action=c1, projection_full=proj)
    for f in ("smoke_PASSED.json", "smoke_FAILED.json"):
        (R / f).unlink(missing_ok=True)
    inv.write_json("smoke_PASSED.json" if ok_all else "smoke_FAILED.json", out)
    for x in rows:
        print(f"  {'OK ' if x['ok'] else 'BAD'} {x['role']:22s} {x['task']:26s} {x['cand']:28s} stored={x['stored_k']} "
              f"A={x['kA']} B={x['kB']} Bruns={x['b_runs']} {x['seconds']}s {x['why'] or ''}")
    if proj:
        print(f"  measured {spp:.4f} s/step, restore {c1:.5f} s/action; full-run projection (12 processes): " +
              ", ".join(f"{k}={proj[k]['wall_hours']}h" for k in ("lower", "central", "upper")))
    print("smoke test", "passed" if ok_all else "failed")
    return 0 if ok_all else 1


# ---------------- statistics helpers ----------------
def boot_mean(vals, groups, n=2000, seed=0):
    vals = np.asarray(vals, float)
    if len(vals) == 0:
        return None, None, None
    g = np.asarray([str(x) for x in groups])
    u = sorted(set(g))
    idx = [np.flatnonzero(g == k) for k in u]
    rng = np.random.RandomState(seed)
    bs = [vals[np.concatenate([idx[p] for p in rng.choice(len(u), len(u))])].mean() for _ in range(n)]
    return float(vals.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def kendall(x, y):
    try:
        from scipy.stats import kendalltau
        t = kendalltau(x, y).correlation
        return None if t is None or np.isnan(t) else float(t)
    except Exception:
        return None


def md_table(headers, rows):
    s = "| " + " | ".join(headers) + " |\n|" + "---|" * len(headers) + "\n"
    for r in rows:
        s += "| " + " | ".join("" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v)) for v in r) + " |\n"
    return s


def pct(x):
    return None if x is None else f"{100 * x:.1f}%"


# ---------------- mirrored label directories ----------------
def link(l, target):
    if not target.exists() or os.path.lexists(l):
        return
    if R not in l.parent.resolve().parents and l.parent.resolve() != R:
        raise SystemExit(f"refusing to create a link outside this directory: {l}")
    l.symlink_to(target)


def write_mirror(J, good):
    mapping = []
    for dname, jfs in inv.LABEL_DIRS.items():
        src = (inv.CV / dname).resolve()
        rel = Path("labels_bdef") / dname
        res_dir = inv.out_path(rel / "results" / ".keep").parent
        res_dir.mkdir(parents=True, exist_ok=True)
        base = res_dir.parent
        for sub in ("screen", "boundaries", "analysis"):
            link(base / sub, src / sub)
        for jf in jfs:
            link(base / jf, src / jf)
        for p in sorted((src / "results").iterdir()):
            if p.name.startswith("label_") and p.name.endswith(".json"):
                d = jload(p)
                new = copy.deepcopy(d)
                new["criterion"] = "B_def"
                new["relabel_root"] = str(R)
                for a in new.get("arms", []):
                    c = a["candidate"]
                    j = J.get((dname, p.name, c))
                    valid = "invalid" not in a and a.get("n") == 8
                    ent = dict(dir=dname, label_file=p.name, job_id=d.get("job_id"), sample_id=d.get("sample_id"),
                               task_b=d.get("task_b"), candidate=c, k_A_stored=a.get("k"),
                               relabel_id=j["id"] if j else None, role=j["role"] if j else None)
                    if not valid:
                        srcl = "stored_invalid"
                    elif j is not None and j["id"] in good:
                        r = good[j["id"]]
                        b = r["arm_B"]
                        a["rollouts"] = [dict(repeat=x["repeat"], success=x["success"], steps=x["steps"]) for x in b["rollouts"]]
                        a["k"], a["n"], a["restore_max_error"] = b["k"], b["n"], b["restore_max_error"]
                        srcl = "B_def_relabel" if c != "none" else "B_def_identity_check"
                        ent["k_A_rerun"] = r["arm_A"]["k"]
                    elif j is not None:
                        srcl = "stored_A_RELABEL_FAILED_OR_MISSING"
                    elif c == "none":
                        srcl = "stored_A_none_identity_by_construction"
                    else:
                        srcl = "stored_A_not_in_relabel_plan_unmoved"
                    a["label_source"] = srcl
                    ent.update(source=srcl, k_B=a.get("k"))
                    mapping.append(ent)
                inv.write_json(rel / "results" / p.name, new)
            elif not p.name.endswith(".tmp"):
                link(res_dir / p.name, p)
    return mapping


# ---------------- full analysis ----------------
def analyze():
    jobs = load_jobs()
    res = load_results("results")
    smoke = load_results("results_smoke")
    branches, problems, files = inv.enumerate_branches()
    J = {(j["dir"], j["label_file"], j["cand"]): j for j in jobs.values()}
    verdict = {i: judge(res.get(i), j) for i, j in jobs.items()}
    good = {i: res[i] for i, (ok, _) in verdict.items() if ok}
    S = dict(created=time.strftime("%F %T"), jobs=len(jobs), results=sum(i in res for i in jobs), good=len(good),
             failed={i: w for i, (ok, w) in verdict.items() if not ok and i in res}, missing=sum(i not in res for i in jobs))
    S["complete_and_verified"] = S["good"] == len(jobs)

    # 1. mode A vs. stored labels
    per = collections.defaultdict(collections.Counter)
    for i, r in res.items():
        if i not in jobs or "executions" not in r:
            continue
        j = jobs[i]
        for key in (f'{j["dir"]}|{j["role"]}', f'ALL|{j["role"]}', "ALL|ALL"):
            c = per[key]
            for e in r["executions"]:
                c["exec"] += 1
                c["exec_match"] += int(e["match_stored"])
                c["success_match"] += int(e["A"]["success"] == e["stored"]["success"])
                c["steps_match"] += int(e["A"]["steps"] == e["stored"]["steps"])
            if "arm_A" in r:
                c["arms"] += 1
                c["k_match"] += int(r["arm_A"]["k"] == j["stored_k"])
                c["n_match"] += int(r["arm_A"]["n"] == 8)
                s_rme = j["stored_restore_max_error"]
                c["rme_both_le_tol"] += int(r["arm_A"]["restore_max_error"] <= TOL and s_rme is not None and s_rme <= TOL)
                c["rme_equal"] += int(r["arm_A"]["restore_max_error"] == s_rme)
    S["match"] = {k: dict(v) for k, v in per.items()}

    # 2. divergence checks
    ck = collections.Counter()
    for i, r in good.items():
        for e in r["executions"]:
            ck["exec"] += 1
            ck["b_source:" + e["b_source"]] += 1
            if "consistency" in e:
                ck["b_runs"] += 1
                ck["b_runs_consistent"] += int(e["consistency"]["ok"])
            if "pred_ok" in e:
                ck["verify_runs"] += 1
                ck["verify_ok"] += int(e["pred_ok"])
            if "identity_ok" in e:
                ck["identity_exec"] += 1
                ck["identity_ok"] += int(e["identity_ok"])
    S["divergence_checks"] = dict(ck)
    rep_sm = collections.Counter()
    for i, r in smoke.items():
        if i in res and "arm_A" in r and "arm_A" in res[i]:
            rep_sm["n"] += 1
            rep_sm["armA_equal"] += int(r["arm_A"]["rollouts"] == res[i]["arm_A"]["rollouts"])
            rep_sm["armB_equal"] += int(r["arm_B"]["rollouts"] == res[i]["arm_B"]["rollouts"])
            rep_sm["handoff_sha_equal"] += int(r["handoff"]["sha"] == res[i]["handoff"]["sha"])
    S["smoke_vs_full"] = dict(rep_sm)

    # 3. branch-level labels and gains
    H = collections.defaultdict(dict)
    for b in branches:
        H[(b["dir"], b["label_file"])][b["cand"]] = b
    rows = []
    for i, r in good.items():
        j = jobs[i]
        if j["cand"] == "none":
            continue
        nb = H[(j["dir"], j["label_file"])].get("none")
        y0 = nb["k"] / 8 if nb is not None and nb["valid"] else None
        yA, yB = j["stored_k"] / 8, r["arm_B"]["k"] / 8
        mv = (j.get("move") or {}).get("moved")
        rows.append(dict(dir=j["dir"], label_file=j["label_file"], task=j["task_b"], cand=j["cand"], role=j["role"], pr=j["pr"],
                         parent=j["parent_id"] or j["sample_id"], moved=mv, dpos=(j.get("move") or {}).get("dpos"),
                         y0=y0, yA=yA, yB=yB, dA=None if y0 is None else yA - y0, dB=None if y0 is None else yB - y0,
                         changed=int(r["arm_B"]["k"] != j["stored_k"])))
    tc = collections.defaultdict(list)
    for x in rows:
        tc[(x["task"], x["cand"])].append(x)
    tc_rows = []
    for (t, c), xs in sorted(tc.items()):
        g = [x for x in xs if x["y0"] is not None]
        tc_rows.append(dict(task=t, cand=c, n=len(xs), moved=sum(bool(x["moved"]) for x in xs),
                            yA=float(np.mean([x["yA"] for x in xs])), yB=float(np.mean([x["yB"] for x in xs])),
                            dA=float(np.mean([x["dA"] for x in g])) if g else None, dB=float(np.mean([x["dB"] for x in g])) if g else None,
                            n_changed=sum(x["changed"] for x in xs)))
    with open(inv.out_path("analysis/task_candidate.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(tc_rows[0].keys()) if tc_rows else ["task"])
        w.writeheader()
        for x in tc_rows:
            w.writerow(x)
    per_task = {}
    for t in sorted({x["task"] for x in rows}):
        xs = [x for x in rows if x["task"] == t]
        per_task[t] = dict(n=len(xs), moved=sum(bool(x["moved"]) for x in xs), changed=sum(x["changed"] for x in xs),
                           yA=float(np.mean([x["yA"] for x in xs])), yB=float(np.mean([x["yB"] for x in xs])),
                           diff_ci=boot_mean([x["yB"] - x["yA"] for x in xs], [x["parent"] for x in xs]))
    S["per_task"] = per_task

    # 4. execution-level transitions
    T = collections.defaultdict(collections.Counter)
    for i, r in good.items():
        j = jobs[i]
        if j["role"] == "none_identity":
            continue
        for e in r["executions"]:
            sa, sb = e["A"]["success"], e["label_B"]["success"]
            c = T[j["task_b"]]
            c[("S" if sa else "F") + ("S" if sb else "F")] += 1
            if sa and sb:
                d = e["label_B"]["steps"] - e["A"]["steps"]
                c["B_same_step" if d == 0 else ("B_later" if d > 0 else "B_earlier")] += 1
    S["transitions"] = {k: dict(v) for k, v in T.items()}

    # 5. rankings (only handoffs whose repair branches were all relabelled and passed the checks)
    rk = []
    for (d, lf), arms in H.items():
        nb = arms.get("none")
        if nb is None or not nb["valid"]:
            continue
        reps = {c: b for c, b in arms.items() if c != "none" and b["valid"]}
        if not reps:
            continue
        yA, yB, ok = {}, {}, True
        for c, b in reps.items():
            j = J.get((d, lf, c))
            if j is None or j["id"] not in good:
                ok = False
                break
            yA[c], yB[c] = b["k"] / 8, good[j["id"]]["arm_B"]["k"] / 8
        if not ok:
            continue
        task, y0 = nb["task_b"], nb["k"] / 8
        mA, mB = max(yA.values()), max(yB.values())
        bA = {c for c, v in yA.items() if v == mA}
        bB = {c for c, v in yB.items() if v == mB}
        cs = sorted(reps)
        ic = INVERSE.get(task)
        ic = ic if ic in reps else None
        rk.append(dict(dir=d, label_file=lf, task=task, pr=task in inv.PR_TASKS, parent=nb["parent_id"] or nb["sample_id"],
                       n_rep=len(reps), y0=y0, maxA=mA, maxB=mB, gainA=mA - y0, gainB=mB - y0, bestA=sorted(bA), bestB=sorted(bB),
                       a_best_still_best=bool(bA & bB), unique_a_best_still_best=(next(iter(bA)) in bB) if len(bA) == 1 else None,
                       regret_A_choice_under_B=mB - float(np.mean([yB[c] for c in bA])),
                       none_best_A=y0 >= mA, none_best_B=y0 >= mB, none_strict_A=y0 > mA, none_strict_B=y0 > mB,
                       tau=kendall([y0] + [yA[c] for c in cs], [y0] + [yB[c] for c in cs]), inverse=ic,
                       inverse_bestA=(ic in bA) if ic else None, inverse_bestB=(ic in bB) if ic else None,
                       inverse_gainA=(yA[ic] - y0) if ic else None, inverse_gainB=(yB[ic] - y0) if ic else None))
    inv.write_json("analysis/rankings.json", rk)
    rsum = {}
    for t in sorted({x["task"] for x in rk if x["pr"]}) + ["ALL_PR"]:
        xs = [x for x in rk if x["pr"] and (t == "ALL_PR" or x["task"] == t)]
        if not xs:
            continue
        mean = lambda k, xs=xs: (float(np.mean([x[k] for x in xs if x[k] is not None])) if any(x[k] is not None for x in xs) else None)
        rsum[t] = dict(n=len(xs), gainA=mean("gainA"), gainB=mean("gainB"),
                       gain_diff_ci=boot_mean([x["gainB"] - x["gainA"] for x in xs], [x["parent"] for x in xs]),
                       a_best_still_best=mean("a_best_still_best"), unique_a_best_still_best=mean("unique_a_best_still_best"),
                       regret=mean("regret_A_choice_under_B"), none_best_A=mean("none_best_A"), none_best_B=mean("none_best_B"),
                       none_strict_A=mean("none_strict_A"), none_strict_B=mean("none_strict_B"), tau=mean("tau"),
                       inverse_bestA=mean("inverse_bestA"), inverse_bestB=mean("inverse_bestB"),
                       inverse_gainA=mean("inverse_gainA"), inverse_gainB=mean("inverse_gainB"))
    S["rankings_pr"] = rsum

    # 6. lemma check (unmoved ⇒ unchanged) and escalation rule
    L = collections.defaultdict(collections.Counter)
    pr_unmoved_changed, esc = [], set()
    for i, r in good.items():
        j = jobs[i]
        if j["role"] == "none_identity":
            continue
        mv = (j.get("move") or {}).get("moved")
        key = f'{j["role"]}|{"moved" if mv else ("unmoved" if mv is False else "n/a")}'
        ch = int(r["arm_B"]["k"] != r["arm_A"]["k"])
        ex = sum(e["A"]["success"] != e["label_B"]["success"] for e in r["executions"])
        L[key]["branches"] += 1; L[key]["branches_changed"] += ch
        L[key]["exec"] += 8; L[key]["exec_changed"] += ex
        if mv is False and ex:
            if j["pr"]:
                pr_unmoved_changed.append(dict(id=i, task=j["task_b"], cand=j["cand"], dpos=j["move"]["dpos"], dorn=j["move"]["dorn"]))
            else:
                esc.add(j["task_b"])
    S["lemma"] = {k: dict(v) for k, v in L.items()}
    S["pr_unmoved_changed"] = pr_unmoved_changed
    S["escalate_tasks"] = sorted(esc)

    # 7. mirrored directories, mapping, prior table
    mapping = write_mirror(J, good)
    inv.write_json("mapping.json", dict(created=S["created"], verified=S["complete_and_verified"], entries=mapping))
    S["label_sources"] = dict(collections.Counter(m["source"] for m in mapping))
    pri = collections.defaultdict(lambda: collections.defaultdict(lambda: [[], []]))
    for m in mapping:
        if m["candidate"] != "none" and m["source"] != "stored_invalid":
            pri[m["task_b"]][m["candidate"]][0].append(m["k_A_stored"] / 8)
            pri[m["task_b"]][m["candidate"]][1].append(m["k_B"] / 8)
    prior = {}
    for t, cs in pri.items():
        tab = {c: (float(np.mean(v[0])), float(np.mean(v[1])), len(v[0])) for c, v in cs.items() if len(v[0]) >= 5}
        prior[t] = dict(topA=sorted(tab, key=lambda c: -tab[c][0])[:3], topB=sorted(tab, key=lambda c: -tab[c][1])[:3],
                        table={c: dict(A=a, B=b, n=n) for c, (a, b, n) in tab.items()})
    S["prior_table"] = prior

    # 8. cost
    spp, c1 = rates(good)
    S["cost"] = dict(seconds=sum(r.get("seconds", 0) for r in good.values()), steps_run=sum(r["checks"]["steps_run"] for r in good.values()),
                     sec_per_step=spp, restore_sec_per_action=c1, b_runs=sum(r["checks"]["n_b_runs"] for r in good.values()))
    inv.write_json("analysis/summary.json", S)
    inv.write_json("analysis/branch_rows.json", rows)
    write_report(S, tc_rows)
    print("wrote report.md, analysis/, labels_bdef/, mapping.json; verified =", S["complete_and_verified"])
    if not S["complete_and_verified"]:
        print(f"warning: {len(S['failed'])} branches failed the checks, {S['missing']} missing results; their labels keep the original values in the mirror and are marked RELABEL_FAILED_OR_MISSING")
    if S["escalate_tasks"]:
        print("escalation rule triggered: bash run.sh escalate " + ",".join(S["escalate_tasks"]))


def write_report(S, tc_rows):
    m = S["match"].get("ALL|ALL", {})
    ck = S["divergence_checks"]
    L = S["lemma"]
    s = "# B_def relabelling results (round 6a)\n\n"
    s += f"Generated {S['created']}. Data directory: `{R}`.\n\n## 0. Summary\n\n"
    s += f"- Jobs {S['jobs']}, with results {S['results']}, all checks passed {S['good']}, missing {S['missing']}, failed {len(S['failed'])}. "
    s += f"Complete with all checks passed: **{S['complete_and_verified']}**.\n"
    if m:
        s += (f"- Mode A vs. stored labels: execution level {m.get('exec_match', 0)}/{m.get('exec', 0)} exactly identical (success and steps); "
              f"branch-level k {m.get('k_match', 0)}/{m.get('arms', 0)}.\n")
    s += (f"- Mode B re-runs {ck.get('b_runs', 0)}, consistent before the divergence point {ck.get('b_runs_consistent', 0)}; verify sample {ck.get('verify_runs', 0)} runs, "
          f"\"B inferred from A\" agrees with the actual run {ck.get('verify_ok', 0)}; none identity {ck.get('identity_ok', 0)}/{ck.get('identity_exec', 0)}.\n")
    if "ALL_PR" in S["rankings_pr"]:
        a = S["rankings_pr"]["ALL_PR"]
        s += (f"- push/rotate handoffs (n={a['n']}): best-repair gain (max_u ŷ_u − ŷ_∅) {a['gainA']:+.3f} under A, {a['gainB']:+.3f} under B_def; "
              f"best repair under A is still best under B_def {pct(a['a_best_still_best'])}; \"no repair\" weakly best A {pct(a['none_best_A'])} → B {pct(a['none_best_B'])}.\n")
    s += f"- Escalation rule (label change on unmoved branches ⇒ relabel that task fully): {'triggered ' + ','.join(S['escalate_tasks']) if S['escalate_tasks'] else 'not triggered'}.\n\n"
    s += "## 1. Mode A vs. stored labels (field by field)\n\n"
    rows = [[k, v.get("exec", 0), v.get("exec_match", 0), v.get("success_match", 0), v.get("steps_match", 0), v.get("arms", 0),
             v.get("k_match", 0), v.get("n_match", 0), v.get("rme_both_le_tol", 0), v.get("rme_equal", 0)] for k, v in sorted(S["match"].items())]
    s += md_table(["dir|role", "executions", "exact match", "success match", "steps match", "branches", "k match", "n=8", "restore error both ≤1e-6", "restore error equal"], rows)
    s += f"\nSmoke vs. full run on the same branch: {S['smoke_vs_full']}\n\n## 2. Divergence checks\n\n"
    s += "".join(f"- {k}: {v}\n" for k, v in sorted(ck.items()))
    if S["failed"]:
        s += "\nFailed branches (first 20):\n\n" + "".join(f"- {i}: {w}\n" for i, w in list(S["failed"].items())[:20])
    s += "\n## 3. Label changes (per task)\n\n"
    rows = [[t, v["n"], v["moved"], v["changed"], v["yA"], v["yB"], v["diff_ci"][0],
             f"[{v['diff_ci'][1]:+.3f}, {v['diff_ci'][2]:+.3f}]" if v["diff_ci"][1] is not None else ""] for t, v in S["per_task"].items()]
    s += md_table(["task", "relabelled branches", "reference block moved", "k changed", "ŷ_A", "ŷ_B", "ŷ_B−ŷ_A", "95% CI (clustered by initial state)"], rows)
    s += "\nExecution-level transitions (S/F = success/failure, A→B):\n\n"
    rows = [[t, v.get("SS", 0), v.get("SF", 0), v.get("FS", 0), v.get("FF", 0), v.get("B_same_step", 0), v.get("B_later", 0),
             v.get("B_earlier", 0)] for t, v in sorted(S["transitions"].items())]
    s += md_table(["task", "SS", "SF", "FS", "FF", "SS same step", "SS B later", "SS B earlier"], rows)
    s += "\nPer task × candidate (n ≥ 5, top 40 by |Δ̂_B − Δ̂_A| descending; full table in analysis/task_candidate.csv):\n\n"
    tr = [x for x in tc_rows if x["n"] >= 5 and x["dA"] is not None]
    tr.sort(key=lambda x: -abs(x["dB"] - x["dA"]))
    s += md_table(["task", "candidate", "n", "moved", "ŷ_A", "ŷ_B", "Δ̂_A", "Δ̂_B", "k changed"],
                  [[x["task"], x["cand"], x["n"], x["moved"], x["yA"], x["yB"], x["dA"], x["dB"], x["n_changed"]] for x in tr[:40]])
    s += "\n## 4. Candidate ranking per handoff (push/rotate)\n\n"
    rows = [[t, v["n"], v["gainA"], v["gainB"], pct(v["a_best_still_best"]), pct(v["unique_a_best_still_best"]), v["regret"],
             pct(v["none_best_A"]), pct(v["none_best_B"]), pct(v["none_strict_A"]), pct(v["none_strict_B"]), v["tau"],
             pct(v["inverse_bestA"]), pct(v["inverse_bestB"]), v["inverse_gainA"], v["inverse_gainB"]] for t, v in S["rankings_pr"].items()]
    s += md_table(["task", "handoffs", "best-repair gain A", "best-repair gain B", "A-best still best", "(unique A-best)", "regret of A choice under B",
                   "none weakly best A", "none weakly best B", "none strictly best A", "none strictly best B", "Kendall τ", "inverse skill best A", "inverse skill best B",
                   "inverse skill Δ̂ A", "inverse skill Δ̂ B"], rows)
    s += "\n## 5. Lemma check: repair does not move the reference block ⇒ label unchanged\n\n"
    rows = [[k, v["branches"], v["branches_changed"], v["exec"], v["exec_changed"]] for k, v in sorted(L.items())]
    s += md_table(["role|moved", "branches", "k changed", "executions", "outcome changed"], rows)
    s += f"\npr branches unmoved but with changed labels: {len(S['pr_unmoved_changed'])} (details in analysis/summary.json; all should be threshold-edge cases).\n"
    s += f"Escalation rule: {'triggered: bash run.sh escalate ' + ','.join(S['escalate_tasks']) if S['escalate_tasks'] else 'not triggered; unmoved non-pr branches keep their original labels'}.\n"
    s += "\n## 6. Prior table (top 3 candidates per task by mean label)\n\n"
    rows = [[t, ", ".join(v["topA"]), ", ".join(v["topB"])] for t, v in sorted(S["prior_table"].items())]
    s += md_table(["task", "top 3 under A", "top 3 under B_def"], rows)
    s += "\n## 7. Outputs and stage-2 usage\n\n"
    s += ("- `labels_bdef/<dir>/results/label_<id>.json`: same format as the original files (job_id, sample_id, task_b, arms[candidate, rollouts[repeat, success, steps], "
          "k, n, restore_max_error]), plus `criterion` and a per-arm `label_source`. The remaining result files (j8_*, screen_*, etc.), "
          "screen/, boundaries/, analysis/ and the jobs files are symlinks to the original directories. For stage 2, point MT/GT/NT at these directories.\n")
    s += f"- Label source counts: {S['label_sources']}\n- `mapping.json`: per-arm source, original k, re-run k_A, k_B.\n"
    s += "- Note: the gate/nt feature caches of train_benefit_v3.py use fixed file names; stage 2 must use new cache names or confirm the caches are path-independent.\n"
    c = S["cost"]
    s += (f"\n## 8. Cost\n\nTotal {c['seconds'] / 3600:.1f} process-hours, {c['steps_run']} steps executed, {c['b_runs']} mode-B re-runs; "
          f"{(c['sec_per_step'] or 0):.4f} s/step (excluding restore), restore {(c['restore_sec_per_action'] or 0):.5f} s/action.\n")
    out = inv.out_path("report.md")
    out.write_text(s, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--smoke-check", action="store_true")
    a = ap.parse_args()
    if a.status:
        status()
    elif a.smoke_check:
        sys.exit(smoke_check())
    else:
        analyze()


if __name__ == "__main__":
    main()
