# -*- coding: utf-8 -*-
"""T2 job planning: writes jobs.json (full run), jobs_smoke.json (smoke test), or jobs_escalate.json with --escalate.

Default relabelling scope (rationale in report.md):
  pr_rep               all repair branches with a push/rotate target;
  none_identity        10% sample of the none branches at pr handoffs (by handoff hash), mode A only, checking B_def ≡ B_orc ≡ A;
  other_moved          all branches with a non-pr target whose repair moved the reference block;
  other_unmoved_sample 10% sample of unmoved branches with a non-pr target (lemma check);
  verify_b             a further 10% of the repair branches above, re-running mode B for all 8 executions.
--escalate t1,t2: add the remaining unmoved branches of these tasks to jobs_escalate.json (used when the escalation rule triggers).
Jobs are sorted by estimated cost in descending order (longest first). Rewriting jobs.json is refused once results exist (keeps runs resumable).
Usage: python -B plan_jobs.py [--none-frac 0.1] [--other-unmoved-frac 0.1] [--verify-frac 0.1]
       python -B plan_jobs.py --escalate lift_pink_block_drawer
"""
import argparse, collections, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import inventory as inv  # noqa: E402


def pick_smoke(jobs):
    key = lambda j: inv.frac("smoke|" + j["id"])
    ests = sorted(j["est"] for j in jobs)
    med = ests[len(ests) // 2] if ests else 0
    moved = lambda j: (j.get("move") or {}).get("moved", True)
    want = [("pr_rep", lambda j: j["stored_k"] >= 1 and moved(j), 2),
            ("pr_rep", lambda j: j["stored_k"] >= 1 and not moved(j), 1),
            ("pr_rep", lambda j: j["stored_k"] == 0, 1),
            ("none_identity", lambda j: j["stored_k"] >= 1, 1),
            ("other_moved", lambda j: j["stored_k"] >= 1, 1),
            ("other_moved", lambda j: True, 1),
            ("other_unmoved_sample", lambda j: True, 1)]
    sel, used = [], set()
    for role, cond, n in want:
        pool = [j for j in sorted(jobs, key=key) if j["role"] == role and cond(j) and j["id"] not in used]
        pool = [j for j in pool if j["est"] <= med] + [j for j in pool if j["est"] > med]
        for j in pool[:n]:
            sel.append(dict(j, verify_b=True))
            used.add(j["id"])
    return sel


def write_jobs(name, jobs, meta):
    return inv.write_json(name, dict(meta, created=time.strftime("%F %T"), jobs=jobs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--none-frac", type=float, default=inv.SCOPE["none_frac"])
    ap.add_argument("--other-unmoved-frac", type=float, default=inv.SCOPE["unmoved_frac"])
    ap.add_argument("--verify-frac", type=float, default=inv.SCOPE["verify_frac"])
    ap.add_argument("--escalate", default=None, help="comma-separated non-pr task names")
    a = ap.parse_args()
    R = inv.rroot()
    branches, problems, files = inv.enumerate_branches()
    inv.check_files(branches)
    if a.escalate:
        tasks = [t for t in a.escalate.split(",") if t]
        if set(tasks) & inv.PR_TASKS:
            raise SystemExit("pr tasks are already fully relabelled; no escalation needed")
        have = {j["id"] for f in ("jobs.json", "jobs_escalate.json") if (R / f).exists()
                for j in json.loads((R / f).read_text())["jobs"]}
        old = json.loads((R / "jobs_escalate.json").read_text())["jobs"] if (R / "jobs_escalate.json").exists() else []
        new = []
        for b in branches:
            if b["cls"] == "other_rep" and b["valid"] and not b.get("file_problem") and b["task_b"] in tasks:
                j = inv.make_job(b, "other_unmoved_escalated", False)
                if j["id"] not in have:
                    new.append(j)
        allj = sorted(old + new, key=lambda j: -j["est"])
        write_jobs("jobs_escalate.json", allj, dict(purpose="escalation of unmoved non-pr branches", tasks=tasks))
        print(f"jobs_escalate.json: {len(new)} added, {len(allj)} in total")
        return
    jobs = sorted(inv.select_scope(branches, a.none_frac, a.other_unmoved_frac, a.verify_frac), key=lambda j: -j["est"])
    meta = dict(purpose="B_def relabel of label branches (round 6a)",
                params=dict(none_frac=a.none_frac, other_unmoved_frac=a.other_unmoved_frac, verify_frac=a.verify_frac,
                            pos_tol=inv.POS_TOL, orn_tol=inv.ORN_TOL),
                label_worker_sha256=inv.sha_file(inv.LABEL_WORKER),
                n_file_problems=sum(1 for b in branches if b.get("file_problem")), n_problems=len(problems))
    jp = R / "jobs.json"
    if jp.exists():
        old_ids = [j["id"] for j in json.loads(jp.read_text())["jobs"]]
        if set(old_ids) == {j["id"] for j in jobs}:
            print("jobs.json already exists with the same job set; left unchanged")
        elif any((R / "results").glob("*.json")) if (R / "results").exists() else False:
            raise SystemExit("jobs.json already has results and the new plan differs; refusing to rewrite")
        else:
            write_jobs("jobs.json", jobs, meta)
    else:
        write_jobs("jobs.json", jobs, meta)
    sp = R / "jobs_smoke.json"
    if not sp.exists():
        smoke = pick_smoke(jobs)
        write_jobs("jobs_smoke.json", smoke, dict(purpose="smoke gate", verify_b_forced=True))
    else:
        smoke = json.loads(sp.read_text())["jobs"]
    c = collections.Counter((j["dir"], j["role"]) for j in jobs)
    print(f"jobs.json: {len(jobs)} branches; smoke test {len(smoke)}")
    for k in sorted(c):
        print(f"  {k[0]:28s} {k[1]:22s} {c[k]:5d}")
    t = collections.Counter((j["task_b"], j["role"]) for j in jobs)
    for k in sorted(t):
        print(f"  {k[0]:28s} {k[1]:22s} {t[k]:5d}")
    print("smoke test:", [(j["role"], j["task_b"], j["cand"], j["stored_k"]) for j in smoke])


if __name__ == "__main__":
    main()
