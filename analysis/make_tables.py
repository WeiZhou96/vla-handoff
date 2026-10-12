# -*- coding: utf-8 -*-
"""Write the numeric LaTeX table rows of the paper (rows only; environments, captions and headers are in the manuscript).

Sources
  data/derived/stats.json            compute_stats.py: decomposition, LIBERO-Goal, cross-scene transfer (original
                                     criterion), offline labels of the added tasks, pre-registered round-2 analysis (A)
  data/derived/fullchain_stats.json  compute_fullchain.py: closed-loop results under the original criterion (A), post hoc
                                     rescoring (i) and the handoff-referenced counterfactual re-execution (ii)
Outputs (tables/)
  tab_round1_rows, tab_round2arms_rows, tab_round2_rows (hypotheses), tab_audit_rows, tab_criterion_rows,
  tab_replay_rows, tab_across_rows, tab_scenes_rows, tab_libero_rows, tab_decomp_rows, tab_vocabnew_rows  (*.tex)
Round-3 tables (tab_r3hyp_rows, tab_r3arms_rows, tab_relabel_rows, tab_offlineB_rows, tab_r3lock_rows) are written from
data/derived/round3_stats.json (compute_round3.py); the tables above are unchanged.
The offline selection table under criterion A (tab_offline_rows.tex) is no longer written: its pooled labels include push and rotation
targets, which are scored relative to the post-repair state.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
S = json.loads((ROOT / "data/derived/stats.json").read_text(encoding="utf-8"))
F = json.loads((ROOT / "data/derived/fullchain_stats.json").read_text(encoding="utf-8"))
OUT = ROOT / "tables"
OUT.mkdir(exist_ok=True)
WRITTEN = []
# Task display names, identical to the figures (analysis/figcommon.py) and to the task names in the manuscript
TASK = {
    "push_blue_block_right": "push blue block right", "push_red_block_right": "push red block right",
    "push_pink_block_right": "push pink block right", "lift_blue_block_slider": "lift blue block (slider)",
    "lift_pink_block_slider": "lift pink block (slider)", "lift_red_block_slider": "lift red block (slider)",
    "lift_pink_block_drawer": "lift pink block (drawer)", "rotate_red_block_left": "rotate red block left",
    "rotate_pink_block_right": "rotate pink block right", "push_into_drawer": "push into drawer", "stack_block": "stack blocks",
}
ARMNAME = {"none": "No intervention", "retry": "Retry once", "fixed": "Fixed repair per task",
           "home_all": "Home before every target", "prior": "Task-prior repair", "inverse": "Inverse-skill rule",
           "risk": "Risk gate", "benefit": "Benefit gate (ours)", "full": "Ours + selective retry",
           "benefit_noretry": "Ours, no retry", "noloc": "Ours, locator-free", "benefit_m5": "Ours, no prior input",
           "benefit_alt": "Ours, no prior input"}
# Semantic shading (see the table notes): warm = benefit gate (ours), cool = rule-based comparators, grey = pooled rows.
WARM, COOL, GROUP = "\\rowcolor{qaetint}", "\\rowcolor{retrtint}", "\\rowcolor{grouptint}"


def write(name, lines):
    (OUT / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    WRITTEN.append(name)


def bold_if(s, cond):
    return f"\\textbf{{{s}}}" if cond else s


def excl0(v):
    return v[1] > 0 or v[2] < 0


def pp(x, sign=True, nd=1):
    v = 100 * x
    s = f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"
    return s.replace("-", "$-$").replace("+", "$+$") if sign else s


def ci(v, nd=1):
    lo = f"{100*v[1]:.{nd}f}".replace("-", "$-$")
    hi = f"{100*v[2]:.{nd}f}".replace("-", "$-$")
    return f"[{lo}, {hi}]"


def est_ci(v, nd=1, bold=False):
    e = pp(v[0], nd=nd)
    if bold:
        e = f"\\textbf{{{e}}}"
    return f"{e} {{\\scriptsize{ci(v, nd)}}}"


def muted(s):
    return f"\\textcolor{{muted}}{{{s}}}"


# ---------------- Closed-loop tables of rounds 1 and 2: (ii) main columns, A in grey ----------------
POSTHOC1 = {"prior", "inverse", "benefit_noretry", "home_all", "noloc", "benefit_m5"}
GROUPS1 = [("Baselines", ["none", "retry", "fixed", "home_all"]), ("Repair rules", ["prior", "inverse"]),
           ("Learned selector", ["risk", "benefit", "full", "benefit_noretry", "noloc", "benefit_m5"])]
GROUPS2 = [("Baselines", ["none", "retry", "home_all"]), ("Repair rules", ["prior", "inverse"]),
           ("Learned selector", ["benefit", "benefit_noretry", "noloc", "benefit_alt"])]


def closed_loop_rows(rd, groups, posthoc=()):
    arms = [a for _, g in groups for a in g]
    A = {a: F["arms"][f"r{rd}|{a}"] for a in arms}
    levels = sorted({round(A[a]["S_ii"], 9) for a in arms}, reverse=True)
    lines = []
    for g, items in groups:
        lines.append(f"\\multicolumn{{8}}{{@{{}}l}}{{\\textit{{{g}}}}} \\\\")
        for a in items:
            s = A[a]
            lev = round(s["S_ii"], 9)
            sv = f"{s['S_ii']:.3f}"
            if lev == levels[0]:
                sv = f"\\textbf{{{sv}}}"
            elif len(levels) > 1 and lev == levels[1]:
                sv = f"\\underline{{{sv}}}"
            bar = "barblue" if a == "benefit" else "bargrey"
            note = "\\tnote{a}" if a in posthoc else ""
            if a == "retry":
                dii, dA = "--", muted("--")
            else:
                P = F["paired"][f"r{rd}|all|{a}-retry"]
                v = P["S_ii"]
                dii = f"\\marg{{{pp(v[0])}}} {{\\scriptsize{ci(v)}}}" if a == "benefit" else est_ci(v)
                dA = muted(est_ci(P["S_A"]))
            shade = WARM if a == "benefit" else COOL if a in ("prior", "inverse") else ""
            sA = muted(f"{s['S_A']:.3f}")
            succ_bar = f"\\npbar[{bar}]{{{sv}}}{{{100*s['S_ii']:.1f}}}"
            lines.append(f"{shade}\\quad {ARMNAME[a]}{note} & {succ_bar} & {sA} & {s['N_ii']:.0f} & {s['U_ii']:.3f} & "
                         f"{s['rep_ii']:.2f} & {dii} & {dA} \\\\")
        lines.append("\\addlinespace[2pt]")
    return lines[:-1]


write("tab_round1_rows.tex", closed_loop_rows(1, GROUPS1, POSTHOC1))
write("tab_round2arms_rows.tex", closed_loop_rows(2, GROUPS2))

# ---------------- Round-2 hypotheses: pre-registered A (stats.json) and post hoc (ii) ----------------
R2S = S["round2"]["paired"]
FAM = S["round2"]["first_target"]["by_family"]["new_noninverse"]
FH = F["H"]
LOW0 = lambda v: v[1] > 0
HSPEC = [
    ("H1", "Ours $-$ task-prior repair, chain success", R2S["benefit-prior"]["success"], FH["H1"]["S_ii"], "$>0$", LOW0, 1),
    ("H2", "Ours $-$ inverse-skill rule, chain success", R2S["benefit-inverse"]["success"], FH["H2"]["S_ii"], "$>0$", LOW0, 1),
    ("H3", f"Ours $-$ task prior, first handoff, no-inverse tasks ($n={FAM['n']}$)", FAM["benefit-prior"], FH["H3"]["f_Bii"],
     "$>0$", LOW0, 1),
    ("H4", "Locator-free $-$ ours, chain success", R2S["noloc-benefit"]["success"], FH["H4"]["S_ii"], "lower bound $>-5$",
     lambda v: v[1] > -0.05, 2),
    ("H5a", "Ours without retry $-$ no intervention", R2S["benefit_noretry-none"]["success"], FH["H5a"]["S_ii"], "$>0$", LOW0, 1),
    ("H5b", "Ours without retry $-$ retry once", R2S["benefit_noretry-retry"]["success"], FH["H5b"]["S_ii"], "$\\ge 0$",
     lambda v: v[0] >= 0, 1),
]
if FH["H3"]["n"] != FAM["n"]:
    print(f"note: H3 n differs between the pre-registered analysis ({FAM['n']}) and the replay ({FH['H3']['n']})")


def outcome(v, rule, h):
    if not rule(v):
        return "\\textcolor{muted}{not supported}"
    if h == "H4" and v[1] < -0.045:
        return "\\cellcolor{supptint}supported (narrowly)"
    return "\\cellcolor{supptint}\\textbf{supported}"


lines = []
for h, desc, va, vii, crit, rule, nd in HSPEC:
    eii = pp(vii[0], nd=nd)
    if h == "H1":
        eii = f"\\marg{{{eii}}}"
    lines.append(f"{h} & {desc} & {crit} & {pp(va[0], nd=nd)} & {ci(va, nd)} & {outcome(va, rule, h)} & "
                 f"{eii} & {ci(vii, nd)} & {outcome(vii, rule, h)} \\\\")
write("tab_round2_rows.tex", lines)

# ---------------- Predicate audit ----------------
AUD, AP = F["audit"], F["audit_pooled"]
BDEF = {"push": "$x_{\\mathrm{end}}-x_{h}>10$\\,cm; contact condition at attempt start",
        "rotate": "CALVIN predicate with start $:=$ handoff ($=B_{\\mathrm{orc}}$)",
        "lift": "height gain from handoff; on source surface at attempt start",
        "push_into": "CALVIN predicate; block not in drawer at handoff",
        "stack": "CALVIN predicate; blocks apart at handoff and at attempt start"}
FLAGLAB = {"push": "no non-robot contact", "lift": "not on source surface", "push_into": "block already in drawer",
           "stack": "blocks already touching"}
PR_ORDER = ["push_red_block_right", "push_blue_block_right", "push_pink_block_right", "rotate_pink_block_right",
            "rotate_red_block_left"]
OTHER_ORDER = ["lift_red_block_slider", "lift_blue_block_slider", "lift_pink_block_slider", "lift_pink_block_drawer",
               "push_into_drawer", "stack_block"]


def flag_text(a):
    if a["kind"] == "rotate":
        return f"$|$cumulative yaw$|\\ge180^\\circ$: {a['flag_yaw180']}"
    return f"{FLAGLAB[a['kind']]}: {a['flag_handoff']} / {a['flag_start']}"


def share(x, n):
    return f"{{\\scriptsize({100*x/n:.1f}\\%)}}" if n else ""


lines = []
for block in (PR_ORDER, OTHER_ORDER):
    for t in block:
        a = AUD.get(t)
        if a is None:
            print("audit: task missing", t)
            continue
        lines.append(f"{TASK[t]} & {BDEF[a['kind']]} & {a['n']} & {a['repaired']} & {a['A_ne_Bdef']} {share(a['A_ne_Bdef'], a['n'])} & "
                     f"{a['A_ne_Borc']} & {a['Bdef_ne_Borc']} & {flag_text(a)} \\\\")
    lines.append("\\midrule")
for key, name in (("pr", "Push and rotation targets"), ("other", "Other targets")):
    a = AP[key]
    num = f"\\marg{{{a['A_ne_Bdef']}}}" if key == "pr" else f"{a['A_ne_Bdef']}"
    lines.append(f"{GROUP}{name} & & {a['n']} & {a['repaired']} & {num} {share(a['A_ne_Bdef'], a['n'])} & "
                 f"{a['A_ne_Borc']} & {a['Bdef_ne_Borc']} & \\\\")
write("tab_audit_rows.tex", lines)

# ---------------- All paired comparisons under A, (i), (ii) (appendix) ----------------
NM = {"benefit": "ours", "retry": "retry", "none": "no intervention", "prior": "task prior", "inverse": "inverse rule",
      "home_all": "home before every target", "benefit_noretry": "ours without retry", "noloc": "locator-free",
      "benefit_m5": "no prior input", "benefit_alt": "no prior input", "risk": "risk gate", "full": "selective retry",
      "fixed": "fixed repair"}
R1C = [("benefit", "retry"), ("benefit", "none"), ("benefit", "prior"), ("benefit", "inverse"), ("retry", "none"),
       ("inverse", "none"), ("prior", "none"), ("inverse", "retry"), ("prior", "retry"), ("benefit_noretry", "retry"),
       ("benefit", "benefit_noretry"), ("benefit", "home_all"), ("home_all", "retry"), ("benefit", "fixed"),
       ("fixed", "retry"), ("benefit", "risk"), ("full", "benefit"), ("noloc", "benefit"), ("benefit_m5", "benefit")]
R2C = [("benefit", "retry"), ("benefit", "none"), ("benefit", "prior"), ("benefit", "inverse"), ("retry", "none"),
       ("inverse", "none"), ("prior", "none"), ("inverse", "retry"), ("prior", "retry"), ("benefit_noretry", "none"),
       ("benefit_noretry", "retry"), ("benefit", "benefit_noretry"), ("benefit", "home_all"), ("home_all", "retry"),
       ("noloc", "benefit"), ("benefit_alt", "benefit")]


def label(X, Y):
    x = NM[X]
    return f"{x[0].upper()}{x[1:]} $-$ {NM[Y]}"


def cell(v):
    return est_ci(v, bold=excl0(v)) if v else "--"


lines = []
for rd, comps in ((1, R1C), (2, R2C)):
    a0 = F["paired"][f"r{rd}|all|benefit-retry"]
    npr, nnp = F["paired"][f"r{rd}|pr|benefit-retry"]["n"], F["paired"][f"r{rd}|nonpr|benefit-retry"]["n"]
    lines.append(f"\\multicolumn{{7}}{{@{{}}l}}{{\\textit{{Round {rd}: {a0['n']} chains ({npr} with and {nnp} without a "
                 f"push or rotation target), {a0['bases']} base configurations}}}} \\\\")
    for X, Y in comps:
        a = F["paired"].get(f"r{rd}|all|{X}-{Y}")
        if not a:
            print("criterion table: missing", rd, X, Y)
            continue
        p = F["paired"].get(f"r{rd}|pr|{X}-{Y}")
        q = F["paired"].get(f"r{rd}|nonpr|{X}-{Y}")
        lines.append(f"\\quad {label(X, Y)} & {cell(a['S_A'])} & {cell(a['S_i'])} & {cell(a['S_ii'])} & {cell(a['L_ii'])} & "
                     f"{cell(p['S_ii'] if p else None)} & {cell(q['S_ii'] if q else None)} \\\\")
    lines.append("\\addlinespace[3pt]")
write("tab_criterion_rows.tex", lines[:-1])

# ---------------- Replay verification per round and arm (appendix) ----------------
ORDER = {1: ["none", "retry", "fixed", "home_all", "prior", "inverse", "risk", "benefit", "full", "benefit_noretry", "noloc",
             "benefit_m5"],
         2: ["none", "retry", "home_all", "prior", "inverse", "benefit", "benefit_noretry", "noloc", "benefit_alt"]}
lines = []
for rd in (1, 2):
    for arm in ORDER[rd]:
        a, d = F["arms"].get(f"r{rd}|{arm}"), F["dv"].get(f"r{rd}|{arm}")
        if a is None or d is None:
            print("replay table: missing", rd, arm)
            continue
        shade = WARM if arm == "benefit" else ""
        lines.append(f"{shade}{rd} & {ARMNAME[arm]} & {a['n']} & {a['match']:.3f} & {d['ok']:.3f} & {100*d['identical']:.1f} & "
                     f"{a['S_A']:.3f} & {a['S_i']:.3f} & {a['S_io']:.3f} & {a['S_ii']:.3f} & {a['A0B1_stop']} \\\\")
    if rd == 1:
        lines.append("\\midrule")
write("tab_replay_rows.tex", lines)

# ---------------- Across-task correlations under (ii), A in grey parentheses (appendix, conditional) ----------------
AC = F["across"]


def rho_txt(x):
    return "--" if x["rho"] is None else f"{x['rho']:.2f}".replace("-", "$-$")


lines = []
for arm, name in (("retry", "Retry once"), ("prior", "Task-prior repair"), ("benefit", "Benefit gate (ours)")):
    cells = []
    for c in ("state_share", "mixed", "all_fail", "success"):
        x, a = AC["cells"][f"{arm}|f_Bii|{c}"], AC["cells"][f"{arm}|f_A|{c}"]
        cells.append(f"{bold_if(rho_txt(x), x['bold'])} {{\\scriptsize\\textcolor{{muted}}{{({rho_txt(a)})}}}}")
    shade = WARM if arm == "benefit" else COOL if arm == "prior" else ""
    lines.append(f"{shade}{name} & " + " & ".join(cells) + " \\\\")
write("tab_across_rows.tex", lines)
print("across: any bold cell under (ii) =", AC["any_bold_ii"], "(place tab:across only if True)")

# ---------------- Table: cross-scene (original criterion, not re-evaluated; appendix) ----------------
SC = S["scenes"]
arms = [("none", "No intervention"), ("retry", "Retry once"), ("inverse", "Inverse-skill rule"), ("prior", "Task-prior repair"),
        ("benefit", "Benefit gate (ours)"), ("benefit_m5", "Ours, no prior input")]
lines = []
for a, name in arms:
    cells = []
    for t in ("scA", "scB", "scC"):
        s = SC[t]["summary"][a]
        bestv = max(SC[t]["summary"][x]["success"] for x, _ in arms)
        v = f"{s['success']:.3f}"
        if s["success"] == bestv:
            v = f"\\textbf{{{v}}}"
        cells.append(f"{v} & {s['steps']:.0f}")
    shade = WARM if a == "benefit" else COOL if a in ("prior", "inverse") else ""
    lines.append(f"{shade}{name} & " + " & ".join(cells) + " \\\\")
lines.append("\\midrule")
for cmp_, name in [("retry", r"Ours $-$ retry"), ("inverse", r"Ours $-$ inverse"), ("prior", r"Ours $-$ prior")]:
    cells = []
    for t in ("scA", "scB", "scC"):
        v = SC[t]["paired"][f"benefit-{cmp_}"]["success"]
        cells.append(f"\\multicolumn{{2}}{{c}}{{{bold_if(pp(v[0]), excl0(v))}\\,{{\\scriptsize{ci(v).replace(', ', ',')}}}}}")
    lines.append(f"{name} & " + " & ".join(cells) + " \\\\")
write("tab_scenes_rows.tex", lines)

# ---------------- Table: LIBERO gate ----------------
G = S["libero"]["gate"]
rows = [("No repair (fresh seeds)", None, None, G["none_fresh_mean_success"]),
        ("Vocabulary repair, best of 9 (optimistic)", G["vocab_best_of_9_optimistic"], "o", None),
        ("Vocabulary repair, split estimate (pre-specified)", G["vocab_split_pick_seed8_eval_seed9"], "s", None),
        ("Vocabulary repair, split estimate (swapped)", G["vocab_split_swapped"], "s", None),
        ("Scripted homing", G["homing"], "h", G["homing"]["mean_success"]),
        ("Wait 50 steps", G["wait50"], "w", G["wait50"]["mean_success"])]
lines = []
for name, g, kind, ms in rows:
    if g is None:
        lines.append(f"{name} & -- & -- & {ms:.3f} \\\\")
        continue
    sh = 100 * g["fraction_gain_ge_0_25"]
    bar = "barblue" if kind in ("o", "s") else "bargrey"
    fr = f"\\npbar[{bar}]{{{sh:.1f}}}{{{sh:.1f}}}\\ {{\\scriptsize[{100*g['ci95'][0]:.1f}, {100*g['ci95'][1]:.1f}]}}"
    mg = f"{pp(g['mean_gain'])} {{\\scriptsize[{100*g['mean_gain_ci95'][0]:.1f}, {100*g['mean_gain_ci95'][1]:.1f}]}}".replace("[-", "[$-$")
    shade = WARM if "pre-specified" in name else ""
    lines.append(f"{shade}{name} & {fr} & {mg} & {('%.3f' % ms) if ms is not None else '--'} \\\\")
write("tab_libero_rows.tex", lines)

# ---------------- Appendix: per-task decomposition ----------------
D = S["decomposition"]
order = sorted(D["by_task"], key=lambda t: -D["by_task"][t]["state_share"])
lines = []
for t in order:
    d = D["by_task"][t]
    ss = 100 * d["state_share"]
    lines.append(f"{TASK[t]} & {d['n']} & {d['success']:.3f} & \\npbar[barstate]{{{ss:.1f}}}{{{ss:.1f}}}\\ "
                 f"{{\\scriptsize[{100*d['state_share_ci'][0]:.1f}, {100*d['state_share_ci'][1]:.1f}]}} & {100*d['noise_share']:.1f} & "
                 f"{d['oracle_auroc']:.3f} & {100*d['mixed']:.1f} & {100*d['all_fail']:.1f} \\\\")
p = D["pooled"]
L = S["libero"]["decomposition"]
lines.append("\\midrule")
for lab, x in (("CALVIN, 11 tasks pooled", p), ("LIBERO-Goal, pooled", L)):
    ss = 100 * x["state_share"]
    lines.append(f"{GROUP}{lab} & {x['n']} & {x['success']:.3f} & \\npbar[barstate]{{{ss:.1f}}}{{{ss:.1f}}} & {100*x['noise_share']:.1f} & "
                 f"{x['oracle_auroc']:.3f} & {100*x['mixed']:.1f} & {100*x['all_fail']:.1f} \\\\")
write("tab_decomp_rows.tex", lines)

# ---------------- Appendix: offline labels of the added tasks without a push or rotation target ----------------
V = S["vocab_newtasks"]["by"]
KEEP = ["lift_blue_block_slider", "lift_pink_block_slider", "lift_pink_block_drawer", "stack_block"]
ALL6 = ["push_pink_block_right", "rotate_red_block_left"] + KEEP
VCOLS = ("none", "top1", "rand", "rule", "oracle")


def vrow(x):
    """Bold the best of the four realisable choices; the split-half hindsight best is a reference and is not ranked."""
    keys = ("none", "top1", "rand", "rule")
    b = max(x[k] for k in keys)
    return " & ".join(bold_if(f"{x[k]:.3f}", x[k] == b) for k in keys) + f" & \\textcolor{{muted}}{{{x['oracle']:.3f}}}"


def wmean(tasks, grp):
    recs = [V[f"{t}|{grp}"] for t in tasks]
    n = sum(r["n"] for r in recs)
    return dict(n=n, **{k: sum(r["n"] * r[k] for r in recs) / n for k in VCOLS})


lines = []
for t in KEEP:
    x = V[f"{t}|all"]
    lines.append(f"{TASK[t]} & {x['n']} & " + vrow(x) + " \\\\")
pooled = []
for grp, name in (("low", "Low-success handoffs, four tasks"), ("natural", "Natural-distribution handoffs, four tasks"),
                  ("all", "All, four tasks")):
    if not all(f"{t}|{grp}" in V for t in ALL6) or f"ALL|{grp}" not in V:
        print(f"vocabnew: per-task '{grp}' entries not available; pooled row omitted")
        continue
    six, ref = wmean(ALL6, grp), V[f"ALL|{grp}"]
    if six["n"] != ref["n"] or any(abs(six[k] - ref[k]) > 1e-6 for k in VCOLS):
        print(f"vocabnew: handoff-weighted six-task mean does not reproduce ALL|{grp}; pooled row omitted")
        continue
    x = wmean(KEEP, grp)
    pooled.append(f"{GROUP}{name} & {x['n']} & " + vrow(x) + " \\\\")
if pooled:
    lines.append("\\midrule")
    lines += pooled
write("tab_vocabnew_rows.tex", lines)

# ---------------- Round 3 (pre-registered, B_def), relabelling and offline B labels: data/derived/round3_stats.json ----------------
R3 = json.loads((ROOT / "data/derived/round3_stats.json").read_text(encoding="utf-8"))
R3_ARMNAME = {"none": "No intervention", "retry": "Retry once", "prior": "Task-prior repair",
              "benefit_B": "Benefit gate (ours)", "benefit_noretry_B": "Ours, no retry", "noloc_B": "Ours, locator-free",
              "benefit_A": "Rounds-1--2 gate (A labels)"}
R3_GROUPS = [("Baselines", ["none", "retry"]), ("Repair rule (B-label prior table)", ["prior"]),
             ("Learned selector, B labels", ["benefit_B", "benefit_noretry_B", "noloc_B"]),
             ("Learned selector, A labels (round-2 policy, frozen)", ["benefit_A"])]


def r3_int(x):
    s = f"{x:.0f}"
    if s in ("0", "-0"):
        return "0"
    return (s if s.startswith("-") else "+" + s).replace("-", "$-$").replace("+", "$+$")


def r3_int_ci(v):
    lo, hi = (f"{x:.0f}".replace("-0", "0") if f"{x:.0f}" == "-0" else f"{x:.0f}" for x in (v[1], v[2]))
    return f"[{lo.replace('-', '$-$')}, {hi.replace('-', '$-$')}]"


def r3_f3(x):
    return f"{x:+.3f}".replace("-", "$-$").replace("+", "$+$")


def r3_tex(s):
    return s.replace("_", "\\_")


# Pre-registered hypotheses H1-H6
R3_HYP = [("H1", "Ours $-$ retry once", "$>0$"), ("H2", "Ours $-$ task-prior repair", "$>0$"),
          ("H3", "Ours $-$ no intervention", "$>0$"), ("H4", "Ours $-$ rounds-1--2 gate (A labels)", "$>0$"),
          ("H5", "Ours without retry $-$ no intervention", "$>0$"),
          ("H6", "Locator-free $-$ ours (non-inferiority)", "$>-5$")]
lines = []
for h, desc, crit in R3_HYP:
    x = R3["hyp"][h]
    v = x["S"]
    est = pp(v[0])
    if h == "H1":
        est = f"\\marg{{{est}}}"
        holm_txt = "primary, $\\alpha=0.025$"
        dec = ("\\cellcolor{supptint}\\textbf{supported}" if x["decision"] == "supported"
               else "\\textcolor{muted}{not supported}")
    else:
        holm_txt = f"{x['holm_rank']} ({x['holm_threshold']:.4f})"
        dec = ("\\cellcolor{supptint}\\textbf{rejected}" if x["decision"] == "rejected"
               else "\\textcolor{muted}{not rejected}")
    lines.append(f"{h} & {desc} & {crit} & {est} & {ci(v)} & {x['p']:.4f} & {holm_txt} & {dec} & "
                 f"{r3_int(x['N'][0])} {{\\scriptsize{r3_int_ci(x['N'])}}} \\\\")
write("tab_r3hyp_rows.tex", lines)

# Round-3 arms
lines = []
for g, items in R3_GROUPS:
    lines.append(f"\\multicolumn{{7}}{{@{{}}l}}{{\\textit{{{g}}}}} \\\\")
    for a in items:
        m = R3["arms"][a]
        bar = "barblue" if a == "benefit_B" else "bargrey"
        succ = f"\\npbar[{bar}]{{{m['S']:.3f}}}{{{100*m['S']:.1f}}}"
        if a == "retry":
            gs = "--"
        else:
            v = R3["vs_retry"][a]["S"]
            gs = f"\\marg{{{pp(v[0])}}} {{\\scriptsize{ci(v)}}}" if a == "benefit_B" else est_ci(v)
        shade = WARM if a == "benefit_B" else COOL if a == "prior" else ""
        lines.append(f"{shade}\\quad {R3_ARMNAME[a]} & {succ} & {m['C']:.2f} & {m['N']:.0f} & {m['R']:.2f} & {m['T']:.2f} & "
                     f"{gs} \\\\")
    lines.append("\\addlinespace[2pt]")
write("tab_r3arms_rows.tex", lines[:-1])

# Relabelling: ranking of candidates at push and rotation handoffs
RLR = R3["relabel"]["rankings_pr"]
lines = []
for t in PR_ORDER + ["ALL_PR"]:
    x = RLR[t]
    d = x["gain_diff_ci"]
    if t == "ALL_PR":
        lines.append("\\midrule")
        name = f"{GROUP}Push and rotation targets"
    else:
        name = TASK[t]
    lines.append(f"{name} & {x['n']} & {x['gainA']:.3f} & {x['gainB']:.3f} & "
                 f"{r3_f3(d[0])} {{\\scriptsize[{r3_f3(d[1])}, {r3_f3(d[2])}]}} & {100*x['a_best_still_best']:.1f} & "
                 f"{100*x['none_best_A']:.1f} & {100*x['none_best_B']:.1f} \\\\")
write("tab_relabel_rows.tex", lines)

# Offline selection under B labels (offline only)
OF = R3["offline_v4"]
R3_OFF = [("No repair", "p0", None, "", False), ("Random labelled candidate", "rand", None, "", False),
          ("Benefit model, B labels (ours)", "M7", "ref", WARM, False),
          ("Locator-free model, B labels", "M7n", "M7_minus_M7n", "", False),
          ("Benefit model, A labels", "M7_A", "M7_minus_M7_A", "", False),
          ("Round-2 model, A labels (in-sample)", "M7_v3all", "M7_minus_M7_v3all", "", True),
          ("Hindsight best, split half", "oracle_split", None, "", True)]
lines = []
for name, key, dkey, shade, grey in R3_OFF:
    val = f"{OF[key]:.3f}"
    if grey:
        val = muted(val)
    if dkey is None or dkey == "ref":
        dtxt = "--"
    else:
        dv = OF[dkey]
        dtxt = est_ci(dv, bold=excl0(dv))
        if grey:
            dtxt = muted(dtxt)
    lines.append(f"{shade}{name} & {val} & {dtxt} \\\\")
write("tab_offlineB_rows.tex", lines)

# Round-3 timeline
TL = R3["timeline"]
R3_TL = [("relabel_complete", "Relabelling complete and verified", "relabel summary.json"),
         ("delta_tuned", "$\\delta_B$ tuned on development data", "tune_report.json"),
         ("prereg_locked", "Pre-registration locked (sha256 recorded)", "lock_record.json"),
         ("code_hashed", "Stage-3 worker and analysis code hashed", "lock_addendum.json"),
         ("first_job", "First round-3 job started", "start records"),
         ("analysis", "Pre-registered analysis run", "summary.json")]
lines = []
for key, label, rec in R3_TL:
    t = TL[key]
    lines.append(f"{label} & {t[5:10]} {t[11:19]} & \\texttt{{{r3_tex(rec)}}} \\\\")
write("tab_r3lock_rows.tex", lines)
print("round 3: H1", R3["hyp"]["H1"]["decision"], "| Holm rejected:",
      [h for h in ("H2", "H3", "H4", "H5", "H6") if R3["hyp"][h]["decision"] == "rejected"])

print("tables written:", WRITTEN)
