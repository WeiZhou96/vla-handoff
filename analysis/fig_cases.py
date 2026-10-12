# -*- coding: utf-8 -*-
"""Figure: real handoff states and every labelled repair branch for three handoffs (CALVIN scene D, static camera).

Selection (fixed before inspection, see server_export_cases.py): per task, the first labelled low-score handoff of the
five-task vocabulary study in sha256("paper-case|" + sample_id) order with all branches valid. All labelled branches of
that handoff are shown; none is omitted. Frames are the stored 200x200 observations (handoff state, or the state after the
repair finished), shown without retouching (display resampling only). Under each frame: the eight executions of the target
task from that exact state, same seeds across branches (filled = success), and k/8. A dagger marks the candidate ranked
first by the post-repair verifier that chose which candidates to label.
Labels are those of the handoff-referenced criterion B_def (relabelled branches, data/raw/case_frames/labels_bdef/,
copied from relabel_bdef_20261011/labels_bdef); where the label under the start-relative criterion A differs, it is
given in grey below the frame.
Data: data/raw/case_frames/*.png, cases.json, labels_bdef/label_*.json."""
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from figcommon import *  # noqa

setup()
CF = ROOT / "data/raw/case_frames"
cases = json.loads((CF / "cases.json").read_text())
BDEF = {}
for p in sorted((CF / "labels_bdef").glob("label_*.json")):
    r = json.loads(p.read_text(encoding="utf8"))
    assert r.get("criterion") == "B_def", p
    for a in r["arms"]:
        w = "".join("1" if x["success"] else "0" for x in sorted(a["rollouts"], key=lambda x: x["repeat"]))
        assert w.count("1") == a["k"], (p, a["candidate"])
        BDEF[(r["sample_id"], a["candidate"])] = (a["k"], w)
for t, c in cases.items():
    for b in c["branches"]:
        kb, wb = BDEF[(c["sample_id"], b["candidate"])]  # every shown branch must have a B_def label
        b["kA"], b["k"], b["wins"] = b["k"], kb, wb
        if b["kA"] != kb:
            print(f"{t}: {b['candidate']} A {b['kA']}/8 -> B_def {kb}/8")
ORDER = ["rotate_pink_block_right", "lift_red_block_slider", "push_red_block_right"]
TITLE = {"rotate_pink_block_right": "Target: rotate the pink block right", "lift_red_block_slider": "Target: lift the red block from the slider",
         "push_red_block_right": "Target: push the red block to the right"}
SHORTC = lambda c: {"none": "no repair (handoff)", "retreat": "retreat (scripted)", "home": "home (scripted)"}.get(
    c, c.replace("_block", "").replace("_", " "))


def wrap(name):
    if len(name) <= 13 or " " not in name:
        return name + "\n"
    cut = name.rfind(" ", 0, 14)
    if cut <= 0:
        cut = name.find(" ")
    return name[:cut] + "\n" + name[cut + 1:]


NCOL = 7
fig = plt.figure(figsize=(TEXT_W, 5.25))
gs = fig.add_gridspec(len(ORDER), NCOL, left=0.005, right=0.995, top=0.9, bottom=0.075, hspace=1.0, wspace=0.07)
for ri, t in enumerate(ORDER):
    c = cases[t]
    br = c["branches"]
    none = next(b for b in br if b["candidate"] == "none")
    top = max((b for b in br if b["candidate"] != "none" and b["verifier_score"] is not None), key=lambda b: b["verifier_score"])
    rest = sorted([b for b in br if b["candidate"] != "none"], key=lambda b: (-b["k"], b["candidate"]))
    seq = [none] + rest
    for ci, b in enumerate(seq[:NCOL]):
        ax = fig.add_subplot(gs[ri, ci])
        img = Image.open(CF / f"{c['key']}__{b['candidate']}.png")
        ax.imshow(np.asarray(img), interpolation="lanczos")
        ax.set_xticks([]); ax.set_yticks([])
        d = b["k"] - none["k"]
        col = "#333333" if b["candidate"] == "none" else (OKABE_ITO["blue"] if d > 0 else (OURS if d < 0 else "#9E9E9E"))
        for s in ax.spines.values():
            s.set_color(col); s.set_linewidth(1.6 if b["candidate"] == "none" else 1.1)
        name = wrap(SHORTC(b["candidate"]) + ("†" if b is top else ""))
        ax.set_title(name, fontsize=6.8, pad=3.2, color=INK, linespacing=0.95,
                     fontweight="bold" if b["candidate"] == "none" else "normal")
        wins = [int(x) for x in b["wins"]]
        for j, wv in enumerate(wins):
            ax.add_patch(plt.Rectangle((j * 0.088, -0.175), 0.075, 0.09, transform=ax.transAxes, clip_on=False,
                                       facecolor=(col if wv else "white"), edgecolor=col, lw=0.6))
        ax.text(0.72, -0.13, f"{b['k']}/8", transform=ax.transAxes, fontsize=7.0, va="center", ha="left",
                color=col, fontweight="bold")
        if b["kA"] != b["k"]:
            ax.text(0.0, -0.30, f"A: {b['kA']}/8", transform=ax.transAxes, fontsize=6.5, va="center", ha="left",
                    color="#767C82")
        if ci == 0:
            ax.text(0.0, 1.36, TITLE[t], transform=ax.transAxes, fontsize=7.6, ha="left", va="bottom", color=INK, fontweight="bold")
print(save(fig, "fig_cases"))
