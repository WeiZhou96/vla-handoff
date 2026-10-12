# -*- coding: utf-8 -*-
"""Figure (rules, b): paired difference in five-subtask chain success for seven comparisons.
Rounds 1 and 2: original start-relative criterion A (grey) and handoff-referenced counterfactual re-execution, estimate (ii)
(orange; post hoc); 95% percentile intervals from a cluster bootstrap over base configurations (2,000 resamples, seed 17,
shared by A and (ii)); data/derived/fullchain_stats.json.
Round 3: pre-registered confirmation under B_def (blue triangles); 95% percentile intervals from the pre-registered cluster
bootstrap (5,000 resamples, seed 20261011); data/derived/round3_stats.json (compute_round3.py). Round 3 has no inverse-rule arm;
its gate uses B labels and its task prior uses the B-label prior table.
Filled markers: interval excludes zero. Dashed segment: non-inferiority margin (-5 pp) of the locator-free comparison (H6).
The panel letter is set in the manuscript. The script checks text overlap, legend placement and the minimum font size before saving.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.text import Text  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figcommon import ROOT, COL_W, OURS, GREY, OKABE_ITO, setup, save, style_ax  # noqa: E402

setup()
plt.rcParams["savefig.dpi"] = 300
F = json.loads((ROOT / "data/derived/fullchain_stats.json").read_text(encoding="utf-8"))
R3 = json.loads((ROOT / "data/derived/round3_stats.json").read_text(encoding="utf-8"))
R3COL = OKABE_ITO["blue"]
FS = 6.6
MINFS = 6.5
M = "\u2212"
NI_MARGIN = -5.0
# label, rounds-1-2 arms (X, Y), round-3 pairwise key
COMPS = [(f"Ours {M} no intervention", "benefit", "none", "benefit_B-none"),
         (f"Ours {M} retry once", "benefit", "retry", "benefit_B-retry"),
         (f"Ours {M} task prior", "benefit", "prior", "benefit_B-prior"),
         (f"Ours {M} inverse rule", "benefit", "inverse", None),
         (f"Inverse rule {M} retry", "inverse", "retry", None),
         (f"Ours, no retry {M} retry", "benefit_noretry", "retry", "benefit_noretry_B-retry"),
         (f"Locator-free {M} ours", "noloc", "benefit", "noloc_B-benefit_B")]
SER = [(1, "S_A", "Round 1, criterion A", "o", GREY), (1, "S_ii", "Round 1, (ii)", "o", OURS),
       (2, "S_A", "Round 2, criterion A", "s", GREY), (2, "S_ii", "Round 2, (ii)", "s", OURS),
       (3, None, r"Round 3, $B_{\mathrm{def}}$ (pre-registered)", "^", R3COL)]
STEP = len(SER) + 1.4


def value(rd, k, X, Y, key3):
    if rd == 3:
        if key3 is None:
            return None
        v = R3["pairwise"].get(key3)
        if v is None:
            raise KeyError(f"round3_stats.json: pairwise {key3} missing")
        return v["S"]
    v = F["paired"].get(f"r{rd}|all|{X}-{Y}")
    if v is None:
        print(f"fig_forest: r{rd}|all|{X}-{Y} not in fullchain_stats.json; omitted")
        return None
    return v[k]


fig, ax = plt.subplots(figsize=(COL_W, 3.9))
fig.subplots_adjust(left=0.36, right=0.97, top=0.985, bottom=0.2)
yt, vals = [], [NI_MARGIN]
for gi, (name, X, Y, key3) in enumerate(COMPS):
    base = (len(COMPS) - 1 - gi) * STEP
    yt.append(base + (len(SER) - 1) / 2)
    if gi % 2 == 0:
        ax.axhspan(base - 0.7, base + len(SER) - 0.3, color="#F2F4F7", lw=0, zorder=0)
    if key3 == "noloc_B-benefit_B":
        ax.plot([NI_MARGIN, NI_MARGIN], [base - 0.6, base + len(SER) - 0.4], color=GREY, lw=0.6, ls=(0, (2.0, 1.6)), zorder=1)
    for si, (rd, k, _, mk, col) in enumerate(SER):
        v = value(rd, k, X, Y, key3)
        if v is None:
            continue
        e, lo, hi = (100 * x for x in v)
        vals += [lo, hi]
        yy = base + len(SER) - 1 - si
        ax.plot([lo, hi], [yy, yy], color=col, lw=0.9, zorder=2)
        ax.scatter([e], [yy], marker=mk, s=15, color=col if (lo > 0 or hi < 0) else "white", edgecolor=col, lw=0.8, zorder=3)
ax.axvline(0, color="#333333", lw=0.6, zorder=1)
ax.set_yticks(yt)
ax.set_yticklabels([c[0] for c in COMPS], fontsize=FS)
ax.tick_params(axis="y", length=0)
lo_, hi_ = np.floor(min(vals) / 5) * 5 - 1, np.ceil(max(vals) / 5) * 5 + 1
ax.set_xlim(lo_, hi_)
ax.set_xticks([v for v in range(-30, 61, 10) if lo_ <= v <= hi_])
ax.set_xlabel("Paired difference in chain success (pp)")
ax.set_ylim(-1, len(COMPS) * STEP - 1.3)
style_ax(ax)
ax.spines["left"].set_visible(False)

# legend handles ordered so that with two columns each row pairs criterion A with (ii) of the same round
H_ = {lab: Line2D([], [], marker=mk, color=col, lw=0.9, markersize=3.6, label=lab) for _, _, lab, mk, col in SER}
ORDER = [SER[0][2], SER[2][2], SER[4][2], SER[1][2], SER[3][2]]
handles = [H_[lab] for lab in ORDER]


def make_legend(ncol):
    return fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.1), bbox_transform=fig.transFigure,
                      ncol=ncol, frameon=False, fontsize=FS, handlelength=1.4, columnspacing=0.9, labelspacing=0.3,
                      borderaxespad=0.0, borderpad=0.2)


ncol = 2
leg = make_legend(ncol)
for _ in range(12):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    below = [ax.xaxis.label.get_window_extent(r)] + [t.get_window_extent(r) for t in ax.get_xticklabels() if t.get_text()]
    ybot = min(b.y0 for b in below)
    leg.set_bbox_to_anchor((0.5, (ybot - 2.0) / fig.bbox.height), transform=fig.transFigure)
    fig.canvas.draw()
    lb = leg.get_window_extent(r)
    if lb.width > fig.bbox.width - 2.0 and ncol > 1:
        leg.remove()
        ncol = 1
        leg = make_legend(ncol)
        continue
    if lb.y0 >= 1.0:
        break
    fig.subplots_adjust(bottom=fig.subplotpars.bottom + (1.0 - lb.y0) / fig.bbox.height + 0.005)
else:
    raise RuntimeError("fig_forest: the legend could not be placed inside the figure area")


def _overlap(a, b, pad=0.5):
    return (min(a.x1, b.x1) - max(a.x0, b.x0) > pad) and (min(a.y1, b.y1) - max(a.y0, b.y0) > pad)


def check_layout():
    """Raise if texts overlap, the legend leaves the figure or covers the axes or a label, or a font is below MINFS."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    items = [("tick:" + t.get_text(), t.get_window_extent(r))
             for t in ax.get_yticklabels() + ax.get_xticklabels() if t.get_visible() and t.get_text()]
    items.append(("xlabel", ax.xaxis.label.get_window_extent(r)))
    items += [("legend:" + t.get_text(), t.get_window_extent(r)) for t in leg.get_texts()]
    probs = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if _overlap(items[i][1], items[j][1]):
                probs.append(f"'{items[i][0]}' overlaps '{items[j][0]}'")
    W, Hh = fig.bbox.width, fig.bbox.height
    lb = leg.get_window_extent(r)
    for name, b in items + [("legend frame", lb)]:
        if b.x0 < -0.5 or b.y0 < -0.5 or b.x1 > W + 0.5 or b.y1 > Hh + 0.5:
            probs.append(f"{name} extends outside the figure")
    if _overlap(lb, ax.get_window_extent(r)):
        probs.append("legend overlaps the axes area")
    for name, b in items:
        if not name.startswith("legend:") and _overlap(lb, b):
            probs.append(f"legend overlaps {name}")
    small = [(t.get_text(), t.get_fontsize()) for t in fig.findobj(Text)
             if t.get_visible() and t.get_text() and t.get_fontsize() < MINFS]
    if small:
        probs.append(f"font below {MINFS} pt: {small}")
    if probs:
        raise RuntimeError("fig_forest layout check failed:\n  " + "\n  ".join(probs))


check_layout()
print(save(fig, "fig_forest"))
