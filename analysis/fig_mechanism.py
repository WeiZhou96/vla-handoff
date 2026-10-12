# -*- coding: utf-8 -*-
"""Figure (mechanism): undo-redo at push and rotation targets under the original start-relative criterion A.

One point per target handoff of the benefit gate in the bit-exact mode-A replay (both rounds pooled).
x = displacement of the target block after the repair, relative to the handoff state.
y = displacement at the end of the last attempt, relative to the handoff state.
(a) push: x position in cm. (b) rotation: yaw in degrees, wrapped to (-180, 180].
Colour: outcome under A and under the handoff-referenced criterion B_def at the end of the last attempt.
Black plus: per-task median of the handoffs that succeed under A but not under B_def (only tasks with at least five).
Panel (a) dashed line: B_def threshold (net displacement > 10 cm), labelled directly. Its label is placed
automatically in the first candidate position, starting at the left end of the line, that contains no marker and no
part of the diagonal.
Panel (a) dotted diagonal: A threshold for a single attempt after the repair (y - x > 10 cm). It is keyed in the
figure legend, not labelled inside the axes.
Task marker keys sit below each panel.
After drawing, an automatic check asserts that no two text boxes intersect and that no text box contains a plotted
marker centre. If the check fails, the figure is saved as fig_mechanism_overlapfail and the script exits with code 3.
Data: data/raw/fullchain_replay_20261010/analysis/handoff_displacement.json (server_export_displacement.py). Its medians
are checked against section 8 of the replay report.
"""
import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text
from figcommon import ROOT, TEXT_W, OURS, INK, MUTED, LIGHT, OKABE_ITO, TASK_NAME, setup, save, style_ax

setup()
SRC = ROOT / "data/raw/fullchain_replay_20261010/analysis/handoff_displacement.json"
if not SRC.exists():
    raise SystemExit(f"missing {SRC}: run experiments/fullchain_replay_20261010/server_export_displacement.py on the experiment server and copy its output here")
D = json.loads(SRC.read_text(encoding="utf-8"))
R = [dict(zip(D["columns"], r)) for r in D["rows"]]
HA = [r for r in R if r["mode"] == "A" and r["arm"] == "benefit"]
FS = 6.6
THR = 10.0   # cm, push threshold of A and of B_def (Table audit)
DIAG_LS = (0, (1, 1.5))
# section 8 of the replay report, arm benefit: n, rep p50, end p50, A1B0 n, A1B0 rep p50, A1B0 end p50
REPORT8 = {"push_red_block_right": (95, -9.5, 3.6, 51, -9.7, 3.2), "push_blue_block_right": (53, -10.1, 1.4, 31, -10.2, 0.1),
           "push_pink_block_right": (74, -0.0, 7.6, 13, -9.8, 0.6),
           "rotate_pink_block_right": (71, 87.6, -22.1, 33, 90.2, -9.0), "rotate_red_block_left": (72, 0.0, 93.3, 2, 0.0, 2.0)}


def med(v):
    v = [x for x in v if x is not None]
    return float(np.median(v)) if v else None


def a1b0(rs):
    return [r for r in rs if r["success"] and not r["Bdef"]]


for t, ref in REPORT8.items():
    hs = [r for r in HA if r["task"] == t]
    c = a1b0(hs)
    got = (len(hs), med([r["d_rep"] for r in hs]), med([r["d_end"] for r in hs]), len(c),
           med([r["d_rep"] for r in c]), med([r["d_end"] for r in c]))
    assert got[0] == ref[0] and got[3] == ref[3], (t, got, ref)
    assert all(abs(g - e) <= 0.051 for g, e in zip(got[1:3] + got[4:], ref[1:3] + ref[4:])), (t, got, ref)


def overlap_check(fig, marks):
    """Text-text and text-marker check in display space (renderer of the figure canvas).
    marks: list of (axes, x, y) data coordinates of every plotted marker centre.
    Tick labels are counted only for ticks inside the view interval (the ones that are drawn)."""
    fig.canvas.draw()
    rnd = fig.canvas.get_renderer()
    tick_ids, texts = set(), []
    for ax in fig.axes:
        for axis, lim in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            lo, hi = min(lim), max(lim)
            eps = 1e-9 * max(1.0, hi - lo)
            for tk in list(axis.get_major_ticks()) + list(axis.get_minor_ticks()):
                loc = tk.get_loc()
                inside = loc is not None and lo - eps <= loc <= hi + eps
                for lab in (tk.label1, tk.label2):
                    tick_ids.add(id(lab))
                    if inside and tk.get_visible():
                        texts.append(lab)
    seen = set(tick_ids)
    for t in fig.findobj(Text):
        if id(t) not in seen:
            seen.add(id(t))
            texts.append(t)
    boxes = []
    for t in texts:
        if not t.get_visible() or not t.get_text().strip():
            continue
        b = t.get_window_extent(rnd)
        if b.width > 0 and b.height > 0:
            boxes.append((t.get_text().replace("\n", " | "), b))
    bad = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes[i][1].overlaps(boxes[j][1]):
                bad.append(f"text/text: {boxes[i][0]!r} x {boxes[j][0]!r}")
    for ax, x, y in marks:
        px, py = ax.transData.transform((x, y))
        for s, b in boxes:
            if b.x0 <= px <= b.x1 and b.y0 <= py <= b.y1:
                bad.append(f"text/marker: {s!r} contains marker at ({x:.3g}, {y:.3g})")
    return len(boxes), len(marks), bad


def place_label(txt, ax, cands, pts, rnd):
    """Put txt at the first candidate (x, y, ha, va) whose box lies inside the axes and contains none of pts.
    pts: (x, y, pad in points), where pad is the marker radius plus a margin."""
    k_pt = ax.figure.dpi / 72.0
    P = ax.transData.transform(np.array([(p[0], p[1]) for p in pts], float)) if pts else np.empty((0, 2))
    pad = np.array([p[2] for p in pts], float) * k_pt
    ib = ax.bbox
    for k, (x, y, ha, va) in enumerate(cands):
        txt.set_position((x, y)); txt.set_ha(ha); txt.set_va(va)
        b = txt.get_window_extent(rnd)
        if b.x0 < ib.x0 + 2 * k_pt or b.x1 > ib.x1 - 2 * k_pt or b.y0 < ib.y0 + 2 * k_pt or b.y1 > ib.y1 - 2 * k_pt:
            continue
        hit = (P[:, 0] >= b.x0 - pad) & (P[:, 0] <= b.x1 + pad) & (P[:, 1] >= b.y0 - pad) & (P[:, 1] <= b.y1 + pad)
        if not hit.any():
            return k
    x, y, ha, va = cands[0]
    txt.set_position((x, y)); txt.set_ha(ha); txt.set_va(va)
    return None


CELL = [("A0B0", "both fail", LIGHT), ("A1B1", "A and $B_\\mathrm{def}$ succeed", OKABE_ITO["blue"]),
        ("A0B1", "$B_\\mathrm{def}$ only", OKABE_ITO["green"]), ("A1B0", "A only", OURS)]
PANELS = [("push", ["push_red_block_right", "push_blue_block_right", "push_pink_block_right"], "x", "cm"),
          ("rotation", ["rotate_pink_block_right", "rotate_red_block_left"], "yaw", "deg")]
MK = {"push_red_block_right": "o", "push_blue_block_right": "s", "push_pink_block_right": "^",
      "rotate_pink_block_right": "o", "rotate_red_block_left": "s"}

H, AXB, AXH = 3.4, 0.72, 2.0      # inches: figure height, axes bottom, axes height
fig = plt.figure(figsize=(TEXT_W, H))
rnd = fig.canvas.get_renderer()
marks = []
for i, (pname, tasks, q, unit) in enumerate(PANELS):
    ax = fig.add_axes([0.075 + i * 0.49, AXB / H, 0.40, AXH / H])
    rs = [r for r in HA if r["task"] in tasks]
    ok = [r for r in rs if r["d_rep"] is not None and r["d_end"] is not None]
    if pname == "push":
        xs, ys = [r["d_rep"] for r in ok], [r["d_end"] for r in ok]
        x0, x1 = np.floor(min(xs) - 2), np.ceil(max(xs) + 2)
        y0, y1 = np.floor(min(min(ys), 0) - 2), np.ceil(max(max(ys), THR) + 2)
    else:
        x0, x1, y0, y1 = -185, 185, -185, 185
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    style_ax(ax)
    if pname != "push":
        ax.set_xticks([-180, -90, 0, 90, 180]); ax.set_yticks([-180, -90, 0, 90, 180])
    pts = []
    for key, _, col in CELL:
        for t in tasks:
            s = [r for r in ok if r["task"] == t and f"A{int(bool(r['success']))}B{int(bool(r['Bdef']))}" == key]
            if s:
                px, py = [r["d_rep"] for r in s], [r["d_end"] for r in s]
                ax.scatter(px, py, s=8, marker=MK[t], color=col, lw=0, alpha=0.8,
                           zorder=2 + [c[0] for c in CELL].index(key))
                pts += [(x, y, 2.5) for x, y in zip(px, py)]
    for t in tasks:
        c = [r for r in a1b0(ok) if r["task"] == t]
        if len(c) >= 5:
            mx, my = med([r["d_rep"] for r in c]), med([r["d_end"] for r in c])
            ax.scatter([mx], [my], marker="+", s=70, lw=1.4, color=INK, zorder=8)
            pts.append((mx, my, 5.5))
    marks += [(ax, x, y) for x, y, _ in pts]
    if pname == "push":
        ax.axhline(THR, color=INK, lw=0.6, ls=(0, (4, 2)), zorder=1)
        xx = np.array([x0, x1])
        ax.plot(xx, xx + THR, color=MUTED, lw=0.6, ls=DIAG_LS, zorder=1)
        lab = ax.text(0, 0, "$B_\\mathrm{def}$: net > 10 cm", fontsize=FS, color=INK, zorder=9)
        u = (y1 - y0) / (AXH * 72.0)     # data units per point on y
        cands = []
        for fx in (0.015, 0.15, 0.30, 0.45, 0.60, 0.75):
            for va, sg in (("bottom", 1), ("top", -1)):
                cands.append((x0 + fx * (x1 - x0), THR + sg * 1.5 * u, "left", va))
        for va, sg in (("bottom", 1), ("top", -1)):
            cands.append((x1 - 0.015 * (x1 - x0), THR + sg * 1.5 * u, "right", va))
        diag = [(x, x + THR, 1.0) for x in np.linspace(x0, x1, 400) if y0 <= x + THR <= y1]
        k = place_label(lab, ax, cands, pts + diag, rnd)
        print("B_def label:", "no free candidate (check below will fail)" if k is None
              else f"candidate {k}: x = {cands[k][0]:.1f} cm, {'above' if cands[k][3] == 'bottom' else 'below'} the line")
    ax.axhline(0, color="#C9CCD1", lw=0.5, zorder=0); ax.axvline(0, color="#C9CCD1", lw=0.5, zorder=0)
    ax.set_xlabel(f"Target block {q} after repair $-$ handoff ({unit})")
    ax.set_ylabel(f"{q} at end of target $-$ handoff ({unit})")
    ax.text(0.0, 1.03, f"({'ab'[i]})", transform=ax.transAxes, fontsize=7.5, fontweight="bold", va="bottom")
    ax.text(0.075, 1.03, f"{pname} targets: {len(ok)} handoffs, A only {len(a1b0(ok))}", transform=ax.transAxes,
            fontsize=FS, color=INK, va="bottom")
    ax.legend(handles=[Line2D([], [], marker=MK[t], ls="", color="#6E757C", markersize=3.5, label=TASK_NAME[t]) for t in tasks],
              loc="upper center", bbox_to_anchor=(0.5, -0.235), ncol=len(tasks), frameon=False, fontsize=FS,
              handletextpad=0.2, columnspacing=1.0, borderaxespad=0.0)
    print(pname, "points", len(ok), "of", len(rs), "| A only", len(a1b0(ok)))
fig.legend(handles=[Line2D([], [], marker="o", ls="", color=col, markersize=4, label=lab) for _, lab, col in CELL]
           + [Line2D([], [], marker="+", ls="", color=INK, markersize=6, mew=1.3, label="median of A only, per task"),
              Line2D([], [], color=MUTED, lw=0.8, ls=DIAG_LS, label="(a) A, one attempt: y $-$ x > 10 cm")],
           loc="upper center", bbox_to_anchor=(0.52, 1.0), ncol=3, frameon=False, fontsize=FS, handletextpad=0.3,
           columnspacing=1.2)

nt, nm, bad = overlap_check(fig, marks)
print(f"overlap check: {nt} text boxes, {nm} marker centres, {len(bad)} conflicts -> {'FAIL' if bad else 'PASS'}")
for b in bad:
    print("   ", b)
if bad:
    print(save(fig, "fig_mechanism_overlapfail"))
    sys.exit(3)
print(save(fig, "fig_mechanism"))
