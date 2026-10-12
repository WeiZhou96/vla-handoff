# -*- coding: utf-8 -*-
"""Figure 1: method overview, drawn at the printed width (7.05 in) so that every label keeps its point size.

Coordinates are given in the units of a 1740 x 910 layout canvas. The three images are the original 200x200 CALVIN static-camera
observations of handoff 0383d5c1c5f6 (target lift_red_block_slider): the handoff itself and the stored states after the
repairs rotate_blue_block_left and lift_pink_block_slider (data/raw/case_frames, exported by experiments/paper_export/server_export_cases.py).
Outcome vectors Y and counts k are symbolic; no measured per-seed outcome is drawn.
"""
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle, FancyBboxPatch, Polygon, Circle
from figcommon import ROOT, FIG, TEXT_W, setup, save_figure

setup()
plt.rcParams.update({"mathtext.fontset": "custom", "mathtext.rm": "Calibri", "mathtext.it": "Calibri:italic",
                     "mathtext.bf": "Calibri:bold", "mathtext.cal": "cmsy10", "mathtext.fallback": "stixsans"})

W, H = 1740, 905
NAVY, TITLE, PURPLE, ORANGE = "#00385B", "#00275B", "#7A1AAD", "#D55E00"
BORDER, PALE, INKC, MUTEDC = "#8BB7C7", "#F5F8FA", "#1F2A33", "#5A6670"
S_HEAD, S_SUB, S_TXT, S_SMALL = 8.6, 7.6, 7.0, 6.5

fig = plt.figure(figsize=(TEXT_W, TEXT_W * H / W))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")


def T(x, y, s, size=S_TXT, color=INKC, bold=False, ha="left", va="center", italic=False, bg=None, z=5, **kw):
    t = ax.text(x, y, s, fontsize=size, color=color, fontweight="bold" if bold else "normal", ha=ha, va=va,
                fontstyle="italic" if italic else "normal", zorder=z, **kw)
    if bg:
        t.set_bbox(dict(boxstyle="square,pad=0.12", facecolor=bg, edgecolor="none"))
    return t


def L(pts, color=NAVY, lw=0.9, dashed=False, arrow=False, z=3):
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=color, lw=lw, ls=(0, (3.2, 2.2)) if dashed else "-", solid_capstyle="butt", zorder=z)
    if arrow:
        (x0, y0), (x1, y1) = pts[-2], pts[-1]
        ax.annotate("", xy=(x1, y1), xytext=(x0 + 0.6 * (x1 - x0), y0 + 0.6 * (y1 - y0)),
                    arrowprops=dict(arrowstyle="-|>,head_length=0.45,head_width=0.22", color=color, lw=lw,
                                    shrinkA=0, shrinkB=0), zorder=z)


def B(x, y, w, h, fc="white", ec=BORDER, lw=0.6, r=6, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, zorder=z))


def IMG(name, x, y, side, ec=BORDER):
    im = mpimg.imread(ROOT / f"data/raw/case_frames/0383d5c1c5f6__{name}.png")
    ax.imshow(im, extent=(x, x + side, y + side, y), zorder=4, interpolation="lanczos")
    ax.add_patch(Rectangle((x, y), side, side, fc="none", ec=ec, lw=0.7, zorder=5))


# ---------------------------------------------------------------- handoff (top left)
T(30, 26, "After task A", S_HEAD, TITLE, bold=True)
T(30, 62, "Static RGB", S_SMALL, MUTEDC)
IMG("none", 30, 80, 226)
T(30, 330, r"Handoff state $s$", S_SUB, TITLE, bold=True)
T(30, 362, "Recorded CALVIN observation", S_SMALL, MUTEDC)
T(30, 393, "Next task B: lift red block from the slider", S_SMALL, MUTEDC)
T(282, 62, "Robot state", S_TXT, TITLE, bold=True)
B(282, 82, 112, 170, "white", BORDER, 0.6, 4)
ax.add_patch(Rectangle((282, 82), 10, 170, fc="#BAD6E1", ec="none", zorder=3))
for i, lab in enumerate([r"$q$", r"$\dot q$", "eef pose", "gripper"]):
    y = 82 + 42.5 * i
    if i:
        L([(292, y), (394, y)], BORDER, 0.5)
    T(305, y + 21, lab, S_TXT, INKC)

# ---------------------------------------------------------------- offline paired labels (top)
ax.add_patch(FancyBboxPatch((420, 46), 1290, 352, boxstyle="round,pad=0,rounding_size=8", fc="#FAF8FC", ec="none", zorder=0))
T(432, 26, "Offline paired labels", S_HEAD, TITLE, bold=True)
T(440, 74, "Same saved handoff", S_SMALL, PURPLE)
T(440, 128, "Restore", S_TXT, PURPLE, bold=True)
ax.add_patch(Circle((486, 190), 30, fc="white", ec=PURPLE, lw=0.8, zorder=4))
T(486, 190, r"$s$", 9.5, NAVY, ha="center")
T(486, 250, "restored before\neach branch", S_SMALL, PURPLE, ha="center", va="top", linespacing=1.15)
rows = [140, 240, 340]
for ry in rows:
    L([(517, 190), (592, 190), (592, ry), (636, ry)], PURPLE, 0.85, dashed=True, arrow=True)
T(646, rows[0], r"$\varnothing$", S_SUB, NAVY)
ax.add_patch(Circle((710, rows[0]), 24, fc="white", ec=BORDER, lw=0.7, zorder=4))
T(710, rows[0], r"$s$", 8.5, NAVY, ha="center")
T(780, rows[0], "No repair", S_TXT)
T(646, rows[1], r"$u_1$", S_SUB, NAVY)
IMG("rotate_blue_block_left", 676, rows[1] - 40, 80)
T(780, rows[1], "Rotate blue\nblock left", S_TXT, linespacing=1.15)
T(646, rows[2], r"$u_2$", S_SUB, NAVY)
IMG("lift_pink_block_slider", 676, rows[2] - 40, 80)
T(780, rows[2], "Lift pink block\nfrom slider", S_TXT, linespacing=1.15)
T(780, 392, "recorded post-repair states", S_SMALL, MUTEDC, va="center")

T(1150, 26, r"Same next-task seeds ($J=8$)", S_SUB, TITLE, bold=True, ha="center")
for j in range(8):
    x = 1010 + j * 36
    B(x, 54, 26, 26, "white", BORDER, 0.55, 3)
    T(x + 13, 67.5, str(j + 1), S_SMALL, NAVY, ha="center")
T(1150, 98, "shared by every branch", S_SMALL, MUTEDC, ha="center")
subs = [r"\varnothing", r"u_1", r"u_2"]
for ry, sb in zip(rows, subs):
    L([(935, ry), (998, ry)], PURPLE, 0.85, dashed=True, arrow=True)
    T(1150, ry, rf"$[\,Y_{{{sb},1}},\ \ldots,\ Y_{{{sb},8}}\,]$", 8.8, NAVY, ha="center")
    L([(1300, ry), (1420, ry)], PURPLE, 0.85, dashed=True, arrow=True)
    T(1432, ry, rf"$k_{{{sb}}}/8$", 9.2, NAVY)
T(1150, 385, r"$Y_{u,j}\in\{0,1\}$", 7.4, MUTEDC, ha="center")
T(1560, 26, "Branch success targets", S_SUB, TITLE, bold=True, ha="center")
L([(1528, rows[0]), (1600, rows[0]), (1600, rows[2]), (1528, rows[2])], PURPLE, 0.85, dashed=True)
L([(1528, rows[1]), (1600, rows[1])], PURPLE, 0.85, dashed=True)
L([(1600, 240), (1680, 240), (1680, 428), (509, 428), (509, 512)], PURPLE, 0.95, dashed=True, arrow=True)
T(1240, 428, "fit (weighted by executions)", S_SMALL, PURPLE, ha="center", bg="white", italic=True)

# ---------------------------------------------------------------- repair selector (bottom left)
T(30, 470, "Repair selector", S_HEAD, TITLE, bold=True)
T(30, 502, "trained offline, deployed frozen", S_SMALL, MUTEDC)
L([(30, 193), (14, 193), (14, 590), (50, 590)], NAVY, 0.9, arrow=True)
ax.add_patch(Polygon([[56, 532], [168, 556], [168, 626], [56, 650]], closed=True, fc="#8BBFDF", ec="none", zorder=3))
T(112, 591, "Frozen\nDINOv2", S_TXT, NAVY, bold=True, ha="center", linespacing=1.1)
L([(172, 591), (216, 591)], NAVY, 0.9, arrow=True)
T(222, 552, r"State features $\phi(o)$", S_TXT, TITLE, bold=True)
for i, c in enumerate(["#6F9EAA", "#87ADB9", "#91A2B0", "#9BA0B7", "#A6A1B9", "#AE85A2"]):
    ax.add_patch(Rectangle((224 + i * 22, 579), 20, 22, fc=c, ec="white", lw=0.3, zorder=3))
ctrl = [("Robot state", 676, "#8CA4B4"), (r"Next task $\ell$", 722, "#BAC4AD"),
        (r"Candidate $u$", 768, "#B0A3C8"), (r"Prior $\bar q(\ell,u)$", 814, "#C7B9A9")]
for lab, y, c in ctrl:
    T(30, y, lab, S_TXT, TITLE, bold=True)
    L([(186, y), (216, y)], NAVY, 0.8, arrow=True)
    for k in range(6):
        ax.add_patch(Rectangle((224 + k * 22, y - 11), 20, 22, fc=c, ec="white", lw=0.3, zorder=3))
    L([(358, y), (388, y), (388, 612)], NAVY, 0.8)
L([(358, 590), (400, 590), (400, 660), (432, 660)], NAVY, 0.9, arrow=True)
L([(388, 612), (400, 612)], NAVY, 0.8)

B(438, 516, 150, 316, "#F5F9FA", BORDER, 0.6, 6)
T(513, 540, r"ExtraTrees $q_\theta$", S_SUB, TITLE, bold=True, ha="center")


def tree(cx, top):
    n = [(cx, top), (cx - 26, top + 21), (cx + 26, top + 21), (cx - 40, top + 41), (cx - 12, top + 41), (cx + 13, top + 41),
         (cx + 41, top + 41), (cx - 44, top + 58)]
    for a, b in [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6), (3, 7)]:
        L([n[a], n[b]], "#4D6B81", 0.6)
    for (x, y) in n:
        ax.add_patch(Circle((x, y), 5, fc="#74ADBF", ec=NAVY, lw=0.45, zorder=4))


for top in (580, 664, 748):
    tree(513, top)
L([(594, 660), (634, 660)], NAVY, 0.9, arrow=True)
T(732, 540, r"35 repairs $+\ \varnothing$", S_SUB, TITLE, bold=True, ha="center")
B(640, 560, 186, 262, PALE, BORDER, 0.6, 6)
for i, (lab, y) in enumerate([(r"$q(\varnothing)$", 576), (r"$q(u)$", 638), (r"$q(u^\star)$", 700)]):
    B(654, y, 158, 48, "#FFE6D1" if i == 2 else "white", ORANGE if i == 2 else BORDER, 0.9 if i == 2 else 0.55, 4)
    T(733, y + 24, lab, 8.2, NAVY, ha="center")
T(733, 784, r"$\vdots$", 9, NAVY, ha="center")

# ---------------------------------------------------------------- online decision (bottom right)
T(880, 470, "Online repair decision", S_HEAD, TITLE, bold=True)
T(880, 510, r"$q(u^\star)-q(\varnothing)-\lambda\,\bar c(\ell,u^\star)/H>\delta$", 8.4, NAVY)
T(944, 742, r"$u^\star=\arg\max_{u\in\mathcal{U}(\ell)}\,q(u)$", 7.6, NAVY, ha="center")
L([(832, 668), (1000, 668)], NAVY, 1.0, arrow=True)
ax.add_patch(Polygon([[1074, 594], [1146, 668], [1074, 742], [1002, 668]], closed=True, fc="#D9EBF3", ec=NAVY, lw=1.0,
                     zorder=3))
T(1074, 668, "Benefit\ngate", S_SUB, NAVY, bold=True, ha="center", linespacing=1.05)
L([(1146, 668), (1176, 668), (1176, 591), (1240, 591)], NAVY, 1.0, arrow=True)
L([(1176, 668), (1176, 761), (1240, 761)], NAVY, 1.0, arrow=True)
T(1186, 562, "pass", S_SMALL, NAVY, bold=True)
T(1186, 790, "fail", S_SMALL, NAVY, bold=True)
B(1246, 546, 176, 92, "#FFF4EA", ORANGE, 0.9, 6)
T(1334, 572, r"Execute $u^\star$", S_SUB, ORANGE, bold=True, ha="center")
T(1334, 610, "instruction or\nscripted move", S_SMALL, INKC, ha="center", linespacing=1.1)
B(1246, 726, 176, 70, "#F7F9FA", BORDER, 0.7, 6)
T(1334, 761, "Skip repair", S_SUB, NAVY, bold=True, ha="center")
L([(1426, 591), (1458, 591), (1458, 761), (1426, 761)], NAVY, 1.0)
L([(1458, 676), (1480, 676), (1480, 614), (1504, 614)], NAVY, 1.0, arrow=True)
B(1508, 580, 182, 68, "#EDF5F9", BORDER, 0.8, 6)
T(1599, 614, "Frozen VLA", S_SUB, NAVY, bold=True, ha="center")
L([(1599, 652), (1599, 724)], NAVY, 1.0, arrow=True)
B(1508, 728, 182, 66, "white", BORDER, 0.8, 6)
T(1599, 761, "Task B", S_SUB, NAVY, bold=True, ha="center")

L([(880, 830), (936, 830)], NAVY, 0.9, arrow=True)
T(946, 830, "deployment", S_SMALL, INKC)
L([(880, 866), (936, 866)], PURPLE, 0.85, dashed=True, arrow=True)
T(946, 866, "offline labelling and training", S_SMALL, PURPLE)

save_figure(fig, FIG / "fig_overview.pdf", also_png=True, close=False)
fig.savefig(FIG / "fig_overview.svg")
plt.close(fig)
print(FIG / "fig_overview.pdf")
