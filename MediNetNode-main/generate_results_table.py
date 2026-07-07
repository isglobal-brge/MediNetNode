"""
Generate a publication-quality results table image (booktabs style).
Caption goes in Google Docs separately — image contains only the table.
Output: results_table.png (300 dpi)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.family"]      = "serif"
rcParams["font.serif"]       = ["DejaVu Serif", "Times New Roman", "Georgia"]
rcParams["mathtext.fontset"] = "dejavuserif"
rcParams["font.size"]        = 11

# ── Data ───────────────────────────────────────────────────────────────────────
COL_HEADERS = ["Run", r"$\sigma$", "Batch", "Acc", "Loss", "F1"]

ROWS = [
    [r"Experimental ($b\!=\!32$)", "0.957", "32", "0.655", "1.595", "0.028"],
    [r"Experimental ($b\!=\!16$)", "0.808", "16", "0.654", "1.637", "0.024"],
    ["DP Production",              "0.710", "32", "0.650", "1.959", "0.011"],
    [r"DP cost vs. exp ($b\!=\!32$)", "",   "",   "-0.005", "+0.357", "-0.017"],
]

N_COLS    = len(COL_HEADERS)
N_ROWS    = len(ROWS)

# ── Layout ─────────────────────────────────────────────────────────────────────
COL_W   = [3.0, 0.7, 0.7, 0.72, 0.72, 0.72]   # column widths in inches
ROW_H   = 0.50                                  # row height in inches
HDR_H   = 0.52                                  # header row height
PAD     = 0.12                                  # outer padding

fig_w = sum(COL_W) + PAD * 2
fig_h = HDR_H + N_ROWS * ROW_H + PAD * 2

fig, ax = plt.subplots(figsize=(fig_w, fig_h))
ax.set_xlim(0, fig_w)
ax.set_ylim(0, fig_h)
ax.axis("off")

# ── Column x positions ─────────────────────────────────────────────────────────
def col_left(i):
    return PAD + sum(COL_W[:i])

def col_center(i):
    return col_left(i) + COL_W[i] / 2

# ── Y positions (from top) ────────────────────────────────────────────────────
y_toprule  = fig_h - PAD
y_midrule  = y_toprule - HDR_H
y_midrule2 = y_midrule - ROW_H * 3        # thin rule before delta row
y_botrule  = y_midrule - ROW_H * N_ROWS

x0, x1 = PAD, fig_w - PAD

for y, lw in [
    (y_toprule,  1.6),
    (y_midrule,  0.9),
    (y_midrule2, 0.9),
    (y_botrule,  1.6),
]:
    ax.plot([x0, x1], [y, y], color="black", linewidth=lw,
            solid_capstyle="butt", clip_on=False)

# ── Header ────────────────────────────────────────────────────────────────────
hy = (y_toprule + y_midrule) / 2
for i, hdr in enumerate(COL_HEADERS):
    ha = "left" if i == 0 else "center"
    x  = col_left(i) + 0.06 if ha == "left" else col_center(i)
    ax.text(x, hy, hdr, ha=ha, va="center", fontweight="bold", fontsize=11)

# ── Rows ──────────────────────────────────────────────────────────────────────
for r, row in enumerate(ROWS):
    ry       = y_midrule - ROW_H * r - ROW_H / 2
    is_delta = (r == N_ROWS - 1)
    kw       = dict(fontstyle="italic", color="#555555") if is_delta else {}

    for i, cell in enumerate(row):
        if cell == "":
            continue
        ha = "left" if i == 0 else "center"
        x  = col_left(i) + 0.06 if ha == "left" else col_center(i)
        ax.text(x, ry, cell, ha=ha, va="center", fontsize=11, **kw)

plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig("results_table.png", dpi=300, bbox_inches="tight",
            facecolor="white", edgecolor="none")
print("Saved: results_table.png")
