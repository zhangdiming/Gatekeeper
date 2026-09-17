import os, sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, FancyBboxPatch, Polygon
import figstyle
F, G = figstyle.FCS, figstyle.GREY
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PDF = os.path.normpath(os.path.join(HERE, "..", "figures", "f0_architecture.pdf"))
OUT_PNG = os.path.normpath(os.path.join(HERE, "..", "figures", "f0_architecture_preview.png"))

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 6.8, "pdf.fonttype": 42})
fig = plt.figure(figsize=(7.2, 3.05)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 42); ax.axis("off")
rng = np.random.RandomState(3)

def arrow(x0, y0, x1, y1, color=G["dark"], lw=0.8, ls="-", rad=0.0, ms=7):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=ms, lw=lw, color=color, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))
def label(x, y, s, **kw):
    kw.setdefault("ha", "center"); kw.setdefault("va", "center"); kw.setdefault("fontsize", 6.6); ax.text(x, y, s, **kw)
def panel(x, y, w, h, title=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.8", fc="white", ec=G["pale"], lw=0.7))
    if title: label(x + w / 2, y + h + 1.1, title, fontsize=7.0, color=G["dark"], weight="bold")

panel(1.5, 17, 16, 20, "flow $c$ under ECH")

for i, (txt, col) in enumerate([("dst IP", F["navy"]), ("ASN", F["navy"]), ("port", F["navy"]), ("SNI", G["light"])]):
    x = 2.6 + i * 3.7
    ax.add_patch(FancyBboxPatch((x, 32.6), 3.3, 2.6, boxstyle="round,pad=0,rounding_size=0.5", fc=col, ec="none", alpha=0.95 if col != G["light"] else 0.6))
    label(x + 1.65, 33.9, txt, color="white", fontsize=5.6)
ax.plot([15.8, 17.0], [33.0, 34.9], color=F["red"], lw=1.1); ax.plot([15.8, 17.0], [34.9, 33.0], color=F["red"], lw=1.1)

xs = np.linspace(3.2, 16.3, 14); sizes = rng.gamma(2.0, 1.0, 14); sizes = 1.2 + 4.2 * sizes / sizes.max(); dirs = rng.choice([1, -1], 14, p=[0.45, 0.55]); dirs[:2] = [1, -1]
ax.plot([2.4, 17.0], [25.0, 25.0], color=G["mid"], lw=0.6)
for x, s, d in zip(xs, sizes, dirs):
    ax.add_patch(Rectangle((x - 0.4, 25.0 if d > 0 else 25.0 - s), 0.8, s, fc=F["navy"] if d > 0 else F["teal"], ec="none"))
label(9.5, 18.9, "packet sizes, directions,\ninter-arrival times", fontsize=5.6, color=G["mid"])
label(9.5, 30.6, "visible", fontsize=5.8, color=G["mid"])

panel(21.5, 17, 20, 20, "gate: IP table $\\rightarrow$ allowed set $\\mathcal{A}(c)$")
K = 12
def mask_glyph(x, y, allowed, w=7.6, h=1.4):
    cw = w / K
    for j in range(K):
        ax.add_patch(Rectangle((x + j * cw, y), cw * 0.9, h, fc=F["navy"] if j in allowed else G["pale"], ec="none"))
rows = [("single IP", [4]), ("multi IP", [4, 5]), ("new IP,\nknown ASN", [3, 4, 5, 6]), ("new IP,\nunknown ASN", list(range(K)))]
ys = [33.2, 29.4, 25.6, 21.8]
for (nm, al), y in zip(rows, ys):
    label(22.6, y + 0.7, nm, ha="left", fontsize=5.8, linespacing=1.1)
    mask_glyph(33.2, y, al)
label(37.0, 36.0, "$\\mathcal{A}(c) \\subseteq \\mathcal{K}$", fontsize=5.8, color=G["mid"])
arrow(17.5, 27, 21.5, 27, color=G["dark"])

ax.add_patch(Rectangle((18.2, 28.2), 2.6, 2.4, fc="white", ec=G["mid"], lw=0.6))
for r in range(3):
    ax.plot([18.4, 20.6], [28.6 + r * 0.75] * 2, color=G["mid"], lw=0.45)
label(19.5, 31.4, "$N(\\mathrm{ip},y)$", fontsize=5.4, color=G["mid"])

panel(45.5, 17, 20, 20, "score inside $\\mathcal{A}(c)$")
for i in range(3):
    ax.add_patch(Rectangle((46.8 + i * 1.5, 22 + i * 0.6), 1.0, 11 - i * 1.2, fc=F["teal"], ec="none", alpha=0.85 - 0.2 * i))
label(48.7, 20.0, "1D-CNN", fontsize=5.8, color=G["mid"])
arrow(51.6, 27.5, 53.6, 27.5)

lz = rng.rand(K) * 6 + 1; lz[4] = 8.5; lz[5] = 6.0; lz[3] = 4.5
allowed = {3, 4, 5, 6}
bw = 9.6 / K
for j in range(K):
    ax.add_patch(Rectangle((54.2 + j * bw, 22.0), bw * 0.85, lz[j], fc=F["navy"] if j in allowed else G["pale"], ec="none"))
ax.plot([54.0, 64.2], [22.0, 22.0], color=G["mid"], lw=0.6)
label(59.1, 20.3, "logits $z_y$, $y \\in \\mathcal{A}(c)$", fontsize=5.8, color=G["mid"])
label(59.1, 33.0, "$E_{\\mathcal{A}} = -\\log\\sum_{y\\in\\mathcal{A}(c)} e^{z_y}$", fontsize=6.4)
arrow(41.5, 27, 45.5, 27)

panel(69.5, 17, 29, 20, "reject with per-path budget $b$")
ax.plot([71.5, 96.5], [24.5, 24.5], color=G["dark"], lw=0.8); label(96.8, 24.5, "$E$", ha="left", fontsize=6.4)

xx = np.linspace(71.5, 96.5, 200)
def dens(mu, sd, amp): return amp * np.exp(-0.5 * ((xx - mu) / sd) ** 2)
ax.fill_between(xx, 24.5, 24.5 + dens(80, 3.0, 6.5), color=F["navy"], alpha=0.35, lw=0); ax.fill_between(xx, 24.5, 24.5 + dens(90, 3.2, 5.5), color=G["light"], alpha=0.5, lw=0)
label(76.2, 30.0, "target", fontsize=5.6, color=F["navy"]); label(92.8, 29.2, "calibration-\nunknown", fontsize=5.4, color=G["mid"], linespacing=1.05)
for i, (xt, nm) in enumerate([(83.3, "$\\delta_1$"), (84.4, "$\\delta_2$"), (85.6, "$\\delta_3$"), (82.2, "$\\delta_4$")]):
    ax.plot([xt, xt], [24.5, 32.6], color=F["red"], lw=0.7, ls="--")
label(84.0, 34.0, "$\\delta_p$ = $b$-quantile per path", fontsize=5.6, color=F["red"])
label(76.5, 21.2, "accept $\\Rightarrow \\hat y$", fontsize=6.4, color=F["navy"]); label(91.5, 21.2, "reject", fontsize=6.4, color=F["red"])
arrow(78.0, 22.6, 74.5, 22.6, color=F["navy"], ms=6); arrow(88.5, 22.6, 92.5, 22.6, color=F["red"], ms=6)
arrow(65.5, 27, 69.5, 27)

panel(21.5, 2.0, 77, 10.5, None)
label(22.6, 11.1, "weekly, no labels:", ha="left", fontsize=6.4, weight="bold", color=G["dark"])
tt = np.linspace(0, 1, 40)
def spark(x0, y0, w, h, y, col, name):
    y = (y - y.min()) / (y.max() - y.min() + 1e-9)
    ax.plot(x0 + tt * w, y0 + y * h, color=col, lw=1.0); label(x0 + w / 2, y0 - 1.5, name, fontsize=6.2, color=col)
spark(24, 5.0, 12, 4.2, 0.25 + 0.15 * tt + 0.02 * rng.randn(40), F["navy"], "new-IP rate $r_{\\mathrm{new}}$")
spark(40, 5.0, 12, 4.2, 0.85 - 0.25 * (tt > 0.18) - 0.03 * tt, F["navy"], "IP agreement $a_{\\mathrm{ip}}$")
spark(56, 5.0, 12, 4.2, 0.2 + 1.8 * (tt > 0.18) + 0.05 * rng.randn(40), F["navy"], "energy shift $W_1$")
for x0 in (40, 56):
    ax.plot([x0 + 0.18 * 12] * 2, [4.6, 9.8], color=F["red"], lw=0.6, ls=":")
label(73.5, 9.6, "quantile matching: keep acceptance rate $a_p$", ha="left", fontsize=6.0)
label(73.5, 7.4, "recalibrate $\\delta_p$ on fresh non-target traffic", ha="left", fontsize=6.0)
label(73.5, 5.2, "extend IP table  /  fine-tune  /  retrain", ha="left", fontsize=6.0)

arrow(84.0, 12.5, 84.0, 17.0, color=F["purple"], ls="--", ms=6)
arrow(31.5, 12.5, 31.5, 17.0, color=F["purple"], ls="--", ms=6)
arrow(55.5, 12.5, 55.5, 17.0, color=F["purple"], ls="--", ms=6)
label(32.6, 14.7, "IP table", ha="left", fontsize=5.4, color=F["purple"]); label(56.6, 14.7, "model", ha="left", fontsize=5.4, color=F["purple"]); label(85.1, 14.7, "$\\delta_p$", ha="left", fontsize=5.8, color=F["purple"])
label(1.5, 7.0, "drift\nmonitor", ha="left", fontsize=6.6, weight="bold", color=G["dark"]); arrow(9.5, 7.0, 21.5, 7.0, color=G["mid"])

os.makedirs(os.path.dirname(OUT_PDF), exist_ok=True); fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.02); fig.savefig(OUT_PNG, dpi=220, bbox_inches="tight", pad_inches=0.02)
print("wrote", OUT_PDF)
