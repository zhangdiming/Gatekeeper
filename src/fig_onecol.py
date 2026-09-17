import os, sys, pandas as pd, matplotlib
matplotlib.use("Agg")
sys.path.insert(0, os.path.dirname(__file__))
import figstyle
from figstyle import LAB, style as STY
figstyle.apply()
import matplotlib.pyplot as plt
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAB, FIG = os.path.join(PROJ, "tables"), os.path.join(PROJ, "figures")
W = 3.4

def budget_line(ax):
    ax.axhline(2, ls="--", color=figstyle.GREY["dark"], lw=0.8); ax.text(11.9, 2.03, "budget", ha="right", va="bottom", fontsize=6.5)

up = pd.read_csv(os.path.join(TAB, "t3_update_strategies.csv"))
order = [s for s in ["noupdate", "noupdate+QM", "noupdate+R", "L0", "L0L1_1", "L0L1_5", "L0L1_5+R", "L2", "L2+R"] if s in set(up.strategy)]
fig, ax = plt.subplots(2, 1, figsize=(W, 4.3), sharex=True)
for st in order:
    d = up[up.strategy == st].groupby("month")[["acc_all", "FAR"]].mean()
    ax[0].plot(d.index, d.acc_all * 100, label=LAB.get(st, st), **STY(st)); ax[1].plot(d.index, d.FAR * 100, **STY(st))
budget_line(ax[1])
ax[0].set_title("(a) accepted-and-correct target fraction"); ax[1].set_title("(b) false acceptance on unknown services")
ax[0].set_ylabel("TPR at budget 2% (%)"); ax[1].set_ylabel("FAR (%)"); ax[1].set_xlabel("month of 2022"); ax[1].set_xticks(range(2, 13, 2))
h, l = ax[0].get_legend_handles_labels()
fig.tight_layout(h_pad=0.8, rect=(0, 0.17, 1, 1))
fig.legend(h, l, loc="lower center", ncol=2, columnspacing=1.0, handlelength=1.8, bbox_to_anchor=(0.5, 0.0))
fig.savefig(os.path.join(FIG, "f2_update_strategies_1col.pdf")); plt.close(fig)

tm = pd.read_csv(os.path.join(TAB, "t8_threshold_maintenance.csv"))
RL = {"fixed": "fixed (January)", "QM-week": "quantile matching (label-free)", "R-prev": "recalibrated, previous week", "R-conc": "recalibrated, same week"}
fig, ax = plt.subplots(2, 1, figsize=(W, 4.0), sharex=True)
for rule in ("fixed", "QM-week", "R-prev", "R-conc"):
    d = tm[(tm.method == "IPCOD-mask") & (tm.rule == rule)].sort_values("month")
    ax[0].plot(d.month, d.TPR * 100, label=RL[rule], **STY(rule)); ax[1].plot(d.month, d.FAR * 100, **STY(rule))
budget_line(ax[1])
ax[0].set_title("(a) accepted-and-correct target fraction"); ax[1].set_title("(b) false acceptance on unknown services")
ax[0].set_ylabel("TPR of GateKeeper (%)"); ax[1].set_ylabel("FAR (%)"); ax[1].set_xlabel("month of 2022"); ax[1].set_xticks(sorted(tm.month.unique()))
h, l = ax[0].get_legend_handles_labels()
fig.tight_layout(h_pad=0.8, rect=(0, 0.12, 1, 1))
fig.legend(h, l, loc="lower center", ncol=2, columnspacing=1.0, handlelength=1.8, bbox_to_anchor=(0.5, 0.0))
fig.savefig(os.path.join(FIG, "f5_threshold_maintenance_1col.pdf")); plt.close(fig)
print("written f2_update_strategies_1col.pdf, f5_threshold_maintenance_1col.pdf")
