import os, sys, json, glob, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import figstyle; figstyle.apply(); COL = figstyle.COL; LAB = figstyle.LAB; STY = figstyle.style
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJ = ROOT
TAB, FIG = os.path.join(PROJ, "tables"), os.path.join(PROJ, "figures")
os.makedirs(TAB, exist_ok=True); os.makedirs(FIG, exist_ok=True)
tag = sys.argv[1] if len(sys.argv) > 1 else "main"
R = os.path.join(ROOT, "results", "tls", tag)

mon = pd.concat([pd.read_csv(f).assign(seed=int(os.path.basename(f).split("seed")[1].split(".")[0])) for f in glob.glob(os.path.join(R, "monthly_seed*.csv"))])
sig = pd.concat([pd.read_csv(f).assign(seed=int(os.path.basename(f).split("seed")[1].split(".")[0])) for f in glob.glob(os.path.join(R, "signals_seed*.csv"))])
cost = pd.concat([pd.read_csv(f).assign(seed=int(os.path.basename(f).split("seed")[1].split(".")[0])) for f in glob.glob(os.path.join(R, "costs_seed*.csv"))])
sizes = [json.load(open(f)) for f in glob.glob(os.path.join(R, "sizes_seed*.json"))]
n_seeds_all = mon.seed.nunique(); n_seeds = 1
print("seeds:", n_seeds, "months:", sorted(mon.month.unique()), "methods:", sorted(mon.method.unique()))

def agg(df, keys, cols):
    g = df.groupby(keys)[cols]
    m, s = g.mean(), g.std(ddof=0)
    return m, s

def fmt(m, s, col, pct=True, prec=1):
    v = m[col] * (100 if pct else 1); e = s[col] * (100 if pct else 1)
    return v.map(lambda x: f"{x:.{prec}f}") + (e.map(lambda x: f"$\\pm${x:.{prec}f}") if n_seeds > 1 else "")

nu = mon[(mon.strategy == "noupdate") & (mon.seed == 0)]
cols = ["acc_all", "mF1_all", "FAR", "TPR", "AUROC", "FAR_same", "FAR_cross", "reject_rate_known"]
m, s = agg(nu, ["method", "month"], cols)
order = ["Global-closed", "RF-closed", "Hier-closed", "Mask-closed", "MaskRF-closed", "IP-baseline", "Global-MSP", "OpenMax", "Global-Energy", "RF-MaxProb", "IPCOD-group-pooled", "IPCOD-group", "IPCOD-RF", "IPCOD-mask"]
order = [o for o in order if o in m.index.get_level_values(0)]
rows = []
for meth in order:
    for mo in (2, 7, 12):
        if (meth, mo) in m.index:
            r = m.loc[(meth, mo)]; e = s.loc[(meth, mo)]
            rows.append(dict(method=meth, month=mo, **{c: r[c] for c in cols}, **{c + "_std": e[c] for c in cols}))
T1 = pd.DataFrame(rows); T1.to_csv(os.path.join(TAB, "t1_main_tls.csv"), index=False)
with open(os.path.join(TAB, "t1_main_tls.tex"), "w") as f:
    def c(meth, mo, col):
        if (meth, mo) not in m.index: return "--"
        v = m.loc[(meth, mo), col]
        return "--" if (np.isnan(v) or (col == "FAR" and v >= 1.0)) else f"{v*100:.1f}"
    f.write("\\begin{tabular}{lrrrrrrrrrrrr}\n\\toprule\n & \\multicolumn{3}{c}{Acc$_{\\mathrm{all}}$ / TPR@2\\%} & \\multicolumn{3}{c}{macro-F1} & \\multicolumn{3}{c}{FAR@2\\%} & \\multicolumn{3}{c}{AUROC} \\\\\n\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}\\cmidrule(lr){11-13}\nMethod & M2 & M7 & M12 & M2 & M7 & M12 & M2 & M7 & M12 & M2 & M7 & M12 \\\\\n\\midrule\n")
    closed = [o for o in order if o in ("Global-closed", "RF-closed", "Hier-closed", "Mask-closed", "MaskRF-closed")]
    openm = [o for o in order if o not in closed]
    f.write("\\multicolumn{13}{l}{\\emph{Closed set (rejection disabled)}} \\\\\n")
    for meth in closed:
        f.write(f"{meth} & " + " & ".join(c(meth, mo, "acc_all") for mo in (2, 7, 12)) + " & " + " & ".join(c(meth, mo, "mF1_all") for mo in (2, 7, 12)) + " & -- & -- & -- & -- & -- & -- \\\\\n")
    f.write("\\midrule\n\\multicolumn{13}{l}{\\emph{Open set (thresholds calibrated to a 2\\% budget in January and kept fixed)}} \\\\\n")
    for meth in openm:
        f.write(f"{meth} & " + " & ".join(c(meth, mo, "TPR") for mo in (2, 7, 12)) + " & " + " & ".join(c(meth, mo, "mF1_all") for mo in (2, 7, 12)) + " & " + " & ".join(c(meth, mo, "FAR") for mo in (2, 7, 12)) + " & " + " & ".join(c(meth, mo, "AUROC") for mo in (2, 7, 12)) + " \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

osr = [o for o in order if o not in ("Global-closed", "RF-closed", "Hier-closed", "Mask-closed", "MaskRF-closed")]
m2, s2 = agg(nu[nu.method.isin(osr)], ["method"], ["FAR", "FAR_same", "FAR_cross", "TPR", "AUROC", "reject_rate_known"])
m2.to_csv(os.path.join(TAB, "t2_stratified_avg_over_months.csv"))
mm, ss = agg(nu[(nu.method.isin(osr)) & (nu.month == 2)], ["method"], ["FAR", "FAR_same", "FAR_cross", "TPR", "AUROC", "reject_rate_known"])
with open(os.path.join(TAB, "t2_stratified_tls.tex"), "w") as f:
    f.write("\\begin{tabular}{lrrrrrrrr}\n\\toprule\n & \\multicolumn{4}{c}{Month 2 (one month after calibration)} & \\multicolumn{4}{c}{Average over months 2--12} \\\\\n\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\nMethod & FAR & FAR$_{\\mathrm{same}}$ & FAR$_{\\mathrm{cross}}$ & TPR & FAR & FAR$_{\\mathrm{same}}$ & FAR$_{\\mathrm{cross}}$ & TPR \\\\\n\\midrule\n")
    for meth in osr:
        if meth not in mm.index: continue
        a = mm.loc[meth]; b = m2.loc[meth]
        f.write(f"{meth} & {a['FAR']*100:.1f} & {a['FAR_same']*100:.1f} & {a['FAR_cross']*100:.1f} & {a['TPR']*100:.1f} & {b['FAR']*100:.1f} & {b['FAR_same']*100:.1f} & {b['FAR_cross']*100:.1f} & {b['TPR']*100:.1f} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

up = mon[(mon.method == "IPCOD-mask") & (mon.seed == 0)]
m3, s3 = agg(up, ["strategy", "month"], ["acc_all", "FAR", "TPR", "mF1_all"])
m3.to_csv(os.path.join(TAB, "t3_update_strategies.csv"))
cm = cost[cost.seed == 0].groupby("strategy")[["labels", "train_s", "ip_added"]].sum()
ct = cost[cost.seed == 0].groupby("strategy")["trig"].sum()
strats = [s_ for s_ in ["noupdate", "noupdate+QM", "noupdate+R", "L0", "L0L1_1", "L0L1_5", "L0L1_5+R", "L2", "L2+R"] if s_ in m3.index.get_level_values(0)]
months = sorted(up.month.unique())
with open(os.path.join(TAB, "t3_update_strategies.tex"), "w") as f:
    f.write("\\begin{tabular}{l" + "r" * len(months) + "rrr}\n\\toprule\nStrategy & " + " & ".join(f"M{mo}" for mo in months) + " & Mean & Labels & Train (s) \\\\\n\\midrule\n")
    tex = lambda st: LAB.get(st, st).replace("%", "\\%")
    for st in strats:
        vals = [m3.loc[(st, mo), "acc_all"] * 100 if (st, mo) in m3.index else np.nan for mo in months]
        f.write(f"{tex(st)} & " + " & ".join(f"{v:.1f}" for v in vals) + f" & {np.nanmean(vals):.1f} & {cm.loc[st, 'labels']:.0f} & {cm.loc[st, 'train_s']:.0f} \\\\\n")
    f.write("\\midrule\n")
    for st in strats:
        vals = [m3.loc[(st, mo), "FAR"] * 100 if (st, mo) in m3.index else np.nan for mo in months]
        f.write(f"FAR: {tex(st)} & " + " & ".join(f"{v:.1f}" for v in vals) + f" & {np.nanmean(vals):.1f} & & \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

if sizes:
    S = pd.DataFrame(sizes).mean(numeric_only=True); S.to_csv(os.path.join(TAB, "t4_sizes.csv"))

fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.55))
for meth in ["Mask-closed", "MaskRF-closed", "Hier-closed", "IP-baseline", "RF-closed", "Global-closed"]:
    d = nu[nu.method == meth].groupby("month")[["acc_all"]].mean()
    if d.empty: continue
    ax[0].plot(d.index, d.acc_all * 100, label=figstyle.disp(meth), **STY(meth))
for meth in ["IPCOD-mask", "IPCOD-RF", "IPCOD-group", "RF-MaxProb", "Global-Energy", "Global-MSP", "OpenMax"]:
    d = nu[nu.method == meth].groupby("month")[["acc_all", "FAR"]].mean()
    if d.empty: continue
    ax[1].plot(d.index, d.acc_all * 100, label=figstyle.disp(meth), **STY(meth))
    ax[2].plot(d.index, d.FAR * 100, label=figstyle.disp(meth), **STY(meth))
ax[2].axhline(2, ls="--", color=figstyle.GREY["dark"], lw=0.8); ax[2].text(11.9, 2.08, "budget", ha="right", va="bottom", fontsize=6.5)
ax[0].set_ylabel("closed-set accuracy (%)"); ax[1].set_ylabel("TPR at budget 2% (%)"); ax[2].set_ylabel("FAR, thresholds fixed (%)")
for a in ax: a.set_xlabel("month of 2022"); a.set_xticks(range(2, 13, 2))
ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2, columnspacing=0.8, handlelength=1.8)
ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2, columnspacing=0.8, handlelength=1.8)
ax[0].set_title("(a) no rejection"); ax[1].set_title("(b) open-set, calibrated in January"); ax[2].set_title("(c) achieved FAR")
fig.set_size_inches(7.2, 3.2); fig.tight_layout(w_pad=1.2); fig.savefig(os.path.join(FIG, "f1_monthly_noupdate.pdf")); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))
for st in strats:
    d = up[up.strategy == st].groupby("month")[["acc_all", "FAR"]].mean()
    ax[0].plot(d.index, d.acc_all * 100, label=LAB.get(st, st), **STY(st))
    ax[1].plot(d.index, d.FAR * 100, label=LAB.get(st, st), **STY(st))
ax[1].axhline(2, ls="--", color=figstyle.GREY["dark"], lw=0.8); ax[1].text(11.9, 2.03, "budget", ha="right", va="bottom", fontsize=6.5)
ax[0].set_title("(a) accepted-and-correct target fraction"); ax[1].set_title("(b) false acceptance on unknown services")
ax[0].set_xlabel("month of 2022"); ax[0].set_ylabel("TPR of GateKeeper at budget 2% (%)"); ax[1].set_xlabel("month of 2022"); ax[1].set_ylabel("FAR (%)")
h, l = ax[0].get_legend_handles_labels(); ax[0].set_xticks(range(2, 13, 2)); ax[1].set_xticks(range(2, 13, 2))
fig.set_size_inches(7.2, 3.2); fig.tight_layout(w_pad=1.2, rect=(0, 0.16, 1, 1))
fig.legend(h, l, loc="lower center", ncol=3, columnspacing=1.2, handlelength=1.8, bbox_to_anchor=(0.5, 0.0))
fig.savefig(os.path.join(FIG, "f2_update_strategies.pdf")); plt.close(fig)

sn = sig[(sig.strategy == "noupdate") & (sig.seed == 0)].groupby("week_start")[["r_new", "a_ip", "E_shift", "acc_all_week"]].mean().reset_index()
sn["t"] = pd.to_datetime(sn.week_start.astype(str))
from scipy.stats import spearmanr
corr = {k: spearmanr(sn[k], sn.acc_all_week) for k in ["r_new", "a_ip", "E_shift"]}
json.dump({k: {"rho": float(v.correlation), "p": float(v.pvalue)} for k, v in corr.items()}, open(os.path.join(TAB, "t5_signal_correlation.json"), "w"), indent=1)
print("signal correlations:", {k: (round(v.correlation, 3), round(v.pvalue, 4)) for k, v in corr.items()})
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5))
drift_week = pd.Timestamp("2022-03-08")
for i, (k, lab, pl) in enumerate([("r_new", "new-IP rate $r_{\\mathrm{new}}$", "(a)"), ("a_ip", "IP agreement $a_{\\mathrm{ip}}$", "(b)"), ("E_shift", "energy shift $W_1$", "(c)")]):
    ax[i].plot(sn.t, sn[k], color=figstyle.FCS["navy"], lw=1.3); ax[i].set_ylabel(lab, color=figstyle.FCS["navy"])
    ax[i].xaxis.set_major_locator(matplotlib.dates.MonthLocator(bymonth=[2, 4, 6, 8, 10, 12])); ax[i].xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b"))
    ax[i].axvline(drift_week, color=figstyle.GREY["light"], lw=0.8, ls=":"); ax[i].grid(False)
    a2 = ax[i].twinx(); a2.plot(sn.t, sn.acc_all_week * 100, color=figstyle.FCS["red"], lw=1.2, ls="--"); a2.set_ylabel("weekly Acc$_{\\mathrm{all}}$ (%)", color=figstyle.FCS["red"])
    a2.spines["top"].set_visible(False); a2.grid(False)
    ax[i].set_title(f"{pl} Spearman $\\rho$ = {corr[k].correlation:.2f}")
ax[0].text(drift_week, ax[0].get_ylim()[1], " 8 Mar", fontsize=6.5, va="top", color=figstyle.GREY["mid"])
fig.tight_layout(w_pad=1.6); fig.savefig(os.path.join(FIG, "f3_drift_signals.pdf")); plt.close(fig)
print("tables/figures written")

if n_seeds_all > 1:
    sv = mon[mon.strategy.isin(["noupdate", "noupdate+QM"]) & mon.month.isin([2, 7, 12])]
    meths = [m_ for m_ in ["Global-closed", "Hier-closed", "Mask-closed", "Global-Energy", "Global-MSP", "IPCOD-mask", "IPCOD-group"] if sv[sv.method == m_].seed.nunique() == n_seeds_all]
    g = sv[sv.method.isin(meths)].groupby(["strategy", "method", "month"])[["acc_all", "FAR"]].agg(["mean", "std"])
    g.to_csv(os.path.join(TAB, "t10_seed_variation.csv"))
    with open(os.path.join(TAB, "t10_seed_variation.tex"), "w") as f:
        f.write("\\begin{tabular}{llrrr}\n\\toprule\nStrategy & Method & M2 & M7 & M12 \\\\\n\\midrule\n")
        for st in ["noupdate", "noupdate+QM"]:
            for m_ in (meths if st == "noupdate" else [x for x in meths if x in ("Global-Energy", "IPCOD-mask")]):
                cells = []
                for mo in (2, 7, 12):
                    a = g.loc[(st, m_, mo)]
                    cell = f"{a[('acc_all','mean')]*100:.1f}$\\pm${a[('acc_all','std')]*100:.1f}"
                    if a[("FAR", "mean")] < 1: cell += f" / {a[('FAR','mean')]*100:.1f}$\\pm${a[('FAR','std')]*100:.1f}"
                    cells.append(cell)
                f.write(f"{LAB.get(st, st).replace('%', chr(92)+'%')} & {m_} & " + " & ".join(cells) + " \\\\\n")
            if st == "noupdate": f.write("\\midrule\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print("T10 seeds:", n_seeds_all, meths)
