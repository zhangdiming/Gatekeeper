import os, sys, json, glob, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import figstyle; figstyle.apply(); STY = figstyle.style
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); PROJ = ROOT
tag = sys.argv[1] if len(sys.argv) > 1 else "main"
R = os.path.join(ROOT, "results", "tls", tag); TAB = os.path.join(PROJ, "tables"); FIG = os.path.join(PROJ, "figures"); os.makedirs(TAB, exist_ok=True); os.makedirs(FIG, exist_ok=True)
split = json.load(open(os.path.join(ROOT, "results", "tls", "split.json")))
prov = {int(k): v for k, v in split["provider"].items()}; names = {int(k): v for k, v in split["names"].items()}
cal = dict(np.load(os.path.join(R, "flows_seed0_cal.npz")))
if os.path.exists(os.path.join(R, "flows_rf_seed0_cal.npz")): cal.update(dict(np.load(os.path.join(R, "flows_rf_seed0_cal.npz"))))
months = sorted(int(os.path.basename(f).split("_M")[1].split(".")[0]) for f in glob.glob(os.path.join(R, "flows_seed0_M*.npz")))
print("months with dumps:", months)

METHODS = {
    "Global-MSP": ("msp_score", "global_pred", False), "Global-Energy": ("global_score", "global_pred", False), "OpenMax": ("openmax_score", "openmax_pred", False),
    "IPCOD-mask": ("mask_score", "mask_pred", True), "IPCOD-mask-pooled": ("mask_score", "mask_pred", False), "IPCOD-group": ("group_score", "group_pred", True), "RF-MaxProb": ("rf_score", "rf_pred", False), "IPCOD-RF": ("rfmask_score", "rfmask_pred", True)}
BUDGETS = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20]

def thresholds(score, path, is_ucal, b, per_path):
    pooled = np.quantile(score[is_ucal], b)
    if not per_path: return np.full(4, pooled)
    d = np.full(4, pooled)
    for p in range(4):
        m = is_ucal & (path == p)
        if m.sum() >= 50: d[p] = np.quantile(score[m], b)
    return d

def load_month(m):
    z = dict(np.load(os.path.join(R, f"flows_seed0_M{m}.npz")))
    rf = os.path.join(R, f"flows_rf_seed0_M{m}.npz")
    if os.path.exists(rf): z.update(dict(np.load(rf)))
    return z

cal_rf = None
rows = []; sweep = {}
for m in months:
    z = load_month(m); path = z["path"]; k = z["is_known"]; u = z["is_utest"]; same = z["is_same"]; y = z["y"]
    for name, (sk, pk, pp) in METHODS.items():
        if sk not in z: continue
        for b in BUDGETS:
            if sk in cal:
                d = thresholds(cal[sk], cal["path"], cal["is_ucal"], b, pp)
            else:
                d = thresholds(z[sk], path, z["is_ucal"], b, pp)
            acc = z[sk] < d[path]; pred = z[pk]
            rows.append(dict(month=m, method=name, budget=b, calib_source=("cal-period" if sk in cal else "same-month-ucal"),
                             TPR=float((acc[k] & (pred[k] == y[k])).mean()), FAR=float(acc[u].mean()),
                             FAR_same=float(acc[same].mean()), FAR_cross=float(acc[u & ~same].mean()),
                             **{f"TPR_p{p}": float((acc & (pred == y))[k & (path == p)].mean()) if (k & (path == p)).any() else np.nan for p in range(4)}))
df = pd.DataFrame(rows); df.to_csv(os.path.join(TAB, "t6_budget_sweep.csv"), index=False)

z = load_month(months[0]); u = z["is_utest"]; raw = z["raw_y"]
brk = []
for name in ("IPCOD-mask", "IPCOD-RF", "RF-MaxProb", "Global-Energy"):
    sk, pk, pp = METHODS[name]
    if sk not in z: continue
    d = thresholds(cal[sk], cal["path"], cal["is_ucal"], 0.02, pp) if sk in cal else thresholds(z[sk], z["path"], z["is_ucal"], 0.02, pp)
    acc = z[sk] < d[z["path"]]
    for cls in np.unique(raw[u]):
        mm = u & (raw == cls)
        brk.append(dict(method=name, service=names.get(int(cls), str(cls)), provider=prov.get(int(cls), "?"), same_provider=bool(z["is_same"][mm][0]),
                        n=int(mm.sum()), FAR=float(acc[mm].mean()), share_single_ip=float((z["path"][mm] == 0).mean()), share_new_ip=float((z["path"][mm] >= 2).mean())))
pd.DataFrame(brk).to_csv(os.path.join(TAB, "t7_unknown_service_breakdown.csv"), index=False)

fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.55))
for i, m in enumerate([months[0], months[len(months) // 2]]):
    for name in ["IPCOD-mask", "IPCOD-RF", "RF-MaxProb", "Global-Energy", "Global-MSP", "OpenMax"]:
        d = df[(df.month == m) & (df.method == name)].sort_values("budget")
        if d.empty: continue
        ax[i].plot(d.FAR * 100, d.TPR * 100, label=figstyle.disp(name), **STY(name))
    ax[i].set_xlabel("achieved FAR on unknown services (%)"); ax[i].set_ylabel("TPR on target services (%)"); ax[i].set_title(f"({'ab'[i]}) month {m}"); ax[i].set_xscale("log")
h, l = ax[0].get_legend_handles_labels()
for name in ["IPCOD-mask", "IPCOD-RF", "RF-MaxProb", "Global-Energy"]:
    d = df[(df.method == name) & (df.budget == 0.02)].sort_values("month")
    ax[2].plot(d.month, d.FAR * 100, label=figstyle.disp(name), **STY(name))
ax[2].axhline(2, ls="--", color=figstyle.GREY["dark"], lw=0.8); ax[2].text(11.9, 2.05, "budget", ha="right", va="bottom", fontsize=6.5)
ax[2].set_xlabel("month of 2022"); ax[2].set_ylabel("FAR at budget 2% (%)"); ax[2].set_title("(c) thresholds fixed in January"); ax[2].set_xticks(range(2, 13, 2))
fig.set_size_inches(7.2, 3.0); fig.tight_layout(w_pad=1.2, rect=(0, 0.12, 1, 1))
fig.legend(h, l, loc="lower center", ncol=6, columnspacing=1.2, handlelength=1.8, bbox_to_anchor=(0.5, 0.0))
fig.savefig(os.path.join(FIG, "f4_budget_sweep.pdf")); plt.close(fig)
print(df[(df.month == months[0])].pivot_table(index="method", columns="budget", values="TPR").round(3))
print(df[(df.month == months[0])].pivot_table(index="method", columns="budget", values="FAR").round(3))

def acc_fracs(score, path, d):
    return np.array([float((score < d[path])[path == p].mean()) if (path == p).any() else np.nan for p in range(4)])
tm_rows = []
for name in ["IPCOD-mask", "IPCOD-RF", "RF-MaxProb", "Global-Energy", "Global-MSP", "OpenMax"]:
    sk, pk, pp = METHODS[name]
    if sk not in cal: continue
    d0 = thresholds(cal[sk], cal["path"], cal["is_ucal"], 0.02, pp); a0 = acc_fracs(cal[sk], cal["path"], d0); a0_all = float((cal[sk] < d0[cal["path"]]).mean())
    prev = None
    for m in months:
        z = load_month(m); days = np.unique(z["day"]); k = z["is_known"]; u = z["is_utest"]; y = z["y"]; pred = z[pk]; sc = z[sk]; path = z["path"]
        agg = {r: dict(acc=np.zeros(len(sc), bool)) for r in ("fixed", "QM-week", "R-prev", "R-conc")}
        for w in range(0, len(days), 7):
            sel = np.isin(z["day"], days[w:w + 7])
            agg["fixed"]["acc"][sel] = sc[sel] < d0[path[sel]]
            dq = d0.copy()
            for p in range(4):
                mp = sel & (path == p)
                if mp.sum() >= 50 and not np.isnan(a0[p]): dq[p] = np.quantile(sc[mp], a0[p]) if pp else dq[p]
            if not pp: dq[:] = np.quantile(sc[sel], a0_all)
            agg["QM-week"]["acc"][sel] = sc[sel] < dq[path[sel]]
            dc = thresholds(sc[sel], path[sel], z["is_ucal"][sel], 0.02, pp); agg["R-conc"]["acc"][sel] = sc[sel] < dc[path[sel]]
            dp = thresholds(*prev, 0.02, pp) if prev is not None else d0; agg["R-prev"]["acc"][sel] = sc[sel] < dp[path[sel]]
            prev = (sc[sel], path[sel], z["is_ucal"][sel])
        for rule, v in agg.items():
            acc = v["acc"]
            tm_rows.append(dict(method=name, rule=rule, month=m, TPR=float((acc[k] & (pred[k] == y[k])).mean()), FAR=float(acc[u].mean()),
                                FAR_same=float(acc[z["is_same"]].mean()), FAR_cross=float(acc[u & ~z["is_same"]].mean())))
tm = pd.DataFrame(tm_rows); tm.to_csv(os.path.join(TAB, "t8_threshold_maintenance.csv"), index=False)
piv = tm.pivot_table(index=["method", "rule"], columns="month", values=["TPR", "FAR"]).round(3)
print(piv.to_string())
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.55))
RL = {"fixed": "fixed (January)", "QM-week": "quantile matching (label-free)", "R-prev": "recalibrated, previous week", "R-conc": "recalibrated, same week"}
for rule in ("fixed", "QM-week", "R-prev", "R-conc"):
    d = tm[(tm.method == "IPCOD-mask") & (tm.rule == rule)].sort_values("month")
    ax[0].plot(d.month, d.TPR * 100, label=RL[rule], **STY(rule)); ax[1].plot(d.month, d.FAR * 100, label=RL[rule], **STY(rule))
ax[1].axhline(2, ls="--", color=figstyle.GREY["dark"], lw=0.8); ax[1].text(11.9, 2.03, "budget", ha="right", va="bottom", fontsize=6.5)
ax[0].set_ylabel("TPR of GateKeeper (%)"); ax[1].set_ylabel("FAR (%)"); ax[0].set_xlabel("month of 2022"); ax[1].set_xlabel("month of 2022"); ax[0].legend(loc="center right", bbox_to_anchor=(1.0, 0.47), handlelength=1.8)
ax[0].set_title("(a) accepted-and-correct target fraction"); ax[1].set_title("(b) false acceptance on unknown services")
ax[0].set_xticks(months); ax[1].set_xticks(months)
fig.tight_layout(w_pad=1.2); fig.savefig(os.path.join(FIG, "f5_threshold_maintenance.pdf")); plt.close(fig)
print("threshold maintenance written")

sel_months = [m for m in (3, 7, 12) if m in months]
with open(os.path.join(TAB, "t8_threshold_maintenance.tex"), "w") as f:
    f.write("\\begin{tabular}{ll" + "r" * len(sel_months) + "}\n\\toprule\nMethod & Rule & " + " & ".join(f"M{m}" for m in sel_months) + " \\\\\n\\midrule\n")
    for name in ["IPCOD-mask", "IPCOD-RF", "RF-MaxProb", "Global-Energy", "Global-MSP", "OpenMax"]:
        for rule in ("fixed", "QM-week", "R-prev", "R-conc"):
            d = tm[(tm.method == name) & (tm.rule == rule)].set_index("month")
            if d.empty: continue
            cells = " & ".join(f"{d.loc[m, 'TPR']*100:.1f} / {d.loc[m, 'FAR']*100:.1f}" for m in sel_months)
            f.write(f"{name if rule == 'fixed' else ''} & {rule} & {cells} \\\\\n")
        f.write("\\midrule\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
print("t8 tex written")

from scipy.stats import ks_2samp
rows_p1 = []
for m in months:
    z = load_month(m); k = z["is_known"]; y = z["y"]
    for name, gk, mk_ in (("CNN", "global_pred", "mask_pred"), ("RF", "rf_pred", "rfmask_pred")):
        if gk not in z: continue
        g, mm_ = z[gk][k], z[mk_][k]; yy = y[k]
        rows_p1.append(dict(month=m, scorer=name, acc_global=float((g == yy).mean()), acc_mask=float((mm_ == yy).mean()),
                            correction=float(((g != yy) & (mm_ == yy)).mean()), violation_lb=float(((g == yy) & (mm_ != yy)).mean())))
P1 = pd.DataFrame(rows_p1); P1["gain"] = P1.acc_mask - P1.acc_global; P1["check"] = P1.correction - P1.violation_lb - P1.gain
P1.to_csv(os.path.join(TAB, "t11_masking_decomposition.csv"), index=False)
with open(os.path.join(TAB, "t11_masking_decomposition.tex"), "w") as f:
    f.write("\\begin{tabular}{lrrrrr}\n\\toprule\nScorer & Month & Acc$_{\\mathcal{K}}$ & Acc$_{\\mathcal{A}}$ & Correction & Violation \\\\\n\\midrule\n")
    for name in ("CNN", "RF"):
        for m in (2, 3, 7, 12):
            r = P1[(P1.scorer == name) & (P1.month == m)]
            if r.empty: continue
            r = r.iloc[0]; f.write(f"{name} & M{m} & {r.acc_global*100:.1f} & {r.acc_mask*100:.1f} & {r.correction*100:.1f} & {r.violation_lb*100:.1f} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
print(P1.round(4).to_string())

z2 = load_month(months[0]); gap = z2["mask_score"] - z2["global_score"]
print("Prop2: min gap", float(gap.min()), " mean gap by path:", [float(gap[z2["path"] == p].mean()) for p in range(4)])

rows_p3 = []
d0 = thresholds(cal["mask_score"], cal["path"], cal["is_ucal"], 0.02, True)
for m in months:
    z = load_month(m); u = z["is_utest"]; path = z["path"]; sc = z["mask_score"]
    tot = 0.0
    for p in range(4):
        cu = cal["is_ucal"] & (cal["path"] == p); tu = u & (path == p)
        if cu.sum() < 50 or tu.sum() < 50: continue
        ks = ks_2samp(cal["mask_score"][cu], sc[tu]).statistic
        pi = tu.sum() / u.sum(); far_p = float((sc[tu] < d0[p]).mean())
        rows_p3.append(dict(month=m, path=p, pi_test=float(pi), FAR_p=far_p, KS=float(ks), contrib=float(pi * (far_p - 0.02))))
P3 = pd.DataFrame(rows_p3); P3.to_csv(os.path.join(TAB, "t12_path_ks.csv"), index=False)
print(P3[P3.month.isin([2, 3, 7, 12])].round(3).to_string())

z = load_month(months[0]); u = z["is_utest"]; raw = z["raw_y"]; k = z["is_known"]; path = z["path"]; sc = z["mask_score"]
d0 = thresholds(cal["mask_score"], cal["path"], cal["is_ucal"], 0.02, True); acc = sc < d0[path]
rows_p4 = []
for cls in np.unique(raw[u]):
    mu = u & (raw == cls)
    if mu.sum() < 200: continue
    pv = prov.get(int(cls), "?"); pdom = int(np.bincount(path[mu]).argmax())
    mt = k & (path == pdom) & np.isin(raw, [a for a, p_ in prov.items() if p_ == pv])
    if mt.sum() < 200:
        mt = k & (path == pdom)
    lo, hi = min(sc[mu].min(), sc[mt].min()), max(sc[mu].max(), sc[mt].max()); bins = np.linspace(lo, hi, 200)
    hu, _ = np.histogram(sc[mu], bins); ht, _ = np.histogram(sc[mt], bins)
    tv = 0.5 * np.abs(hu / hu.sum() - ht / ht.sum()).sum()
    tpr_t = float(((sc < d0[path]) & (z["mask_pred"] == z["y"]))[mt].mean())
    rows_p4.append(dict(service=names.get(int(cls), str(cls)), provider=pv, same_provider=bool(z["is_same"][mu][0]), n=int(mu.sum()), dominant_path=pdom,
                        FAR=float(acc[mu].mean()), TV=float(tv), bound=float(acc[mu].mean() + tv), TPR_targets_on_path=tpr_t, n_targets=int(mt.sum())))
P4 = pd.DataFrame(rows_p4).sort_values("FAR", ascending=False); P4.to_csv(os.path.join(TAB, "t7b_unknown_tv.csv"), index=False)
print(P4.head(8).round(3).to_string())
PATHN = {0: "single IP", 1: "multi IP", 2: "new IP, known ASN", 3: "new IP, unknown ASN"}
with open(os.path.join(TAB, "t7b_unknown_tv.tex"), "w") as f:
    f.write("\\begin{tabular}{llrrrrr}\n\\toprule\nService & Provider & Flows & Path & Accepted & TV & TPR bound / observed \\\\\n\\midrule\n")
    for _, r in P4.head(6).iterrows():
        f.write(f"{r.service} & {r.provider} & {r.n:,} & {PATHN[r.dominant_path]} & {r.FAR*100:.1f}\\% & {r.TV:.2f} & {min(r.bound,1)*100:.0f}\\% / {r.TPR_targets_on_path*100:.1f}\\% \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

rows_p5 = []
a0 = acc_fracs(cal["mask_score"], cal["path"], thresholds(cal["mask_score"], cal["path"], cal["is_ucal"], 0.02, True))
for m in months:
    z = load_month(m); days = np.unique(z["day"]); u = z["is_utest"]; k = z["is_known"]; path = z["path"]; sc = z["mask_score"]
    lab = k | u
    for p in range(4):
        num = den = 0.0; n_acc_u = n_acc = n_u = n_p = 0
        for w in range(0, len(days), 7):
            sel = np.isin(z["day"], days[w:w + 7]) & (path == p)
            if sel.sum() < 50 or np.isnan(a0[p]): continue
            dq = np.quantile(sc[sel], a0[p]); acc_w = sel & (sc < dq)
            n_acc_u += int((acc_w & u).sum()); n_acc += int((acc_w & lab).sum()); n_u += int((sel & u).sum()); n_p += int((sel & lab).sum())
        if n_p == 0 or n_u == 0: continue
        rows_p5.append(dict(month=m, path=p, a_p=float(a0[p]), unknown_share_path=n_u / n_p, unknown_share_accepted=(n_acc_u / n_acc if n_acc else np.nan),
                            FAR_p=n_acc_u / n_u, identity_rhs=(a0[p] * (n_acc_u / n_acc) / (n_u / n_p) if n_acc else np.nan)))
P5 = pd.DataFrame(rows_p5); P5.to_csv(os.path.join(TAB, "t13_qm_identity.csv"), index=False)
print(P5[P5.month.isin([2, 3, 7, 12])].round(3).to_string())
print("properties written")
