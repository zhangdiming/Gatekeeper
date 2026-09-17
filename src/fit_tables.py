import os, re, glob
TAB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tables")
SINGLE = {"t11_masking_decomposition.tex", "t15_trigger_sensitivity.tex", "t8_threshold_maintenance.tex"}
DOUBLE = {"t1_main_tls.tex", "t2_stratified_tls.tex", "t7b_unknown_tv.tex", "t3_update_strategies.tex", "t14_extra_baselines.tex", "t9_quic.tex", "t10_seed_variation.tex"}
for f in glob.glob(os.path.join(TAB, "*.tex")):
    name = os.path.basename(f); width = "\\columnwidth" if name in SINGLE else "\\textwidth" if name in DOUBLE else None
    if width is None: continue
    s = open(f).read()
    s = re.sub(r"\\begin\{tabular\*?\}(\{[^}]*\})?\{(@\{\\extracolsep\{\\fill\}\})?([lrc|]+)(@\{\})?\}",
               lambda m: "\\begin{tabular*}{%s}{@{\\extracolsep{\\fill}}%s@{}}" % (width, m.group(3)), s, count=1)
    s = s.replace("\\end{tabular}", "\\end{tabular*}")
    for a, b in (("IPCOD-group-pooled", "GateKeeper-group (pooled)"), ("IPCOD-group", "GateKeeper-group"), ("IPCOD-RF", "GateKeeper-RF"), ("IPCOD-mask + threshold", "GateKeeper + threshold"), ("IPCOD-mask", "GateKeeper")):
        s = s.replace(a, b)
    open(f, "w").write(s)
    print(name, "->", width)
