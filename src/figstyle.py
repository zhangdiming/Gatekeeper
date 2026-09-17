import matplotlib
import matplotlib.pyplot as plt

FCS = {"navy": "#004385", "orange": "#C66219", "amber": "#FFC000", "purple": "#742984", "teal": "#1A7C6B", "red": "#B7282C"}
GREY = {"dark": "#222222", "mid": "#666666", "light": "#A6A6A6", "pale": "#D9D9D9"}

COL = {
    "IPCOD-mask": FCS["navy"], "IPCOD-RF": FCS["purple"], "IPCOD-group": FCS["orange"], "IPCOD-group-pooled": FCS["amber"],
    "RF-MaxProb": FCS["red"], "Global-Energy": FCS["teal"], "Global-MSP": FCS["amber"], "OpenMax": GREY["mid"],
    "IP-baseline": GREY["light"], "Mask-closed": FCS["navy"], "MaskRF-closed": FCS["purple"], "Hier-closed": FCS["orange"],
    "RF-closed": FCS["red"], "Global-closed": FCS["teal"],

    "noupdate": GREY["mid"], "noupdate+QM": FCS["navy"], "noupdate+R": FCS["teal"], "L0": GREY["light"], "L0L1_1": FCS["amber"],
    "L0L1_5": FCS["orange"], "L0L1_5+R": FCS["red"], "L2": "#7FB3A5", "L2+R": FCS["purple"],

    "fixed": GREY["mid"], "QM-week": FCS["navy"], "R-prev": FCS["orange"], "R-conc": FCS["teal"],
}
LS = {"Mask-closed": "-", "MaskRF-closed": "-", "Hier-closed": "--", "RF-closed": "--", "Global-closed": "--", "IP-baseline": ":"}
MK = {"IPCOD-mask": "o", "IPCOD-RF": "s", "IPCOD-group": "^", "IPCOD-group-pooled": "v", "RF-MaxProb": "D", "Global-Energy": "x",
      "Global-MSP": "+", "OpenMax": "*", "IP-baseline": ".", "Mask-closed": "o", "MaskRF-closed": "s", "Hier-closed": "^",
      "RF-closed": "D", "Global-closed": "x",
      "noupdate": "o", "noupdate+QM": "s", "noupdate+R": "^", "L0L1_5": "v", "L0L1_5+R": "D", "L2": "x", "L2+R": "P",
      "fixed": "o", "QM-week": "s", "R-prev": "^", "R-conc": "D"}
LAB = {"noupdate": "No update", "noupdate+QM": "Threshold QM (label-free)", "noupdate+R": "Threshold recal. (fresh non-target)",
       "L0": "L0: IP-table self-labelling", "L0L1_1": "L0+L1 (1% labels)", "L0L1_5": "L0+L1 (5% labels)",
       "L0L1_5+R": "L0+L1 (5% labels) + recal.", "L2": "L2: full monthly retraining", "L2+R": "L2 + recal."}

def apply():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5, "legend.fontsize": 6.8,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.7, "lines.linewidth": 1.3, "lines.markersize": 3.2, "legend.frameon": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "axes.grid": True,
        "grid.color": GREY["pale"], "grid.linewidth": 0.5, "axes.axisbelow": True})

DISP = {"IPCOD-mask": "GateKeeper", "IPCOD-RF": "GateKeeper-RF", "IPCOD-group": "GateKeeper-group", "IPCOD-group-pooled": "GateKeeper-group (pooled)"}
def disp(name):
    return DISP.get(name, name)

def style(name):
    return dict(color=COL.get(name, GREY["dark"]), marker=MK.get(name, "o"), ls=LS.get(name, "-"))
