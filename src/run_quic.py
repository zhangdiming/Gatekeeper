import sys, os, json, time, argparse, copy, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from data import *
from common import *
from sklearn.ensemble import RandomForestClassifier

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--budget", type=float, default=0.02)
ap.add_argument("--months", default="1-3")
ap.add_argument("--strategies", default="noupdate,L0")
ap.add_argument("--epochs", type=int, default=6)
ap.add_argument("--tau", type=float, default=0.95)
ap.add_argument("--ppi_len", type=int, default=30)
ap.add_argument("--tag", default="main")
ap.add_argument("--skip_trees", action="store_true")
args = ap.parse_args()

NAME = "CESNET-QUIC22"
RES = os.path.join(ROOT, "results", "quic", args.tag)
os.makedirs(RES, exist_ok=True)

enum_q = app_enum(NAME); sm_q = servicemap(NAME); days_q = dates(NAME)
train_days_q, cal_days_q = days_q[:5], days_q[5:7]
_D = load_days(NAME, train_days_q)
_cnt = pd.Series(_D["y"]).value_counts()
bg = {i for i, n in enum_q.items() if n.endswith("-background")}
elig = sorted(int(c) for c in _cnt.index if _cnt[c] >= 200 and int(c) not in bg)
_r = np.random.RandomState(args.seed); _r.shuffle(elig)
prov_q = {a: (sm_q.loc[enum_q[a], "Service Provider"] if enum_q[a] in sm_q.index and isinstance(sm_q.loc[enum_q[a], "Service Provider"], str) else enum_q[a]) for a in elig}
ucal_l, utest_l, known_l = elig[:10], elig[10:20], elig[20:]
known_prov = {prov_q[a] for a in known_l}
utest_bg = sorted(bg)
split = {"known": known_l, "ucal": ucal_l, "utest": utest_l + utest_bg,
         "utest_same_provider": [a for a in utest_l if prov_q[a] in known_prov] + [a for a in utest_bg if enum_q[a] in ("google-background", "facebook-background")],
         "train_days": train_days_q, "cal_days": cal_days_q, "names": {int(a): enum_q[a] for a in elig + utest_bg}}
os.makedirs(RES, exist_ok=True); dump(split, os.path.join(RES, "split.json")); del _D
known = sorted(int(a) for a in split["known"]); ucal = set(int(a) for a in split["ucal"]); utest = set(int(a) for a in split["utest"])
utest_same = set(int(a) for a in split["utest_same_provider"])
kidx = {a: i for i, a in enumerate(known)}; K = len(known)
days = dates(NAME); rng = np.random.RandomState(args.seed)
m0, m1 = [int(x) for x in args.months.split("-")]
months = list(range(m0, m1 + 1))
strategies = args.strategies.split(",")
B = args.budget
log = open(os.path.join(RES, f"log_seed{args.seed}.txt"), "a")
def P_(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); log.write(s + "\n"); log.flush()

def prep(D):
    P, F = transform(D["X_ppi"], D["X_fs"])
    yk = np.array([kidx.get(int(c), -1) for c in D["y"]], np.int64)
    is_known = yk >= 0
    is_ucal = np.isin(D["y"], list(ucal)); is_utest = np.isin(D["y"], list(utest))
    return dict(P=P, F=F, y=yk, is_known=is_known, is_ucal=is_ucal, is_utest=is_utest,
                is_same=np.isin(D["y"], list(utest_same)), ip=D["ip"], asn=D["asn"], day=D["day"], raw_y=D["y"])

t0 = time.time()
Dtr = prep(load_days(NAME, split["train_days"], ppi_len=args.ppi_len))
Dca = prep(load_days(NAME, split["cal_days"], ppi_len=args.ppi_len))
P_(f"train flows {len(Dtr['y'])} known {Dtr['is_known'].sum()} ucal {Dtr['is_ucal'].sum()} | cal flows {len(Dca['y'])} known {Dca['is_known'].sum()} ucal {Dca['is_ucal'].sum()} ({time.time()-t0:.0f}s)")

def proba_chunks(mdl, X, bs=50000):
    out = [mdl.predict_proba(X[i:i + bs]).astype(np.float32) for i in range(0, len(X), bs)]
    return np.concatenate(out)

def cap_per_class(y, cap=20000, rng=rng):
    keep = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        keep.append(idx if len(idx) <= cap else rng.choice(idx, cap, replace=False))
    return np.sort(np.concatenate(keep))

class State:
    pass

def build_groups(ip_tab, y, asn, min_share=0.005):
    tot = len(y); groups = []
    for a, c in sorted(ip_tab.asn.items(), key=lambda kv: -sum(kv[1].values())):
        if sum(c.values()) / tot >= min_share:
            groups.append(int(a))
    return groups

def fit_state(D, seed, epochs, prev=None, sel=None, lr=1e-3, tag="", train_groups=True):
    st = State()
    m = D["is_known"] if sel is None else (D["is_known"] & sel)
    idx = np.where(m)[0]
    st.ip_tab = IPTable(tau=args.tau, min_count=20).fit(D["ip"][idx], D["asn"][idx], D["y"][idx])

    cidx = idx[np.isin(np.arange(len(idx)), cap_per_class(D["y"][idx]))]
    P_(f"  [{tag}] global model on {len(cidx)} flows")
    st.model = train_net(D["P"][cidx], D["F"][cidx], D["y"][cidx], K, epochs=epochs, seed=seed, lr=lr,
                         model=copy.deepcopy(prev.model) if prev is not None else None, verbose=False)
    st.groups = build_groups(st.ip_tab, D["y"][idx], D["asn"][idx]) if prev is None else prev.groups
    st.gmodels = {} if train_groups else (dict(prev.gmodels) if prev is not None else {})
    for a in (st.groups if train_groups else []):
        gi = cidx[D["asn"][cidx] == a]
        if len(gi) < 500:
            continue
        cls = np.unique(D["y"][gi])
        if len(cls) < 2:
            st.gmodels[a] = ("const", int(cls[0]) if len(cls) else -1); continue
        gm = train_net(D["P"][gi], D["F"][gi], D["y"][gi], K, epochs=epochs, seed=seed, lr=lr,
                       model=copy.deepcopy(prev.gmodels[a][1]) if (prev is not None and a in prev.gmodels and prev.gmodels[a][0] == "net") else None, verbose=False)
        st.gmodels[a] = ("net", gm)
    P_(f"  [{tag}] groups {len(st.groups)} trained {sum(1 for v in st.gmodels.values() if v[0]=='net')} (global params {n_params(st.model)})")
    return st

PATHS = ["single", "multi", "newip_asn", "newip_noasn"]
_GATE_CACHE = {}
def gate_lookup(st, D):
    key = (id(D["ip"]), id(st.ip_tab), st.ip_tab.version)
    if key in _GATE_CACHE:
        return _GATE_CACHE[key]
    tab = st.ip_tab
    uniq, inv = np.unique(D["ip"], return_inverse=True)
    u_path = np.full(len(uniq), 3, np.int8); u_lab = np.full(len(uniq), -1, np.int64); U = np.ones((len(uniq), K), bool)
    for j, ip in enumerate(uniq):
        s, c = tab.lookup(ip)
        if s == "new": continue
        u_path[j] = 0 if s == "single" else 1; u_lab[j] = c
        U[j] = False; U[j, list(tab.ip[ip].keys())] = True
    path = u_path[inv]; iplab = u_lab[inv]; mask = U[inv]
    ua, ainv = np.unique(D["asn"], return_inverse=True)
    A = np.ones((len(ua), K), bool); a_known = np.zeros(len(ua), bool)
    for j, a in enumerate(ua):
        if a in tab.asn:
            a_known[j] = True; A[j] = False; A[j, list(tab.asn_classes(a))] = True
    newip = path == 3
    sel = newip & a_known[ainv]
    path[sel] = 2; mask[sel] = A[ainv[sel]]
    if len(_GATE_CACHE) > 3: _GATE_CACHE.clear()
    _GATE_CACHE[key] = (path, iplab, mask)
    return path, iplab, mask

_LOGIT_CACHE = {}
def model_logits(st, D):
    key = (id(st.model), tuple(sorted((a, id(v[1])) for a, v in st.gmodels.items())), id(D["P"]))
    if key in _LOGIT_CACHE:
        return _LOGIT_CACHE[key]
    glog = logits_of(st.model, D["P"], D["F"]); L = glog.copy(); asn = D["asn"]
    for a, (kind, gm) in st.gmodels.items():
        if kind != "net": continue
        gi = np.where(asn == a)[0]
        if len(gi): L[gi] = logits_of(gm, D["P"][gi], D["F"][gi])
    if len(_LOGIT_CACHE) > 2: _LOGIT_CACHE.clear()
    _LOGIT_CACHE[key] = (glog, L)
    return glog, L

def decode_all(st, D):
    glog, L = model_logits(st, D); n = len(D["y"])
    path, iplab, mask = gate_lookup(st, D)
    res = {"global": (glog.argmax(1), energy(glog), np.full(n, 3, np.int8)), "glog": glog}
    Lm = masked(glog, mask); p = Lm.argmax(1); p[path == 0] = iplab[path == 0]; res["mask"] = (p, energy(Lm), path)
    ph = L.argmax(1); ph[path == 0] = iplab[path == 0]; res["hier"] = (ph, energy(L), path)
    Lg = masked(L, mask); pg = Lg.argmax(1); pg[path == 0] = iplab[path == 0]; res["group"] = (pg, energy(Lg), path)
    return res

def decode(st, D, variant="group"):
    return decode_all(st, D)[variant]

def calibrate_paths(score, path, is_ucal, budget):
    pooled = calibrate(score[is_ucal], budget); delta = np.full(4, pooled)
    for p in range(4):
        m = is_ucal & (path == p)
        if m.sum() >= 50:
            delta[p] = calibrate(score[m], budget)
    return delta

def evaluate(pred, score, delta_vec, path, D):
    delta = delta_vec[path] if np.ndim(delta_vec) else np.full(len(path), delta_vec)
    m = D["is_known"] | D["is_utest"]
    r = osr_metrics(pred[m], score[m], delta[m], D["y"][m], D["is_known"][m], K)
    for nm, sel in (("same", D["is_same"]), ("cross", D["is_utest"] & ~D["is_same"])):
        mm = sel
        r[f"FAR_{nm}"] = float((score[mm] < delta[mm]).mean()) if mm.any() else np.nan
        r[f"n_{nm}"] = int(mm.sum())
    for p in range(4):
        mp = m & (path == p)
        r[f"share_{PATHS[p]}"] = float(mp.sum() / m.sum())
        if (mp & D["is_known"]).any():
            r[f"TPR_{PATHS[p]}"] = float(((score < delta) & (pred == D["y"]))[mp & D["is_known"]].mean())
    return r

def ip_baseline(st, D, prefix=24):
    pref = defaultdict(Counter)
    for ip, c in st.ip_tab.ip.items():
        pref[ip.rsplit(".", 1)[0] if "." in ip else ip.rsplit(":", 4)[0]][c.most_common(1)[0][0]] += sum(c.values())
    pred = np.full(len(D["y"]), -1, np.int64)
    for i, ip in enumerate(D["ip"]):
        c = st.ip_tab.ip.get(ip)
        if c: pred[i] = c.most_common(1)[0][0]; continue
        pc = pref.get(ip.rsplit(".", 1)[0] if "." in ip else ip.rsplit(":", 4)[0])
        if pc: pred[i] = pc.most_common(1)[0][0]
    m = D["is_known"] | D["is_utest"]; k = D["is_known"][m]; u = ~k; pm = pred[m]; y = D["y"][m]
    return dict(FAR=float((pm[u] >= 0).mean()), TPR=float((pm[k] == y[k]).mean()), acc_all=float((pm[k] == y[k]).mean()),
                reject_rate_known=float((pm[k] < 0).mean()), acc_accepted=float((pm[k] == y[k])[pm[k] >= 0].mean()),
                FAR_same=float((pred[D["is_same"]] >= 0).mean()), FAR_cross=float((pred[D["is_utest"] & ~D["is_same"]] >= 0).mean()),
                mF1_all=float(f1_score(y[k], pm[k], average="macro")))

def drift_signals(st, D, ref=None, sel=None):
    res = decode_all(st, D); pred, score, path = res["mask"]; glog = res["glog"]
    if sel is None: sel = np.ones(len(path), bool)
    out = {"r_new": float(np.mean(path[sel] >= 2))}
    s = sel & (path == 0)
    if s.any():
        _, iplab, _ = gate_lookup(st, D)
        out["a_ip"] = float((glog[s].argmax(1) == iplab[s]).mean())
    else:
        out["a_ip"] = np.nan
    out["E_shift"] = w1(score[sel], ref) if ref is not None else 0.0
    out["mean_E"] = float(score[sel].mean())
    return out, score[sel]

st0 = fit_state(Dtr, args.seed, args.epochs, tag="M1")
val = {}
res0 = decode_all(st0, Dca)
for variant in ("global", "mask", "group", "hier"):
    pred, score, path = res0[variant]
    val[variant] = dict(pred=pred, score=score, path=path)
    mk = Dca["is_known"]
    P_(f"  val closed-set {variant}: {closed_metrics(pred[mk], Dca['y'][mk])}")

deltas = {}
deltas["Global-Energy"] = calibrate(val["global"]["score"][Dca["is_ucal"]], B)
gl = res0["glog"]
msp_cal = msp(gl); deltas["Global-MSP"] = calibrate(msp_cal[Dca["is_ucal"]], B)
om = OpenMax().fit(logits_of(st0.model, Dtr["P"][Dtr["is_known"]], Dtr["F"][Dtr["is_known"]]), Dtr["y"][Dtr["is_known"]], K)
om_s, om_p = om.score(gl); deltas["OpenMax"] = calibrate(om_s[Dca["is_ucal"]], B)
deltas["IPCOD-mask"] = calibrate_paths(val["mask"]["score"], val["mask"]["path"], Dca["is_ucal"], B)
deltas["IPCOD-group"] = calibrate_paths(val["group"]["score"], val["group"]["path"], Dca["is_ucal"], B)
deltas["IPCOD-group-pooled"] = calibrate(val["group"]["score"][Dca["is_ucal"]], B)

acc_frac = {}
for key, variant in (("IPCOD-mask", "mask"), ("Global-Energy", "global")):
    sc, pa = val[variant]["score"], val[variant]["path"]; dl = deltas[key][pa] if np.ndim(deltas[key]) else np.full(len(pa), deltas[key])
    acc_frac[key] = np.array([float((sc < dl)[pa == p].mean()) if (pa == p).any() else np.nan for p in range(4)])
P_("calibration-time acceptance fractions per path:", {k: v.round(3).tolist() for k, v in acc_frac.items()})
np.savez_compressed(os.path.join(RES, f"flows_seed{args.seed}_cal.npz"), y=Dca["y"], raw_y=Dca["raw_y"], is_known=Dca["is_known"], is_ucal=Dca["is_ucal"], is_utest=Dca["is_utest"], is_same=Dca["is_same"],
                    **{f"{v}_pred": res0[v][0] for v in ("global", "mask", "group", "hier")}, **{f"{v}_score": res0[v][1].astype(np.float32) for v in ("global", "mask", "group", "hier")},
                    path=res0["group"][2], msp_score=msp_cal.astype(np.float32), openmax_score=om_s.astype(np.float32))
P_("deltas:", {k: (v.tolist() if np.ndim(v) else v) for k, v in deltas.items()})

trees = {}
if not args.skip_trees:
    cidx = np.where(Dtr["is_known"])[0]; cidx = cidx[np.isin(np.arange(len(cidx)), cap_per_class(Dtr["y"][cidx]))]
    Xtr = flat(Dtr["P"][cidx], Dtr["F"][cidx]); t1 = time.time()
    rf = RandomForestClassifier(n_estimators=100, max_depth=24, min_samples_leaf=3, n_jobs=-1, random_state=args.seed).fit(Xtr, Dtr["y"][cidx])
    P_(f"  RF trained ({time.time()-t1:.0f}s)"); t1 = time.time()
    Xca = flat(Dca["P"], Dca["F"])
    rf_s = -proba_chunks(rf, Xca).max(1); deltas["RF-MaxProb"] = calibrate(rf_s[Dca["is_ucal"]], B)

    pr_ca = proba_chunks(rf, Xca); path_ca, iplab_ca, mask_ca = gate_lookup(st0, Dca)
    rf_m = np.where(mask_ca, pr_ca, -1.0); rf_ms = -rf_m.max(1)
    deltas["IPCOD-RF"] = calibrate_paths(rf_ms, path_ca, Dca["is_ucal"], B)
    trees = dict(rf=rf)
    np.savez_compressed(os.path.join(RES, f"flows_rf_seed{args.seed}_cal.npz"), rf_pred=pr_ca.argmax(1), rf_score=rf_s.astype(np.float32), rfmask_pred=rf_m.argmax(1), rfmask_score=rf_ms.astype(np.float32))
    size_rf = sum(t.tree_.node_count for t in rf.estimators_) * 5 * 8 / 1e6
    sizes_rf_nodes = int(sum(t.tree_.node_count for t in rf.estimators_))
else:
    size_rf = np.nan
sizes = dict(cnn_params=n_params(st0.model), cnn_MB=n_params(st0.model) * 4 / 1e6,
             ipcod_group_MB=(n_params(st0.model) + sum(n_params(g) for k, g in st0.gmodels.values() if k == "net")) * 4 / 1e6,
             n_group_models=sum(1 for k, g in st0.gmodels.values() if k == "net"), RF_MB=size_rf,
             ip_table_entries=len(st0.ip_tab.ip))
dump(sizes, os.path.join(RES, f"sizes_seed{args.seed}.json")); P_("sizes:", sizes)

ref_sig, ref_score = drift_signals(st0, Dca)
P_("reference signals:", ref_sig)

with torch.no_grad():
    st0.model.eval(); P1, F1 = torch.from_numpy(Dca["P"][:1]), torch.from_numpy(Dca["F"][:1])
    for _ in range(50): st0.model(P1, F1)
    t1 = time.time()
    for _ in range(500): st0.model(P1, F1)
    lat_ms = (time.time() - t1) / 500 * 1e3
P_(f"single-flow latency (batch=1, CPU): {lat_ms:.3f} ms"); sizes["latency_ms"] = lat_ms; dump(sizes, os.path.join(RES, f"sizes_seed{args.seed}.json"))

def eval_all_methods(st, D, deltas, om, trees, month, strategy, rows):
    res = decode_all(st, D); gl = res["glog"]; mk = D["is_known"]
    for variant, name in (("global", "Global-Energy"), ("mask", "IPCOD-mask"), ("group", "IPCOD-group"), ("hier", "Hier-closed")):
        pred, score, path = res[variant]
        if name == "Hier-closed":
            r = closed_metrics(pred[mk], D["y"][mk]); r = dict(acc_all=r["acc"], mF1_all=r["mF1"], FAR=1.0, TPR=r["acc"])
        else:
            r = evaluate(pred, score, deltas[name], path, D)
        rows.append(dict(month=month, strategy=strategy, method=name, **r))
    r = closed_metrics(gl.argmax(1)[mk], D["y"][mk]); rows.append(dict(month=month, strategy=strategy, method="Global-closed", acc_all=r["acc"], mF1_all=r["mF1"], FAR=1.0))
    rows.append(dict(month=month, strategy=strategy, method="Global-MSP", **evaluate(gl.argmax(1), msp(gl), deltas["Global-MSP"], res["global"][2], D)))
    if om is not None:
        s_, p_ = om.score(gl); rows.append(dict(month=month, strategy=strategy, method="OpenMax", **evaluate(p_, s_, deltas["OpenMax"], res["global"][2], D)))
    pred, score, path = res["group"]
    rows.append(dict(month=month, strategy=strategy, method="IPCOD-group-pooled", **evaluate(pred, score, deltas["IPCOD-group-pooled"], path, D)))
    rows.append(dict(month=month, strategy=strategy, method="IP-baseline", **ip_baseline(st, D)))
    mkm = closed_metrics(res["mask"][0][mk], D["y"][mk]); rows.append(dict(month=month, strategy=strategy, method="Mask-closed", acc_all=mkm["acc"], mF1_all=mkm["mF1"], FAR=1.0))
    if strategy == "noupdate":
        s_om, p_om = om.score(gl) if om is not None else (np.zeros(len(gl), np.float32), gl.argmax(1))
        np.savez_compressed(os.path.join(RES, f"flows_seed{args.seed}_M{month}.npz"), y=D["y"], raw_y=D["raw_y"], is_known=D["is_known"], is_utest=D["is_utest"],
                            is_ucal=D["is_ucal"], is_same=D["is_same"], day=D["day"].astype("U8"),
                            **{f"{v}_pred": res[v][0] for v in ("global", "mask", "group", "hier")}, **{f"{v}_score": res[v][1].astype(np.float32) for v in ("global", "mask", "group", "hier")},
                            path=res["group"][2], msp_score=msp(gl).astype(np.float32), openmax_score=s_om.astype(np.float32), openmax_pred=p_om)
    if trees:
        X = flat(D["P"], D["F"]); pr = proba_chunks(trees["rf"], X); pred = pr.argmax(1); sc = -pr.max(1)
        r = closed_metrics(pred[mk], D["y"][mk]); rows.append(dict(month=month, strategy=strategy, method="RF-closed", acc_all=r["acc"], mF1_all=r["mF1"], FAR=1.0))
        rows.append(dict(month=month, strategy=strategy, method="RF-MaxProb", **evaluate(pred, sc, deltas["RF-MaxProb"], np.zeros(len(pred), np.int8), D)))
        path, iplab, mask = gate_lookup(st, D); prm = np.where(mask, pr, -1.0); predm = prm.argmax(1); predm[path == 0] = iplab[path == 0]; scm = -prm.max(1)
        rows.append(dict(month=month, strategy=strategy, method="IPCOD-RF", **evaluate(predm, scm, deltas["IPCOD-RF"], path, D)))
        r = closed_metrics(predm[mk], D["y"][mk]); rows.append(dict(month=month, strategy=strategy, method="MaskRF-closed", acc_all=r["acc"], mF1_all=r["mF1"], FAR=1.0))
        if strategy == "noupdate":
            np.savez_compressed(os.path.join(RES, f"flows_rf_seed{args.seed}_M{month}.npz"), rf_pred=pred, rf_score=sc.astype(np.float32), rfmask_pred=predm, rfmask_score=scm.astype(np.float32))

def self_label_ip_table(st, D, deltas, min_n=50, purity=0.9, pconf=0.9):
    res = decode_all(st, D); pred, score, path = res["mask"]
    pmax = torch.softmax(torch.from_numpy(res["glog"]), 1).amax(1).numpy()
    delta = deltas["IPCOD-mask"][path]
    ok = (path >= 2) & (score < delta) & (pmax >= pconf)
    cnt = defaultdict(Counter)
    for ip, c in zip(D["ip"][ok], pred[ok]):
        cnt[ip][int(c)] += 1
    added = 0
    for ip, c in cnt.items():
        tot = sum(c.values()); lab, n = c.most_common(1)[0]
        if tot >= min_n and n / tot >= purity:
            st.ip_tab.add(ip, lab, n); added += 1
    return added

def triggered(sig, ref):
    return (sig["r_new"] >= 1.5 * ref["r_new"]) or (sig["a_ip"] <= ref["a_ip"] - 0.03) or (sig["E_shift"] >= 0.5)

rows, sig_rows, cost_rows = [], [], []
def clone_state(st):
    c = State(); c.model = st.model; c.gmodels = st.gmodels; c.groups = st.groups; c.ip_tab = copy.deepcopy(st.ip_tab)
    return c
states = {s: (clone_state(st0) if s != "noupdate" else st0) for s in strategies}
sdeltas = {s: copy.deepcopy(deltas) for s in strategies}
for month in months:
    mdays = days[7 * month:7 * month + 7]
    if not mdays: continue
    Dm = prep(load_days(NAME, mdays, ppi_len=args.ppi_len))
    P_(f"== test week {month}: {len(Dm['y'])} flows known {Dm['is_known'].sum()} utest {Dm['is_utest'].sum()} ucal {Dm['is_ucal'].sum()}")

    tw = time.time()
    for s in strategies:
        pred, score, path = decode(states[s], Dm, "mask"); dl = sdeltas[s]["IPCOD-mask"][path]
        P_(f"  [{s}] decode done ({time.time()-tw:.0f}s)")
        for w in range(0, len(mdays), 7):
            sel = np.isin(Dm["day"], mdays[w:w + 7])
            sig, _ = drift_signals(states[s], Dm, ref_score, sel)
            mk = sel & Dm["is_known"]
            sig["acc_all_week"] = float(((score < dl) & (pred == Dm["y"]))[mk].mean())
            sig_rows.append(dict(month=month, week_start=mdays[w], strategy=s, **sig))
        P_(f"  [{s}] weekly signals done ({time.time()-tw:.0f}s)")
    for s in strategies:
        t1 = time.time()
        base, mode = (s.split("+") + [""])[:2]
        if mode == "R":
            wk = np.isin(Dm["day"], mdays[:7])
            for key, variant in (("IPCOD-mask", "mask"), ("Global-Energy", "global"), ("IPCOD-group", "group")):
                pred, score, path = decode(states[s], Dm, variant)
                sdeltas[s][key] = calibrate_paths(score, path, wk & Dm["is_ucal"], B) if key != "Global-Energy" else calibrate(score[wk & Dm["is_ucal"]], B)
            if trees and base == "noupdate":
                Xw = flat(Dm["P"][wk], Dm["F"][wk]); prw = proba_chunks(trees["rf"], Xw); pathw, _, maskw = gate_lookup(states[s], {k: v[wk] for k, v in Dm.items()})
                sdeltas[s]["RF-MaxProb"] = calibrate(-prw.max(1)[Dm["is_ucal"][wk]], B)
                sdeltas[s]["IPCOD-RF"] = calibrate_paths(-np.where(maskw, prw, -1.0).max(1), pathw, Dm["is_ucal"][wk], B)
        elif mode == "QM":
            for key, variant in (("IPCOD-mask", "mask"), ("Global-Energy", "global")):
                pred, score, path = decode(states[s], Dm, variant)
                if key == "Global-Energy":
                    sdeltas[s][key] = float(np.quantile(score, acc_frac[key][3]))
                else:
                    d = np.array(sdeltas[s][key], dtype=float)
                    for p in range(4):
                        if (path == p).sum() >= 50 and not np.isnan(acc_frac[key][p]): d[p] = float(np.quantile(score[path == p], acc_frac[key][p]))
                    sdeltas[s][key] = d
        eval_all_methods(states[s], Dm, sdeltas[s], om if base == "noupdate" else None, trees if (base == "noupdate") else {}, month, s, rows)
        last = [r for r in rows if r["month"] == month and r["strategy"] == s and r["method"] == "IPCOD-mask"][-1]
        P_(f"  [{s}] IPCOD-mask acc_all {last['acc_all']:.4f} FAR {last['FAR']:.4f} TPR {last['TPR']:.4f} | eval {time.time()-t1:.0f}s")

        cost = dict(month=month, strategy=s, labels=0, train_s=0.0, ip_added=0, trig=False)
        if base == "noupdate":
            pass
        elif base.startswith("L0"):
            cost["ip_added"] = self_label_ip_table(states[s], Dm, sdeltas[s])
            if base.startswith("L0L1"):
                beta = float(base.split("_")[1]) / 100
                msig = [r for r in sig_rows if r["month"] == month and r["strategy"] == s]
                trig = any(triggered(r, ref_sig) for r in msig); cost["trig"] = trig
                if trig:
                    t2 = time.time()
                    sel = rng.rand(len(Dm["y"])) < beta
                    cost["labels"] = int((sel & Dm["is_known"]).sum())
                    old_tab = states[s].ip_tab
                    states[s] = fit_state(Dm, args.seed, epochs=3, prev=states[s], sel=sel, lr=3e-4, tag=f"{s}-M{month}", train_groups=False)

                    for ip, c in states[s].ip_tab.ip.items():
                        for cls, n in c.items(): old_tab.ip[ip][cls] += n
                    for a, c in states[s].ip_tab.asn.items():
                        for cls, n in c.items(): old_tab.asn[a][cls] += n
                    old_tab.version += 1; states[s].ip_tab = old_tab
                    pred, score, path = decode(states[s], Dm, "mask")
                    sdeltas[s]["IPCOD-mask"] = calibrate_paths(score, path, Dm["is_ucal"], B)
                    cost["train_s"] = time.time() - t2
        elif base == "L2":
            t2 = time.time()
            cost["labels"] = int(Dm["is_known"].sum())
            states[s] = fit_state(Dm, args.seed, epochs=args.epochs, tag=f"L2-M{month}", train_groups=False)
            pred, score, path = decode(states[s], Dm, "global"); sdeltas[s]["Global-Energy"] = calibrate(score[Dm["is_ucal"]], B)
            pred, score, path = decode(states[s], Dm, "mask"); sdeltas[s]["IPCOD-mask"] = calibrate_paths(score, path, Dm["is_ucal"], B)
            sdeltas[s]["IPCOD-group"] = sdeltas[s]["IPCOD-mask"]; sdeltas[s]["IPCOD-group-pooled"] = calibrate(score[Dm["is_ucal"]], B)
            cost["train_s"] = time.time() - t2
        cost_rows.append(cost)
    pd.DataFrame(rows).to_csv(os.path.join(RES, f"monthly_seed{args.seed}.csv"), index=False)
    pd.DataFrame(sig_rows).to_csv(os.path.join(RES, f"signals_seed{args.seed}.csv"), index=False)
    pd.DataFrame(cost_rows).to_csv(os.path.join(RES, f"costs_seed{args.seed}.csv"), index=False)
P_("done", f"{time.time()-t0:.0f}s")
