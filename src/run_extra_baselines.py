import sys, os, json, time, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as Fn
sys.path.insert(0, os.path.dirname(__file__))
from data import *
from common import IPTable, train_net, logits_of, n_params
from numpy.lib.recfunctions import structured_to_unstructured
from sklearn.neighbors import KNeighborsClassifier

NAME = "CESNET-TLS-Year22"; RES = os.path.join(ROOT, "results", "tls", "main"); SEED = 0
split = json.load(open(os.path.join(ROOT, "results", "tls", "split.json")))
known = sorted(int(a) for a in split["known"]); kidx = {a: i for i, a in enumerate(known)}; K = len(known)
days = dates(NAME); rng = np.random.RandomState(SEED)
PH = ["PHIST_SRC_SIZES", "PHIST_DST_SIZES", "PHIST_SRC_IPT", "PHIST_DST_IPT"]

def load(dl):
    D = load_days(NAME, dl, extra_fields=tuple(PH))
    P, F = transform(D["X_ppi"], D["X_fs"])
    H = np.concatenate([D[k].astype(np.float32) for k in PH], axis=1)
    H = np.log1p(H)
    y = np.array([kidx.get(int(c), -1) for c in D["y"]], np.int64)
    return dict(P=P, F=np.concatenate([F, H], axis=1), F10=F, y=y, ip=D["ip"], asn=D["asn"])

def gate_mask(tab, D):
    uniq, inv = np.unique(D["ip"], return_inverse=True); U = np.ones((len(uniq), K), bool); known_ip = np.zeros(len(uniq), bool)
    for j, ip in enumerate(uniq):
        if ip in tab.ip: known_ip[j] = True; U[j] = False; U[j, list(tab.ip[ip].keys())] = True
    mask = U[inv]; new = ~known_ip[inv]
    ua, ainv = np.unique(D["asn"], return_inverse=True); A = np.ones((len(ua), K), bool); ak = np.zeros(len(ua), bool)
    for j, a in enumerate(ua):
        if a in tab.asn: ak[j] = True; A[j] = False; A[j, list(tab.asn_classes(a))] = True
    sel = new & ak[ainv]; mask[sel] = A[ainv[sel]]
    return mask

t0 = time.time()
Dtr = load(split["train_days"]); mk = Dtr["y"] >= 0
tab = IPTable(tau=0.95, min_count=20).fit(Dtr["ip"][mk], Dtr["asn"][mk], Dtr["y"][mk])

idx = np.where(mk)[0]; keep = []
for c in np.unique(Dtr["y"][idx]):
    ii = idx[Dtr["y"][idx] == c]; keep.append(ii if len(ii) <= 20000 else rng.choice(ii, 20000, replace=False))
cidx = np.sort(np.concatenate(keep)); print("train flows", len(cidx), f"({time.time()-t0:.0f}s)", flush=True)

class BigNet(nn.Module):
    def __init__(self, n_ppi_ch, n_fs, n_classes, width=128, emb=256):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(n_ppi_ch, width, 3, padding=1), nn.BatchNorm1d(width), nn.ReLU(),
                                  nn.Conv1d(width, width, 3, padding=1), nn.BatchNorm1d(width), nn.ReLU(),
                                  nn.Conv1d(width, width, 3, padding=1), nn.BatchNorm1d(width), nn.ReLU(),
                                  nn.Conv1d(width, width, 3, padding=1), nn.BatchNorm1d(width), nn.ReLU())
        self.fs = nn.Sequential(nn.Linear(n_fs, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(2 * width + 128, emb), nn.ReLU(), nn.Dropout(0.1)); self.cls = nn.Linear(emb, n_classes)
    def forward(self, P, F):
        h = self.conv(P); h = torch.cat([h.mean(2), h.amax(2), self.fs(F)], 1); return self.cls(self.head(h))
big = BigNet(4, Dtr["F"].shape[1], K)
t1 = time.time(); big = train_net(Dtr["P"][cidx], Dtr["F"][cidx], Dtr["y"][cidx], K, epochs=12, seed=SEED, model=big, verbose=True)
print(f"big CNN params {n_params(big)} trained in {time.time()-t1:.0f}s", flush=True)

keep2 = []
for c in np.unique(Dtr["y"][cidx]):
    ii = cidx[Dtr["y"][cidx] == c]; keep2.append(ii if len(ii) <= 2000 else rng.choice(ii, 2000, replace=False))
kidx_ = np.sort(np.concatenate(keep2))
def knn_feats(D): return np.concatenate([D["P"].reshape(len(D["P"]), -1), D["F10"]], axis=1)
Xk = knn_feats(Dtr)[kidx_]; yk = Dtr["y"][kidx_]
knn = KNeighborsClassifier(n_neighbors=10, metric="manhattan", n_jobs=-1, algorithm="brute").fit(Xk, yk)
print("kNN train size", len(kidx_), flush=True)

rows = []
for month in range(2, 13):
    md = month_days(days, month)
    if not md: continue
    Dm = load(md); m = Dm["y"] >= 0; y = Dm["y"]; mask = gate_mask(tab, Dm)

    L = logits_of(big, Dm["P"], Dm["F"]); Lm = np.where(mask, L, -1e4)
    r = dict(month=month, bigcnn_global=float((L.argmax(1) == y)[m].mean()), bigcnn_mask=float((Lm.argmax(1) == y)[m].mean()))

    sel = rng.choice(np.where(m)[0], min(100000, int(m.sum())), replace=False)
    t2 = time.time(); Xs = knn_feats(Dm)[sel]
    dist, nbr = knn.kneighbors(Xs, n_neighbors=10)
    votes = np.zeros((len(sel), K), np.float32)
    for j in range(10): np.add.at(votes, (np.arange(len(sel)), yk[nbr[:, j]]), 1.0)
    pred = votes.argmax(1); vm = np.where(mask[sel], votes, -1.0); predm = vm.argmax(1)

    noallow = vm.max(1) <= 0
    r.update(knn_global=float((pred == y[sel]).mean()), knn_mask=float((predm == y[sel]).mean()), knn_n=int(len(sel)), knn_no_allowed_neighbor=float(noallow.mean()), knn_s=time.time() - t2)
    rows.append(r); print(r, flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(RES, "extra_baselines_seed0.csv"), index=False)
json.dump(dict(bigcnn_params=n_params(big), bigcnn_MB=n_params(big) * 4 / 1e6, knn_train=len(kidx_)), open(os.path.join(RES, "extra_baselines_meta.json"), "w"))
print("done", f"{time.time()-t0:.0f}s")
