import time, json, os, numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn
from collections import defaultdict, Counter
from sklearn.metrics import f1_score, roc_auc_score
from scipy.stats import wasserstein_distance

torch.set_num_threads(max(1, os.cpu_count() - 2))

class PPINet(nn.Module):
    def __init__(self, n_ppi_ch, n_fs, n_classes, width=96, emb=192):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_ppi_ch, width, 3, padding=1), nn.BatchNorm1d(width), nn.ReLU(),
            nn.Conv1d(width, width, 3, padding=1), nn.BatchNorm1d(width), nn.ReLU(),
            nn.Conv1d(width, width, 3, padding=1), nn.BatchNorm1d(width), nn.ReLU())
        self.fs = nn.Sequential(nn.Linear(n_fs, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(2 * width + 64, emb), nn.ReLU(), nn.Dropout(0.1))
        self.cls = nn.Linear(emb, n_classes)
    def embed(self, P, F):
        h = self.conv(P)
        h = torch.cat([h.mean(2), h.amax(2), self.fs(F)], 1)
        return self.head(h)
    def forward(self, P, F):
        return self.cls(self.embed(P, F))

def n_params(model):
    return sum(p.numel() for p in model.parameters())

def train_net(P, F, y, n_classes, epochs=6, bs=2048, lr=1e-3, seed=0, model=None, verbose=True):
    torch.manual_seed(seed); np.random.seed(seed)
    if model is None:
        model = PPINet(P.shape[1], F.shape[1], n_classes)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    Pt, Ft, yt = torch.from_numpy(P), torch.from_numpy(F), torch.from_numpy(y.astype(np.int64))
    n = len(yt); idx = np.arange(n)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * ((n + bs - 1) // bs))
    model.train(); t0 = time.time()
    for ep in range(epochs):
        np.random.shuffle(idx); tot = 0.0
        for i in range(0, n, bs):
            b = idx[i:i + bs]
            opt.zero_grad()
            loss = Fn.cross_entropy(model(Pt[b], Ft[b]), yt[b])
            loss.backward(); opt.step(); sched.step(); tot += loss.item() * len(b)
        if verbose:
            print(f"    epoch {ep+1}/{epochs} loss {tot/n:.4f} ({time.time()-t0:.0f}s)", flush=True)
    model.eval()
    return model

@torch.no_grad()
def logits_of(model, P, F, bs=8192):
    model.eval(); out = []
    for i in range(0, len(P), bs):
        out.append(model(torch.from_numpy(P[i:i + bs]), torch.from_numpy(F[i:i + bs])).numpy())
    return np.concatenate(out) if out else np.zeros((0, model.cls.out_features), np.float32)

def energy(logits, T=1.0):
    z = torch.from_numpy(logits) / T
    return (-T * torch.logsumexp(z, 1)).numpy()

def msp(logits):
    return -torch.softmax(torch.from_numpy(logits), 1).amax(1).numpy()

def masked(logits, mask):
    out = logits.copy(); out[~mask] = -1e4
    return out

class OpenMax:
    def __init__(self, tail=20, alpha=5):
        self.tail, self.alpha = tail, alpha
    def fit(self, logits, y, n_classes):
        from scipy.stats import weibull_min
        self.mav = np.zeros((n_classes, logits.shape[1]), np.float32); self.wb = {}
        for c in range(n_classes):
            L = logits[(y == c) & (logits.argmax(1) == c)]
            if len(L) < 5:
                continue
            self.mav[c] = L.mean(0)
            d = np.sort(np.linalg.norm(L - self.mav[c], axis=1))[-self.tail:]
            if d.max() <= 0:
                continue
            self.wb[c] = weibull_min.fit(d, floc=0)
        return self
    def score(self, logits):
        from scipy.stats import weibull_min
        n, K = logits.shape
        rank = np.argsort(-logits, 1)[:, :self.alpha]
        rev = logits.copy(); unk = np.zeros(n, np.float32)
        for j in range(self.alpha):
            c = rank[:, j]
            w = np.zeros(n, np.float32)
            for cc in np.unique(c):
                if cc in self.wb:
                    m = c == cc
                    d = np.linalg.norm(logits[m] - self.mav[cc], axis=1)
                    w[m] = weibull_min.cdf(d, *self.wb[cc]) * (self.alpha - j) / self.alpha
            rev[np.arange(n), c] = logits[np.arange(n), c] * (1 - w)
            unk += logits[np.arange(n), c] * w
        full = np.concatenate([rev, unk[:, None]], 1)
        p = torch.softmax(torch.from_numpy(full), 1).numpy()
        return p[:, -1] - p[:, :-1].max(1), p[:, :-1].argmax(1)

class IPTable:
    def __init__(self, tau=0.95, min_count=20):
        self.tau, self.min_count = tau, min_count
        self.ip = defaultdict(Counter); self.asn = defaultdict(Counter); self.version = 0
    def fit(self, ips, asns, y):
        for i, a, c in zip(ips, asns, y):
            self.ip[i][c] += 1; self.asn[a][c] += 1
        self.version += 1
        return self
    def add(self, ip, c, n=1):
        self.ip[ip][c] += n; self.version += 1
    def lookup(self, ip):
        cnt = self.ip.get(ip)
        if not cnt:
            return "new", -1
        tot = sum(cnt.values()); c, m = cnt.most_common(1)[0]
        if tot >= self.min_count and m / tot >= self.tau:
            return "single", c
        return "multi", c
    def asn_classes(self, a):
        return set(self.asn[a].keys()) if a in self.asn else set()
    def ip_stats(self):
        n = len(self.ip); single = sum(1 for v in self.ip.values() if sum(v.values()) >= self.min_count and v.most_common(1)[0][1] / sum(v.values()) >= self.tau)
        return dict(n_ips=n, n_single=single)

def calibrate(scores_unknown, budget=0.02):
    if len(scores_unknown) == 0:
        return -np.inf
    return float(np.quantile(scores_unknown, budget))

def osr_metrics(pred, score, delta, y, is_known, n_classes):
    acc_mask = score < delta
    k = is_known; u = ~is_known
    out = {}
    out["n_known"], out["n_unknown"] = int(k.sum()), int(u.sum())
    out["FAR"] = float(acc_mask[u].mean()) if u.any() else np.nan
    out["TPR"] = float((acc_mask[k] & (pred[k] == y[k])).mean()) if k.any() else np.nan
    out["acc_all"] = float((acc_mask[k] & (pred[k] == y[k])).mean()) if k.any() else np.nan
    out["reject_rate_known"] = float((~acc_mask[k]).mean()) if k.any() else np.nan
    ka = k & acc_mask
    out["acc_accepted"] = float((pred[ka] == y[ka]).mean()) if ka.any() else np.nan
    out["mF1_all"] = float(f1_score(y[k], np.where(acc_mask[k], pred[k], -1), average="macro")) if k.any() else np.nan
    if k.any() and u.any():
        out["AUROC"] = float(roc_auc_score(u.astype(int), score))
    else:
        out["AUROC"] = np.nan
    return out

def closed_metrics(pred, y):
    return dict(acc=float((pred == y).mean()), mF1=float(f1_score(y, pred, average="macro")))

def w1(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.nan
    rng = np.random.RandomState(0)
    a = a if len(a) <= 50000 else rng.choice(a, 50000, replace=False)
    b = b if len(b) <= 50000 else rng.choice(b, 50000, replace=False)
    return float(wasserstein_distance(a, b))

def dump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
