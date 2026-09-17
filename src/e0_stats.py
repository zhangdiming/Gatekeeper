import sys, os, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from data import *
from common import IPTable, dump

NAME = "CESNET-TLS-Year22"
RES = os.path.join(ROOT, "results", "tls")
os.makedirs(RES, exist_ok=True)
rng = np.random.RandomState(0)

days = dates(NAME)
enum = app_enum(NAME)
sm = servicemap(NAME)
jan = month_days(days, 1)
train_days, cal_days = jan[:21], jan[21:]
print("days:", len(days), "jan:", len(jan), "train:", train_days[0], "-", train_days[-1], "cal:", cal_days[0], "-", cal_days[-1])

D = load_days(NAME, train_days, extra_fields=("TLS_SNI",))
print("train flows:", len(D["y"]))
cnt = pd.Series(D["y"]).value_counts()
names = pd.Series({i: enum[i] for i in cnt.index})
prov = names.map(lambda n: sm.loc[n, "Service Provider"] if n in sm.index and isinstance(sm.loc[n, "Service Provider"], str) else n)
df = pd.DataFrame({"app": names, "provider": prov, "n_train": cnt}).sort_values("n_train", ascending=False)
df.to_csv(os.path.join(RES, "e0_class_counts.csv"))
elig = df[df.n_train >= 200]
print("classes total:", len(df), "eligible (>=200):", len(elig), "providers:", elig.provider.nunique())

apps = list(elig.index); rng.shuffle(apps)
prov_of = elig.provider.to_dict()
known, ucal, utest = [], [], []

by_prov = {}
for a in apps:
    by_prov.setdefault(prov_of[a], []).append(a)
for p, lst in by_prov.items():
    if len(lst) >= 3:
        known.extend(lst[:-2]); ucal.append(lst[-2]); utest.append(lst[-1])
    elif len(lst) == 2:
        known.append(lst[0]); utest.append(lst[1])
    else:
        known.extend(lst)

n_k, n_c, n_t = 120, 20, 40
def move(src, dst, k):
    for _ in range(k):
        if src:
            dst.append(src.pop())

singles = [a for a in known if len(by_prov[prov_of[a]]) == 1]
while len(utest) < n_t and singles:
    a = singles.pop(); known.remove(a); utest.append(a)
while len(ucal) < n_c and singles:
    a = singles.pop(); known.remove(a); ucal.append(a)
while len(known) > n_k and singles:
    a = singles.pop(); known.remove(a); (ucal if len(ucal) < n_c else utest).append(a)
known_prov = {prov_of[a] for a in known}
split = {"known": sorted(known), "ucal": sorted(ucal), "utest": sorted(utest),
         "utest_same_provider": sorted([a for a in utest if prov_of[a] in known_prov]),
         "utest_cross_provider": sorted([a for a in utest if prov_of[a] not in known_prov]),
         "ucal_same_provider": sorted([a for a in ucal if prov_of[a] in known_prov]),
         "names": {int(a): enum[a] for a in apps}, "provider": {int(a): prov_of[a] for a in apps},
         "train_days": train_days, "cal_days": cal_days}
dump(split, os.path.join(RES, "split.json"))
print("known", len(known), "ucal", len(ucal), "utest", len(utest), "same-prov utest", len(split["utest_same_provider"]), "cross-prov utest", len(split["utest_cross_provider"]))

mk = np.isin(D["y"], known)
tab = IPTable(tau=0.95, min_count=1).fit(D["ip"][mk], D["asn"][mk], D["y"][mk])
st = tab.ip_stats()
single_ips = {ip for ip, c in tab.ip.items() if c.most_common(1)[0][1] / sum(c.values()) >= 0.95}
flows_on_single = float(np.mean([ip in single_ips for ip in D["ip"][mk]]))
tab20 = IPTable(tau=0.95, min_count=20).fit(D["ip"][mk], D["asn"][mk], D["y"][mk])
single20 = {ip for ip, c in tab20.ip.items() if sum(c.values()) >= 20 and c.most_common(1)[0][1] / sum(c.values()) >= 0.95}
flows_on_single20 = float(np.mean([ip in single20 for ip in D["ip"][mk]]))
n_apps_per_ip = pd.Series([len(c) for c in tab.ip.values()]).value_counts(normalize=True).sort_index()
stats = {"n_train_flows": int(len(D["y"])), "n_known_flows": int(mk.sum()), "n_ips_known": st["n_ips"],
         "frac_ips_single_app_tau095": float(len(single_ips) / st["n_ips"]),
         "frac_flows_on_single_app_ip_tau095": flows_on_single,
         "frac_flows_on_single_app_ip_tau095_min20": flows_on_single20,
         "ips_by_n_apps": {int(k): float(v) for k, v in n_apps_per_ip.items() if k <= 5},
         "n_asn_known": len(tab.asn),
         "top_asn_share": {int(a): float(sum(c.values()) / mk.sum()) for a, c in sorted(tab.asn.items(), key=lambda kv: -sum(kv[1].values()))[:15]}}

prov_ip = {}
for ip, c in tab.ip.items():
    ps = {}
    for cls, n in c.items():
        ps[prov_of[cls]] = ps.get(prov_of[cls], 0) + n
    prov_ip[ip] = max(ps.values()) / sum(ps.values())
stats["frac_ips_single_provider_tau095"] = float(np.mean([v >= 0.95 for v in prov_ip.values()]))
stats["frac_flows_on_single_provider_ip"] = float(np.mean([prov_ip[ip] >= 0.95 for ip in D["ip"][mk]]))
dump(stats, os.path.join(RES, "e0_ip_stats.json"))
print(json.dumps(stats, indent=1))
