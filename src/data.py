import os, numpy as np, pandas as pd, tables as tb
from numpy.lib.recfunctions import structured_to_unstructured

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "XS")
PPI_LEN = 30
FLOWSTATS = ["BYTES", "BYTES_REV", "PACKETS", "PACKETS_REV", "DURATION", "PPI_LEN", "PPI_ROUNDTRIPS", "PPI_DURATION"]

def h5_path(name):
    return os.path.join(DATA, f"{name}-XS.h5")

def servicemap(name):
    return pd.read_csv(os.path.join(DATA, f"{name}-XS-servicemap.csv"), index_col="Tag")

def app_enum(name):
    with tb.open_file(h5_path(name), "r") as f:
        t = next(iter(f.get_node("/flows")))
        return {v: k for k, v in dict(t.get_enum("APP")).items()}

def dates(name):
    with tb.open_file(h5_path(name), "r") as f:
        return [n._v_name[1:] for n in f.get_node("/flows")]

def load_days(name, days, ppi_len=PPI_LEN, extra_fields=()):
    ppi_list, fs_list, y_list, ip_list, asn_list, day_list, extra = [], [], [], [], [], [], {k: [] for k in extra_fields}
    with tb.open_file(h5_path(name), "r") as f:
        for d in days:
            t = f.get_node(f"/flows/D{d}")
            rows = t.read()
            ppi = rows["PPI"]
            ppi_list.append(ppi[:, :, :ppi_len].astype(np.float32))
            fs = structured_to_unstructured(rows[FLOWSTATS], dtype=np.float32)
            port = rows["DST_PORT"].astype(np.float32)[:, None]
            proto = rows["PROTOCOL"].astype(np.float32)[:, None] if "PROTOCOL" in rows.dtype.names else np.zeros_like(port)
            fs_list.append(np.concatenate([fs, port, proto], axis=1))
            y_list.append(rows["APP"].astype(np.int32))
            ip_list.append(rows["DST_IP"].astype(str))
            asn_list.append(rows["DST_ASN"].astype(np.int64))
            day_list.append(np.full(len(rows), d))
            for k in extra_fields:
                extra[k].append(rows[k])
    out = dict(X_ppi=np.concatenate(ppi_list), X_fs=np.concatenate(fs_list), y=np.concatenate(y_list),
               ip=np.concatenate(ip_list), asn=np.concatenate(asn_list), day=np.concatenate(day_list))
    for k in extra_fields:
        out[k] = np.concatenate(extra[k])
    return out

def transform(X_ppi, X_fs):
    P = X_ppi.copy()
    P[:, 0] = np.log1p(np.clip(P[:, 0], 0, 65000)) / np.log1p(65000)
    P[:, 2] = np.clip(P[:, 2], 0, 1500) / 1500.0
    if P.shape[1] > 3:
        P[:, 3] = np.clip(P[:, 3], 0, 1)
    F = X_fs.copy()
    F[:, :8] = np.log1p(np.clip(F[:, :8], 0, None))
    F[:, 8] = F[:, 8] / 65535.0
    F[:, 9] = (F[:, 9] == 17).astype(np.float32)
    return P.astype(np.float32), F.astype(np.float32)

def flat(P, F):
    return np.concatenate([P.reshape(len(P), -1), F], axis=1)

def month_days(all_days, month):
    return [d for d in all_days if d[4:6] == f"{month:02d}"]

def week_days(all_days, start_idx, n=7):
    return all_days[start_idx:start_idx + n]
