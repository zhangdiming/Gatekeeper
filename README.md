# GateKeeper

Server-IP priors for open-set and drift-resilient application identification under Encrypted Client Hello (ECH).

GateKeeper routes each flow through an IP-derived gate that restricts the candidate classes of a lightweight packet-sequence classifier, applies an energy-based reject rule calibrated per gate path to a false-acceptance budget, and derives label-free drift statistics from the IP-to-service mapping. This repository contains the code that produces every number, table and figure of the paper.

## Layout

```
src/data.py                 dataset loading (CESNET-TLS-Year22 / CESNET-QUIC22, ECH-observable features only)
src/common.py               model, IP table, open-set scores, calibration, metrics
src/e0_stats.py             dataset statistics and the provider-stratified known/unknown split
src/run_tls.py              main protocol on CESNET-TLS-Year22: monthly evaluation, maintenance strategies
src/run_quic.py             replication on CESNET-QUIC22 (weekly test periods)
src/run_extra_baselines.py  larger PHIST-augmented network and input-space k-NN (closed set)
src/run_extra_knn.py        k-NN rerun with the restricted-vote tie-break
src/make_tables.py          tables and figures from the result CSVs
src/analyze_flows.py        budget sweeps, threshold maintenance, Propositions 1-5 diagnostics
src/fit_tables.py           sets every table to exact column/text width
src/figstyle.py             shared palette and display names
src/draw_architecture.py    Figure 1
scripts/run_all.sh          full pipeline
```

## Environment

Python 3.11, CPU only (32 cores / 31 GB RAM were used).

```
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

## Data

Public datasets obtained through `cesnet-datazoo` into `data/XS/`:

```python
from cesnet_datazoo.datasets import CESNET_TLS_Year22, CESNET_QUIC22
CESNET_TLS_Year22("data", size="XS")
CESNET_QUIC22("data", size="XS")
```

Copy the downloaded `data/XS/servicemap.csv` of each dataset to `data/XS/CESNET-TLS-Year22-XS-servicemap.csv` and `data/XS/CESNET-QUIC22-XS-servicemap.csv` respectively (the tool writes both to the same file name). The SNI and JA3 fields are never used as features.

## Run

```
bash scripts/run_all.sh
```

Runtime on the reference machine: main TLS run 3.1 h, QUIC 0.5 h, extra seeds 0.7 h each, extra baselines 0.6 h. Outputs: `results/` (CSV, per-flow score dumps), `tables/` (CSV + LaTeX), `figures/` (PDF).

## Method names

Result files use the internal keys `IPCOD-mask`, `IPCOD-RF`, `IPCOD-group`, `IPCOD-group-pooled`; they are displayed as `GateKeeper`, `GateKeeper-RF`, `GateKeeper-group`, `GateKeeper-group (pooled)` in tables and figures (`src/figstyle.py`).
