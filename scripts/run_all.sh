#!/bin/bash
set -e
cd "$(dirname "$0")/.."
mkdir -p results
python src/e0_stats.py
python src/run_tls.py --seed 0 --months 2-12 --epochs 6 --tag main > results/main_seed0.out 2>&1
python src/run_quic.py --seed 0 --months 1-3 --epochs 6 --tag main --strategies noupdate,noupdate+QM,noupdate+R > results/quic_seed0.out 2>&1
for s in 1 2; do
  python src/run_tls.py --seed $s --months 2-12 --epochs 6 --tag main --strategies noupdate,noupdate+QM --skip_trees > results/main_seed$s.out 2>&1
done
python src/run_extra_baselines.py > results/extra_baselines.out 2>&1
python src/run_extra_knn.py > results/extra_knn.out 2>&1
python src/make_tables.py main
python src/analyze_flows.py main
python src/fit_tables.py
python src/draw_architecture.py
python src/fig_onecol.py
