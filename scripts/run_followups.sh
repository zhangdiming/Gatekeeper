#!/bin/bash
cd "$(dirname "$0")/.."
until grep -q "^done" results/main_seed0.out; do sleep 60; done
rm -rf results/quic/main
python src/run_quic.py --seed 0 --months 1-3 --epochs 6 --tag main --strategies noupdate,noupdate+QM,noupdate+R > results/quic_seed0.out 2>&1
for s in 1 2; do
  python src/run_tls.py --seed $s --months 2-12 --epochs 6 --tag main --strategies noupdate,noupdate+QM --skip_trees > results/main_seed$s.out 2>&1
done
echo ALL-DONE > results/followups.done
