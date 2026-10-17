#!/bin/bash
# Full pilot: 90 UIEB val images x 20 steps x 3 modes, then 22 NR-IQA metrics, then plots.
# Resumable: finished metrics are skipped. One process per metric so a CUDA crash kills only that metric.
# Watch: tail -f pilot.log
cd "$(dirname "$0")"
PY=~/miniforge3/envs/uwiqa/bin/python
[ -f out/attack.csv ] || $PY attack.py data/uieb/raw --n 90 --steps 20 --workers "$(nproc)" || exit 1
echo "ATTACK DONE"
for m in $($PY -c "from score import METRICS; print(*METRICS)"); do
  $PY score.py --out out --metrics "$m" || echo "FAILED $m"
done
$PY plot.py --out out
echo "PILOT DONE"
