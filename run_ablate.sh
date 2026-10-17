#!/bin/bash
# Ablation: 38 conditions x 90 UIEB images, scored with UIQM, UCIQE + 22 pyiqa metrics, then heatmap.
# Resumable; one process per metric. Waits for the pilot to free the GPU. Watch: tail -f ablate.log
cd "$(dirname "$0")"
PY=~/miniforge3/envs/uwiqa/bin/python
[ -d out_ablate/clean ] || $PY ablate.py data/uieb/raw --n 90 || exit 1
echo "IMAGES DONE"
while pgrep -f run_pilot.sh > /dev/null; do sleep 60; done
for m in uiqm uciqe $($PY -c "from score import METRICS; print(*METRICS)"); do
  $PY score.py --out out_ablate --metrics "$m" || echo "FAILED $m"
done
$PY ablate_report.py --out out_ablate
echo "ABLATE DONE"
