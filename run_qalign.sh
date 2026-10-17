#!/bin/bash
# Q-Align (8-bit) for pilot + ablation, then refresh both reports. Watch: tail -f qalign.log
cd "$(dirname "$0")"
PY=~/miniforge3/envs/uwiqa/bin/python
~/miniforge3/envs/uwiqa/bin/pip install -q icecream
for o in out out_ablate; do $PY score.py --out $o --metrics qalign_8bit || echo "FAILED qalign_8bit $o"; done
$PY plot.py --out out
$PY ablate_report.py --out out_ablate
echo "QALIGN DONE"
