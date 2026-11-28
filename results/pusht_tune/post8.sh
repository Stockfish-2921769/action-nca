#!/bin/sh
# 第八轮（DART，临时）：收集数据 → 从头训练（训练轮数随数据量放大）→ 彻底失败率、吸引子检验 → 500 回合复评
# → 补数据诊断（--fit_suffix _dart）→ 汇总。在服务器上自动跑，不依赖本地的 ssh 连接
cd ~/action-nca || exit 1
PY=~/venvs/action-nca/bin/python
T=runs/pusht_tune
log() { echo "$(date +%H:%M) $*" >> $T/post8.log; }

log start
$PY pusht_tune.py --stage dart --seeds 0 1 2 --jobs 3 > $T/dart.log 2>&1
log "dart data done"
$PY pusht_tune.py --stage train --variants se_global_dart dp_small_dart --seeds 0 1 2 --jobs 6 --gpus 0 1 1 0 > $T/train_r8.log 2>&1
log "training done"
$PY pusht_tune.py --stage cat --jobs 6 > $T/cat_r8.log 2>&1
$PY pusht_tune.py --stage attract --probe_models se_global_dart --seeds 0 1 2 --jobs 6 > $T/attract_r8.log 2>&1
log "cat/attract done"
$PY pusht_tune.py --stage final --probe_models se_global_dart dp_small_dart --seeds 0 1 2 --jobs 9 > $T/final_r8.log 2>&1
log "final done"
$PY pusht_tune.py --stage dagger_fit --fit_suffix _dart --probe_models se_global dp_small --seeds 0 1 2 --jobs 12 > $T/dart_fit.log 2>&1
$PY pusht_tune.py --stage summary > $T/summary_r8.log 2>&1
log POST8DONE
