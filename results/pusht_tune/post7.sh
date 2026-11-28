#!/bin/sh
# 第七轮（DAgger，临时）：收集数据 → 从头训练 → 彻底失败率、吸引子检验、500 回合复评、汇总。在服务器上自动跑，不依赖本地的 ssh 连接
cd ~/action-nca || exit 1
PY=~/venvs/action-nca/bin/python
T=runs/pusht_tune
log() { echo "$(date +%H:%M) $*" >> $T/post7.log; }

log start
$PY pusht_tune.py --stage dagger --probe_models se_global dp_small --seeds 0 1 2 --jobs 6 > $T/dagger.log 2>&1
log "dagger data done"
$PY pusht_tune.py --stage train --variants se_global_dag dp_small_dag --seeds 0 1 2 --jobs 6 --gpus 0 > $T/train_r7.log 2>&1
log "training done"
$PY pusht_tune.py --stage cat --jobs 6 > $T/cat_r7.log 2>&1
$PY pusht_tune.py --stage attract --probe_models se_global_dag --seeds 0 1 2 --jobs 6 > $T/attract_r7.log 2>&1
log "cat/attract done"
$PY pusht_tune.py --stage final --probe_models se_global_dag dp_small_dag --seeds 0 1 2 --jobs 6 > $T/final_r7.log 2>&1
$PY pusht_tune.py --stage summary > $T/summary_r7.log 2>&1
log POST7DONE
