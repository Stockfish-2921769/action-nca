#!/bin/sh
# 第九轮（按结果挑标签，临时）：收集数据 → 从头训练（示范 + DART + 这些样本，训练轮数随数据量放大）→ 彻底失败率、吸引子检验
# → 500 回合复评 → 补数据诊断 → 接管检验 → 汇总。在服务器上自动跑，不依赖本地的 ssh 连接
cd ~/action-nca || exit 1
PY=~/venvs/action-nca/bin/python
T=runs/pusht_tune
log() { echo "$(date +%H:%M) $*" >> $T/post9.log; }

log start
$PY pusht_tune.py --stage rescue --probe_models se_global_dart dp_small_dart --seeds 0 1 2 --rescue_episodes 4000 --jobs 16 > $T/rescue.log 2>&1
log "rescue data done"
$PY pusht_tune.py --stage train --variants se_global_rescue dp_small_rescue --seeds 0 1 2 --jobs 6 --gpus 0 1 1 0 > $T/train_r9.log 2>&1
log "training done"
$PY pusht_tune.py --stage cat --jobs 6 > $T/cat_r9.log 2>&1
$PY pusht_tune.py --stage attract --probe_models se_global_rescue --seeds 0 1 2 --jobs 6 > $T/attract_r9.log 2>&1
log "cat/attract done"
$PY pusht_tune.py --stage final --probe_models se_global_rescue dp_small_rescue --seeds 0 1 2 --jobs 9 > $T/final_r9.log 2>&1
log "final done"
$PY pusht_tune.py --stage rescue_fit --seeds 0 1 2 --jobs 6 > $T/rescue_fit.log 2>&1
$PY pusht_tune.py --stage takeover --probe_models se_global_rescue dp_small_rescue --seeds 0 1 2 --jobs 6 > $T/takeover_r9.log 2>&1
log "fit/takeover done"
$PY pusht_tune.py --stage summary > $T/summary_r9.log 2>&1
log POST9DONE
