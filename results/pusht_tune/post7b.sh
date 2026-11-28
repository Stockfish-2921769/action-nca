#!/bin/sh
# 第七轮补充（临时）：se_global_dag 种子 2 因 GPU 0 显存不够改到 GPU 1 重训。等它和 post7.sh 都完成后，
# 补算它的彻底失败率、吸引子检验和 500 回合复评（各阶段只算没算过的），再重新汇总
cd ~/action-nca || exit 1
PY=~/venvs/action-nca/bin/python
T=runs/pusht_tune
log() { echo "$(date +%H:%M) $*" >> $T/post7.log; }
until grep -q POST7DONE $T/post7.log && [ -f $T/se_global_dag_seed2/metrics.json ]; do sleep 60; done
log "se_global_dag seed 2 ready"
$PY pusht_tune.py --stage cat --jobs 6 > $T/cat_r7b.log 2>&1
$PY pusht_tune.py --stage attract --probe_models se_global_dag --seeds 0 1 2 --jobs 6 > $T/attract_r7b.log 2>&1
$PY pusht_tune.py --stage final --probe_models se_global_dag dp_small_dag --seeds 0 1 2 --jobs 6 > $T/final_r7b.log 2>&1
$PY pusht_tune.py --stage summary > $T/summary_r7b.log 2>&1
log POST7BDONE
