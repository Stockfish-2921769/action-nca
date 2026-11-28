#!/bin/sh
# 第六轮的后续评估（临时）：在服务器上等模型训完，自动跑 cat、attract、final 和画图，不依赖本地的 ssh 连接。
cd ~/action-nca || exit 1
PY=~/venvs/action-nca/bin/python
T=runs/pusht_tune
log() { echo "$(date +%H:%M) $*" >> $T/post6.log; }

log start
# 0. 等之前手动启动的阶段跑完（比如 se_global 种子 1 的吸引子检验），避免两个进程同时写同一个 json
while pgrep -f "pusht_tune.py --[s]tage" > /dev/null; do sleep 60; done
log "no other stage running"
# 1. se_nopad 种子 0 一训完就先看
until [ -f $T/se_nopad_seed0/metrics.json ]; do sleep 60; done
log "se_nopad seed 0 done"
$PY pusht_tune.py --stage cat --jobs 4 > $T/cat_p6a.log 2>&1
$PY pusht_tune.py --stage attract --probe_models se_nopad se_global --seeds 0 1 2 --jobs 4 > $T/attract_p6a.log 2>&1
log "early cat/attract done"
grep -E "se_nopad|se_global" $T/cat_p6a.log >> $T/post6.log

# 2. 全部训完后补齐，并做 500 回合复评
until [ -f $T/se_nopad_seed1/metrics.json ] && [ -f $T/se_nopad_seed2/metrics.json ] && [ -f $T/se_global_seed2/metrics.json ]; do
    sleep 60
done
log "all round-6 models done"
$PY pusht_tune.py --stage cat --jobs 6 > $T/cat_p6b.log 2>&1
$PY pusht_tune.py --stage attract --probe_models se_nopad se_global --seeds 0 1 2 --jobs 6 > $T/attract_p6b.log 2>&1
$PY pusht_tune.py --stage final --probe_models se_nopad se_global --seeds 0 1 2 --jobs 6 > $T/final_p6.log 2>&1
$PY pusht_tune.py --stage summary > $T/summary_p6.log 2>&1
$PY viz_tune.py > $T/viz_p6.log 2>&1
log POST6DONE
