#!/bin/sh
# 第九轮（临时）：NCA 种子 1 先训完，先把它的 500 回合复评、接管检验、吸引子、彻底失败率算掉；
# 这几个阶段都只补没算过的，post9.sh 之后会跳过种子 1。四个阶段写不同的 json，可以同时跑（都在 CPU 上）
cd ~/action-nca || exit 1
PY=~/venvs/action-nca/bin/python
T=runs/pusht_tune
log() { echo "$(date +%H:%M) $*" >> $T/pre9.log; }
log start
$PY pusht_tune.py --stage final --probe_models se_global_rescue --seeds 1 --jobs 1 > $T/final_r9_s1.log 2>&1 &
$PY pusht_tune.py --stage takeover --probe_models se_global_rescue --seeds 1 --jobs 1 > $T/takeover_r9_s1.log 2>&1 &
$PY pusht_tune.py --stage attract --probe_models se_global_rescue --seeds 1 --jobs 1 > $T/attract_r9_s1.log 2>&1 &
$PY pusht_tune.py --stage cat --seeds 1 --jobs 2 > $T/cat_r9_s1.log 2>&1 &
wait
log PRE9DONE
