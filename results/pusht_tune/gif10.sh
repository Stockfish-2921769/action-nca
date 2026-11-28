#!/bin/sh
# 第十轮（临时）：等种子 0 的第十轮 NCA 和小 DP 训完，用它们画轨迹动画（viz_gif.py 的默认模型）
cd ~/action-nca || exit 1
T=runs/pusht_tune
until [ -e $T/se_global_rescue2_seed0/metrics.json ] && [ -e $T/dp_small_rescue2_seed0/metrics.json ]; do sleep 60; done
echo "$(date +%H:%M) start" > $T/gif_r10.log
nice -n 5 ~/venvs/action-nca/bin/python viz_gif.py >> $T/gif_r10.log 2>&1
echo "$(date +%H:%M) GIFDONE" >> $T/gif_r10.log
