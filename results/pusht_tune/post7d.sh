#!/bin/sh
# 第七轮补充（临时）：DAgger 之后学会了没有，加上测试回合里卡住的状态有没有被覆盖
cd ~/action-nca || exit 1
~/venvs/action-nca/bin/python pusht_tune.py --stage dagger_fit --probe_models se_global dp_small --seeds 0 1 2 --jobs 12 > runs/pusht_tune/dagger_fit.log 2>&1
echo "$(date +%H:%M) POST7DDONE" >> runs/pusht_tune/post7.log
