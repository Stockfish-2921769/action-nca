#!/bin/sh
# 路径图的数据（临时）
cd ~/action-nca || exit 1
~/venvs/action-nca/bin/python pusht_tune.py --stage paths --seeds 0 --jobs 8 > runs/pusht_tune/paths.log 2>&1
echo "$(date +%H:%M) PATHSDONE" >> runs/pusht_tune/post7.log
