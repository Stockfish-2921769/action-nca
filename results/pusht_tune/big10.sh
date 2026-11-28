#!/bin/sh
# 大样本对比（临时）：第十轮和第九轮的 NCA、小 DP，每个种子 8000 个新回合（种子 s 的环境种子 600000 + 10000·s 起），
# 都用每次新 z / 新噪声；同一个种子的模型跑同样的回合，逐回合配对比较。在服务器上自动跑，不依赖本地的 ssh 连接
cd ~/action-nca || exit 1
T=runs/pusht_tune
{
  date
  flock $T/.lock_big ~/venvs/action-nca/bin/python pusht_tune.py --stage big \
      --probe_models se_global_rescue2 dp_small_rescue2 se_global_rescue dp_small_rescue \
      --seeds 0 1 2 --big_episodes 8000 --jobs 24
  echo "rc=$? $(date)"
  echo BIGDONE
} >> $T/big_r10.log 2>&1
