#!/bin/bash
# 第六轮排队：某张卡空闲显存 ≥ 5000 MiB（给同机其他用户留余量）且我们自己的训练进程 < 9 个时，依次启动
cd ~/action-nca
PY=~/venvs/action-nca/bin/python
QUEUE="se_global:0 se_nopad:0 se_k16:2 se_global:1 se_nopad:1 se_global:2 se_nopad:2"
for job in $QUEUE; do
  v=${job%%:*}; s=${job##*:}
  if [ -f runs/pusht_tune/${v}_seed${s}/metrics.json ]; then continue; fi
  while true; do
    n=$(pgrep -fc "pusht.py --job nca")
    best=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -n 1)
    gi=${best%%,*}; free=${best##*, }
    if [ "$n" -lt 9 ] && [ "$free" -ge 5000 ]; then break; fi
    sleep 60
  done
  off=$(( gi == 1 ? 0 : 1 ))
  echo "$(date +%H:%M) launch $v seed $s on cuda:$gi (free ${free} MiB, our jobs $n)"
  setsid nohup sh -c "$PY pusht_tune.py --stage train --variants $v --seeds $s --device_offset $off > runs/pusht_tune/train_q_${v}_${s}.log 2>&1" < /dev/null > /dev/null 2>&1 &
  sleep 180
done
echo QUEUEDONE
