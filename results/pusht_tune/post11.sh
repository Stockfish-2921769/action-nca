#!/bin/sh
# 第十一轮（从噪声出发长，临时）：σ = 1 训 3 个种子，σ = 0.25 训种子 0（只用示范，其余同 se_global）。
# 每个模型训完就在 CPU 上评估（500 回合复评、彻底失败率和卡住剖面、吸引子检验），和下一个训练同时跑。
# 在服务器上自动跑，不依赖本地的 ssh 连接
cd ~/action-nca || exit 1
PY=~/venvs/action-nca/bin/python
T=runs/pusht_tune
log() { echo "$(date +%H:%M) $*" >> $T/post11.log; }
# 同一个阶段同时只跑一个：各阶段的结果累积在一个 json 里，两个进程同时写会丢结果
st() { name=$1; shift; flock $T/.lock_$name $PY pusht_tune.py --stage $name "$@" >> $T/${name}_r11.log 2>&1; }
evals() {   # 评估一个（变体, 种子）：500 回合复评、彻底失败率和卡住剖面、吸引子（都只补没算过的）
  st final --probe_models $1 --seeds $2 --jobs 1
  st cat --seeds $2 --jobs 1
  st attract --probe_models $1 --seeds $2 --jobs 1
  log "$1 seed $2 evaluated"
}
train() {   # 在 GPU $3 上训练变体 $1 种子 $2；训完在后台评估，GPU 马上接着训下一个
  $PY pusht_tune.py --stage train --variants $1 --seeds $2 --jobs 1 --gpus $3 >> $T/train_r11.log 2>&1
  log "$1 seed $2 trained"
  evals $1 $2 &
}

log start
# 两张卡各依次训两个（两个挤一张卡会各慢一倍）
( train se_global_ns1 0 0; train se_global_ns1 2 0; wait ) &
( train se_global_ns1 1 1; train se_global_ns025 0 1; wait ) &
wait
log "training done"
for s in 0 1 2; do evals se_global_ns1 $s; done      # 补上没算完的（正常情况下都已算过，直接跳过）
evals se_global_ns025 0
$PY pusht_tune.py --stage summary > $T/summary_r11.log 2>&1
log POST11DONE
