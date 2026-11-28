#!/bin/sh
# 第十轮（再收集一轮，临时）：在第九轮的模型卡住的地方收集（新的环境种子 500000 起，接手的加上第九轮的两个模型）
# → 从头训练（示范 + DART + 两轮挑出来的样本，训练轮数随数据量放大）→ 每个种子训完就在 CPU 上评估
# → 补数据诊断 → 汇总。在服务器上自动跑，不依赖本地的 ssh 连接
cd ~/action-nca || exit 1
PY=~/venvs/action-nca/bin/python
T=runs/pusht_tune
log() { echo "$(date +%H:%M) $*" >> $T/post10.log; }
# 同一个阶段同时只跑一个：各阶段的结果累积在一个 json 里，两个进程同时写会丢结果
st() { name=$1; shift; flock $T/.lock_$name $PY pusht_tune.py --stage $name "$@" >> $T/${name}_r10.log 2>&1; }
evals() {   # 评估一个种子：500 回合复评、彻底失败率、吸引子、接管检验（都只补没算过的）
  st final --probe_models se_global_rescue2 dp_small_rescue2 --seeds $1 --jobs 3
  st cat --seeds $1 --jobs 1
  st attract --probe_models se_global_rescue2 --seeds $1 --jobs 1
  st takeover --probe_models se_global_rescue2 dp_small_rescue2 --seeds $1 --jobs 2
  log "seed $1 evaluated"
}
nca() {     # 在 GPU $2 上训练 NCA 种子 $1，训完马上评估这个种子
  $PY pusht_tune.py --stage train --variants se_global_rescue2 --seeds $1 --jobs 1 --gpus $2 >> $T/train_r10.log 2>&1
  log "nca seed $1 trained"
  evals $1
}

log start
$PY pusht_tune.py --stage rescue --probe_models se_global_rescue dp_small_rescue --seeds 0 1 2 --rescue_episodes 8000 \
    --rescue_seed 500000 --rescue_ctrl dp dp_noise dp_small nca dp_small_r9 nca_r9 --jobs 16 > $T/rescue_r10.log 2>&1
log "rescue data done"
# GPU 0 依次训 NCA 种子 0、2（两个挤一张卡会各慢一倍）；GPU 1 训 NCA 种子 1，同时训三个小 DP
( nca 0 0; nca 2 0 ) &
nca 1 1 &
( $PY pusht_tune.py --stage train --variants dp_small_rescue2 --seeds 0 1 2 --jobs 3 --gpus 1 >> $T/train_r10.log 2>&1
  log "dp trained" ) &
wait
log "training done"
for s in 0 1 2; do evals $s; done     # 补上训练时没赶上的
st rescue_fit --seeds 0 1 2 --jobs 6
$PY pusht_tune.py --stage summary > $T/summary_r10.log 2>&1
log POST10DONE
