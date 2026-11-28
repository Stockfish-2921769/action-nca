#!/usr/bin/env python3
"""
Push-T 上 NCA 的调参：找 NCA 在这个任务上的上限，目标是超过参数量相当的小 DP（NOTES.md 第 19 节）。

三步：
  --stage eval     不重训：在 runs/pusht/nca_seed0–2 上扫冷启动步数、再生步数、评估时的 fire_rate、再生时定期冷启动
                   （EVAL_GRID）。结果在 runs/pusht_tune/eval_sweep.json 和 eval_sweep.md。--eval_model 换成调参变体
                   （runs/pusht_tune/<变体>_seed<s>），--grid 只跑其中几项，这时结果写到 eval_<变体>.json / .md。
  --stage train    训练变体（TRAIN_VARIANTS，名字 → pusht.py 的额外参数），每个变体用 pusht.py --job nca（TRAIN_JOB
                   里的用 --job dp）训练和评估三种执行方式，输出在 runs/pusht_tune/<变体>_seed<s>/。--variants、--seeds
                   选择要跑的，--gpus 选 GPU。
  --stage summary  汇总成 runs/pusht_tune/summary.md，和基线 NCA、小 DP、DP（runs/pusht/）并排。
  --stage probe    离线探针：一次再生能否跟上新观测（runs/pusht_tune/probe.md）。
  --stage chain    漂移探针：连续再生时误差和隐藏通道是否累积，并试再生前清零隐藏通道（runs/pusht_tune/chain.md）。
  --stage blend    部分冷启动：再生前把旧计划往冷启动的初始状态拉一部分（runs/pusht_tune/blend.md）。
  --stage diag     冷启动诊断：失败回合的覆盖率分布、采样的多模态程度，NCA 和小 DP、DP 并排（runs/pusht_tune/diag.md）。
  --stage noise    冷启动计划加随机平移，看随机性能否减少彻底失败（runs/pusht_tune/noise.md）。
  --stage stall    慢启动：卡住时计划的逐步剖面，以及初始状态加噪声、卡住时跳过前 8 步（runs/pusht_tune/stall.md）。
  --stage ens      训练种子集成（计划取平均），看彻底失败是不是单个模型的方差（runs/pusht_tune/ens.md）。
  --stage cat      所有已训练 NCA 的彻底失败率和卡住时的慢启动剖面（runs/pusht_tune/cat.md，结果累积）。
  --stage attract  吸引子检验：卡住的状态上从 DP 的计划出发长，前 8 步会不会被拉回原地（runs/pusht_tune/attract.md，结果累积）。
  --stage steps    不重训：冷启动少长几步能否减少彻底失败（runs/pusht_tune/steps.md，结果累积）。
  --stage final    最终对比：选定的 NCA 变体和小 DP、DP，每个种子 500 个测试回合（runs/pusht_tune/final.md，结果累积）。
  --stage dagger   DAgger 数据：学生（--probe_models）自己闭环跑 --dagger_episodes 个回合（训练用的环境种子 200000 起），
                   每次规划时的观测用同种子大 DP 的计划做标签，存到 runs/pusht_tune/dagger/<学生>_seed<s>.npz，
                   诊断在 dagger.md。之后用 --stage train --variants se_global_dag dp_small_dag 训练。
  --stage dagger_fit  补数据之后学会了没有：在学生的 DAgger 数据上，比较补数据前后的学生和老师（大 DP）的计划；
                   再看测试回合里卡住的状态有没有被额外数据覆盖（runs/pusht_tune/dagger_fit.md；--fit_suffix _dart
                   检查 DART，还加上 DART 数据本身，结果在 dart_fit.md）。
  --stage paths    闭环路径：DAgger 前后的 NCA 和小 DP 在同一批测试回合上的轨迹，以及卡住时的计划
                   （runs/pusht_tune/paths.json，图由 viz_tune.py 画成 results/pusht_tune/paths.png）。
  --stage dart     DART 数据：老师（同种子大 DP）自己闭环 --dart_episodes 个回合（环境种子 300000 起），执行前给动作块
                   加扰动（整体平移、放慢），扰动前的计划作标签，存到 runs/pusht_tune/dart/dp_seed<s>.npz，诊断在 dart.md。
                   之后用 --stage train --variants se_global_dart dp_small_dart 训练（训练轮数随数据量放大）。
  --stage takeover 接管检验：--probe_models 里的模型在测试回合里第一次卡住时，换自己（新随机数）、大 DP、小 DP 接手，
                   看能不能脱困（runs/pusht_tune/takeover.md，结果累积）。
  --stage rescue   按结果挑标签：学生（--probe_models，第九轮是 se_global_dart dp_small_dart）自己闭环 --rescue_episodes 个回合
                   （环境种子 --rescue_seed 起，默认 400000），彻底失败回合里第一次卡住时，--rescue_ctrl 里的规划器（RESCUE_CTRL）
                   各接手 --rescue_tries 次；挑脱困（最好覆盖率 ≥ 0.5）里最好的一次，从接手到覆盖率最高的那一步，每一步一个样本，
                   标签是实际执行的 16 个动作。存到 runs/pusht_tune/rescue/<学生>_seed<s>.npz，诊断在 rescue.md。之后用
                   --stage train --variants se_global_rescue dp_small_rescue 训练（示范 + DART + 这些样本）。
                   第十轮（再收集一轮）：学生换成第九轮的模型 se_global_rescue dp_small_rescue，--rescue_seed 500000
                   --rescue_episodes 8000，--rescue_ctrl 加上第九轮的两个模型；之后训练 se_global_rescue2 dp_small_rescue2
                   （示范 + DART + 两轮的样本）。
  --stage rescue_fit  补数据之后学会了没有：救出来的卡住状态（见过）、没救出来的（没见过）、测试回合里卡住的（没见过）上，
                   比较补数据前后的学生、老师和小 DP 的计划（runs/pusht_tune/rescue_fit.md）。
  --stage big      大样本对比：--probe_models 里的模型每个种子闭环 --big_episodes 个新回合（默认 8000，种子 s 的环境种子从
                   600000 + 10000·s 起，和以上所有阶段都不重叠），都用每次新 z / 新噪声；同一个种子的模型跑同样的回合，
                   BIG_PAIRS 里的两个模型逐回合配对比较（runs/pusht_tune/big.md，逐回合结果在 big/<模型>_seed<s>.npz）。

用法
  python pusht_tune.py --stage eval
  python pusht_tune.py --stage eval --eval_model se --grid base_fresh base_warm refresh2 refresh4 refresh8
  python pusht_tune.py --stage train --variants sync ema smooth big --seeds 0
  python pusht_tune.py --stage summary
"""
import argparse
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np

import pusht as P

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "runs", "pusht")
OUT = os.path.join(HERE, "runs", "pusht_tune")

# (名字, 执行方式, 冷启动步数, 再生步数, 评估时 fire_rate；None = 用训练时的 0.5[, 再生时每隔几次规划冷启动一次, z 的缩放])
EVAL_GRID = [("base_fresh", "fresh", 32, 10, None), ("base_fixed", "fixed", 32, 10, None), ("base_warm", "warm", 32, 10, None)]
EVAL_GRID += [(f"cold{c}", "fresh", c, 10, None) for c in (24, 48, 64)]
EVAL_GRID += [(f"warm{w}", "warm", 32, w, None) for w in (6, 16, 24, 32)]
EVAL_GRID += [("sync_fresh16", "fresh", 16, 5, 1.0), ("sync_fresh32", "fresh", 32, 10, 1.0),
              ("sync_warm5", "warm", 16, 5, 1.0), ("sync_warm10", "warm", 16, 10, 1.0)]
EVAL_GRID += [(f"refresh{r}", "warm", 32, 10, None, r) for r in (2, 4, 8)]   # 检验再生是否随连续次数漂移
EVAL_GRID += [(f"z{s:g}_{m}", m, 32, 10, None, 0, s) for s in (0.0, 0.5) for m in ("fresh", "fixed")]  # z 缩小 → 偏向常见模态
EVAL_GRID = [tuple(g) + (0, 1.0)[len(g) - 5:] for g in EVAL_GRID]          # 缺省：从不冷启动，z 不缩放

# 名字 → pusht.py 的额外参数（基线 = pusht.py 默认：fire_rate 0.5，无 EMA，无平滑项，hidden 62，width 256，2 万轮）
TRAIN_VARIANTS = {
    "sync": ["--fire_rate", "1.0"],                                  # 同步更新：每步所有细胞都更新
    "ema": ["--nca_ema", "0.999"],                                   # 参数 EMA（DP 一直在用）
    "smooth": ["--smooth_weight", "1.0"],                            # 平滑项：计划二阶差分的平方和
    "big": ["--nca_hidden", "126", "--nca_width", "512"],            # 约 43 万参数（超过小 DP 的 12.5 万）
    "rnoise": ["--regrow_noise", "0.03"],                            # 再生训练时扰动旧计划（约 8 像素，和衔接跳变同量级）
}
# 第二轮：以平滑项 + EMA 为新基础（第一轮里这两项都没有降低成功率，平滑项把粗糙度降到小 DP 的水平）
SE = ["--smooth_weight", "1.0", "--nca_ema", "0.999"]
TRAIN_VARIANTS.update({
    "se": SE,
    "se_noise": SE + ["--obs_noise", "0.02"],                        # 训练时位置观测加噪声（约 5 像素），模拟闭环里的状态偏移
    "se_k16": SE + ["--k", "16"],                                    # min-of-K 的候选数 8 → 16
    "se_long": SE + ["--nca_iters", "40000"],                        # 训练轮数加倍
})
# 第三轮：再生的训练 / 评估错配。默认每轮把 256 个冷启动样本全部换进 4096 的样本池，样本平均只再生 1–2 次就被换掉，
# 而闭环评估一回合要连续再生约 30 次；另外样本池里的 z 都是 min-of-K 选出来的，评估时的 z 是随机的
TRAIN_VARIANTS.update({
    "se_persist": SE + ["--pool_inject", "64"],                      # 每轮只换进 64 个，样本能一直再生到示范回合结束
    "se_prand": SE + ["--pool_random", "1", "--pool_inject", "256"],  # 随机 z 的计划也进样本池（换进总数不变）
    "se_pp": SE + ["--pool_inject", "64", "--pool_random", "1"],     # 两者都用
})
# 第四轮：第三轮没有改善；探针（--stage probe）显示再生只把旧计划往新目标拉了一半（旧计划落后一个块时误差 30–36 像素，
# 冷启动 8 像素）。训练时新观测总是正好 8 步后，目标和“照抄旧计划”只差约 10 像素，再生从没学过大幅纠正
TRAIN_VARIANTS.update({
    "se_jit4": SE + ["--regrow_jitter", "4"],                        # 再生时新观测取 8 ± 4 步后，目标随之改变
    "se_jit8": SE + ["--regrow_jitter", "8"],                        # 8 ± 8 步
})
# 第五轮：回到冷启动。诊断（--stage diag）显示得分差全在彻底失败（最好覆盖率 < 0.5）的回合：NCA 9–10%，小 DP 4%，
# DP 3%；失败大多只在一个种子上出现，推杆没有卡住。另外换 z 的散布（3.3 像素）不比同 z 重算（3.9）大，z 没被用上
TRAIN_VARIANTS.update({
    "se_zemb": SE + ["--z_emb", "1"],                                # z 广播到每个细胞，让 z 更容易被用上
    "se_exec": SE + ["--exec_weight", "4"],                          # 前 8 步（会执行的部分）误差权重 ×4
})
# 第六轮：彻底失败来自“慢启动”吸引子（--stage stall）：分布外状态上计划前 8 步停在当前位置附近，
# 即使从 DP 的好计划出发，长 8 步也会被拉回去；后面的细胞其实已经朝正确方向走了。改变早期细胞的锚定方式
TRAIN_VARIANTS.update({
    "se_global": SE + ["--nca_global", "1"],                         # 全局感知：每个细胞也看到所有细胞的均值
    "se_nopad": SE + ["--seed_pad", "0", "--z_emb", "1"],            # 左边界不再是种子细胞（当前位置），z 走观测编码
})
# 第七轮：DAgger。前六轮的结论是彻底失败来自分布外状态上的慢启动吸引子，18 个训练变体都没消除，而示范里没有这些状态。
# 学生自己闭环跑 500 个回合（--stage dagger，训练用的环境种子），每次规划时的观测用同种子大 DP 的计划做标签，
# 和示范合在一起从头训练（pusht.py --extra）。小 DP 做同样的处理（它自己的回合、同一个老师、同样多的回合），比较才公平
TRAIN_VARIANTS.update({
    "se_global_dag": SE + ["--nca_global", "1", "--extra", "runs/pusht_tune/dagger/se_global_seed{seed}.npz"],
    "dp_small_dag": P.VARIANTS["dp_small"][1] + ["--extra", "runs/pusht_tune/dagger/dp_small_seed{seed}.npz"],
})
# 第八轮：DART。DAgger 只补到了学生自己到过的状态，测试时又卡在没见过的地方（NOTES.md 19.8）。这里不靠学生：老师（同种子
# 大 DP）自己闭环，执行前给动作块加扰动（整体平移、放慢），记下扰动前的计划作标签（--stage dart）。NCA 和小 DP 用同一份
# 数据；训练轮数随数据量放大（--scale_iters 1），每个示范样本被看到的次数和原来一样
TRAIN_VARIANTS.update({
    "se_global_dart": SE + ["--nca_global", "1", "--extra", "runs/pusht_tune/dart/dp_seed{seed}.npz", "--scale_iters", "1"],
    "dp_small_dart": P.VARIANTS["dp_small"][1] + ["--extra", "runs/pusht_tune/dart/dp_seed{seed}.npz", "--scale_iters", "1"],
})
# 第九轮：按结果挑标签。DART 后剩下的彻底失败卡在老师自己也出不来的地方（接管检验：老师只能救出 10%，小 DP 40%，
# NOTES.md 19.9），同一个老师的标签再多也没用。这里在学生（DART 后的模型）自己卡住的状态上，让几个规划器各接手几次
# （--stage rescue），只留脱困的那一段实际执行过的动作作标签，和示范 + DART 数据一起从头训练（训练轮数随数据量放大）
RESCUE_X = "runs/pusht_tune/rescue/{m}_seed{{seed}}.npz"
TRAIN_VARIANTS.update({
    "se_global_rescue": SE + ["--nca_global", "1", "--extra", "runs/pusht_tune/dart/dp_seed{seed}.npz",
                              RESCUE_X.format(m="se_global_dart"), "--scale_iters", "1"],
    "dp_small_rescue": P.VARIANTS["dp_small"][1] + ["--extra", "runs/pusht_tune/dart/dp_seed{seed}.npz",
                                                    RESCUE_X.format(m="dp_small_dart"), "--scale_iters", "1"],
})
# 第十轮：再收集一轮。第九轮后卡住的回合少了 60%，但新卡住的地方离第九轮挑出来的状态更远，那里计划第 8 步仍只有 7–9 像素
# （NOTES.md 19.10）。学生换成第九轮的模型，在新的环境种子上收集（--rescue_seed 500000），接手的规划器加上第九轮的两个模型，
# 和前两轮的数据合在一起从头训练（训练轮数随数据量放大）
TRAIN_VARIANTS.update({
    "se_global_rescue2": SE + ["--nca_global", "1", "--extra", "runs/pusht_tune/dart/dp_seed{seed}.npz",
                               RESCUE_X.format(m="se_global_dart"), RESCUE_X.format(m="se_global_rescue"), "--scale_iters", "1"],
    "dp_small_rescue2": P.VARIANTS["dp_small"][1] + ["--extra", "runs/pusht_tune/dart/dp_seed{seed}.npz",
                                                     RESCUE_X.format(m="dp_small_dart"), RESCUE_X.format(m="dp_small_rescue"),
                                                     "--scale_iters", "1"],
})
TRAIN_JOB = {"dp_small_dag": "dp", "dp_small_dart": "dp", "dp_small_rescue": "dp", "dp_small_rescue2": "dp"}   # 其余都是 NCA
DAGGER_SEED = 200000                                                 # 收集 DAgger 数据的环境种子（测试种子 100000–100499，不重叠）
DART_SEED = 300000                                                   # 收集 DART 数据的环境种子
DART_SIGMAS = (0.0, 15.0, 30.0, 45.0)                                # 每个回合抽一个 σ：每个动作块整体平移 N(0, σ²)（像素）
DART_SLOW = (0.0, 0.25, 0.5)                                         # 每个回合抽一个概率：每个动作块以这个概率“放慢”
DART_SLOW_SCALE = 0.2                                                # 放慢 = 计划相对当前推杆位置的位移乘 0.2（模拟学生的慢启动）
DART_FAR = 0.25                                                      # 离最近的示范观测 ≥ 这个距离算“远”（卡住的状态 0.33–0.47）
EXTRA_DATA = {"_dag": "dagger/{m}_seed{s}.npz", "_dart": "dart/dp_seed{s}.npz"}   # 变体后缀 → 额外训练数据（相对 OUT）
RESCUE_SEED = 400000                                                 # 收集“按结果挑标签”数据的环境种子（第九轮）
RESCUE_SEED2 = 500000                                                # 第十轮（--rescue_seed）
RESCUE_CTRL = {"dp": ("dp", 0.0), "dp_noise": ("dp", 15.0), "dp_small": ("dp_small", 0.0), "nca": ("se_global", 0.0),
               "dp_small_r9": ("dp_small_rescue", 0.0), "nca_r9": ("se_global_rescue", 0.0)}
# ↑ 接手的规划器 → (模型, 每个动作块整体平移 N(0, σ²) 的 σ，像素)：老师、加扰动的老师、只用示范训练的小 DP 和 NCA、
#   补了第九轮数据的小 DP 和 NCA（第十轮才用），都是同种子的
RESCUE_BASE = ["dp", "dp_noise", "dp_small", "nca"]                  # 第九轮的接手者（--rescue_ctrl 的默认）
RESCUE_MIN = 0.5                                                     # 脱困 = 整个回合的最好覆盖率 ≥ 0.5（和接管检验一样）
RESCUE_PAIRS = {"se_global_rescue": "se_global_dart", "dp_small_rescue": "dp_small_dart",       # 补数据后的模型 → 收集数据的学生
                "se_global_rescue2": "se_global_rescue", "dp_small_rescue2": "dp_small_rescue"}
BIG_SEED = 600000                                                    # 大样本对比的环境种子：种子 s 用 BIG_SEED + 10000·s 起
BIG_PAIRS = [("se_global_rescue2", "dp_small_rescue2"), ("se_global_rescue", "dp_small_rescue"),   # 同一轮的 NCA 对小 DP
             ("se_global_rescue2", "se_global_rescue"), ("dp_small_rescue2", "dp_small_rescue")]   # 第十轮对第九轮


def model_kind(m):
    return "dp" if m in ("dp", "dp_small") or TRAIN_JOB.get(m) == "dp" else "nca"


def eval_one(job):
    name, seed, mode, cold, warm, fire, refresh, zs, model = job
    a = dict(cold_steps=cold, warm_steps=warm, dp_steps=10, dp_warm_steps=3, eval_fire_rate=fire, warm_refresh=refresh,
             z_scale=zs, episodes=100, seed=seed)
    path = os.path.join(BASE if model == "nca" else OUT, f"{model}_seed{seed}", "nca.pt")
    return name, seed, P.eval_setting(("nca", path, mode, a))[1]


def stage_eval(a):
    os.makedirs(OUT, exist_ok=True)
    grid = [g for g in EVAL_GRID if a.grid is None or g[0] in a.grid]
    tag = "eval_sweep" if a.eval_model == "nca" and a.grid is None else f"eval_{a.eval_model}"
    jobs = [(g[0], s, *g[1:], a.eval_model) for s in a.seeds for g in grid]
    jpath = os.path.join(OUT, f"{tag}.json")
    R = json.load(open(jpath, encoding="utf-8")) if os.path.exists(jpath) else {}   # 同一个模型多次 --grid 的结果累积
    t0 = time.time()
    with ProcessPoolExecutor(a.jobs, mp_context=mp.get_context("spawn")) as ex:
        for n, s, m in ex.map(eval_one, jobs):
            R[f"{n}/{s}"] = m
            print(f"{n:14s} seed {s}  score {m['score']:.3f}  success {m['success']:.2f}  rough {m['action_rough']:.2f}"
                  f"  jump {m['chunk_jump']:.2f}  ms {m['ms_per_plan']:.0f}  ({time.time() - t0:.0f}s)", flush=True)
    json.dump(R, open(jpath, "w", encoding="utf-8"), indent=1)
    root = "runs/pusht/nca" if a.eval_model == "nca" else f"runs/pusht_tune/{a.eval_model}"
    L = [f"# NCA 评估时的设置（不重训；{root}_seed{{{','.join(map(str, a.seeds))}}}，每格 100 回合）\n",
         "| 设置 | 执行方式 | 冷启动步数 | 再生步数 | fire_rate | 每几次冷启动 | z 缩放 | 得分 | 成功率 | 动作粗糙度 | 衔接跳变 | 毫秒 / 次规划 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for n, mode, c, w, f, rf, zs in EVAL_GRID:
        if not all(f"{n}/{s}" in R for s in a.seeds):
            continue
        r = [R[f"{n}/{s}"] for s in a.seeds]
        L.append(f"| {n} | {mode} | {c} | {w if mode == 'warm' else '–'} | {f or 0.5} | {rf or '–'} | {zs:g}"
                 f" | {P.fmt([x['score'] for x in r], nd=3)}"
                 f" | {P.fmt([x['success'] for x in r], True)} | {P.fmt([x['action_rough'] for x in r])}"
                 f" | {P.fmt([x['chunk_jump'] for x in r])} | {np.mean([x['ms_per_plan'] for x in r]):.0f} |")
    text = "\n".join(L) + "\n"
    open(os.path.join(OUT, f"{tag}.md"), "w", encoding="utf-8").write(text)
    print(text)


def stage_train(a, extra):
    jobs = [(v, s) for s in a.seeds for v in a.variants]

    def launch(i_job):
        i, (v, s) = i_job
        d = os.path.join(OUT, f"{v}_seed{s}")
        if os.path.exists(os.path.join(d, "metrics.json")):
            return
        os.makedirs(d, exist_ok=True)
        cmd = [sys.executable, os.path.join(HERE, "pusht.py"), "--job", TRAIN_JOB.get(v, "nca"), "--seed", str(s), "--out", d,
               "--device", f"cuda:{a.gpus[(i + 1 + a.device_offset) % len(a.gpus)]}"] + extra + TRAIN_VARIANTS[v]
        env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONIOENCODING="utf-8")
        t0 = time.time()
        with open(os.path.join(d, "log.txt"), "w", encoding="utf-8") as f:
            f.write(" ".join(cmd) + "\n")
            f.flush()
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=HERE).returncode
        print(f"{'完成' if rc == 0 else '失败'} {v}_seed{s}，{(time.time() - t0) / 60:.1f} 分钟", flush=True)

    with ThreadPoolExecutor(a.jobs) as ex:
        list(ex.map(launch, enumerate(jobs)))


def stage_summary(a):
    rows = [("NCA 基线", BASE, "nca"), ("小 DP", BASE, "dp_small"), ("DP", BASE, "dp")]
    found = sorted({d.rsplit("_seed", 1)[0] for d in os.listdir(OUT) if "_seed" in d}) if os.path.isdir(OUT) else []
    rows += [(v, OUT, v) for v in found]                                   # runs/pusht_tune/ 里所有变体（含组合）
    L = ["# Push-T 上 NCA 变体和 DP 的对比（每种执行方式 100 个测试回合，测试种子 100000 起；均值 ± 标准差）\n",
         "| 方法 | 种子 | 参数量 | 执行方式 | 得分 | 成功率 | 动作粗糙度 | 推杆粗糙度 | 衔接跳变 | 毫秒 / 次规划 |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for name, root, v in rows:
        M = [json.load(open(os.path.join(root, f"{v}_seed{s}", "metrics.json"), encoding="utf-8"))
             for s in range(5) if os.path.exists(os.path.join(root, f"{v}_seed{s}", "metrics.json"))]
        if not M:
            continue
        seeds = [s for s in range(5) if os.path.exists(os.path.join(root, f"{v}_seed{s}", "metrics.json"))]
        for st in ("fresh", "fixed", "warm"):
            r = [m[st] for m in M]
            L.append(f"| {name} | {','.join(map(str, seeds))} | {M[0]['params']} | {st} | {P.fmt([x['score'] for x in r], nd=3)}"
                     f" | {P.fmt([x['success'] for x in r], True)} | {P.fmt([x['action_rough'] for x in r])}"
                     f" | {P.fmt([x['agent_rough'] for x in r])} | {P.fmt([x['chunk_jump'] for x in r])}"
                     f" | {np.mean([x['ms_per_plan'] for x in r]):.0f} |")
    text = "\n".join(L) + "\n"
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, "summary.md"), "w", encoding="utf-8").write(text)
    print(text)


def stage_probe(a):
    """离线探针：再生出来的前 8 步是跟着新观测走，还是照抄旧计划？在示范数据上换三种新观测：
    matched = 8 步后的真实状态（训练时就是这样），stale = 16 步后（旧计划落后一个块），random = 随机另一个样本。"""
    import torch
    torch.set_num_threads(8)
    O, Y, NXT = P.load_data()
    rng = np.random.default_rng(0)
    ok = np.flatnonzero((NXT >= 0) & (NXT[np.maximum(NXT, 0)] >= 0))
    i = rng.choice(ok, 2048, replace=False)
    conds = {"matched": NXT[i], "stale": NXT[NXT[i]], "random": rng.choice(len(O), len(i))}
    px = lambda d: float((d.norm(dim=-1) * 256).mean())                  # 每步欧氏距离的均值（像素）
    R = {}
    for model in a.probe_models:
        for s in a.seeds:
            pl = P.make_planner("nca", os.path.join(BASE if model == "nca" else OUT, f"{model}_seed{s}", "nca.pt"), "warm")
            m = pl.m
            torch.manual_seed(s)
            with torch.no_grad():
                z = torch.randn(len(i), pl.zd)                                # 和评估时一样用随机 z
                oi = torch.as_tensor(O[i])
                old = m.grow(m.blank(oi), oi, z, 32)
                copy = P.A.plan_of(old)[:, P.T_A:]                            # 照抄：旧计划的后 8 步
                for c, j in conds.items():
                    oj, yj = torch.as_tensor(O[j]), torch.as_tensor(Y[j])[:, :P.T_A]
                    w = P.A.plan_of(m.grow(P.A.shift(old, P.T_A), oj, z, 10))[:, :P.T_A]
                    cold = P.A.plan_of(m.grow(m.blank(oj), oj, z, 32))[:, :P.T_A]
                    R.setdefault((model, c), []).append((px(w - yj), px(w - copy), px(cold - yj), px(yj - copy)))
            print(model, s, "done", flush=True)
    L = [f"# 再生探针（示范数据上 2048 个样本，随机 z，种子 {','.join(map(str, a.seeds))}；前 8 步动作的平均误差，像素）\n",
         "| 模型 | 新观测 | 再生 vs 新目标 | 再生 vs 照抄旧计划 | 冷启动 vs 新目标 | 新目标 vs 照抄旧计划 |",
         "|---|---|---|---|---|---|"]
    for (model, c), v in R.items():
        v = np.array(v)
        L.append(f"| {model} | {c} | " + " | ".join(P.fmt(v[:, k], nd=1) for k in range(4)) + " |")
    text = "\n".join(L) + "\n"
    open(os.path.join(OUT, "probe.md"), "w", encoding="utf-8").write(text)
    print(text)


class ChainPlan(P.NCAPlan):
    """再生模式的规划器，额外记录：每次再生的前 8 步和同一观测、同一 z 下冷启动的差距，以及两次冷启动之间的差距（噪声底）。
    reset_hidden = 再生前把隐藏通道清零，只保留动作通道里的旧计划。"""

    def __init__(self, *args, reset_hidden=False):
        super().__init__(*args)
        self.rh, self.log = reset_hidden, []

    def plan(self, obs):
        import torch
        with torch.no_grad():
            if self.s is None:
                self.s = self.m.grow(self.m.blank(obs), obs, self.z, self.cs)
                self.calls += self.cs
            else:
                copy = P.A.plan_of(self.s)[:, P.T_A:]                    # 照抄：旧计划的后 8 步
                s = P.A.shift(self.s, P.T_A)
                if self.rh:
                    s[:, 2:] = 0.0
                self.s = self.m.grow(s, obs, self.z, self.ws)
                self.calls += self.ws
                c1 = P.A.plan_of(self.m.grow(self.m.blank(obs), obs, self.z, self.cs))[:, :P.T_A]
                c2 = P.A.plan_of(self.m.grow(self.m.blank(obs), obs, self.z, self.cs))[:, :P.T_A]
                w = P.A.plan_of(self.s)[:, :P.T_A]
                live = (obs[:, :6] != obs[:, 6:12]).any(1)      # 已经成功结束的回合状态不再变化，不计入
                d = lambda x, y: ((x - y).norm(dim=-1).mean(1) * 256)[live].tolist()
                self.log.append((d(w, c1), d(c1, c2), float(self.s[:, 2:].pow(2).mean().sqrt()), d(copy, c1), d(w, copy)))
            self.n += 1
            return P.A.plan_of(self.s)


def chain_one(job):
    """一个（模型, 种子, 是否清零隐藏通道）：A. 示范数据上沿真实轨迹连续再生 n 次（新观测总是正好 8 步后），
    每次比较再生和冷启动对新目标的误差；B. 闭环 100 回合再生，每次比较再生和冷启动（同观测同 z）的差距。"""
    import torch
    model, s, rh, n, O, Y, chains = job
    torch.set_num_threads(2)
    path = os.path.join(BASE if model == "nca" else OUT, f"{model}_seed{s}", "nca.pt")
    pl = P.make_planner("nca", path, "warm")
    m, px = pl.m, lambda d: float((d.norm(dim=-1) * 256).mean())
    torch.manual_seed(s)
    A_err, A_cold, A_hid, A_hid_cold, A_wc, A_cc = [], [], [], [], [], []
    with torch.no_grad():
        z = torch.randn(chains.shape[1], pl.zd)
        o = torch.as_tensor(O[chains[0]])
        st = m.grow(m.blank(o), o, z, 32)
        for k in range(1, n + 1):
            o, y = torch.as_tensor(O[chains[k]]), torch.as_tensor(Y[chains[k]])[:, :P.T_A]
            st = P.A.shift(st, P.T_A)
            if rh:
                st[:, 2:] = 0.0
            st = m.grow(st, o, z, 10)
            cs = m.grow(m.blank(o), o, z, 32)
            c2 = P.A.plan_of(m.grow(m.blank(o), o, z, 32))[:, :P.T_A]
            A_err.append(px(P.A.plan_of(st)[:, :P.T_A] - y))
            A_cold.append(px(P.A.plan_of(cs)[:, :P.T_A] - y))
            A_wc.append(px(P.A.plan_of(st)[:, :P.T_A] - P.A.plan_of(cs)[:, :P.T_A]))   # 和闭环 B 的同一指标对照
            A_cc.append(px(P.A.plan_of(cs)[:, :P.T_A] - c2))
            A_hid.append(float(st[:, 2:].pow(2).mean().sqrt()))
            A_hid_cold.append(float(cs[:, 2:].pow(2).mean().sqrt()))
    ck = torch.load(path, map_location="cpu")
    cp = ChainPlan(m, "warm", ck["z_dim"], 32, 10, reset_hidden=rh)
    met = P.rollout(cp, 100, s + 1)[0]
    return dict(model=model, seed=s, reset=rh, A_err=A_err, A_cold=A_cold, A_hid=A_hid, A_hid_cold=A_hid_cold,
                A_wc=A_wc, A_cc=A_cc,
                B_wc=[x[0] for x in cp.log], B_cc=[x[1] for x in cp.log], B_hid=[x[2] for x in cp.log],
                B_pc=[x[3] for x in cp.log], B_wp=[x[4] for x in cp.log],
                score=met["score"], success=met["success"], action_rough=met["action_rough"])


def stage_chain(a):
    """漂移探针：闭环里一个回合要连续再生约 37 次，--stage probe 只测了从冷启动计划出发的一次。
    这里看误差和隐藏通道是否随连续再生的次数累积，并试再生前清零隐藏通道（不重训）。"""
    O, Y, NXT = P.load_data()
    n = a.chain_n
    j, idx = np.arange(len(NXT)), [np.arange(len(NXT))]
    for _ in range(n):                                                  # 沿 NXT 走 n 步都还在同一个示范回合里
        j = np.where(j >= 0, NXT[np.maximum(j, 0)], -1)
        idx.append(j)
    idx = np.stack(idx)
    ok = np.flatnonzero(idx[-1] >= 0)
    chains = idx[:, np.random.default_rng(0).choice(ok, min(1024, len(ok)), replace=False)]
    print(f"{len(ok)} 个起点能连续再生 {n} 次，取 {chains.shape[1]} 个", flush=True)
    jobs = [(mdl, s, rh, n, O, Y, chains) for mdl in a.probe_models for s in a.seeds for rh in (False, True)]
    R = []
    t0 = time.time()
    with ProcessPoolExecutor(a.jobs, mp_context=mp.get_context("spawn")) as ex:
        for r in ex.map(chain_one, jobs):
            R.append(r)
            print(f"{r['model']} seed {r['seed']} reset {int(r['reset'])}  success {r['success']:.2f}"
                  f"  score {r['score']:.3f}  ({time.time() - t0:.0f}s)", flush=True)
    json.dump(R, open(os.path.join(OUT, "chain.json"), "w", encoding="utf-8"))
    ks = [k for k in (1, 2, 3, 5, 7, 10, 15, 20) if k <= n]
    B_bins = [(1, 1), (2, 3), (4, 7), (8, 15), (16, 31), (32, 99)]
    L = [f"# 漂移探针（种子 {','.join(map(str, a.seeds))}；误差都是前 8 步动作的平均距离，像素）\n",
         f"## A. 示范数据上连续再生（{chains.shape[1]} 条链，新观测总是正好 8 步后，随机 z）\n",
         "再生 vs 新目标（括号里是同一位置冷启动 vs 新目标）；最后两列是隐藏通道的均方根（再生 / 冷启动）。\n",
         "| 模型 | 隐藏通道 | " + " | ".join(f"第 {k} 次" for k in ks) + " | 隐藏 RMS 第 1 次 | 第 {} 次 |".format(n),
         "|---|---|" + "---|" * (len(ks) + 2)]
    groups = {}
    for r in R:
        groups.setdefault((r["model"], r["reset"]), []).append(r)
    for (mdl, rh), rs in groups.items():
        e = np.array([r["A_err"] for r in rs]).mean(0)
        c = np.array([r["A_cold"] for r in rs]).mean(0)
        h = np.array([r["A_hid"] for r in rs]).mean(0)
        hc = np.array([r["A_hid_cold"] for r in rs]).mean(0)
        L.append(f"| {mdl} | {'清零' if rh else '保留'} | " + " | ".join(f"{e[k - 1]:.1f} ({c[k - 1]:.1f})" for k in ks)
                 + f" | {h[0]:.2f} / {hc[0]:.2f} | {h[-1]:.2f} / {hc[-1]:.2f} |")
    L += ["", "## B. 闭环再生（100 回合，测试种子 100000 起；只算还没结束的回合）\n",
          "再生 vs 同观测同 z 的冷启动（括号里是两次冷启动之间的差距，即噪声底），按这是回合里第几次再生分组。\n",
          "| 模型 | 隐藏通道 | " + " | ".join(f"{lo}–{hi}" if hi < 99 else f"{lo}+" for lo, hi in B_bins)
          + " | 得分 | 成功率 | 动作粗糙度 |",
          "|---|---|" + "---|" * (len(B_bins) + 3)]
    for (mdl, rh), rs in groups.items():
        cells = []
        for lo, hi in B_bins:
            wc = [x for r in rs for k, v in enumerate(r["B_wc"], 1) if lo <= k <= hi for x in v]
            cc = [x for r in rs for k, v in enumerate(r["B_cc"], 1) if lo <= k <= hi for x in v]
            cells.append(f"{np.mean(wc):.1f} ({np.mean(cc):.1f})" if wc else "–")
        L.append(f"| {mdl} | {'清零' if rh else '保留'} | " + " | ".join(cells)
                 + f" | {P.fmt([r['score'] for r in rs], nd=3)} | {P.fmt([r['success'] for r in rs], True)}"
                 f" | {P.fmt([r['action_rough'] for r in rs])} |")
    L += ["", "## C. 离线（A，全部 n 次平均）和闭环（B，全部再生平均）对照\n",
          "照抄 = 旧计划的后 8 步。闭环里“照抄 vs 冷启动”是新观测要求的纠正量，“再生 vs 照抄”是再生实际做的纠正量。\n",
          "| 模型 | 隐藏通道 | 离线 再生 vs 冷启动 | 离线 冷启动 vs 冷启动 | 闭环 再生 vs 冷启动 | 闭环 冷启动 vs 冷启动"
          " | 闭环 照抄 vs 冷启动 | 闭环 再生 vs 照抄 |",
          "|---|---|---|---|---|---|---|---|"]
    flat = lambda rs, key: np.mean([x for r in rs for v in r[key] for x in v])
    for (mdl, rh), rs in groups.items():
        L.append(f"| {mdl} | {'清零' if rh else '保留'} | {np.mean([r['A_wc'] for r in rs]):.1f}"
                 f" | {np.mean([r['A_cc'] for r in rs]):.1f} | {flat(rs, 'B_wc'):.1f} | {flat(rs, 'B_cc'):.1f}"
                 f" | {flat(rs, 'B_pc'):.1f} | {flat(rs, 'B_wp'):.1f} |")
    text = "\n".join(L) + "\n"
    open(os.path.join(OUT, "chain.md"), "w", encoding="utf-8").write(text)
    print(text)


class BlendPlan(P.NCAPlan):
    """部分冷启动（类似 DP 热启动先给旧计划加噪声）：再生前把状态往冷启动的初始状态拉，
    动作通道 = α·旧计划 + (1−α)·当前位置，隐藏通道 × α。α = 1 是普通再生，α = 0 是只长 warm_steps 步的冷启动。"""

    def __init__(self, *args, alpha=1.0):
        super().__init__(*args)
        self.alpha = alpha

    def plan(self, obs):
        import torch
        with torch.no_grad():
            if self.s is None:
                self.s = self.m.grow(self.m.blank(obs), obs, self.z, self.cs)
                self.calls += self.cs
            else:
                s = self.alpha * P.A.shift(self.s, P.T_A) + (1.0 - self.alpha) * self.m.blank(obs)
                self.s = self.m.grow(s, obs, self.z, self.ws)
                self.calls += self.ws
            self.n += 1
            return P.A.plan_of(self.s)


BLEND_GRID = [(0.75, 10), (0.5, 10), (0.25, 10), (0.5, 20), (0.25, 20), (0.0, 16)]   # (α, 再生步数)


def blend_one(job):
    import torch
    model, s, alpha, ws = job
    torch.set_num_threads(1)
    path = os.path.join(BASE if model == "nca" else OUT, f"{model}_seed{s}", "nca.pt")
    pl = P.make_planner("nca", path, "warm")
    bp = BlendPlan(pl.m, "warm", pl.zd, 32, ws, alpha=alpha)
    return model, s, alpha, ws, P.rollout(bp, 100, s + 1)[0]


def stage_blend(a):
    """不重训：再生前部分冷启动，看能否在比冷启动少的步数下接近冷启动的成功率。结果在 runs/pusht_tune/blend.md。"""
    grid = [tuple(map(float, g.split(":"))) for g in a.blend] if a.blend else BLEND_GRID
    grid = [(al, int(ws)) for al, ws in grid]
    jobs = [(mdl, s, al, ws) for mdl in a.probe_models for al, ws in grid for s in a.seeds]
    jpath = os.path.join(OUT, "blend.json")
    R = {}
    if os.path.exists(jpath):                                           # 多次调用的结果累积
        for r in json.load(open(jpath, encoding="utf-8")):
            R[(r.pop("model"), r.pop("alpha"), r.pop("warm_steps"), r.pop("seed"))] = r
    t0 = time.time()
    with ProcessPoolExecutor(a.jobs, mp_context=mp.get_context("spawn")) as ex:
        for mdl, s, al, ws, m in ex.map(blend_one, jobs):
            R[(mdl, al, ws, s)] = m
            print(f"{mdl} α {al} warm {ws} seed {s}  success {m['success']:.2f}  score {m['score']:.3f}"
                  f"  ({time.time() - t0:.0f}s)", flush=True)
    json.dump([dict(model=k[0], alpha=k[1], warm_steps=k[2], seed=k[3], **v) for k, v in R.items()],
              open(jpath, "w", encoding="utf-8"), indent=1)
    L = [f"# 部分冷启动（不重训；种子 {','.join(map(str, a.seeds))}，每格 100 回合，z 每回合固定）\n",
         "再生前：动作通道 = α·旧计划 + (1−α)·当前位置，隐藏通道 × α。α = 1 是普通再生（见 summary.md），α = 0 是短冷启动。\n",
         "| 模型 | α | 再生步数 | 得分 | 成功率 | 动作粗糙度 | 衔接跳变 | NCA 步数 / 次规划 | 毫秒 / 次规划 |",
         "|---|---|---|---|---|---|---|---|---|"]
    for mdl in sorted({k[0] for k in R}):
        for al, ws in sorted({(k[1], k[2]) for k in R if k[0] == mdl}, key=lambda g: (g[1], -g[0])):
            if not all((mdl, al, ws, s) in R for s in a.seeds):
                continue
            r = [R[(mdl, al, ws, s)] for s in a.seeds]
            L.append(f"| {mdl} | {al:g} | {ws} | {P.fmt([x['score'] for x in r], nd=3)}"
                     f" | {P.fmt([x['success'] for x in r], True)} | {P.fmt([x['action_rough'] for x in r])}"
                     f" | {P.fmt([x['chunk_jump'] for x in r])} | {np.mean([x['calls_per_plan'] for x in r]):.1f}"
                     f" | {np.mean([x['ms_per_plan'] for x in r]):.0f} |")
    text = "\n".join(L) + "\n"
    open(os.path.join(OUT, "blend.md"), "w", encoding="utf-8").write(text)
    print(text)


def diag_one(job):
    """一个（模型, 种子）：闭环 100 回合（每次新 z / 新噪声）的逐回合最好覆盖率和失败回合最后 50 步的推杆移动；
    示范观测上每个观测采 8 个计划（换 z / 换噪声），看样本的散布和离示范动作块的距离。"""
    import torch
    kind, model, s, O, Y = job
    torch.set_num_threads(2)
    path = os.path.join(BASE if model in ("nca", "dp", "dp_small") else OUT, f"{model}_seed{s}",
                        "nca.pt" if kind == "nca" else "dp.pt")
    pl = P.make_planner(kind, path, "fresh")
    met, tr = P.rollout(pl, 100, s + 1)
    move = [float(np.linalg.norm(np.diff(q[-50:], axis=0), axis=1).sum()) for q in tr["agent"]]
    torch.manual_seed(s)
    S, n = 8, len(O)
    with torch.no_grad():
        o = torch.as_tensor(O).repeat_interleave(S, 0)
        if kind == "nca":
            z = torch.randn(n * S, pl.zd)
            p = P.A.plan_of(pl.m.grow(pl.m.blank(o), o, z, 32))
            same = P.A.plan_of(pl.m.grow(pl.m.blank(o), o, z, 32))         # 同 z 再长一次：只差随机异步更新
        else:
            x = torch.randn(n * S, 2, P.T_P)
            p = pl.p.denoise(x, o, pl.ts)
            same = pl.p.denoise(x, o, pl.ts)                                 # DDIM 是确定性的，应为 0
    p, same = p.view(n, S, P.T_P, 2), same.view(n, S, P.T_P, 2)
    y = torch.as_tensor(Y)[:, None]
    dist = lambda u, v: (u - v).norm(dim=-1).mean(-1) * 256                  # 16 步平均距离（像素）
    err = dist(p, y)                                                         # (n, S)
    return dict(model=model, seed=s, best=tr["best"].tolist(), success=tr["success"].tolist(), move=move,
                score=met["score"], spread=float(dist(p, p.mean(1, keepdim=True)).mean()),
                same=float(dist(p, same).mean()), err_mean=float(err.mean()), err_min=float(err.min(1).values.mean()))


def stage_diag(a):
    """诊断冷启动的上限：失败回合的形态（彻底失败还是差一点），以及采样的多模态程度（z 是否被用上）。"""
    O, Y, NXT = P.load_data()
    i = np.random.default_rng(0).choice(len(O), 512, replace=False)
    models = [("nca", m) for m in a.probe_models] + [("dp", "dp_small"), ("dp", "dp")]
    jobs = [(k, m, s, O[i], Y[i]) for k, m in models for s in a.seeds]
    R = []
    with ProcessPoolExecutor(a.jobs, mp_context=mp.get_context("spawn")) as ex:
        for r in ex.map(diag_one, jobs):
            R.append(r)
            print(f"{r['model']} seed {r['seed']}  score {r['score']:.3f}  spread {r['spread']:.1f}", flush=True)
    json.dump(R, open(os.path.join(OUT, f"diag{a.tag}.json"), "w", encoding="utf-8"))
    bins = [(0.0, 0.5), (0.5, 0.8), (0.8, 0.9), (0.9, 2.0)]
    L = [f"# 冷启动诊断（每次新 z / 新噪声；种子 {','.join(map(str, a.seeds))}，每个种子 100 回合）\n",
         "## 逐回合最好覆盖率的分布（占回合数的比例）\n",
         "失败 = 没有达到 0.95；“最后 50 步推杆移动”只算失败回合，太小说明卡住不动。\n",
         "| 模型 | 得分 | 成功 | 失败里 <0.5 | 0.5–0.8 | 0.8–0.9 | ≥0.9 | 失败回合最后 50 步推杆移动（像素） |",
         "|---|---|---|---|---|---|---|---|"]
    by = {}
    for r in R:
        by.setdefault(r["model"], []).append(r)
    for mdl, rs in by.items():
        best = np.concatenate([r["best"] for r in rs])
        succ = np.concatenate([r["success"] for r in rs]).astype(bool)
        mv = np.concatenate([r["move"] for r in rs])
        cells = [f"{np.mean(~succ & (best >= lo) & (best < hi)):.0%}" for lo, hi in bins]
        L.append(f"| {mdl} | {best.mean():.3f} | {succ.mean():.0%} | " + " | ".join(cells)
                 + f" | {np.median(mv[~succ]):.0f}（中位数） |")
    L += ["", "## 采样的多模态程度（示范数据上 512 个观测，每个采 8 个计划，16 步平均距离，像素）\n",
          "散布 = 样本到 8 个样本均值的距离；同 z 重算 = NCA 只差随机异步更新的距离（DP 的 DDIM 是确定性的）。"
          "离示范：8 个样本的平均误差 / 8 个里最好的误差，两者差得越多说明样本覆盖的模态越多。\n",
          "| 模型 | 散布 | 同 z 重算 | 离示范（平均） | 离示范（8 选 1 最好） |", "|---|---|---|---|---|"]
    for mdl, rs in by.items():
        f = lambda k: P.fmt([r[k] for r in rs], nd=1)
        L.append(f"| {mdl} | {f('spread')} | {f('same')} | {f('err_mean')} | {f('err_min')} |")
    text = "\n".join(L) + "\n"
    open(os.path.join(OUT, f"diag{a.tag}.md"), "w", encoding="utf-8").write(text)
    print(text)


class NoisyPlan(P.NCAPlan):
    """每次新 z 的冷启动，再给整个计划加一个随机平移（每个回合每次规划各抽一个，σ 像素）：检验随机性能否帮助跳出失败。"""

    def __init__(self, *args, sigma=0.0):
        super().__init__(*args)
        self.sigma = sigma

    def plan(self, obs):
        import torch
        p = super().plan(obs)
        return p + self.sigma / 256.0 * torch.randn(p.shape[0], 1, 2)


NOISE_GRID = [0.0, 4.0, 8.0, 16.0]


def noise_one(job):
    import torch
    model, s, sigma = job
    torch.set_num_threads(1)
    pl = P.make_planner("nca", os.path.join(BASE if model == "nca" else OUT, f"{model}_seed{s}", "nca.pt"), "fresh")
    npl = NoisyPlan(pl.m, "fresh", pl.zd, 32, 10, sigma=sigma)
    met, tr = P.rollout(npl, 100, s + 1)
    met["catastrophic"] = float(np.mean(tr["best"] < 0.5))
    return model, s, sigma, met


def stage_noise(a):
    jobs = [(mdl, s, sg) for mdl in a.probe_models for sg in NOISE_GRID for s in a.seeds]
    R = {}
    with ProcessPoolExecutor(a.jobs, mp_context=mp.get_context("spawn")) as ex:
        for mdl, s, sg, m in ex.map(noise_one, jobs):
            R[(mdl, sg, s)] = m
            print(f"{mdl} σ {sg} seed {s}  success {m['success']:.2f}  score {m['score']:.3f}"
                  f"  catastrophic {m['catastrophic']:.2f}", flush=True)
    json.dump([dict(model=k[0], sigma=k[1], seed=k[2], **v) for k, v in R.items()],
              open(os.path.join(OUT, "noise.json"), "w", encoding="utf-8"), indent=1)
    L = [f"# 冷启动计划加随机平移（不重训；每次新 z；种子 {','.join(map(str, a.seeds))}，每格 100 回合）\n",
         "| 模型 | σ（像素） | 得分 | 成功率 | 彻底失败（< 0.5） | 动作粗糙度 |", "|---|---|---|---|---|---|"]
    for mdl in a.probe_models:
        for sg in NOISE_GRID:
            r = [R[(mdl, sg, s)] for s in a.seeds]
            L.append(f"| {mdl} | {sg:g} | {P.fmt([x['score'] for x in r], nd=3)} | {P.fmt([x['success'] for x in r], True)}"
                     f" | {P.fmt([x['catastrophic'] for x in r], True)} | {P.fmt([x['action_rough'] for x in r])} |")
    text = "\n".join(L) + "\n"
    open(os.path.join(OUT, "noise.md"), "w", encoding="utf-8").write(text)
    print(text)


class StallPlan(P.NCAPlan):
    """每次新 z 的冷启动，两个针对“慢启动”的改动（不重训）：
    blank_sigma > 0：冷启动的初始状态在动作通道上加噪声（默认是所有细胞停在当前位置）；
    skip = True：上一个块执行完推杆位移 < 10 像素（卡住）的回合，这次执行计划的第 9–16 步而不是第 1–8 步。"""

    def __init__(self, *args, blank_sigma=0.0, skip=False):
        super().__init__(*args)
        self.bs, self.skip, self.last = blank_sigma, skip, None

    def plan(self, obs):
        import torch
        with torch.no_grad():
            self.z = self.zs * torch.randn(self.E, self.zd)
            s = self.m.blank(obs)
            if self.bs > 0:
                s[:, :2] += self.bs * torch.randn_like(s[:, :2])
            self.s = self.m.grow(s, obs, self.z, self.cs)
            self.calls += self.cs
            self.n += 1
            p = P.A.plan_of(self.s)
            pos = obs[:, :2]
            if self.skip and self.last is not None:
                stuck = ((pos - self.last).norm(dim=1) * 256 < 10).float()[:, None, None]
                later = torch.cat([p[:, P.T_A:], p[:, -1:].expand(-1, P.T_A, -1)], 1)
                p = stuck * later + (1 - stuck) * p
            self.last = pos.clone()
            return p


STALL_GRID = [("control", 0.0, False), ("blank0.02", 0.02, False), ("blank0.05", 0.05, False),
              ("blank0.1", 0.1, False), ("skip", 0.0, True), ("blank0.05+skip", 0.05, True)]


def stall_one(job):
    import torch
    model, s, name, bs, skip = job
    torch.set_num_threads(1)
    pl = P.make_planner("nca", os.path.join(BASE if model == "nca" else OUT, f"{model}_seed{s}", "nca.pt"), "fresh")
    sp = StallPlan(pl.m, "fresh", pl.zd, 32, 10, blank_sigma=bs, skip=skip)
    met, tr = P.rollout(sp, 100, s + 1)
    met["catastrophic"] = float(np.mean(tr["best"] < 0.5))
    prof = None
    if name == "control":                                              # 慢启动剖面：卡住时的计划 vs 示范状态上的计划
        prof = slow_profile(pl, tr)
    return model, s, name, met, prof


def obs_at(tr, e, u):
    """回合 e 第 u 步的 12 维观测（当前帧和上一帧；u = 0 时上一帧就是当前帧，和 rollout 一致）。"""
    ag, bl, v = tr["agent"][e], tr["block"][e], max(u - 1, 0)
    return np.concatenate([P.feat(np.concatenate([ag[u], bl[u]])), P.feat(np.concatenate([ag[v], bl[v]]))])


def first_stall(ag):
    """推杆轨迹 ag 里第一次卡住（之后 8、16 步推杆位移都 < 10 像素）的规划时刻 t（8 的倍数）；没有卡住时为 None。"""
    for t in range(8, len(ag) - 16, 8):
        if np.linalg.norm(ag[t + 16] - ag[t]) < 10 and np.linalg.norm(ag[t + 8] - ag[t]) < 10:
            return t
    return None


def stall_obs(tr):
    """彻底失败回合里第一次卡住的时刻 t（first_stall），取 t、t+8、t+16 三个观测。"""
    obs = []
    for e in np.flatnonzero(tr["best"] < 0.5):
        ag = tr["agent"][e]
        t = first_stall(ag)
        if t is not None:
            obs += [obs_at(tr, e, u) for u in (t, t + 8, t + 16) if u < len(ag) - 1]
    return obs


def slow_profile(pl, tr):
    """卡住时（stall_obs）的观测上，NCA 计划第 k 步离当前位置多远。"""
    import torch
    obs = stall_obs(tr)
    if not obs:
        return None
    o = torch.as_tensor(np.array(obs, np.float32))
    with torch.no_grad():
        p = P.A.plan_of(pl.m.grow(pl.m.blank(o), o, torch.randn(len(o), pl.zd), 32))
    return dict(n=len(o), dist=np.median(((p - o[:, None, :2]).norm(dim=-1) * 256).numpy(), 0).tolist())


def stage_stall(a):
    """“慢启动”：分布外状态上 NCA 计划的前 8 步几乎不动，动作挤在第 9–16 步，而每次只执行前 8 步 → 卡住。
    不重训地试两种改法（STALL_GRID），结果在 runs/pusht_tune/stall.md。"""
    jobs = [(mdl, s, *g) for mdl in a.probe_models for g in STALL_GRID for s in a.seeds]
    R, PR = {}, {}
    with ProcessPoolExecutor(a.jobs, mp_context=mp.get_context("spawn")) as ex:
        for mdl, s, name, m, prof in ex.map(stall_one, jobs):
            R[(mdl, name, s)] = m
            if prof:
                PR[(mdl, s)] = prof
            print(f"{mdl} {name} seed {s}  success {m['success']:.2f}  score {m['score']:.3f}"
                  f"  catastrophic {m['catastrophic']:.2f}", flush=True)
    json.dump(dict(evals=[dict(model=k[0], setting=k[1], seed=k[2], **v) for k, v in R.items()],
                   profiles=[dict(model=k[0], seed=k[1], **v) for k, v in PR.items()]),
              open(os.path.join(OUT, "stall.json"), "w", encoding="utf-8"), indent=1)
    ks = [0, 1, 3, 5, 7, 11, 15]
    L = [f"# 慢启动（种子 {','.join(map(str, a.seeds))}；每次新 z）\n",
         "## 卡住时 NCA 计划第 k 步离当前推杆位置的距离（像素，中位数）\n",
         "卡住 = 彻底失败（最好覆盖率 < 0.5）的回合里，之后 8 步和 16 步推杆位移都 < 10 像素的规划时刻。"
         "参照：示范状态上 NCA 和示范目标第 1、8、16 步约 18、66、99 像素。\n",
         "| 模型 | 种子 | 状态数 | " + " | ".join(f"第 {k + 1} 步" for k in ks) + " |", "|---|---|---|" + "---|" * len(ks)]
    for (mdl, s), pr in sorted(PR.items()):
        L.append(f"| {mdl} | {s} | {pr['n']} | " + " | ".join(f"{pr['dist'][k]:.1f}" for k in ks) + " |")
    L += ["", "## 不重训的改法（每格 100 回合）\n",
          "blankσ = 冷启动初始状态的动作通道加 N(0, σ²) 噪声（归一化单位，0.05 ≈ 13 像素）；"
          "skip = 上一个块推杆位移 < 10 像素时执行计划的第 9–16 步。\n",
          "| 模型 | 设置 | 得分 | 成功率 | 彻底失败（< 0.5） | 动作粗糙度 |", "|---|---|---|---|---|---|"]
    for mdl in a.probe_models:
        for name, _, _ in STALL_GRID:
            r = [R[(mdl, name, s)] for s in a.seeds]
            L.append(f"| {mdl} | {name} | {P.fmt([x['score'] for x in r], nd=3)} | {P.fmt([x['success'] for x in r], True)}"
                     f" | {P.fmt([x['catastrophic'] for x in r], True)} | {P.fmt([x['action_rough'] for x in r])} |")
    text = "\n".join(L) + "\n"
    open(os.path.join(OUT, "stall.md"), "w", encoding="utf-8").write(text)
    print(text)


class EnsemblePlan:
    """几个模型（同一变体的不同训练种子）各自冷启动，计划取平均。诊断用：彻底失败多半只在某一个种子上出现。"""

    def __init__(self, planners):
        self.ps = planners

    def reset(self, E):
        for p in self.ps:
            p.reset(E)
        self.calls = 0

    def plan(self, obs):
        out = sum(p.plan(obs) for p in self.ps) / len(self.ps)
        self.calls = sum(p.calls for p in self.ps)
        return out


def ens_one(job):
    import torch
    model, members, s = job
    torch.set_num_threads(1)
    pls = [P.make_planner("nca", os.path.join(BASE if model == "nca" else OUT, f"{model}_seed{m}", "nca.pt"), "fresh")
           for m in members]
    met, tr = P.rollout(EnsemblePlan(pls), 100, s + 1)
    met["catastrophic"] = float(np.mean(tr["best"] < 0.5))
    return model, tuple(members), s, met


def stage_ens(a):
    """不重训：同一变体的 2 个或 3 个训练种子集成（计划取平均），每个评估种子 100 回合。结果在 runs/pusht_tune/ens.md。"""
    sets = [(0, 1), (0, 2), (1, 2), (0, 1, 2)]
    jobs = [(mdl, list(ms), s) for mdl in a.probe_models for ms in sets for s in a.seeds]
    R = {}
    with ProcessPoolExecutor(a.jobs, mp_context=mp.get_context("spawn")) as ex:
        for mdl, ms, s, m in ex.map(ens_one, jobs):
            R[(mdl, ms, s)] = m
            print(f"{mdl} {ms} eval-seed {s}  success {m['success']:.2f}  score {m['score']:.3f}"
                  f"  catastrophic {m['catastrophic']:.2f}", flush=True)
    json.dump([dict(model=k[0], members=list(k[1]), seed=k[2], **v) for k, v in R.items()],
              open(os.path.join(OUT, "ens.json"), "w", encoding="utf-8"), indent=1)
    L = [f"# 训练种子集成（不重训；每次新 z；计划取平均；评估种子 {','.join(map(str, a.seeds))}，每格 100 回合）\n",
         "单个模型的对照见 stall.md 的 control 行（3 个模型各自评估）。\n",
         "| 模型 | 集成的训练种子 | 得分 | 成功率 | 彻底失败（< 0.5） | 动作粗糙度 | 毫秒 / 次规划 |", "|---|---|---|---|---|---|---|"]
    for mdl in a.probe_models:
        for ms in sets:
            r = [R[(mdl, ms, s)] for s in a.seeds]
            L.append(f"| {mdl} | {','.join(map(str, ms))} | {P.fmt([x['score'] for x in r], nd=3)}"
                     f" | {P.fmt([x['success'] for x in r], True)} | {P.fmt([x['catastrophic'] for x in r], True)}"
                     f" | {P.fmt([x['action_rough'] for x in r])} | {np.mean([x['ms_per_plan'] for x in r]):.0f} |")
    text = "\n".join(L) + "\n"
    open(os.path.join(OUT, "ens.md"), "w", encoding="utf-8").write(text)
    print(text)


def stage_cat(a):
    """所有已训练的 NCA（基线 + 调参变体，有几个种子算几个）每次新 z 闭环 100 回合：彻底失败率和卡住时的慢启动剖面。
    结果在 runs/pusht_tune/cat.md（合并已有的 cat.json，只补没算过的）。"""
    jpath = os.path.join(OUT, "cat.json")
    R = {tuple(k.split("/")): v for k, v in json.load(open(jpath, encoding="utf-8")).items()} if os.path.exists(jpath) else {}
    models = ["nca"] + [v for v in TRAIN_VARIANTS if model_kind(v) == "nca"]
    jobs = []
    for mdl in models:
        for s in a.seeds:
            if (mdl, str(s)) in R:
                continue
            if os.path.exists(os.path.join(BASE if mdl == "nca" else OUT, f"{mdl}_seed{s}", "metrics.json")):
                jobs.append((mdl, s, "control", 0.0, False))
    with ProcessPoolExecutor(a.jobs, mp_context=mp.get_context("spawn")) as ex:
        for mdl, s, _, m, prof in ex.map(stall_one, jobs):
            R[(mdl, str(s))] = dict(m, prof=prof)
            print(f"{mdl} seed {s}  success {m['success']:.2f}  score {m['score']:.3f}  catastrophic {m['catastrophic']:.2f}",
                  flush=True)
    json.dump({f"{k[0]}/{k[1]}": v for k, v in R.items()}, open(jpath, "w", encoding="utf-8"), indent=1)
    L = ["# 彻底失败率（每次新 z；每个种子 100 回合，测试种子 100000 起）\n",
         "彻底失败 = 最好覆盖率 < 0.5。卡住时第 8 步 = 彻底失败回合里卡住的规划时刻，NCA 计划第 8 步离当前推杆位置的距离"
         "（中位数，像素；示范状态上约 66）。\n",
         "| 模型 | 种子 | 得分 | 成功率 | 彻底失败 | 卡住的状态数 | 卡住时第 8 步 |", "|---|---|---|---|---|---|---|"]
    for mdl in models:
        r = [(s, R[(mdl, s)]) for s in sorted({k[1] for k in R if k[0] == mdl})]
        if not r:
            continue
        pr = [x["prof"] for _, x in r if x.get("prof")]
        L.append(f"| {mdl} | {','.join(s for s, _ in r)} | {P.fmt([x['score'] for _, x in r], nd=3)}"
                 f" | {P.fmt([x['success'] for _, x in r], True)} | {P.fmt([x['catastrophic'] for _, x in r], True)}"
                 f" | {sum(p['n'] for p in pr)} | {np.median([p['dist'][7] for p in pr]):.1f} |" if pr else
                 f"| {mdl} | {','.join(s for s, _ in r)} | {P.fmt([x['score'] for _, x in r], nd=3)}"
                 f" | {P.fmt([x['success'] for _, x in r], True)} | {P.fmt([x['catastrophic'] for _, x in r], True)} | 0 | – |")
    text = "\n".join(L) + "\n"
    open(os.path.join(OUT, "cat.md"), "w", encoding="utf-8").write(text)
    print(text)


ATTRACT_STEPS = (4, 8, 16, 32)


def attract_one(job):
    """一个（模型, 种子）：每次新 z 闭环 100 回合，取卡住的状态（stall_obs）和成功回合里随机的状态（每回合 1 个，最多 30 个）。
    在这些观测上，NCA 分别从默认冷启动（所有细胞停在当前位置）和同种子大 DP 的计划（隐藏通道为 0）出发长 32 步，
    记录长 4、8、16、32 步后计划第 8 步离当前推杆位置的距离（中位数，像素）；参照：同种子大 DP、小 DP 自己的计划第 8 步，
    以及这些观测离最近的示范观测多远（12 维特征的欧氏距离，中位数）。"""
    import torch
    model, s = job
    torch.set_num_threads(2)
    pl = P.make_planner("nca", os.path.join(BASE if model == "nca" else OUT, f"{model}_seed{s}", "nca.pt"), "fresh")
    dp = P.make_planner("dp", os.path.join(BASE, f"dp_seed{s}", "dp.pt"), "fresh")
    dps = P.make_planner("dp", os.path.join(BASE, f"dp_small_seed{s}", "dp.pt"), "fresh")
    demo = torch.as_tensor(P.load_data()[0])
    met, tr = P.rollout(pl, 100, s + 1)
    ok = np.flatnonzero(tr["success"])[:30]
    rng = np.random.default_rng(s)
    normal = [obs_at(tr, e, int(rng.integers(1, len(tr["agent"][e]) - 1))) for e in ok]
    out = dict(catastrophic=float(np.mean(tr["best"] < 0.5)))
    for label, obs in (("stall", stall_obs(tr)), ("normal", normal)):
        if not obs:
            continue
        o = torch.as_tensor(np.array(obs, np.float32))
        torch.manual_seed(s)
        d8 = lambda p: float(np.median(np.linalg.norm(p[:, 7] - o[:, :2].numpy(), axis=1) * 256))
        prof = lambda p: np.median(np.linalg.norm(p - o[:, None, :2].numpy(), axis=2) * 256, axis=0).tolist()  # 第 1–16 步
        with torch.no_grad():
            z = torch.randn(len(o), pl.zd)
            dpp = dp.p.denoise(torch.randn(len(o), 2, P.T_P), o, dp.ts)              # (n, T_P, 2)
            dsp = dps.p.denoise(torch.randn(len(o), 2, P.T_P), o, dps.ts).numpy()
            r = dict(n=len(o), dp=d8(dpp.numpy()), dp_small=d8(dsp), nn=float(torch.cdist(o, demo).min(1).values.median()),
                     prof_dp=prof(dpp.numpy()), prof_dp_small=prof(dsp))
            s_dp = pl.m.blank(o)
            s_dp[:, :2] = dpp.transpose(1, 2)
            for name, s0 in (("blank", pl.m.blank(o)), ("from_dp", s_dp)):
                fr = []
                pl.m.grow(s0, o, z, ATTRACT_STEPS[-1], frames=fr)
                r[name] = [d8(fr[k - 1]) for k in ATTRACT_STEPS]
                r[name + "_curve"] = [d8(f) for f in fr]                              # 每长一步后的第 8 步距离
                if name == "blank":
                    r["prof_nca"] = prof(fr[-1])                                       # 默认冷启动 32 步的计划剖面
        out[label] = r
    return model, s, out


def stage_attract(a):
    """吸引子检验（--probe_models 里的模型，--seeds 里的种子；结果累积在 runs/pusht_tune/attract.json / .md）：
    卡住的状态上，从 DP 的好计划出发，NCA 是否也会把前 8 步拉回原地。"""
    jpath = os.path.join(OUT, "attract.json")
    R = {tuple(k.split("/")): v for k, v in json.load(open(jpath, encoding="utf-8")).items()} if os.path.exists(jpath) else {}
    jobs = [(m, s) for m in a.probe_models for s in a.seeds if (m, str(s)) not in R
            and os.path.exists(os.path.join(BASE if m == "nca" else OUT, f"{m}_seed{s}", "metrics.json"))]   # 训练完的才算
    with ProcessPoolExecutor(a.jobs, mp_context=mp.get_context("spawn")) as ex:
        for m, s, out in ex.map(attract_one, jobs):
            R[(m, str(s))] = out
            print(f"{m} seed {s}  catastrophic {out['catastrophic']:.2f}  "
                  + "  ".join(f"{k}: n {v['n']} nn {v['nn']:.3f} dp {v['dp']:.0f} dp_small {v['dp_small']:.0f}"
                              f" blank {v['blank']} from_dp {v['from_dp']}"
                              for k, v in out.items() if k != "catastrophic"), flush=True)
    json.dump({f"{k[0]}/{k[1]}": v for k, v in R.items()}, open(jpath, "w", encoding="utf-8"), indent=1)
    steps = "/".join(map(str, ATTRACT_STEPS))
    L = ["# 吸引子检验（每次新 z 闭环 100 回合，测试种子 100000 起）\n",
         "卡住 = 彻底失败回合里第一次卡住的时刻及其后 8、16 步（同 stall.md）；正常 = 成功回合里随机的时刻（每回合 1 个，最多 30 个）。"
         f"后两栏是 NCA 长 {steps} 步后计划第 8 步离当前推杆位置的距离（中位数，像素；示范状态上约 66）："
         "从停在原地出发 = 默认冷启动；从 DP 计划出发 = 动作通道换成同种子大 DP 在同一观测上的计划，隐藏通道为 0。"
         "DP、小 DP 第 8 步 = 它们自己在同一观测上的计划。离示范 = 到最近的示范观测的距离（12 维特征，中位数）。\n",
         "| 模型 | 种子 | 状态 | 状态数 | 离示范 | DP 第 8 步 | 小 DP 第 8 步 | 从停在原地出发 | 从 DP 计划出发 |",
         "|---|---|---|---|---|---|---|---|---|"]
    for (m, s), out in sorted(R.items()):
        for label in ("stall", "normal"):
            if label in out:
                v = out[label]
                L.append(f"| {m} | {s} | {'卡住' if label == 'stall' else '正常'} | {v['n']} | {v['nn']:.3f} | {v['dp']:.0f}"
                         f" | {v['dp_small']:.0f} | {'/'.join(f'{x:.0f}' for x in v['blank'])}"
                         f" | {'/'.join(f'{x:.0f}' for x in v['from_dp'])} |")
    text = "\n".join(L) + "\n"
    open(os.path.join(OUT, "attract.md"), "w", encoding="utf-8").write(text)
    print(text)


COLD_GRID = (8, 12, 16, 20, 24, 32)


def steps_one(job):
    import torch
    model, s, cold = job
    torch.set_num_threads(1)
    pl = P.make_planner("nca", os.path.join(BASE if model == "nca" else OUT, f"{model}_seed{s}", "nca.pt"), "fresh")
    met, tr = P.rollout(P.NCAPlan(pl.m, "fresh", pl.zd, cold, 10), 100, s + 1)
    met["catastrophic"] = float(np.mean(tr["best"] < 0.5))
    return model, s, cold, met


def stage_steps(a):
    """不重训：每次新 z 冷启动的生长步数（COLD_GRID）。吸引子检验（attract.md）里，卡住的状态上计划先走出去、
    长得越久越缩回原地；这里看少长几步能否减少彻底失败。结果累积在 runs/pusht_tune/steps.json / .md。"""
    jpath = os.path.join(OUT, "steps.json")
    R = {tuple(k.split("/")): v for k, v in json.load(open(jpath, encoding="utf-8")).items()} if os.path.exists(jpath) else {}
    jobs = [(m, s, c) for m in a.probe_models for c in COLD_GRID for s in a.seeds if (m, str(c), str(s)) not in R
            and os.path.exists(os.path.join(BASE if m == "nca" else OUT, f"{m}_seed{s}", "metrics.json"))]
    with ProcessPoolExecutor(a.jobs, mp_context=mp.get_context("spawn")) as ex:
        for m, s, c, met in ex.map(steps_one, jobs):
            R[(m, str(c), str(s))] = met
            print(f"{m} cold {c} seed {s}  success {met['success']:.2f}  score {met['score']:.3f}"
                  f"  catastrophic {met['catastrophic']:.2f}", flush=True)
    json.dump({"/".join(k): v for k, v in R.items()}, open(jpath, "w", encoding="utf-8"), indent=1)
    L = ["# 冷启动生长步数（不重训；每次新 z；每个种子 100 回合，测试种子 100000 起）\n",
         "训练时冷启动长 24–40 步，默认评估 32 步。彻底失败 = 最好覆盖率 < 0.5。\n",
         "| 模型 | 种子 | 冷启动步数 | 得分 | 成功率 | 彻底失败 | 动作粗糙度 | 毫秒 / 次规划 |", "|---|---|---|---|---|---|---|---|"]
    for m in dict.fromkeys(k[0] for k in R):
        for c in COLD_GRID:
            seeds = sorted(k[2] for k in R if k[0] == m and k[1] == str(c))
            if not seeds:
                continue
            r = [R[(m, str(c), s)] for s in seeds]
            L.append(f"| {m} | {','.join(seeds)} | {c} | {P.fmt([x['score'] for x in r], nd=3)}"
                     f" | {P.fmt([x['success'] for x in r], True)} | {P.fmt([x['catastrophic'] for x in r], True)}"
                     f" | {P.fmt([x['action_rough'] for x in r])} | {np.mean([x['ms_per_plan'] for x in r]):.0f} |")
    text = "\n".join(L) + "\n"
    open(os.path.join(OUT, "steps.md"), "w", encoding="utf-8").write(text)
    print(text)


FINAL_MODES = {"dp_small": ("fresh", "warm"), "dp": ("fresh", "fixed"),   # 各自最好的两种执行方式；NCA 只比冷启动
               "dp_small_dag": ("fresh", "warm"), "dp_small_dart": ("fresh", "warm"), "dp_small_rescue": ("fresh", "warm"),
               "dp_small_rescue2": ("fresh", "warm")}


def final_one(job):
    import torch
    model, mode, s, n = job
    torch.set_num_threads(1)
    kind = model_kind(model)
    root = BASE if model in ("nca", "dp", "dp_small") else OUT
    pl = P.make_planner(kind, os.path.join(root, f"{model}_seed{s}", f"{kind}.pt"), mode)
    met, tr = P.rollout(pl, n, s + 1)
    best, succ = np.asarray(tr["best"]), np.asarray(tr["success"]).astype(bool)
    met.update(catastrophic=float(np.mean(best < 0.5)), near=float(np.mean((best >= 0.9) & ~succ)), episodes=n)
    return model, mode, s, met


def stage_final(a):
    """最终对比：--probe_models 里的模型，每个种子 --episodes 个测试回合（默认 500，测试种子 100000 起）。
    NCA 用每次新 z 的冷启动，小 DP 用每次新噪声和热启动，DP 用每次新噪声和固定噪声。结果累积在 runs/pusht_tune/final.json / .md。"""
    jpath = os.path.join(OUT, "final.json")
    R = {tuple(k.split("/")): v for k, v in json.load(open(jpath, encoding="utf-8")).items()} if os.path.exists(jpath) else {}
    jobs = []
    for m in a.probe_models:
        root = BASE if m in ("nca", "dp", "dp_small") else OUT
        for mode in FINAL_MODES.get(m, ("fresh",)):
            jobs += [(m, mode, s, a.episodes) for s in a.seeds if (m, mode, str(s)) not in R
                     and os.path.exists(os.path.join(root, f"{m}_seed{s}", "metrics.json"))]
    with ProcessPoolExecutor(a.jobs, mp_context=mp.get_context("spawn")) as ex:
        for m, mode, s, met in ex.map(final_one, jobs):
            R[(m, mode, str(s))] = met
            print(f"{m} {mode} seed {s}  score {met['score']:.3f}  success {met['success']:.3f}"
                  f"  catastrophic {met['catastrophic']:.3f}", flush=True)
    json.dump({"/".join(k): v for k, v in R.items()}, open(jpath, "w", encoding="utf-8"), indent=1)
    L = ["# 最终对比（每个种子 500 个测试回合，测试种子 100000–100499；均值 ± 种子间标准差）\n",
         "彻底失败 = 最好覆盖率 < 0.5；差一点 = 最好覆盖率 ≥ 0.9 但没有成功。毫秒数是 500 个环境一批、有负载的服务器上测的，只看量级。\n",
         "| 模型 | 执行方式 | 种子 | 得分 | 成功率 | 彻底失败 | 差一点 | 动作粗糙度 | 毫秒 / 次规划 |", "|---|---|---|---|---|---|---|---|---|"]
    for m, mode in dict.fromkeys((k[0], k[1]) for k in R):
        seeds = sorted(k[2] for k in R if k[:2] == (m, mode))
        r = [R[(m, mode, s)] for s in seeds]
        L.append(f"| {m} | {mode} | {','.join(seeds)} | {P.fmt([x['score'] for x in r], nd=3)}"
                 f" | {P.fmt([x['success'] for x in r], True)} | {P.fmt([x['catastrophic'] for x in r], True)}"
                 f" | {P.fmt([x['near'] for x in r], True)} | {P.fmt([x['action_rough'] for x in r])}"
                 f" | {np.mean([x['ms_per_plan'] for x in r]):.0f} |")
    text = "\n".join(L) + "\n"
    open(os.path.join(OUT, "final.md"), "w", encoding="utf-8").write(text)
    print(text)


def dagger_one(job):
    """一个（学生, 种子）：学生每次新 z / 新噪声闭环跑 n 个回合（环境种子 DAGGER_SEED 起），每次规划时的观测
    用同种子大 DP 每次新噪声的计划做标签，存到 runs/pusht_tune/dagger/<学生>_seed<s>.npz（O、Y，以及诊断用的
    回合号 ep、时刻 t、所在回合是否彻底失败 cat）。"""
    import torch
    student, s, n = job
    torch.set_num_threads(2)
    kind = model_kind(student)
    root = BASE if student in ("nca", "dp", "dp_small") else OUT
    pl = P.make_planner(kind, os.path.join(root, f"{student}_seed{s}", f"{kind}.pt"), "fresh")
    dp = P.make_planner("dp", os.path.join(BASE, f"dp_seed{s}", "dp.pt"), "fresh")
    met, tr = P.rollout(pl, n, s + 1, start=DAGGER_SEED)
    ep = np.array([e for e in range(n) for _ in range(0, len(tr["agent"][e]) - 1, P.T_A)])
    t = np.array([u for e in range(n) for u in range(0, len(tr["agent"][e]) - 1, P.T_A)])
    O = np.array([obs_at(tr, e, u) for e, u in zip(ep, t)], np.float32)
    chunks = lambda x, k: np.array_split(x, -(-len(x) // k))
    torch.manual_seed(s)
    with torch.no_grad():
        Y = np.concatenate([dp.p.denoise(torch.randn(len(c), 2, P.T_P), torch.as_tensor(c), dp.ts).numpy()
                            for c in chunks(O, 1000)]).astype(np.float32)        # (n, T_P, 2)，归一化，和示范的 Y 一样
    cat = tr["best"][ep] < 0.5
    os.makedirs(os.path.join(OUT, "dagger"), exist_ok=True)
    np.savez_compressed(os.path.join(OUT, "dagger", f"{student}_seed{s}.npz"), O=O, Y=Y, ep=ep, t=t, cat=cat)
    # 诊断：学生实际执行的第 8 个动作、老师计划的第 8 步，离当时推杆位置多远（像素）；离最近的示范观测多远
    ag = np.array([tr["agent"][e][u] for e, u in zip(ep, t)])
    own = np.array([np.linalg.norm(tr["acts"][e][u + 7] - tr["agent"][e][u]) if u + 7 < len(tr["acts"][e]) else np.nan
                    for e, u in zip(ep, t)])
    lab = np.linalg.norm((Y[:, 7] + 1.0) * 256.0 - ag, axis=1)
    demo = torch.as_tensor(P.load_data()[0])
    nn_ = np.concatenate([torch.cdist(torch.as_tensor(c), demo).min(1).values.numpy() for c in chunks(O, 1000)])
    out = dict(met, episodes=n, catastrophic=float(np.mean(tr["best"] < 0.5)), n=len(O), n_cat=int(cat.sum()))
    for k, m in (("cat", cat), ("ok", ~cat)):
        if m.any():
            out[k] = dict(own8=float(np.nanmedian(own[m])), label8=float(np.median(lab[m])), nn=float(np.median(nn_[m])))
    return student, s, out


def stage_dagger(a):
    """DAgger 数据：--probe_models 里的学生，--seeds 里的种子，每个 --dagger_episodes 个回合。已有的 npz 不重做
    （避免覆盖训练用过的数据）；诊断累积在 runs/pusht_tune/dagger.json / .md。"""
    jpath = os.path.join(OUT, "dagger.json")
    R = {tuple(k.split("/")): v for k, v in json.load(open(jpath, encoding="utf-8")).items()} if os.path.exists(jpath) else {}
    jobs = [(m, s, a.dagger_episodes) for m in a.probe_models for s in a.seeds
            if not os.path.exists(os.path.join(OUT, "dagger", f"{m}_seed{s}.npz"))]
    with ProcessPoolExecutor(a.jobs, mp_context=mp.get_context("spawn")) as ex:
        for m, s, out in ex.map(dagger_one, jobs):
            R[(m, str(s))] = out
            print(f"{m} seed {s}  success {out['success']:.3f}  catastrophic {out['catastrophic']:.3f}  samples {out['n']}"
                  f" (catastrophic episodes {out['n_cat']})  "
                  + "  ".join(f"{k}: own8 {v['own8']:.0f} label8 {v['label8']:.0f} nn {v['nn']:.3f}"
                              for k, v in out.items() if k in ("cat", "ok")), flush=True)
    json.dump({f"{k[0]}/{k[1]}": v for k, v in R.items()}, open(jpath, "w", encoding="utf-8"), indent=1)
    cell = lambda x: f"{x['own8']:.0f} / {x['label8']:.0f} | {x['nn']:.3f}" if x else "– | –"
    L = [f"# DAgger 数据（学生每次新 z / 新噪声闭环，环境种子 {DAGGER_SEED} 起；标签 = 同种子大 DP 每次新噪声的计划）\n",
         "样本 = 每次规划时的观测（每 8 步一个）。第 8 步 = 离当时推杆位置的距离（中位数，像素）：学生 = 它实际执行的第 8 个动作，"
         "老师 = DP 标签的第 8 步。离示范 = 到最近的示范观测的距离（12 维特征，中位数）。示范共 25650 个样本。\n",
         "| 学生 | 种子 | 回合 | 得分 | 成功率 | 彻底失败 | 样本 | 其中来自彻底失败回合 | 彻底失败回合：学生 / 老师第 8 步 | 离示范"
         " | 其余回合：学生 / 老师第 8 步 | 离示范 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for (m, s), v in sorted(R.items()):
        L.append(f"| {m} | {s} | {v['episodes']} | {v['score']:.3f} | {100 * v['success']:.1f}% | {100 * v['catastrophic']:.1f}%"
                 f" | {v['n']} | {v['n_cat']} | {cell(v.get('cat'))} | {cell(v.get('ok'))} |")
    text = "\n".join(L) + "\n"
    open(os.path.join(OUT, "dagger.md"), "w", encoding="utf-8").write(text)
    print(text)


class DartPlan:
    """DART：老师（大 DP，每次新噪声）照常规划，执行前给动作块加扰动；扰动前的计划记在 clean 里作标签。
    每个回合抽一个平移 σ（DART_SIGMAS）和一个放慢概率（DART_SLOW）。"""

    def __init__(self, inner, rng):
        self.inner, self.rng = inner, rng

    @property
    def calls(self):
        return self.inner.calls

    def reset(self, E):
        self.inner.reset(E)
        self.sigma, self.slow, self.clean = self.rng.choice(DART_SIGMAS, E), self.rng.choice(DART_SLOW, E), []

    def plan(self, obs):
        import torch
        p = self.inner.plan(obs)                                         # (E, T_P, 2)，归一化坐标
        self.clean.append(p.numpy().copy())
        E, cur = len(p), obs[:, None, :2]
        slow = torch.as_tensor(self.rng.random(E) < self.slow, dtype=p.dtype)[:, None, None]
        p = cur + (1 - slow * (1 - DART_SLOW_SCALE)) * (p - cur)
        return p + torch.as_tensor(self.rng.normal(size=(E, 1, 2)) * self.sigma[:, None, None] / 256.0, dtype=p.dtype)


def stall_states_of(d):
    """DAgger 数据（O、ep、t、cat；每 8 步一个观测）里每个彻底失败回合第一次卡住的时刻及其后 8、16 步的观测（规则同 first_stall）。"""
    out = []
    for e in np.unique(d["ep"][d["cat"]]):
        idx = np.flatnonzero(d["ep"] == e)
        idx = idx[np.argsort(d["t"][idx])]
        ag = (d["O"][idx, :2] + 1.0) * 256.0
        for k in range(1, len(ag) - 2):
            if np.linalg.norm(ag[k + 2] - ag[k]) < 10 and np.linalg.norm(ag[k + 1] - ag[k]) < 10:
                out += [d["O"][idx[j]] for j in (k, k + 1, k + 2)]
                break
    return np.array(out, np.float32)


def dart_one(job):
    """一个种子：老师（同种子大 DP）加扰动闭环 n 个回合（环境种子 DART_SEED 起），每次规划时的观测和扰动前的计划存到
    runs/pusht_tune/dart/dp_seed<s>.npz（O、Y，以及诊断用的回合号 ep、时刻 t、离示范 nn、是否远 far、该回合的 σ 和放慢概率）。
    诊断：老师在扰动下的成绩；这些状态离示范多远；NCA（se_global）在 DAgger 数据里卡住的状态，离示范和离示范 + DART 数据多远。"""
    import torch
    s, n = job
    torch.set_num_threads(2)
    pl = DartPlan(P.make_planner("dp", os.path.join(BASE, f"dp_seed{s}", "dp.pt"), "fresh"), np.random.default_rng(s))
    met, tr = P.rollout(pl, n, s + 1, start=DART_SEED)
    ep = np.array([e for e in range(n) for _ in range(0, len(tr["agent"][e]) - 1, P.T_A)])
    t = np.array([u for e in range(n) for u in range(0, len(tr["agent"][e]) - 1, P.T_A)])
    O = np.array([obs_at(tr, e, u) for e, u in zip(ep, t)], np.float32)
    Y = np.array([pl.clean[u // P.T_A][e] for e, u in zip(ep, t)], np.float32)   # 扰动前的计划，(n, T_P, 2) 归一化
    demo = torch.as_tensor(P.load_data()[0])
    chunks = lambda x, k: np.array_split(x, -(-len(x) // k))
    nn_ = lambda q, ref: np.concatenate([torch.cdist(torch.as_tensor(c), ref).min(1).values.numpy() for c in chunks(q, 2000)])
    nn_demo = nn_(O, demo)
    far = nn_demo >= DART_FAR
    os.makedirs(os.path.join(OUT, "dart"), exist_ok=True)
    np.savez_compressed(os.path.join(OUT, "dart", f"dp_seed{s}.npz"), O=O, Y=Y, ep=ep, t=t, nn=nn_demo, far=far,
                        sigma=pl.sigma[ep], slow=pl.slow[ep])
    best, succ = np.asarray(tr["best"]), np.asarray(tr["success"], bool)
    out = dict(score=met["score"], success=met["success"], catastrophic=float(np.mean(best < 0.5)), episodes=n, n=len(O),
               n_far=int(far.sum()), nn_median=float(np.median(nn_demo)), nn_p90=float(np.quantile(nn_demo, 0.9)),
               by_sigma={f"{sg:g}": dict(success=float(succ[pl.sigma == sg].mean()), catastrophic=float((best[pl.sigma == sg] < 0.5).mean()))
                         for sg in DART_SIGMAS},
               by_slow={f"{q:g}": dict(success=float(succ[pl.slow == q].mean()), catastrophic=float((best[pl.slow == q] < 0.5).mean()))
                        for q in DART_SLOW})
    dpath = os.path.join(OUT, "dagger", f"se_global_seed{s}.npz")
    if os.path.exists(dpath):                                            # NCA 卡住的状态（DAgger 数据，不是测试回合）被覆盖到没有
        st = stall_states_of(np.load(dpath))
        if len(st):
            both = torch.cat([demo, torch.as_tensor(O)])
            out.update(n_stall=len(st), stall_nn_demo=float(np.median(nn_(st, demo))), stall_nn_dart=float(np.median(nn_(st, both))))
    return s, out


def stage_dart(a):
    """DART 数据：--seeds 里的种子，每个 --dart_episodes 个回合。已有的 npz 不重做；诊断累积在 runs/pusht_tune/dart.json / .md。"""
    jpath = os.path.join(OUT, "dart.json")
    R = json.load(open(jpath, encoding="utf-8")) if os.path.exists(jpath) else {}
    jobs = [(s, a.dart_episodes) for s in a.seeds if not os.path.exists(os.path.join(OUT, "dart", f"dp_seed{s}.npz"))]
    with ProcessPoolExecutor(a.jobs, mp_context=mp.get_context("spawn")) as ex:
        for s, out in ex.map(dart_one, jobs):
            R[str(s)] = out
            print(f"seed {s}  " + "  ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in out.items()
                                            if not isinstance(v, dict)), flush=True)
    json.dump(R, open(jpath, "w", encoding="utf-8"), indent=1)
    L = [f"# DART 数据（老师 = 同种子大 DP，每次新噪声，执行前加扰动；环境种子 {DART_SEED} 起）\n",
         f"每个回合抽一个平移 σ ∈ {{{', '.join(f'{x:g}' for x in DART_SIGMAS)}}} 像素（每个动作块整体平移 N(0, σ²)）和一个放慢概率"
         f" ∈ {{{', '.join(f'{x:g}' for x in DART_SLOW)}}}（放慢 = 计划相对当前推杆位置的位移乘 {DART_SLOW_SCALE:g}）。"
         "样本 = 每次规划时的观测（每 8 步一个），标签 = 扰动前的计划。离示范 = 到最近的示范观测的距离（12 维特征）；"
         f"远 = 离示范 ≥ {DART_FAR:g}。NCA 卡住的状态 = se_global 的 DAgger 数据里，彻底失败回合第一次卡住的时刻及其后 8、16 步"
         "（环境种子 200000 起，不是测试回合）。\n",
         "| 种子 | 回合 | 老师得分 | 成功率 | 彻底失败 | 样本 | 远 | 离示范：中位数 / 90% | NCA 卡住的状态数 | 它们离示范 → 离示范 + DART |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for s, v in sorted(R.items()):
        L.append(f"| {s} | {v['episodes']} | {v['score']:.3f} | {100 * v['success']:.1f}% | {100 * v['catastrophic']:.1f}% | {v['n']}"
                 f" | {v['n_far']}（{100 * v['n_far'] / v['n']:.0f}%） | {v['nn_median']:.3f} / {v['nn_p90']:.3f}"
                 f" | {v.get('n_stall', '–')} | " + (f"{v['stall_nn_demo']:.3f} → {v['stall_nn_dart']:.3f}" if "stall_nn_dart" in v else "–") + " |")
    L += ["\n老师在不同扰动下的成功率 / 彻底失败（3 个种子合起来）：\n", "| 扰动 | 成功率 | 彻底失败 |", "|---|---|---|"]
    for key, label, vals in (("by_sigma", "平移 σ = {} 像素", DART_SIGMAS), ("by_slow", "放慢概率 {}", DART_SLOW)):
        for x in vals:
            r = [v[key][f"{x:g}"] for v in R.values() if f"{x:g}" in v.get(key, {})]
            if r:
                L.append(f"| {label.format(f'{x:g}')} | {100 * np.mean([q['success'] for q in r]):.1f}%"
                         f" | {100 * np.mean([q['catastrophic'] for q in r]):.1f}% |")
    text = "\n".join(L) + "\n"
    open(os.path.join(OUT, "dart.md"), "w", encoding="utf-8").write(text)
    print(text)


def plans_at(pl, kind, o, chunk=2000):
    """观测 o（n, 12）上每个各采一个每次新 z / 新噪声的冷启动计划，(n, T_P, 2)，归一化坐标（和 rollout 里的 fresh 一样）。"""
    import torch
    out = []
    with torch.no_grad():
        for c in torch.split(o, chunk):
            if kind == "nca":
                out.append(P.A.plan_of(pl.m.grow(pl.m.blank(c), c, torch.randn(len(c), pl.zd), 32)).numpy())
            else:
                out.append(pl.p.denoise(torch.randn(len(c), 2, P.T_P), c, pl.ts).numpy())
    return np.concatenate(out)


def dagger_fit_one(job):
    """一个（学生, 种子）：在它的 DAgger 数据上（彻底失败回合 / 其余回合的状态，各至多 n_max 个），补数据前的学生、
    补数据后的学生（<学生><后缀>，后缀 _dag 或 _dart）、老师（同种子大 DP）各采 R 个计划，算第 8 步离当前推杆位置多远，
    以及离标签多远（16 步逐步距离的平均）。老师离标签的距离是同一个分布两次采样的差，相当于能拟合到的下限。
    后缀是 _dart 时，DAgger 数据不在训练集里（没见过的卡住状态）；另外再在 DART 数据上（远 / 近）算一遍。"""
    import torch
    student, s, R, n_max, suffix = job
    torch.set_num_threads(4)
    d = np.load(os.path.join(OUT, "dagger", f"{student}_seed{s}.npz"))
    kind = model_kind(student)
    root = BASE if student in ("nca", "dp", "dp_small") else OUT
    models = {"before": (P.make_planner(kind, os.path.join(root, f"{student}_seed{s}", f"{kind}.pt"), "fresh"), kind),
              "after": (P.make_planner(kind, os.path.join(OUT, f"{student}{suffix}_seed{s}", f"{kind}.pt"), "fresh"), kind),
              "teacher": (P.make_planner("dp", os.path.join(BASE, f"dp_seed{s}", "dp.pt"), "fresh"), "dp")}
    rng = np.random.default_rng(s)
    sets = [("cat", d, d["cat"]), ("ok", d, ~d["cat"])]
    if suffix == "_dart":
        dd = np.load(os.path.join(OUT, EXTRA_DATA[suffix].format(m=student, s=s)))
        sets += [("far", dd, dd["far"]), ("near", dd, ~dd["far"])]
    out = {}
    for sub, d, mask in sets:
        idx = np.sort(rng.permutation(np.flatnonzero(mask))[:n_max])
        if len(idx) == 0:
            continue
        o, y = torch.as_tensor(d["O"][idx]), d["Y"][idx]
        ag = o[:, :2].numpy()
        r = dict(n=len(idx), label8=float(np.median(np.linalg.norm(y[:, 7] - ag, axis=1) * 256)))
        torch.manual_seed(s)
        for name, (pl, k) in models.items():
            p = plans_at(pl, k, o.repeat_interleave(R, 0)).reshape(len(idx), R, P.T_P, 2)
            d8 = np.linalg.norm(p[:, :, 7] - ag[:, None], axis=-1) * 256                  # (n, R)
            err = np.linalg.norm(p - y[:, None], axis=-1).mean(-1) * 256                 # (n, R)，16 步逐步距离的平均
            r[name] = dict(d8=float(np.median(d8)), err=float(np.median(err.mean(1))), err_min=float(np.median(err.min(1))))
        out[sub] = r
    return student, s, out


def stall_cover_one(job):
    """一个（模型, 种子）：和 --stage final 同样的测试回合（测试种子 100000 起，单线程，每次新 z / 新噪声）再闭环一遍，
    取卡住的状态（stall_obs），算它们离最近的示范观测、离最近的额外训练样本（EXTRA_DATA[后缀]：DAgger 或 DART 数据）多远，
    以及模型自己在这些状态上的计划第 8 步。"""
    import torch
    model, student, s, n, suffix = job
    torch.set_num_threads(1)                                           # 和 final_one 一样，闭环结果才能逐回合对上
    kind = model_kind(model)
    root = BASE if model in ("nca", "dp", "dp_small") else OUT
    pl = P.make_planner(kind, os.path.join(root, f"{model}_seed{s}", f"{kind}.pt"), "fresh")
    met, tr = P.rollout(pl, n, s + 1)
    obs = stall_obs(tr)
    out = dict(catastrophic=float(np.mean(tr["best"] < 0.5)), n=len(obs))
    if obs:
        o = torch.as_tensor(np.array(obs, np.float32))
        nn_ = lambda ref: float(torch.cdist(o, torch.as_tensor(ref)).min(1).values.median())
        torch.manual_seed(s)
        p = plans_at(pl, kind, o)
        out.update(nn_demo=nn_(P.load_data()[0]),
                   nn_extra=nn_(np.load(os.path.join(OUT, EXTRA_DATA[suffix].format(m=student, s=s)))["O"]),
                   d8=float(np.median(np.linalg.norm(p[:, 7] - o[:, :2].numpy(), axis=1) * 256)))
    return model, s, out


def stage_dagger_fit(a):
    """补数据之后学会了没有（--probe_models 里的学生，--seeds 里的种子，--fit_suffix 选 _dag（DAgger）或 _dart（DART）；
    要先有学生的 DAgger 数据、对应的额外数据和 <学生><后缀>_seed<s>）。两部分：在学生的 DAgger 数据（_dart 时还有 DART
    数据）上比较前后的学生和老师（dagger_fit_one）；测试回合里卡住的状态有没有被额外数据覆盖（stall_cover_one）。
    结果在 runs/pusht_tune/dagger_fit.json / .md（_dart 时是 dart_fit.json / .md）。"""
    R_SAMPLES, N_MAX, N_TEST = 8, 1000, 500
    suffix = a.fit_suffix
    name, what = {"_dag": ("dagger_fit", "DAgger"), "_dart": ("dart_fit", "DART")}[suffix]
    ok = lambda m, s: (os.path.exists(os.path.join(OUT, "dagger", f"{m}_seed{s}.npz"))
                       and os.path.exists(os.path.join(OUT, EXTRA_DATA[suffix].format(m=m, s=s)))
                       and os.path.exists(os.path.join(OUT, f"{m}{suffix}_seed{s}", "metrics.json")))
    jobs = [(m, s, R_SAMPLES, N_MAX, suffix) for m in a.probe_models for s in a.seeds if ok(m, s)]
    cover_jobs = [(mm, m, s, N_TEST, suffix) for m in a.probe_models for mm in (m, m + suffix) for s in a.seeds if ok(m, s)]
    F, C = {}, {}
    with ProcessPoolExecutor(a.jobs, mp_context=mp.get_context("spawn")) as ex:
        f_fit = [ex.submit(dagger_fit_one, j) for j in jobs]
        f_cov = [ex.submit(stall_cover_one, j) for j in cover_jobs]
        for f in f_fit:
            m, s, out = f.result()
            F[f"{m}/{s}"] = out
            print(f"fit {m} seed {s}  " + "  ".join(
                f"{k}: n {v['n']} label8 {v['label8']:.0f} d8 " + "/".join(f"{v[x]['d8']:.0f}" for x in ("before", "after", "teacher"))
                + " err " + "/".join(f"{v[x]['err']:.0f}" for x in ("before", "after", "teacher")) for k, v in out.items()), flush=True)
        for f in f_cov:
            m, s, out = f.result()
            C[f"{m}/{s}"] = out
            print(f"cover {m} seed {s}  " + "  ".join(f"{k} {v:.3f}" for k, v in out.items()), flush=True)
    json.dump(dict(fit=F, cover=C), open(os.path.join(OUT, f"{name}.json"), "w", encoding="utf-8"), indent=1)
    trio = lambda v, key: " / ".join(f"{v[x][key]:.0f}" for x in ("before", "after", "teacher"))
    dj = os.path.join(OUT, "dagger.json")                              # --stage dagger 的诊断：DAgger 状态离示范多远，作参照
    DJ = json.load(open(dj, encoding="utf-8")) if os.path.exists(dj) else {}
    span = lambda sub: (f"{min(v[sub]['nn'] for v in DJ.values() if sub in v):.2f}–"
                        f"{max(v[sub]['nn'] for v in DJ.values() if sub in v):.2f}")
    seen = ("这些状态和标签都在 DAgger 后的模型的训练集里" if suffix == "_dag" else
            "DART 后的模型没见过这些状态（它的额外数据是老师加扰动闭环收的），这是推广的检验")
    LAB = {"cat": "DAgger 数据·彻底失败回合", "ok": "DAgger 数据·其余回合",
           "far": f"DART 数据·离示范 ≥ {DART_FAR:g}", "near": "DART 数据·其余"}
    L = [f"# {what} 之后学会了没有\n",
         "## 补数据前后的学生和老师在同一批状态上\n",
         f"DAgger 数据 = `--stage dagger` 收集的样本（学生每次新 z / 新噪声闭环时每 8 步一个观测，环境种子 {DAGGER_SEED} 起），"
         f"分成彻底失败回合和其余回合；{seen}。"
         + (f"DART 数据 = `--stage dart` 收集的样本（环境种子 {DART_SEED} 起），分成离示范远（≥ {DART_FAR:g}）和其余；"
            "这些在 DART 后的模型的训练集里。" if suffix == "_dart" else "")
         + f"每组随机取至多 {N_MAX} 个状态。每个状态上，补数据前的学生、补数据后的学生（<学生>{suffix}）和老师（同种子大 DP）"
         f"各采 {R_SAMPLES} 个每次新 z / 新噪声的冷启动计划。"
         "第 8 步 = 计划第 8 步离当前推杆位置的距离（中位数，像素）；离标签 = 计划和大 DP 标签 16 步逐步距离的平均"
         f"（每个状态先对 {R_SAMPLES} 个计划取平均或最小，再取中位数，像素）。"
         "老师离标签的距离是同一个分布两次采样的差，相当于能拟合到的下限。\n",
         "| 学生 | 种子 | 状态 | 状态数 | 标签第 8 步 | 第 8 步：前 / 后 / 老师 | 离标签（平均）：前 / 后 / 老师"
         f" | 离标签（{R_SAMPLES} 个里最近）：前 / 后 / 老师 |",
         "|---|---|---|---|---|---|---|---|"]
    for k, out in sorted(F.items()):
        m, s = k.split("/")
        for sub, v in out.items():
            L.append(f"| {m} | {s} | {LAB[sub]} | {v['n']} | {v['label8']:.0f}"
                     f" | {trio(v, 'd8')} | {trio(v, 'err')} | {trio(v, 'err_min')} |")
    L += [f"\n## 测试回合里卡住的状态有没有被 {what} 数据覆盖\n",
          f"和 `--stage final` 同样的测试回合（每个种子 {N_TEST} 个，测试种子 100000 起，每次新 z / 新噪声）再闭环一遍，"
          f"取卡住的状态（彻底失败回合里第一次卡住的时刻及其后 8、16 步，同 stall.md）。离示范、离 {what} 数据 = 到最近的示范观测、"
          f"到最近的 {what} 样本（补数据后的模型训练时见过）的距离（12 维特征，中位数）"
          + (f"；参照：DAgger 数据里彻底失败回合的状态离示范 {span('cat')}，其余回合 {span('ok')}" if DJ else "")
          + "。第 8 步 = 模型自己在这些状态上的计划（中位数，像素）。\n",
          f"| 模型 | 种子 | 彻底失败 | 卡住的状态数 | 离示范 | 离 {what} 数据 | 自己的计划第 8 步 |", "|---|---|---|---|---|---|---|"]
    for m in dict.fromkeys(k.split("/")[0] for k in C):
        for s in sorted(k.split("/")[1] for k in C if k.split("/")[0] == m):
            v = C[f"{m}/{s}"]
            L.append(f"| {m} | {s} | {100 * v['catastrophic']:.1f}% | {v['n']} | "
                     + (f"{v['nn_demo']:.3f} | {v['nn_extra']:.3f} | {v['d8']:.0f} |" if v["n"] else "– | – | – |"))
    text = "\n".join(L) + "\n"
    open(os.path.join(OUT, f"{name}.md"), "w", encoding="utf-8").write(text)
    print(text)


# 路径图（viz_tune.py 的 paths.png）：同一批测试回合上 DAgger 前后的 NCA 和小 DP。每列按规则取第一个符合的回合，不另外挑
PATH_MODELS = ("se_global", "se_global_dag", "dp_small", "dp_small_dag")
PATH_RULES = [
    ("fixed", "NCA：DAgger 前彻底失败，后成功", lambda o: o["se_global"]["cat"] & o["se_global_dag"]["succ"]),
    ("new", "NCA：DAgger 前成功，后彻底失败", lambda o: o["se_global"]["succ"] & o["se_global_dag"]["cat"]),
    ("both", "NCA：DAgger 前后都彻底失败", lambda o: o["se_global"]["cat"] & o["se_global_dag"]["cat"]),
    ("dp_new", "小 DP：DAgger 前成功，后彻底失败", lambda o: o["dp_small"]["succ"] & o["dp_small_dag"]["cat"]),
]


def paths_one(job):
    """一个模型：和 --stage final 同样的测试回合（单线程，每次新 z / 新噪声），返回轨迹。两种对照（环境的初始状态不变）：
    tag = "@rerun"：同一个模型，规划器的随机数种子换成 s + 1001；tag = "#other"：换成下一个训练种子的模型。"""
    import torch
    model, s, n, tag = job
    torch.set_num_threads(1)
    kind = model_kind(model)
    root = BASE if model in ("nca", "dp", "dp_small") else OUT
    ms = (s + 1) % 3 if tag == "#other" else s
    pl = P.make_planner(kind, os.path.join(root, f"{model}_seed{ms}", f"{kind}.pt"), "fresh")
    return model + tag, P.rollout(pl, n, s + (1001 if tag == "@rerun" else 1))[1]


def stage_paths(a):
    """闭环路径（--seeds 的第一个种子，每个模型 --episodes 个测试回合，和 final 逐回合相同）：按 PATH_RULES 每条规则取
    第一个符合的回合，存四个模型在这些回合上的推杆轨迹、块的位姿；彻底失败的回合再存第一次卡住时（first_stall）
    模型自己和同种子大 DP 在同一观测上各采 3 个的计划。另外数彻底失败的回合重合多少：DAgger 前后，对照是同一个模型
    换个随机数重跑（@rerun）、换一个训练种子的模型（#other）。结果在 runs/pusht_tune/paths.json，图由 viz_tune.py 画。"""
    import torch
    s, n, R = a.seeds[0], a.episodes, 3
    jobs = [(m, s, n, "") for m in PATH_MODELS] + [(m, s, n, tag) for m in ("se_global", "dp_small") for tag in ("@rerun", "#other")]
    with ProcessPoolExecutor(min(a.jobs, len(jobs)), mp_context=mp.get_context("spawn")) as ex:
        TR = dict(ex.map(paths_one, jobs))
    cat_set = lambda k: set(np.flatnonzero(TR[k]["best"] < 0.5).tolist())
    overlap = {f"{x}|{y}": dict(a=len(cat_set(x)), b=len(cat_set(y)), both=len(cat_set(x) & cat_set(y)))
               for m in ("se_global", "dp_small") for x, y in ((m, m + "@rerun"), (m, m + "#other"), (m, m + "_dag"))}
    o = {m: dict(cat=TR[m]["best"] < 0.5, succ=np.asarray(TR[m]["success"], bool)) for m in PATH_MODELS}
    cols = []
    for name, label, rule in PATH_RULES:
        idx = np.flatnonzero(rule(o))
        cols.append(dict(rule=name, label=label, n_match=int(len(idx)), episode=int(idx[0]) if len(idx) else None))
    path_of = lambda m: os.path.join(BASE if m in ("dp", "dp_small") else OUT, f"{m}_seed{s}", f"{model_kind(m)}.pt")
    planners = {m: P.make_planner(model_kind(m), path_of(m), "fresh") for m in PATH_MODELS}
    dp = P.make_planner("dp", path_of("dp"), "fresh")
    px = lambda p: np.round((p + 1.0) * 256.0, 1).tolist()
    torch.manual_seed(s)
    traj = {}
    for c in cols:
        e = c["episode"]
        if e is None:
            continue
        for m in PATH_MODELS:
            tr = TR[m]
            ag, bl = tr["agent"][e], tr["block"][e]
            r = dict(agent=np.round(ag, 1).tolist(), block=np.round(bl[::25], 2).tolist(), block_end=np.round(bl[-1], 2).tolist(),
                     best=float(tr["best"][e]), success=bool(tr["success"][e]), steps=len(tr["acts"][e]))
            t = first_stall(ag) if tr["best"][e] < 0.5 else None
            if t is not None:
                ob = torch.as_tensor(np.array([obs_at(tr, e, t)] * R, np.float32))
                r["stall"] = dict(t=t, block=np.round(bl[t], 2).tolist(), plan=px(plans_at(planners[m], model_kind(m), ob)),
                                  dp_plan=px(plans_at(dp, "dp", ob)))
            traj[f"{m}/{e}"] = r
    out = dict(seed=s, episodes=n, models=list(PATH_MODELS), cols=cols, traj=traj, overlap=overlap,
               counts={m: dict(cat=int(o[m]["cat"].sum()), succ=int(o[m]["succ"].sum())) for m in PATH_MODELS})
    json.dump(out, open(os.path.join(OUT, "paths.json"), "w", encoding="utf-8"))
    for c in cols:
        print(f"{c['rule']}: {c['n_match']} episodes match, showing episode {c['episode']}", flush=True)
    print({m: v for m, v in out["counts"].items()}, flush=True)
    print("catastrophic overlap:", overlap, flush=True)


# 接管检验：DART 后 NCA 还会卡住的状态上，老师（大 DP）自己的计划第 8 步也只有 13–16 像素（attract.md），
# DART 前是 41–62。这里直接看：卡住的时候换别人接手，能不能脱困
TAKEOVER_CTRL = {"self": None, "dp": "dp", "dp_small": "dp_small"}     # 接手的规划器；None = 卡住的那个模型自己（换随机数）


def continue_from(pl, seeds, prefix, rng_seed):
    """回合 i 用环境种子 seeds[i] 重置，重放 prefix[i]（之前执行过的动作，长度 t_i 是 8 的倍数）回到同一个状态，
    再交给规划器 pl（每次新 z / 新噪声）闭环到第 MAX_STEPS 步，和 rollout 一样每 8 步规划一次。
    返回 best（覆盖率奖励最大值，含重放的部分）、success、接手后的推杆轨迹、接手时的第一个计划（像素），
    以及接手后的完整记录：states（第 t 步起的状态，(n+1, 5)）、acts（执行的动作，(n, 2) 像素）、rew（每步之后的奖励）、
    prev0（第 t−1 步的状态）。"""
    import gym_pusht  # noqa: F401  注册环境
    import gymnasium as gym
    import torch
    torch.manual_seed(rng_seed)
    E = len(seeds)
    envs = [gym.make("gym_pusht/PushT-v0", obs_type="state").unwrapped for _ in range(E)]
    states, prev, best = np.zeros((E, 5), np.float32), np.zeros((E, 5), np.float32), np.zeros(E)
    for e, env in enumerate(envs):
        s = p = env.reset(seed=int(seeds[e]))[0]
        for act in prefix[e]:
            p = s
            s, r, *_ = env.step(act)
            best[e] = max(best[e], r)
        states[e], prev[e] = s, p                                        # 第 t 步和第 t−1 步，和 rollout 的观测一致
    t0 = np.array([len(q) for q in prefix])
    alive, success = np.ones(E, bool), np.zeros(E, bool)
    agent = [[states[e, :2].copy()] for e in range(E)]
    traj, acts, rew, prev0 = [[states[e].copy()] for e in range(E)], [[] for _ in range(E)], [[] for _ in range(E)], prev.copy()
    first = np.full((E, P.T_P, 2), np.nan, np.float32)
    pl.reset(E)                                                          # 规划器按整批 E 个生成随机数，没轮到的也一起算
    for u in range(int(t0.min()), P.MAX_STEPS, P.T_A):
        if not alive.any():
            break
        act = alive & (t0 <= u)
        if not act.any():
            continue
        plan = (pl.plan(torch.from_numpy(np.concatenate([P.feat(states), P.feat(prev)], 1))).numpy() + 1.0) * 256.0
        first[t0 == u] = plan[t0 == u]
        for k in range(min(P.T_A, P.MAX_STEPS - u)):
            for e in np.flatnonzero(act & alive):
                prev[e] = states[e]
                s, r, term, _, _ = envs[e].step(plan[e, k])
                states[e] = s
                best[e] = max(best[e], r)
                agent[e].append(s[:2].copy())
                traj[e].append(np.asarray(s, np.float32).copy())
                acts[e].append(plan[e, k].copy())
                rew[e].append(float(r))
                if term:
                    success[e], alive[e] = True, False
    return dict(best=best, success=success, agent=[np.array(q) for q in agent], first=first,
                states=[np.array(q, np.float32) for q in traj], acts=[np.array(q, np.float32).reshape(-1, 2) for q in acts],
                rew=[np.array(q) for q in rew], prev0=prev0)


def takeover_one(job):
    """一个（卡住的来源模型, 种子）：和 --stage final 同样的测试回合闭环（单线程，逐回合和 final 相同），取彻底失败
    回合里第一次卡住的时刻 t（first_stall），从那里分别交给 TAKEOVER_CTRL 里的规划器跑到第 300 步：self = 来源模型自己
    （换随机数种子，看是不是运气），dp = 同种子大 DP（DAgger / DART 的老师），dp_small = 同种子小 DP（只用示范训练）。"""
    import torch
    src, s, n = job
    torch.set_num_threads(1)
    path_of = lambda m: os.path.join(BASE if m in ("nca", "dp", "dp_small") else OUT, f"{m}_seed{s}", f"{model_kind(m)}.pt")
    tr = P.rollout(P.make_planner(model_kind(src), path_of(src), "fresh"), n, s + 1)[1]
    sel = [(int(e), first_stall(tr["agent"][e])) for e in np.flatnonzero(tr["best"] < 0.5)]
    sel = [(e, t) for e, t in sel if t is not None]
    out = dict(catastrophic=int(np.sum(tr["best"] < 0.5)), n=len(sel), episode=[e for e, _ in sel], t=[t for _, t in sel])
    if not sel:
        return src, s, out
    ag0 = np.array([tr["agent"][e][t] for e, t in sel])
    for c, m in TAKEOVER_CTRL.items():
        m = m or src
        if c != "self" and m == src:
            continue
        r = continue_from(P.make_planner(model_kind(m), path_of(m), "fresh"),
                          [P.TEST_SEED + e for e, _ in sel], [tr["acts"][e][:t] for e, t in sel], s + 1001)
        disp = lambda k: np.array([np.linalg.norm(q[min(k, len(q) - 1)] - q[0]) for q in r["agent"]])
        out[c] = dict(model=m, best=np.round(r["best"], 4).tolist(), success=r["success"].astype(int).tolist(),
                      moved=((disp(8) >= 10) | (disp(16) >= 10)).astype(int).tolist(), disp16=np.round(disp(16), 1).tolist(),
                      d8=np.round(np.linalg.norm(r["first"][:, 7] - ag0, axis=1), 1).tolist(),
                      replay_dev=float(np.abs(np.array([q[0] for q in r["agent"]]) - ag0).max()))   # 重放回到的状态对不对
    return src, s, out


def stage_takeover(a):
    """接管检验（--probe_models 里的来源模型，--seeds，每个种子 --episodes 个测试回合）：彻底失败回合里第一次卡住时
    换别的规划器接手，能不能脱困。结果累积在 runs/pusht_tune/takeover.json / .md（已有的（来源, 种子）不重算）。"""
    jpath = os.path.join(OUT, "takeover.json")
    R = json.load(open(jpath, encoding="utf-8")) if os.path.exists(jpath) else {}
    jobs = [(m, s, a.episodes) for m in a.probe_models for s in a.seeds if f"{m}/{s}" not in R
            and os.path.exists(os.path.join(BASE if m in ("nca", "dp", "dp_small") else OUT, f"{m}_seed{s}", "metrics.json"))]
    if jobs:
        with ProcessPoolExecutor(min(a.jobs, len(jobs)), mp_context=mp.get_context("spawn")) as ex:
            for m, s, out in ex.map(takeover_one, jobs):
                R[f"{m}/{s}"] = out
                print(f"{m} seed {s}  catastrophic {out['catastrophic']}  stalled {out['n']}  " + "  ".join(
                    f"{c}: recovered {np.mean(np.array(v['best']) >= 0.5):.2f} moved {np.mean(v['moved']):.2f}"
                    f" d8 {np.median(v['d8']):.0f} dev {v['replay_dev']:.2g}" for c, v in out.items() if c in TAKEOVER_CTRL),
                      flush=True)
    json.dump(R, open(jpath, "w", encoding="utf-8"), indent=1)
    name = {"self": "自己（换随机数）", "dp": "大 DP（老师）", "dp_small": "小 DP（只用示范）"}
    pct = lambda x: f"{100 * np.mean(x):.0f}%"
    L = ["# 接管检验：卡住的时候换别人接手，能不能脱困\n",
         f"每个种子 {a.episodes} 个测试回合（测试种子 100000 起，和 final 逐回合相同），取彻底失败回合里第一次卡住的时刻"
         "（之后 8、16 步推杆位移都 < 10 像素，同 stall.md），重放之前的动作回到那个状态，再交给接手的规划器"
         "（每次新 z / 新噪声）跑到第 300 步。自己 = 卡住的那个模型，只换随机数种子；大 DP、小 DP 是同种子的，"
         "小 DP 只用示范训练。接手时第 8 步 = 接手者第一个计划第 8 步离推杆多远（中位数，像素；示范状态上约 60）；"
         "动起来了 = 之后 8 或 16 步推杆位移 ≥ 10 像素；脱离彻底失败 = 整个回合的最好覆盖率 ≥ 0.5。\n",
         "## 三个种子合在一起\n",
         "| 卡住的来源 | 卡住的回合 | 接手的 | 接手时第 8 步 | 动起来了 | 脱离彻底失败 | 成功 |", "|---|---|---|---|---|---|---|"]
    srcs = list(dict.fromkeys(k.split("/")[0] for k in R))
    for m in srcs:
        outs = [R[k] for k in R if k.split("/")[0] == m and R[k]["n"]]
        for c in TAKEOVER_CTRL:
            vs = [o[c] for o in outs if c in o]
            if not vs:
                continue
            cat = lambda key: np.concatenate([v[key] for v in vs])
            L.append(f"| {m} | {len(cat('best'))} | {name[c]} | {np.median(cat('d8')):.0f} | {pct(cat('moved'))}"
                     f" | {pct(cat('best') >= 0.5)} | {pct(cat('success'))} |")
    L += ["\n## 逐种子\n", "| 卡住的来源 | 种子 | 彻底失败 | 卡住的回合 | 接手的 | 接手时第 8 步 | 动起来了 | 脱离彻底失败 | 成功 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for m in srcs:
        for k in sorted(k for k in R if k.split("/")[0] == m):
            o = R[k]
            for c in TAKEOVER_CTRL:
                if c in o:
                    v = o[c]
                    L.append(f"| {m} | {k.split('/')[1]} | {o['catastrophic']} | {o['n']} | {name[c]} | {np.median(v['d8']):.0f}"
                             f" | {pct(v['moved'])} | {pct(np.array(v['best']) >= 0.5)} | {pct(v['success'])} |")
    text = "\n".join(L) + "\n"
    open(os.path.join(OUT, "takeover.md"), "w", encoding="utf-8").write(text)
    print(text)


# ----------------------------------------------------------------------------- 第九轮：按结果挑标签

def model_path(m, s):
    return os.path.join(BASE if m in ("nca", "dp", "dp_small") else OUT, f"{m}_seed{s}", f"{model_kind(m)}.pt")


class NoisyPlan:
    """加扰动的接手者：每个动作块整体平移 N(0, σ²)（像素）。和 DART 不同，这里执行的就是平移后的动作，标签也用它。"""

    def __init__(self, inner, sigma, rng):
        self.inner, self.sigma, self.rng = inner, sigma, rng

    def reset(self, E):
        self.inner.reset(E)

    def plan(self, obs):
        import torch
        p = self.inner.plan(obs)
        return p + torch.as_tensor(self.rng.normal(size=(len(p), 1, 2)) * self.sigma / 256.0, dtype=p.dtype)


def rescue_collect_one(job):
    """学生每次新 z / 新噪声闭环 n 个回合（环境种子 start 起）：彻底失败回合里第一次卡住的时刻 t（first_stall）、
    之前执行过的动作（重放用）和第 t 步的观测，以及这批回合的成绩。"""
    import torch
    student, s, start, n, rng_seed = job
    torch.set_num_threads(1)
    met, tr = P.rollout(P.make_planner(model_kind(student), model_path(student, s), "fresh"), n, rng_seed, start=start)
    sel = [(int(e), first_stall(tr["agent"][e])) for e in np.flatnonzero(tr["best"] < 0.5)]
    sel = [(e, t) for e, t in sel if t is not None]
    return student, s, dict(episodes=n, score=met["score"], success=met["success"], catastrophic=int(np.sum(tr["best"] < 0.5)),
                            env_seed=[start + e for e, _ in sel], t=[t for _, t in sel],
                            prefix=[tr["acts"][e][:t] for e, t in sel], obs=[obs_at(tr, e, t) for e, t in sel])


def rescue_try_one(job):
    """一次接手：RESCUE_CTRL 里的规划器 ctrl 用随机数种子 rng_seed，从学生每个卡住的状态接着跑到第 300 步。"""
    import torch
    student, s, ctrl, k, env_seed, prefix, rng_seed = job
    torch.set_num_threads(2)
    m, sigma = RESCUE_CTRL[ctrl]
    pl = P.make_planner(model_kind(m), model_path(m, s), "fresh")
    if sigma > 0:
        pl = NoisyPlan(pl, sigma, np.random.default_rng(rng_seed))
    r = continue_from(pl, env_seed, prefix, rng_seed)
    return student, s, ctrl, k, {x: r[x] for x in ("best", "success", "states", "acts", "rew", "prev0")}


def rescue_segment(r, i):
    """接手记录 r（continue_from 的结果）里第 i 个回合：从接手时刻到覆盖率最高的那一步，每一步一个样本。观测和示范一样是
    （当前帧, 上一帧）；标签是从这一步起实际执行的 16 个动作（越过记录末尾时重复最后一个，和示范的回合末尾一样），归一化。"""
    S, act, rw = r["states"][i], r["acts"][i], r["rew"][i]
    j = int(np.argmax(rw))                                              # 第 j 个动作之后覆盖率最高
    f = P.feat(S)
    fp = np.concatenate([P.feat(r["prev0"][i])[None], f[:-1]])          # 上一帧：接手时刻的上一帧是学生的
    idx = np.arange(j + 1)
    a = np.clip(act, 0.0, 512.0) / 256.0 - 1.0
    Y = a[np.minimum(idx[:, None] + np.arange(P.T_P), len(a) - 1)]
    return np.concatenate([f[idx], fp[idx]], 1).astype(np.float32), Y.astype(np.float32)


def stage_rescue(a):
    """按结果挑标签（--probe_models 里的学生，--seeds，每个 --rescue_episodes 个回合，环境种子 --rescue_seed 起，每批 1000 个并行）：
    学生闭环找卡住的状态（rescue_collect_one）→ --rescue_ctrl 里每个规划器各接手 --rescue_tries 次（rescue_try_one）
    → 每个卡住的状态挑最好的一次（成功的里步数最少的；都没成功时最好覆盖率最高的），整个回合的最好覆盖率 ≥ RESCUE_MIN
    才留下，切成样本（rescue_segment）。存到 runs/pusht_tune/rescue/<学生>_seed<s>.npz（O、Y 是训练样本，seg、step 是
    样本所属的卡住状态和离接手时刻的步数；stall_* 是所有卡住的状态，best、success 是每次接手的结果）。
    已有的 npz 不重做；诊断累积在 runs/pusht_tune/rescue.json / .md。"""
    jpath = os.path.join(OUT, "rescue.json")
    R = json.load(open(jpath, encoding="utf-8")) if os.path.exists(jpath) else {}
    todo = [(m, s) for m in a.probe_models for s in a.seeds if not os.path.exists(os.path.join(OUT, "rescue", f"{m}_seed{s}.npz"))]
    ctrls, K, B, E0 = list(a.rescue_ctrl), a.rescue_tries, 1000, a.rescue_seed
    t_start = time.time()
    C = {}
    with ProcessPoolExecutor(a.jobs, mp_context=mp.get_context("spawn")) as ex:
        col_jobs = [(m, s, E0 + b, min(B, a.rescue_episodes - b), E0 + 100 * s + b // B)
                    for m, s in todo for b in range(0, a.rescue_episodes, B)]
        for m, s, out in ex.map(rescue_collect_one, col_jobs):
            c = C.setdefault((m, s), dict(episodes=0, score=0.0, success=0.0, catastrophic=0, env_seed=[], t=[], prefix=[], obs=[]))
            c["episodes"] += out["episodes"]
            c["score"] += out["score"] * out["episodes"]
            c["success"] += out["success"] * out["episodes"]
            c["catastrophic"] += out["catastrophic"]
            for k in ("env_seed", "t", "prefix", "obs"):
                c[k] += out[k]
        for (m, s), c in C.items():
            print(f"collect {m} seed {s}: {c['episodes']} episodes, catastrophic {c['catastrophic']}, stalled {len(c['t'])}"
                  f"  ({time.time() - t_start:.0f}s)", flush=True)
        try_jobs = [(m, s, ctrl, k, c["env_seed"], c["prefix"], E0 + 1000 * s + 10 * ci + k)
                    for (m, s), c in C.items() if c["t"] for ci, ctrl in enumerate(ctrls) for k in range(K)]
        T = {}
        for m, s, ctrl, k, r in ex.map(rescue_try_one, try_jobs):
            T[(m, s, ctrl, k)] = r
            print(f"try {m} seed {s} {ctrl} #{k}: escaped {np.mean(r['best'] >= RESCUE_MIN):.2f}"
                  f" success {np.mean(r['success']):.2f}  ({time.time() - t_start:.0f}s)", flush=True)
    os.makedirs(os.path.join(OUT, "rescue"), exist_ok=True)
    for (m, s), c in C.items():
        n_st = len(c["t"])
        best, succ, steps = (np.zeros((n_st, len(ctrls), K), dt) for dt in (float, bool, int))
        for ci, ctrl in enumerate(ctrls):
            for k in range(K):
                if (m, s, ctrl, k) in T:
                    r = T[(m, s, ctrl, k)]
                    best[:, ci, k], succ[:, ci, k] = r["best"], r["success"]
                    steps[:, ci, k] = [len(q) for q in r["acts"]]
        CK = len(ctrls) * K
        key = np.where(succ, 10.0 - steps / 1000.0, best).reshape(n_st, CK)   # 成功的（> 9）排在没成功的（≤ 1）前面
        flat = key.argmax(1) if n_st else np.zeros(0, int)
        pick_c, pick_k = flat // K, flat % K
        pick_best = best.reshape(n_st, CK)[np.arange(n_st), flat]
        pick_succ = succ.reshape(n_st, CK)[np.arange(n_st), flat]
        rescued = pick_best >= RESCUE_MIN
        O, Y, seg, step = [], [], [], []
        for i in np.flatnonzero(rescued):
            o, y = rescue_segment(T[(m, s, ctrls[pick_c[i]], pick_k[i])], i)
            O.append(o), Y.append(y)
            seg += [i] * len(o)
            step += list(range(len(o)))
        O = np.concatenate(O) if O else np.zeros((0, 12), np.float32)
        Y = np.concatenate(Y) if Y else np.zeros((0, P.T_P, 2), np.float32)
        np.savez_compressed(os.path.join(OUT, "rescue", f"{m}_seed{s}.npz"), O=O, Y=Y, seg=np.array(seg, int),
                            step=np.array(step, int), stall_O=np.array(c["obs"], np.float32).reshape(-1, 12),
                            stall_t=np.array(c["t"], int), stall_env=np.array(c["env_seed"], int), rescued=rescued,
                            pick_ctrl=np.where(rescued, pick_c, -1), pick_best=pick_best, best=best, success=succ,
                            ctrls=np.array(ctrls))
        seg_len = np.bincount(np.array(seg, int), minlength=n_st)[rescued] if len(seg) else np.zeros(0)
        base = [ctrls.index(c) for c in RESCUE_BASE if c in ctrls]
        R[f"{m}/{s}"] = dict(
            env_start=E0, tries=K,
            episodes=c["episodes"], score=c["score"] / c["episodes"], success=c["success"] / c["episodes"],
            catastrophic=c["catastrophic"], n_stall=n_st, n_rescued=int(rescued.sum()), n_success=int((pick_succ & rescued).sum()),
            t_median=float(np.median(c["t"])) if n_st else None, n_samples=len(O),
            seg_median=float(np.median(seg_len)) if len(seg_len) else None,
            by_ctrl={ctrl: dict(any=float((best[:, ci] >= RESCUE_MIN).any(1).mean()) if n_st else None,
                                per_try=float((best[:, ci] >= RESCUE_MIN).mean()) if n_st else None,
                                success_any=float(succ[:, ci].any(1).mean()) if n_st else None,
                                picked=int(((pick_c == ci) & rescued).sum())) for ci, ctrl in enumerate(ctrls)},
            base_any=float((best[:, base] >= RESCUE_MIN).any((1, 2)).mean()) if n_st and base else None,   # 只用第九轮的接手者
            dp_or_small=float(((best[:, ctrls.index("dp")] >= RESCUE_MIN).any(1)
                               | (best[:, ctrls.index("dp_small")] >= RESCUE_MIN).any(1)).mean())
            if n_st and {"dp", "dp_small"} <= set(ctrls) else None)
        print(f"{m} seed {s}: rescued {int(rescued.sum())}/{n_st}, samples {len(O)}", flush=True)
    json.dump(R, open(jpath, "w", encoding="utf-8"), indent=1)
    name = {"dp": "老师", "dp_noise": f"老师 + 扰动（σ = {RESCUE_CTRL['dp_noise'][1]:g}）", "dp_small": "小 DP", "nca": "NCA",
            "dp_small_r9": "第九轮的小 DP", "nca_r9": "第九轮的 NCA"}
    cols = list(dict.fromkeys(c for v in R.values() for c in v["by_ctrl"]))   # 各轮用过的接手者的并集
    Ks = "/".join(map(str, sorted({v.get("tries", 3) for v in R.values()})))
    pct = lambda x: "–" if x is None else f"{100 * x:.0f}%"
    L = ["# 按结果挑标签的数据（学生每次新 z / 新噪声闭环）\n",
         "卡住的状态 = 彻底失败回合（最好覆盖率 < 0.5）里第一次卡住的时刻（之后 8、16 步推杆位移都 < 10 像素）。"
         f"每个卡住的状态上，表里有数的规划器各接手 {Ks} 次（每次新 z / 新噪声，换随机数种子），跑到第 300 步；"
         "老师 = 同种子大 DP，小 DP、NCA = 同种子、只用示范训练的（dp_small、se_global），第九轮的小 DP、NCA = 同种子、"
         "补了第九轮数据的（dp_small_rescue、se_global_rescue，第十轮才用）。救出 = 整个回合的最好覆盖率 ≥ "
         f"{RESCUE_MIN:g}；每个规划器那一栏是它自己的几次里任一次救出；原来四个 = 第九轮的接手者（老师、老师 + 扰动、小 DP、NCA）。"
         "每个卡住的状态挑最好的一次（成功的里步数最少的，都没成功时最好覆盖率最高的），救出来的才留下：从接手到覆盖率最高的那一步，"
         "每一步一个样本，标签是实际执行的 16 个动作。示范共 25650 个样本，DART 数据每个种子约 3.6 万个。\n",
         "| 学生 | 种子 | 环境种子 | 回合 | 得分 | 彻底失败 | 卡住的状态 | 接手时刻（中位数） | "
         + " | ".join(f"{name[c]}救出" for c in cols)
         + " | 老师或小 DP | 原来四个里任一次 | 全部里任一次 | 其中成功 | 样本 | 每段步数（中位数） |",
         "|---|---|---|---|---|---|---|---|" + "---|" * len(cols) + "---|---|---|---|---|---|"]
    for k, v in R.items():
        m, s = k.split("/")
        n = max(v["n_stall"], 1)
        tm = "–" if v["t_median"] is None else f"{v['t_median']:.0f}"
        sm = "–" if v["seg_median"] is None else f"{v['seg_median']:.0f}"
        L.append(f"| {m} | {s} | {v.get('env_start', RESCUE_SEED)} 起 | {v['episodes']} | {v['score']:.3f}"
                 f" | {100 * v['catastrophic'] / v['episodes']:.1f}% | {v['n_stall']} | {tm} | "
                 + " | ".join(pct(v["by_ctrl"][c]["any"]) if c in v["by_ctrl"] else "–" for c in cols)
                 + f" | {pct(v['dp_or_small'])} | {pct(v.get('base_any', v['n_rescued'] / n))} | {pct(v['n_rescued'] / n)}"
                 f" | {pct(v['n_success'] / n)} | {v['n_samples']} | {sm} |")
    L += ["\n挑出来的那一次来自哪个规划器（救出来的状态数）：\n", "| 学生 | 种子 | " + " | ".join(name[c] for c in cols) + " |",
          "|---|---|" + "---|" * len(cols)]
    for k, v in R.items():
        L.append(f"| {k.split('/')[0]} | {k.split('/')[1]} | "
                 + " | ".join(str(v["by_ctrl"][c]["picked"]) if c in v["by_ctrl"] else "–" for c in cols) + " |")
    text = "\n".join(L) + "\n"
    open(os.path.join(OUT, "rescue.md"), "w", encoding="utf-8").write(text)
    print(text)


def rescue_fit_one(job):
    """一个（补数据后的模型, 种子）：几组卡住的状态上，补数据前的学生（before）、补数据后的模型（after）、老师（同种子大 DP）、
    小 DP（只用示范）各采 R 个每次新 z / 新噪声的冷启动计划，算第 8 步离当前推杆位置多远；救出来的状态上还算离标签多远
    （16 步逐步距离的平均）。seen = 救出来的卡住状态（after 训练时见过它们和标签）；unrescued = 接手都没救出来的卡住状态
    （训练集里没有）；test = before 在测试回合（和 final 相同）里第一次卡住的状态；own = after 自己在测试回合里第一次卡住的
    状态。test、own 两组还算离最近的“救出来的 / 没救出来的”卡住状态多远（12 维特征）。"""
    import torch
    after, s, R = job
    before = RESCUE_PAIRS[after]
    d = np.load(os.path.join(OUT, "rescue", f"{before}_seed{s}.npz"))
    torch.set_num_threads(1)                                            # 测试回合和 final 逐回合相同

    def test_stalls(m):
        tr = P.rollout(P.make_planner(model_kind(m), model_path(m, s), "fresh"), 500, s + 1)[1]
        o = [obs_at(tr, e, t) for e in np.flatnonzero(tr["best"] < 0.5) for t in [first_stall(tr["agent"][e])] if t is not None]
        return np.array(o, np.float32).reshape(-1, 12)

    sets = {"seen": (d["stall_O"][d["rescued"]], d["Y"][d["step"] == 0]), "unrescued": (d["stall_O"][~d["rescued"]], None),
            "test": (test_stalls(before), None), "own": (test_stalls(after), None)}
    torch.set_num_threads(4)
    models = {"before": before, "after": after, "teacher": "dp", "dp_small": "dp_small"}
    pls = {k: P.make_planner(model_kind(m), model_path(m, s), "fresh") for k, m in models.items()}
    ref = {"rescued": d["stall_O"][d["rescued"]], "unrescued": d["stall_O"][~d["rescued"]]}
    out = {}
    for name, (o, y) in sets.items():
        if len(o) == 0:
            continue
        ot = torch.as_tensor(o)
        ag = o[:, :2]
        r = dict(n=len(o))
        if y is not None:
            r["label8"] = float(np.median(np.linalg.norm(y[:, 7] - ag, axis=1) * 256))
        if name in ("test", "own"):
            r.update({f"nn_{k}": float(torch.cdist(ot, torch.as_tensor(v)).min(1).values.median()) for k, v in ref.items() if len(v)})
        torch.manual_seed(s)
        for k, pl in pls.items():
            p = plans_at(pl, model_kind(models[k]), ot.repeat_interleave(R, 0)).reshape(len(o), R, P.T_P, 2)
            d8 = np.linalg.norm(p[:, :, 7] - ag[:, None], axis=-1) * 256                  # (n, R)
            r[k] = dict(d8=float(np.median(d8)))
            if y is not None:
                err = np.linalg.norm(p - y[:, None], axis=-1).mean(-1) * 256
                r[k].update(err=float(np.median(err.mean(1))), err_min=float(np.median(err.min(1))))
        out[name] = r
    return after, s, out


def stage_rescue_fit(a):
    """补数据之后学会了没有（RESCUE_PAIRS 里训练完的模型，--seeds）。结果累积在 runs/pusht_tune/rescue_fit.json / .md
    （已有的（模型, 种子）不重算）。"""
    R_SAMPLES = 8
    jpath = os.path.join(OUT, "rescue_fit.json")
    F = json.load(open(jpath, encoding="utf-8")) if os.path.exists(jpath) else {}
    jobs = [(m, s, R_SAMPLES) for m in RESCUE_PAIRS for s in a.seeds if f"{m}/{s}" not in F
            and os.path.exists(os.path.join(OUT, m + f"_seed{s}", "metrics.json"))
            and os.path.exists(os.path.join(OUT, "rescue", f"{RESCUE_PAIRS[m]}_seed{s}.npz"))]
    with ProcessPoolExecutor(min(a.jobs, max(len(jobs), 1)), mp_context=mp.get_context("spawn")) as ex:
        for m, s, out in ex.map(rescue_fit_one, jobs):
            F[f"{m}/{s}"] = out
            print(f"fit {m} seed {s}  " + "  ".join(
                f"{k}: n {v['n']} d8 " + "/".join(f"{v[x]['d8']:.0f}" for x in ("before", "after", "teacher", "dp_small"))
                + (f" err {v['before']['err']:.0f}/{v['after']['err']:.0f}" if "err" in v["before"] else "")
                for k, v in out.items()), flush=True)
    json.dump(F, open(jpath, "w", encoding="utf-8"), indent=1)
    LAB = {"seen": "救出来的卡住状态（见过）", "unrescued": "没救出来的卡住状态", "test": "测试回合里补数据前卡住的",
           "own": "测试回合里补数据后卡住的"}
    four = lambda v, key: " / ".join(f"{v[x][key]:.0f}" if key in v[x] else "–" for x in ("before", "after", "teacher", "dp_small"))
    L = ["# 按结果挑标签之后学会了没有\n",
         f"补数据前 = 收集数据的学生（{', '.join(f'{v} → {k}' for k, v in RESCUE_PAIRS.items())}）。四组卡住的状态（都是第一次卡住的时刻）："
         f"救出来的 = `--stage rescue` 里这个学生救出来的（第九轮环境种子 {RESCUE_SEED} 起，第十轮 {RESCUE_SEED2} 起；"
         "它们和标签在补数据后的训练集里，第十轮的模型还见过第九轮的）；"
         "没救出来的 = 同一批里所有接手都没救出来的（不在训练集里）；测试回合里补数据前 / 后卡住的 = 和 final 同样的 500 个测试回合"
         "（测试种子 100000 起）里，补数据前的学生 / 补数据后的模型第一次卡住的状态。"
         f"每个状态上四个模型各采 {R_SAMPLES} 个每次新 z / 新噪声的冷启动计划。第 8 步 = 计划第 8 步离当前推杆位置的距离"
         "（中位数，像素；示范状态上约 60）；离标签 = 计划和标签（实际执行的 16 个动作）逐步距离的平均（像素，中位数）。"
         "离救出来的 / 没救出来的 = 到最近的那类卡住状态的距离（12 维特征，中位数）。\n",
         "| 模型 | 种子 | 状态 | 状态数 | 标签第 8 步 | 第 8 步：前 / 后 / 老师 / 小 DP | 离标签：前 / 后 / 老师 / 小 DP | 离救出来的 / 没救出来的 |",
         "|---|---|---|---|---|---|---|---|"]
    for k, out in F.items():
        m, s = k.split("/")
        for sub, v in out.items():
            nn_ = (f"{v['nn_rescued']:.3f} / {v.get('nn_unrescued', float('nan')):.3f}" if "nn_rescued" in v else "–")
            L.append(f"| {m} | {s} | {LAB[sub]} | {v['n']} | {v['label8']:.0f}" if "label8" in v else
                     f"| {m} | {s} | {LAB[sub]} | {v['n']} | –")
            L[-1] += f" | {four(v, 'd8')} | {four(v, 'err') if 'err' in v['before'] else '–'} | {nn_} |"
    text = "\n".join(L) + "\n"
    open(os.path.join(OUT, "rescue_fit.md"), "w", encoding="utf-8").write(text)
    print(text)


def big_one(job):
    """大样本对比的一批：模型 m（种子 s）每次新 z / 新噪声闭环 n 个回合（环境种子 start 起）。返回逐回合的最好覆盖率、
    是否成功、执行的步数、彻底失败的回合是不是卡住了（first_stall），以及这批的动作粗糙度和规划耗时。"""
    import torch
    m, s, start, n = job
    torch.set_num_threads(1)
    met, tr = P.rollout(P.make_planner(model_kind(m), model_path(m, s), "fresh"), n, start, start=start)
    best = np.asarray(tr["best"], float)
    stall = np.array([b < 0.5 and first_stall(ag) is not None for b, ag in zip(best, tr["agent"])])
    return m, s, start, dict(best=best, success=np.asarray(tr["success"], bool), steps=np.array([len(q) for q in tr["acts"]]),
                             stall=stall, rough=met["action_rough"], ms=met["ms_per_plan"])


def big_stats(r):
    """一个（模型, 种子）或几个种子合在一起的逐回合结果 r → 指标。"""
    b, ok = r["best"], r["success"].astype(bool)
    cat = b < 0.5
    return dict(episodes=len(b), score=float(b.mean()), success=float(ok.mean()), catastrophic=float(cat.mean()),
                n_cat=int(cat.sum()), near=float(np.mean((b >= 0.9) & ~ok)), stalled=int(r["stall"].sum()),
                steps_success=float(r["steps"][ok].mean()) if ok.any() else float("nan"))


def big_pair(ra, rb):
    """同样的回合上两个模型逐回合配对：得分差（前者 − 后者）和它的标准误；成功、彻底失败各数只在前者 / 只在后者 / 两个都
    发生的回合，z = (b − c) / √(b + c)（McNemar）。"""
    n = len(ra["best"])
    d = ra["best"] - rb["best"]
    out = dict(n=n, d_score=float(d.mean()), se_score=float(d.std(ddof=1) / np.sqrt(n)))
    for key, xa, xb in (("success", ra["success"].astype(bool), rb["success"].astype(bool)),
                        ("catastrophic", ra["best"] < 0.5, rb["best"] < 0.5)):
        b_, c_ = int(np.sum(xa & ~xb)), int(np.sum(~xa & xb))
        out[key] = dict(only_a=b_, only_b=c_, both=int(np.sum(xa & xb)), diff=(b_ - c_) / n,
                        z=float((b_ - c_) / np.sqrt(b_ + c_)) if b_ + c_ else 0.0)
    return out


def stage_big(a):
    """大样本对比：--probe_models 里的模型，--seeds，每个种子 --big_episodes 个新回合（种子 s 的环境种子从 BIG_SEED + 10000·s
    起，和测试、DAgger、DART、两轮收集都不重叠；每批 1000 个并行），都用每次新 z / 新噪声（和第九轮收集时一样）。同一个种子的
    模型跑同样的回合，BIG_PAIRS 里的两个模型逐回合配对比较。逐回合结果存在 runs/pusht_tune/big/<模型>_seed<s>.npz（已有的
    不重做），汇总在 runs/pusht_tune/big.json / .md（big/ 里所有的 npz 都算进去）。"""
    d = os.path.join(OUT, "big")
    os.makedirs(d, exist_ok=True)
    npz = lambda m, s: os.path.join(d, f"{m}_seed{s}.npz")
    B, N = 1000, a.big_episodes
    todo = [(m, s) for m in a.probe_models for s in a.seeds if not os.path.exists(npz(m, s)) and os.path.exists(model_path(m, s))]
    jobs = [(m, s, BIG_SEED + 10000 * s + b, min(B, N - b)) for m, s in todo for b in range(0, N, B)]
    t0, parts = time.time(), {}
    with ProcessPoolExecutor(a.jobs, mp_context=mp.get_context("spawn")) as ex:
        for m, s, start, out in ex.map(big_one, jobs):
            parts.setdefault((m, s), []).append(out)
            print(f"  {m} seed {s} 环境种子 {start} 起：彻底失败 {int(np.sum(out['best'] < 0.5))}  ({time.time() - t0:.0f}s)", flush=True)
            if len(parts[(m, s)]) == -(-N // B):                         # 这个（模型, 种子）的所有批都回来了（ex.map 按顺序返回）
                ps = parts.pop((m, s))
                r = {k: np.concatenate([o[k] for o in ps]) for k in ("best", "success", "steps", "stall")}
                np.savez(npz(m, s), **r, rough=np.array([o["rough"] for o in ps]), ms=np.array([o["ms"] for o in ps]),
                         start=BIG_SEED + 10000 * s)
                st = big_stats(r)
                print(f"{m} seed {s}: {st['episodes']} 回合  得分 {st['score']:.3f}  成功 {st['success']:.3f}"
                      f"  彻底失败 {st['catastrophic']:.4f}（{st['n_cat']} 个，卡住 {st['stalled']}）", flush=True)
    # 汇总 big/ 里所有的结果：模型按 --probe_models、BIG_PAIRS 的顺序
    D = {}
    for f in sorted(os.listdir(d)):
        if f.endswith(".npz") and "_seed" in f:
            m, s = f[:-4].rsplit("_seed", 1)
            D[(m, int(s))] = dict(np.load(os.path.join(d, f)))
    order = list(dict.fromkeys(a.probe_models + [x for p in BIG_PAIRS for x in p] + sorted(m for m, _ in D)))
    J = dict(models={}, pairs={})
    L = [f"# 大样本对比（种子 s 的环境种子从 {BIG_SEED} + 10000·s 起，和测试、DAgger、DART、两轮收集都不重叠）\n",
         "都用每次新 z / 新噪声的冷启动（小 DP 也是每次新噪声），和第九轮收集时一样。彻底失败 = 最好覆盖率 < 0.5；差一点 = 最好"
         "覆盖率 ≥ 0.9 但没有成功；卡住 = 彻底失败的回合里推杆停住（之后 8、16 步位移都 < 10 像素，同接管检验）。"
         "均值 ± 种子间标准差；“合起来”把所有种子的回合放在一起算（不同种子的回合是不同的初始状态）。\n",
         "| 模型 | 种子 | 每个种子的回合 | 得分 | 成功率 | 彻底失败 | 彻底失败（合起来） | 其中卡住的 | 差一点 | 成功回合的步数 | 动作粗糙度 |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in order:
        seeds = sorted(s for mm, s in D if mm == m)
        if not seeds:
            continue
        S = {s: big_stats(D[(m, s)]) for s in seeds}
        for s in seeds:
            J["models"][f"{m}/{s}"] = dict(S[s], rough=float(np.mean(D[(m, s)]["rough"])), ms=float(np.mean(D[(m, s)]["ms"])))
        allr = {k: np.concatenate([D[(m, s)][k] for s in seeds]) for k in ("best", "success", "steps", "stall")}
        A_ = big_stats(allr)
        J["models"][f"{m}/all"] = A_
        L.append(f"| {m} | {','.join(map(str, seeds))} | {'/'.join(str(S[s]['episodes']) for s in seeds)}"
                 f" | {P.fmt([S[s]['score'] for s in seeds], nd=3)} | {P.fmt([S[s]['success'] for s in seeds], True)}"
                 f" | {P.fmt([S[s]['catastrophic'] for s in seeds], True)} | {100 * A_['catastrophic']:.2f}%（{A_['n_cat']} 个）"
                 f" | {A_['stalled'] / max(A_['n_cat'], 1):.0%} | {P.fmt([S[s]['near'] for s in seeds], True)}"
                 f" | {A_['steps_success']:.0f} | {np.mean([np.mean(D[(m, s)]['rough']) for s in seeds]):.2f} |")
    L += ["\n逐种子（得分 / 成功率 / 彻底失败个数）：\n"]
    for m in order:
        seeds = sorted(s for mm, s in D if mm == m)
        if seeds:
            L.append(f"- {m}：" + "；".join(f"种子 {s} {J['models'][f'{m}/{s}']['score']:.3f} / {100 * J['models'][f'{m}/{s}']['success']:.1f}%"
                                           f" / {J['models'][f'{m}/{s}']['n_cat']}" for s in seeds))
    # 彻底失败分成卡住 / 没卡住；得分分解：按最好覆盖率分四类，丢分 = 这类回合比满分少的部分（三类丢分加起来 = 1 − 得分）
    L += ["\n## 彻底失败的类型和得分分解（所有种子合起来）\n",
          "比例后面括号里是这类回合的平均最好覆盖率。中间 = 最好覆盖率 0.5–0.9。\n",
          "| 模型 | 卡住（逐种子） | 没卡住的彻底失败（逐种子） | 成功 | 差一点 | 中间 | 彻底失败 | 丢分：差一点 / 中间 / 彻底失败 |",
          "|---|---|---|---|---|---|---|---|"]
    J["decomp"] = {}
    for m in order:
        seeds = sorted(s for mm, s in D if mm == m)
        if not seeds:
            continue
        stl = [int(D[(m, s)]["stall"].sum()) for s in seeds]
        non = [int(np.sum((D[(m, s)]["best"] < 0.5) & ~D[(m, s)]["stall"].astype(bool))) for s in seeds]
        b = np.concatenate([D[(m, s)]["best"] for s in seeds])
        ok = np.concatenate([D[(m, s)]["success"] for s in seeds]).astype(bool)
        bins = [ok, (b >= 0.9) & ~ok, (b >= 0.5) & (b < 0.9), b < 0.5]
        loss = [float(np.sum(1 - b[v]) / len(b)) for v in bins[1:]]
        J["decomp"][m] = dict(stall=stl, nonstall=non, frac=[float(v.mean()) for v in bins],
                              mean=[float(b[v].mean()) if v.any() else float("nan") for v in bins], loss=loss)
        L.append(f"| {m} | {sum(stl)}（{'/'.join(map(str, stl))}） | {sum(non)}（{'/'.join(map(str, non))}） | "
                 + " | ".join(f"{100 * v.mean():.2f}%（{b[v].mean():.3f}）" if v.any() else "0%" for v in bins)
                 + " | " + " / ".join(f"{x:.4f}" for x in loss) + " |")
    L += ["\n## 逐回合配对（同一个种子的两个模型跑的是同样的回合）\n",
          "差 = 前者 − 后者。得分差的 ± 是逐回合差的标准误；“只有前者 / 只有后者”是只在一个模型上发生的回合数，"
          "z = (b − c) / √(b + c)（McNemar）。“合起来”那行只算了回合的随机性，没算训练种子之间的差别。\n",
          "| 前者 | 后者 | 种子 | 回合 | 得分差 | 成功率差 | 只有前者成功 / 只有后者成功 | z | 彻底失败差 | 只有前者彻底失败 / 只有后者 | 两个都彻底失败 | z |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    pair_seeds = lambda ma, mb: sorted(s for mm, s in D if mm == ma and (mb, s) in D
                                       and len(D[(mb, s)]["best"]) == len(D[(ma, s)]["best"]))
    for ma, mb in BIG_PAIRS:
        seeds = pair_seeds(ma, mb)
        if not seeds:
            continue
        rows = [(str(s), D[(ma, s)], D[(mb, s)]) for s in seeds]
        if len(seeds) > 1:
            cat = lambda X, k: np.concatenate([x[k] for x in X])
            rows.append(("合起来", {k: cat([D[(ma, s)] for s in seeds], k) for k in ("best", "success")},
                         {k: cat([D[(mb, s)] for s in seeds], k) for k in ("best", "success")}))
        for lab, ra, rb in rows:
            p = big_pair(ra, rb)
            J["pairs"][f"{ma}|{mb}/{lab if lab != '合起来' else 'all'}"] = p
            sc, ca = p["success"], p["catastrophic"]
            L.append(f"| {ma} | {mb} | {lab} | {p['n']} | {p['d_score']:+.4f} ± {p['se_score']:.4f}"
                     f" | {100 * sc['diff']:+.1f} | {sc['only_a']} / {sc['only_b']} | {sc['z']:+.1f}"
                     f" | {100 * ca['diff']:+.2f} | {ca['only_a']} / {ca['only_b']} | {ca['both']} | {ca['z']:+.1f} |")
    # 只在一个模型上彻底失败的回合里，那个模型是不是卡住了；只在一个模型上成功的回合里，另一个模型差多少
    L += ["\n配对时只在一个模型上发生的回合（所有种子合起来）：彻底失败的那个模型卡住的比例；成功时另一个模型的最好覆盖率"
          "（中位数，≥ 0.9 = 差一点的比例）。\n",
          "| 前者 | 后者 | 只有前者彻底失败 | 其中前者卡住 | 只有后者彻底失败 | 其中后者卡住 | 只有前者成功：后者中位数 / 差一点 | 只有后者成功：前者中位数 / 差一点 |",
          "|---|---|---|---|---|---|---|---|"]
    J["pairs_detail"] = {}
    for ma, mb in BIG_PAIRS:
        seeds = pair_seeds(ma, mb)
        if not seeds:
            continue
        g = lambda m, k: np.concatenate([D[(m, s)][k] for s in seeds])
        ba, bb = g(ma, "best"), g(mb, "best")
        sa, sb = g(ma, "stall").astype(bool), g(mb, "stall").astype(bool)
        ua, ub = g(ma, "success").astype(bool), g(mb, "success").astype(bool)
        oa, ob, wa, wb = (ba < 0.5) & (bb >= 0.5), (bb < 0.5) & (ba >= 0.5), ua & ~ub, ub & ~ua
        q = dict(only_a_cat=int(oa.sum()), stall_a=int(sa[oa].sum()), only_b_cat=int(ob.sum()), stall_b=int(sb[ob].sum()),
                 only_a_succ=int(wa.sum()), b_med=float(np.median(bb[wa])), b_near=float(np.mean(bb[wa] >= 0.9)),
                 only_b_succ=int(wb.sum()), a_med=float(np.median(ba[wb])), a_near=float(np.mean(ba[wb] >= 0.9)))
        J["pairs_detail"][f"{ma}|{mb}"] = q
        L.append(f"| {ma} | {mb} | {q['only_a_cat']} | {q['stall_a'] / max(q['only_a_cat'], 1):.0%} | {q['only_b_cat']}"
                 f" | {q['stall_b'] / max(q['only_b_cat'], 1):.0%} | {q['b_med']:.3f} / {q['b_near']:.0%}（{q['only_a_succ']} 个）"
                 f" | {q['a_med']:.3f} / {q['a_near']:.0%}（{q['only_b_succ']} 个） |")
    json.dump(J, open(os.path.join(OUT, "big.json"), "w", encoding="utf-8"), indent=1)
    text = "\n".join(L) + "\n"
    open(os.path.join(OUT, "big.md"), "w", encoding="utf-8").write(text)
    print(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True,
                    choices=["eval", "train", "summary", "probe", "chain", "blend", "diag", "noise", "stall", "ens", "cat",
                             "attract", "steps", "final", "dagger", "dagger_fit", "paths", "dart", "takeover", "rescue",
                             "rescue_fit", "big"])
    ap.add_argument("--probe_models", nargs="+", default=["nca", "se", "se_persist"], help="--stage probe / chain 检查的模型")
    ap.add_argument("--chain_n", type=int, default=10, help="--stage chain 在示范数据上连续再生的次数")
    ap.add_argument("--blend", nargs="+", default=None, help="--stage blend 的设置，形如 0.5:20（α:再生步数）；缺省用 BLEND_GRID")
    ap.add_argument("--tag", default="", help="--stage diag 输出文件名的后缀（diag<tag>.md），避免覆盖")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--variants", nargs="+", default=list(TRAIN_VARIANTS))
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--device_offset", type=int, default=0, help="GPU 轮流分配的起点（两次调用时用来平衡两张卡）")
    ap.add_argument("--eval_model", default="nca", help="--stage eval 评估哪个模型：nca = runs/pusht 的基线，否则是调参变体名")
    ap.add_argument("--grid", nargs="+", default=None, help="--stage eval 只跑 EVAL_GRID 里这几项")
    ap.add_argument("--episodes", type=int, default=500, help="--stage final 每个种子的测试回合数")
    ap.add_argument("--dagger_episodes", type=int, default=500, help="--stage dagger 每个（学生, 种子）闭环收集的回合数")
    ap.add_argument("--dart_episodes", type=int, default=1000, help="--stage dart 每个种子老师加扰动闭环的回合数")
    ap.add_argument("--rescue_episodes", type=int, default=4000, help="--stage rescue 每个（学生, 种子）闭环找卡住状态的回合数")
    ap.add_argument("--rescue_tries", type=int, default=3, help="--stage rescue 每个接手的规划器在每个卡住的状态上试几次")
    ap.add_argument("--rescue_seed", type=int, default=RESCUE_SEED, help=f"--stage rescue 的环境种子起点（第十轮用 {RESCUE_SEED2}）")
    ap.add_argument("--rescue_ctrl", nargs="+", default=RESCUE_BASE, choices=list(RESCUE_CTRL),
                    help="--stage rescue 接手的规划器（RESCUE_CTRL 的键；默认是第九轮的四个）")
    ap.add_argument("--fit_suffix", default="_dag", choices=["_dag", "_dart"], help="--stage dagger_fit 检查哪种补数据")
    ap.add_argument("--big_episodes", type=int, default=8000, help="--stage big 每个（模型, 种子）的回合数")
    ap.add_argument("--gpus", type=int, nargs="+", default=[0, 1], help="--stage train 轮流使用的 GPU 编号")
    a, extra = ap.parse_known_args()          # 其余参数原样传给 pusht.py（例如 --nca_iters 40000）
    if a.stage == "eval":
        stage_eval(a)
    elif a.stage == "train":
        stage_train(a, extra)
    elif a.stage == "probe":
        stage_probe(a)
    elif a.stage == "chain":
        stage_chain(a)
    elif a.stage == "blend":
        stage_blend(a)
    elif a.stage == "diag":
        stage_diag(a)
    elif a.stage == "noise":
        stage_noise(a)
    elif a.stage == "stall":
        stage_stall(a)
    elif a.stage == "ens":
        stage_ens(a)
    elif a.stage == "cat":
        stage_cat(a)
    elif a.stage == "attract":
        stage_attract(a)
    elif a.stage == "steps":
        stage_steps(a)
    elif a.stage == "final":
        stage_final(a)
    elif a.stage == "dagger":
        stage_dagger(a)
    elif a.stage == "dagger_fit":
        stage_dagger_fit(a)
    elif a.stage == "paths":
        stage_paths(a)
    elif a.stage == "dart":
        stage_dart(a)
    elif a.stage == "takeover":
        stage_takeover(a)
    elif a.stage == "rescue":
        stage_rescue(a)
    elif a.stage == "rescue_fit":
        stage_rescue_fit(a)
    elif a.stage == "big":
        stage_big(a)
    else:
        stage_summary(a)


if __name__ == "__main__":
    main()
