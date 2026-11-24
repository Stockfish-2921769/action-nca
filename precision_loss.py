#!/usr/bin/env python3
"""
精度损失实验：隐变量 NCA 每次换新 z 时碰撞多，是因为 min-of-K 只训练离示范最近的那个 z，
其余 z 长出的计划不受约束（模态已经确定的状态上，仍有不少 z 会长出另一侧、穿过障碍的计划）。
action_nca.py --prec_weight w 另采几个随机 z，要求它们的计划离附近状态上的某条示范动作块足够近。

本脚本对每个种子 s 成对比较 --prec_weight 0（原方法）和给定的权重：
同一份示范、同一个初始化种子、同一组评估起点。各次训练并行，每个进程单线程。
已有 metrics.json 的运行会跳过，中断后可以直接重跑。

用法
  python precision_loss.py                           # 种子 0–4，权重 0 和 1，每种方式 500 回合
  python precision_loss.py --weights 0 0.3 1 3       # 扫描权重
  python precision_loss.py --aggregate_only
输出：runs/precision/w<权重>_seed<s>/（不纳入版本控制）和 runs/precision/summary.md。
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SETTINGS = [("NCA 确定性，冷启动", "deterministic, cold start"),
            ("隐变量，每次新 z", "latent z, cold start, new z each plan"),
            ("隐变量，z 固定", "latent z, cold start, same z"),
            ("隐变量，再生", "latent z, warm start (regrow)")]


def out_dir(args, w, s):
    return os.path.join(args.runs, f"w{w:g}_seed{s}")


def train_eval(args, w, s):
    d = out_dir(args, w, s)
    if os.path.exists(os.path.join(d, "metrics.json")):
        print(f"跳过 w={w:g} seed {s}（已完成）", flush=True)
        return
    os.makedirs(d, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONIOENCODING="utf-8")
    cmd = [sys.executable, "action_nca.py", "--seed", str(s), "--out", d, "--episodes", str(args.episodes),
           "--prec_weight", str(w), "--prec_samples", str(args.prec_samples), "--prec_radius", str(args.prec_radius)]
    t0 = time.time()
    with open(os.path.join(d, "train_log.txt"), "w", encoding="utf-8") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=HERE).returncode
    if rc != 0:
        raise RuntimeError(f"w={w:g} seed {s} 失败（返回码 {rc}），见 {d}/train_log.txt")
    print(f"完成 w={w:g} seed {s}，{(time.time() - t0) / 60:.1f} 分钟", flush=True)


def fmt(v, pct):
    v = np.asarray(v, float)
    if pct:
        return f"{100 * v.mean():.1f}% ± {100 * v.std():.1f}%（{100 * v.min():.1f}–{100 * v.max():.1f}）"
    return f"{v.mean():.2f} ± {v.std():.2f}"


def aggregate(args):
    M = {w: {s: json.load(open(os.path.join(out_dir(args, w, s), "metrics.json"), encoding="utf-8"))
             for s in args.seeds} for w in args.weights}
    L = [f"# 精度损失实验（种子 {', '.join(map(str, args.seeds))}，每种方式 {args.episodes} 回合，"
         f"训练 1500 轮，评估种子 = 种子 + 1）\n",
         f"精度损失：每个观测另采 {args.prec_samples} 个 z，附近半径 {args.prec_radius}。"
         "单元格是均值 ± 标准差（最小–最大）。\n",
         "| 执行方式 | 精度损失权重 | 成功率 | 碰撞率 | 计划翻转 | 粗糙度 | 成功回合左/右 |",
         "|---|---|---|---|---|---|---|"]
    for name, key in SETTINGS:
        for w in args.weights:
            r = [M[w][s][key] for s in args.seeds]
            lr = "、".join(f"{x['left']}/{x['right']}" for x in r)
            L.append(f"| {name} | {w:g} | {fmt([x['success'] for x in r], True)} | "
                     f"{fmt([x['collision'] for x in r], True)} | {fmt([x['plan_flips'] for x in r], False)} | "
                     f"{fmt([x['roughness_x100'] for x in r], False)} | {lr} |")
    L += ["", "起点处第一段计划（2000 个 z）：", "", "| 权重 | 向左 | 向右 | 未决 |", "|---|---|---|---|"]
    for w in args.weights:
        f = [M[w][s]["first_plan_at_start"] for s in args.seeds]
        L.append(f"| {w:g} | {fmt([x['left'] for x in f], True)} | {fmt([x['right'] for x in f], True)} | "
                 f"{fmt([x['undecided'] for x in f], True)} |")
    base = args.weights[0]
    L += ["", f"逐种子：隐变量 NCA 的碰撞率（权重 {base:g} → 其他权重）", ""]
    for w in args.weights[1:]:
        for name, key in SETTINGS[1:]:
            pairs = [(M[base][s][key]["collision"], M[w][s][key]["collision"]) for s in args.seeds]
            better = sum(b < a for a, b in pairs)
            L.append(f"- 权重 {w:g}，{name}：" + "，".join(f"{100 * a:.1f}%→{100 * b:.1f}%" for a, b in pairs)
                     + f"；{better}/{len(pairs)} 个种子下降")
    text = "\n".join(L) + "\n"
    with open(os.path.join(args.runs, "summary.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--weights", type=float, nargs="+", default=[0.0, 1.0], help="第一个是对照")
    ap.add_argument("--prec_samples", type=int, default=2)
    ap.add_argument("--prec_radius", type=float, default=0.04)
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--runs", default=os.path.join(HERE, "runs", "precision"))
    ap.add_argument("--aggregate_only", action="store_true")
    args = ap.parse_args()
    if not args.aggregate_only:
        with ThreadPoolExecutor(args.jobs) as ex:
            for fut in [ex.submit(train_eval, args, w, s) for w in args.weights for s in args.seeds]:
                fut.result()
    aggregate(args)


if __name__ == "__main__":
    main()
