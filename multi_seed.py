#!/usr/bin/env python3
"""
多种子实验：用几个随机种子重复 NCA 和 Diffusion Policy 的主要对比，给出均值和波动范围，
并逐个种子检验 NOTES.md 第 9 节的几条结论。

每个种子 s（同时决定示范数据、模型初始化和训练采样；评估种子是 s + 1，NCA 和 DP 相同，
所以同一种子内是成对比较：同一份示范、同一组评估起点）：
  1. NCA：python action_nca.py --seed s（确定性 + 隐变量两个模型），CPU 单线程，各种子并行。
  2. 小 DP（固定用种子 0 上按训练损失选出的配置，不再按种子调参）和大 DP：在 --dp_device 上依次训练。
  3. 评估（CPU）：小 DP 的三种执行方式 + 步数扫描，并在同一个 rollout() 下评估该种子的 NCA；
     大 DP 只评估三种执行方式，作为参照。
  4. 汇总：每个指标的均值 ± 标准差（最小–最大），以及 H1–H5 在几个种子上成立。
     小 DP 的配置是在种子 0 上挑的，所以另给一张去掉种子 0 的表。

已经跑完的步骤（输出文件已存在）会跳过，中断后可以直接重跑。

用法
  python multi_seed.py                         # 种子 0–4，每种方式 500 回合，约 1.5 小时
  python multi_seed.py --aggregate_only        # 只重新汇总 runs/ 里已有的结果
输出：runs/seed_<s>/（模型、日志、各自的 metrics.json，不纳入版本控制），
     runs/summary.md、runs/summary.json、runs/summary.png。
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SMALL_DP = ["--dims", "8", "16", "32", "--kernel", "3", "--t_dim", "16", "--groups", "2", "--lr", "2e-3"]

# 汇总表里的执行方式：(显示名, 来源, metrics.json 里的名字)
SETTINGS = [
    ("DP 小，每次新噪声", "small", "DP, DDIM 10, new noise each plan"),
    ("DP 小，固定噪声", "small", "DP, DDIM 10, same noise"),
    ("DP 小，热启动（3 步）", "small", "DP, warm start (t=50, 3 steps)"),
    ("DP 大，每次新噪声", "large", "DP, DDIM 10, new noise each plan"),
    ("DP 大，固定噪声", "large", "DP, DDIM 10, same noise"),
    ("DP 大，热启动（3 步）", "large", "DP, warm start (t=50, 3 steps)"),
    ("NCA 确定性，冷启动", "small", "NCA, deterministic, cold start"),
    ("NCA 隐变量，每次新 z", "small", "NCA, latent z, cold start, new z each plan"),
    ("NCA 隐变量，z 固定", "small", "NCA, latent z, cold start, same z"),
    ("NCA 隐变量，再生", "small", "NCA, latent z, warm start (regrow)"),
]

_print_lock = threading.Lock()


def log(msg):
    with _print_lock:
        print(time.strftime("%H:%M:%S"), msg, flush=True)


def run(name, cmd, logfile, done_file, threads):
    """跑一个子进程；done_file 已存在就跳过。"""
    if os.path.exists(done_file):
        log(f"跳过 {name}（已完成）")
        return
    os.makedirs(os.path.dirname(logfile), exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads))
    log(f"开始 {name}")
    t0 = time.time()
    with open(logfile, "w") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=HERE).returncode
    if rc != 0:
        raise RuntimeError(f"{name} 失败（返回码 {rc}），见 {logfile}")
    log(f"完成 {name}，{(time.time() - t0) / 60:.1f} 分钟")


def run_all(args):
    py = sys.executable
    d = lambda s, *p: os.path.join(args.runs, f"seed_{s}", *p)
    nca_extra = ["--iters", str(args.nca_iters)] if args.nca_iters else []
    dp_extra = ["--iters", str(args.dp_iters)] if args.dp_iters else []
    E = ["--episodes", str(args.episodes)]

    def nca(s):
        run(f"NCA 训练 seed {s}", [py, "action_nca.py", "--seed", str(s), "--out", d(s, "nca"), *E, *nca_extra],
            d(s, "nca.log"), d(s, "nca", "metrics.json"), args.nca_threads)

    def dp_train(s, size):
        cfg = SMALL_DP if size == "small" else []
        run(f"DP {size} 训练 seed {s}",
            [py, "diffusion_baseline.py", "--train_only", "--device", args.dp_device, "--seed", str(s),
             "--out", d(s, f"dp_{size}"), *cfg, *dp_extra],
            d(s, f"dp_{size}_train.log"), d(s, f"dp_{size}", "dp_policy.pt"), 2)

    def dp_eval(s, size, waits):
        for w in waits:
            w.result()                                  # 等依赖的训练完成（失败会在这里抛出）
        extra = (["--nca_dir", d(s, "nca"), "--sweep_spacing", "leading", "--sweep_steps", *map(str, args.sweep_steps),
                  "--sweep_warm", *map(str, args.sweep_warm), "--sweep_nca", *map(str, args.sweep_nca)]
                 if size == "small" else ["--nca_dir", "none", "--no_sweep"])
        run(f"DP {size} 评估 seed {s}",
            [py, "diffusion_baseline.py", "--load", d(s, f"dp_{size}", "dp_policy.pt"), "--seed", str(s),
             "--out", d(s, f"dp_{size}"), *E, *extra],
            d(s, f"dp_{size}_eval.log"), d(s, f"dp_{size}", "metrics.json"), args.eval_threads)

    with ThreadPoolExecutor(len(args.seeds)) as cpu, ThreadPoolExecutor(1) as gpu, \
            ThreadPoolExecutor(args.eval_jobs) as ev:
        nf = {s: cpu.submit(nca, s) for s in args.seeds}
        sf = {s: gpu.submit(dp_train, s, "small") for s in args.seeds}   # 先训完所有小模型，评估能早点开始
        lf = {s: gpu.submit(dp_train, s, "large") for s in args.seeds}
        evs = [ev.submit(dp_eval, s, "small", [nf[s], sf[s]]) for s in args.seeds]
        evs += [ev.submit(dp_eval, s, "large", [lf[s]]) for s in args.seeds]
        for f in [*nf.values(), *sf.values(), *lf.values(), *evs]:
            f.result()


# ----------------------------------------------------------------------------- 汇总

def load(args):
    out = {}
    for s in args.seeds:
        p = lambda *x: os.path.join(args.runs, f"seed_{s}", *x)
        try:
            out[s] = dict(small=json.load(open(p("dp_small", "metrics.json"))),
                          large=json.load(open(p("dp_large", "metrics.json"))),
                          nca=json.load(open(p("nca", "metrics.json"))))
        except FileNotFoundError as e:
            print(f"seed {s} 结果不全，跳过：{e.filename}")
    return out


def sweep_point(m, key_prefix, steps):
    for k, pts in m["sweep"].items():
        if k.startswith(key_prefix):
            for p in pts:
                if p["steps"] == steps:
                    return p
    raise KeyError(f"{key_prefix} steps={steps}")


def stat(vals, fmt="{:.1%}"):
    v = np.asarray(vals, float)
    v = v[~np.isnan(v)]                      # 没有成功回合时粗糙度是 nan，跳过
    if len(v) == 0:
        return "–"
    sd = fmt.format(v.std(ddof=1)) if len(v) > 1 else "–"
    return f"{fmt.format(v.mean())} ± {sd}（{fmt.format(v.min())}–{fmt.format(v.max())}）"


def imbalance(m):
    return abs(m["left"] - m["right"]) / max(1, m["left"] + m["right"])


def table(R, seeds):
    cols = [("成功率", "success", "{:.1%}"), ("碰撞率", "collision", "{:.1%}"), ("计划翻转", "plan_flips", "{:.2f}"),
            ("粗糙度", "roughness_x100", "{:.2f}"), ("最大拐折", "max_kink_x100", "{:.1f}")]
    lines = ["| 执行方式 | " + " | ".join(c[0] for c in cols) + " | 左右失衡 | 每次规划调用 | 每次规划乘加 (M) |",
             "|---|" + "---|" * (len(cols) + 3)]
    for name, src, key in SETTINGS:
        ms = [R[s][src]["settings"][key] for s in seeds]
        cells = [stat([m[k] for m in ms], f) for _, k, f in cols]
        cells.append(stat([imbalance(m) for m in ms], "{:.2f}"))
        cells.append(f"{np.mean([m['calls_per_plan'] for m in ms]):.1f}")
        cells.append(f"{np.mean([m['macs_per_plan'] for m in ms]) / 1e6:.2f}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return lines


def hypotheses(R, seeds, args):
    S = lambda s, src, key: R[s][src]["settings"][key]
    rows, res = [], {}

    def check(hid, text, per_seed):
        ok = [bool(v[0]) for v in per_seed.values()]
        res[hid] = dict(text=text, holds=sum(ok), of=len(ok), per_seed={s: v[1] for s, v in per_seed.items()})
        detail = "；".join(f"s{s}: {v[1]}" for s, v in per_seed.items())
        rows.append(f"| {hid} | {text} | **{sum(ok)}/{len(ok)}** | {detail} |")

    check("H1", "NCA 再生的计划翻转 < 小 DP 热启动（3 步）",
          {s: (S(s, "small", SETTINGS[9][2])["plan_flips"] < S(s, "small", SETTINGS[2][2])["plan_flips"],
               f"{S(s, 'small', SETTINGS[9][2])['plan_flips']:.3f} 对 {S(s, 'small', SETTINGS[2][2])['plan_flips']:.3f}")
           for s in seeds})
    check("H2", "每次重新采随机性时，NCA 碰撞率 > 小 DP",
          {s: (S(s, "small", SETTINGS[7][2])["collision"] > S(s, "small", SETTINGS[0][2])["collision"],
               f"{S(s, 'small', SETTINGS[7][2])['collision']:.1%} 对 {S(s, 'small', SETTINGS[0][2])['collision']:.1%}")
           for s in seeds})
    nca4 = {s: sweep_point(R[s]["small"], "NCA regrow", 4) for s in seeds}
    for ws in (1, 2):
        dpw = {s: sweep_point(R[s]["small"], "DP warm start", ws) for s in seeds}
        check(f"H3.{ws}", f"小 DP 热启动 {ws} 步的成功率 ≥ NCA 再生续长 4 步",
              {s: (dpw[s]["success"] >= nca4[s]["success"],
                   f"{dpw[s]['success']:.1%}（{dpw[s]['calls_per_plan']:.1f} 次）对 {nca4[s]['success']:.1%}"
                   f"（{nca4[s]['calls_per_plan']:.1f} 次）") for s in seeds})
    det = {s: S(s, "small", SETTINGS[6][2]) for s in seeds}
    check("H4", "确定性 NCA 偏向哪边不受控：左右失衡 > 隐变量再生的左右失衡",
          {s: (imbalance(det[s]) > imbalance(S(s, "small", SETTINGS[9][2])),
               f"左/右 {det[s]['left']}/{det[s]['right']}") for s in seeds})
    n_left = sum(det[s]["left"] > det[s]["right"] for s in seeds)
    for who, get in (("隐变量 NCA", lambda s: R[s]["nca"]["first_plan_at_start"]),
                     ("小 DP", lambda s: R[s]["small"]["first_plan_at_start"])):
        check(f"H5 {who}", f"{who}：起点处第一段计划双峰（未决 < 5%，左右各 > 35%）",
              {s: (get(s)["undecided"] < 0.05 and min(get(s)["left"], get(s)["right"]) > 0.35,
                   f"{get(s)['left']:.0%}/{get(s)['right']:.0%}/{get(s)['undecided']:.1%}") for s in seeds})
    return rows, res, n_left


def plot(R, seeds, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    ax = axes[0]
    curves = [("DP small, fresh noise", "DP fresh noise", "#378ADD"), ("DP small, warm start", "DP warm start", "#1D9E75"),
              ("NCA regrow", "NCA regrow", "#E24B4A")]
    for label, prefix, c in curves:
        steps = [p["steps"] for k, v in R[seeds[0]]["small"]["sweep"].items() if k.startswith(prefix) for p in v]
        x = np.array([[sweep_point(R[s]["small"], prefix, st)["calls_per_plan"] for st in steps] for s in seeds])
        y = np.array([[sweep_point(R[s]["small"], prefix, st)["success"] for st in steps] for s in seeds])
        for xi, yi in zip(x, y):
            ax.plot(xi, yi, "o", color=c, alpha=0.25, ms=3)
        ax.plot(x.mean(0), y.mean(0), "-o", color=c, ms=5, label=label)
        ax.fill_between(x.mean(0), y.min(0), y.max(0), color=c, alpha=0.15)
    ax.set_xscale("log")
    ax.set_xlabel("network calls per plan (log)")
    ax.set_ylabel("success rate")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.set_title(f"success vs. compute ({len(seeds)} seeds; band = min–max)", fontsize=10)
    ax.legend(fontsize=8)

    short = ["DP-s fresh", "DP-s fixed", "DP-s warm", "DP-L fresh", "DP-L fixed", "DP-L warm",
             "NCA det", "NCA new z", "NCA same z", "NCA regrow"]
    for ax, key, title in ((axes[1], "plan_flips", "plan flips per episode"), (axes[2], "collision", "collision rate")):
        for i, (_, src, k) in enumerate(SETTINGS):
            v = [R[s][src]["settings"][k][key] for s in seeds]
            c = "#E24B4A" if k.startswith("NCA") else "#378ADD"
            ax.bar(i, np.mean(v), color=c, alpha=0.35)
            ax.plot([i] * len(v), v, "o", color=c, ms=4)
        ax.set_xticks(range(len(SETTINGS)))
        ax.set_xticklabels(short, rotation=45, ha="right", fontsize=8)
        ax.set_title(f"{title} (bar = mean, dots = seeds)", fontsize=10)
        ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def aggregate(args):
    R = load(args)
    seeds = sorted(R)
    if len(seeds) < 2:
        print("完成的种子不足 2 个，无法汇总")
        return
    cfg = R[seeds[0]]["small"]["config"]
    lines = [f"# 多种子实验汇总", "",
             f"种子 {', '.join(map(str, seeds))}（每个种子同时决定示范数据、初始化和训练采样；评估种子 = 种子 + 1）。"
             f"每种执行方式评估 {cfg['episodes']} 回合。NCA：默认参数（1500 轮，除非用 --nca_iters 改过）；"
             f"DP：{cfg['iters']} 轮，batch {cfg['batch']}。小 DP 的配置是在种子 0 上按训练损失选出的，之后不再调整。", "",
             "单元格是 均值 ± 标准差（最小–最大）。左右失衡 = |左 − 右| / (左 + 右)，只计成功回合。", "",
             f"## 全部种子（{len(seeds)} 个）", "", *table(R, seeds), ""]
    rest = [s for s in seeds if s != 0]
    if 0 in seeds and len(rest) >= 2:
        lines += [f"## 去掉种子 0（{len(rest)} 个；种子 0 参与了小 DP 的配置选择）", "", *table(R, rest), ""]
    hrows, hres, n_left = hypotheses(R, seeds, args)
    lines += ["## 逐种子检验", "", "| 编号 | 结论 | 成立的种子数 | 各种子的数值 |", "|---|---|---|---|", *hrows, "",
              f"确定性 NCA 在 {n_left}/{len(seeds)} 个种子上偏左（成功回合里左多于右），其余偏右。", ""]
    with open(os.path.join(args.runs, "summary.md"), "w") as f:
        f.write("\n".join(lines))
    with open(os.path.join(args.runs, "summary.json"), "w") as f:
        json.dump(dict(seeds=seeds, hypotheses=hres, det_left_majority=n_left,
                       settings={name: {s: R[s][src]["settings"][k] for s in seeds} for name, src, k in SETTINGS}),
                  f, indent=2, ensure_ascii=False)
    plot(R, seeds, os.path.join(args.runs, "summary.png"))
    print("\n".join(lines))
    print(f"\nsaved to {args.runs}/summary.md, summary.json, summary.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--dp_device", default="mps", help="DP 训练用的设备（mps / cuda / cpu）")
    ap.add_argument("--nca_threads", type=int, default=1, help="每个 NCA 训练进程的 CPU 线程数")
    ap.add_argument("--eval_threads", type=int, default=2)
    ap.add_argument("--eval_jobs", type=int, default=2, help="同时跑几个评估")
    ap.add_argument("--sweep_steps", type=int, nargs="+", default=[3, 5, 10])
    ap.add_argument("--sweep_warm", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--sweep_nca", type=int, nargs="+", default=[2, 4, 6, 10])
    ap.add_argument("--nca_iters", type=int, default=None, help="只用于快速检查流程")
    ap.add_argument("--dp_iters", type=int, default=None, help="只用于快速检查流程")
    ap.add_argument("--aggregate_only", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.runs, exist_ok=True)
    if not args.aggregate_only:
        t0 = time.time()
        run_all(args)
        log(f"全部完成，{(time.time() - t0) / 60:.1f} 分钟")
    aggregate(args)


if __name__ == "__main__":
    main()
