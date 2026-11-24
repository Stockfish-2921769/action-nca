#!/usr/bin/env python3
"""
障碍随机化，并让策略看见障碍。

任务：起点、终点不变；每条示范 / 每个回合的矩形障碍随机（中心 x、y，半宽，半高）。
每个障碍生成左绕、右绕两条示范（多模态保留），绕行幅度随障碍而变：离障碍外缘留 0.12–0.3 的余量。
观测 = 当前位置 + 障碍参数（4 维）；“看不见”的对照组只给位置（同样的随机障碍示范）。

模型（每个种子成对比较，同一份示范、同一组评估障碍和起点）：
  NCA：隐变量 NCA + 精度损失（权重 1），--nca_iters 轮；
  DP： 小 DP（multi_seed.py 的配置），DDIM 10 步，--dp_iters 轮。
评估：每次新噪声 / z、固定噪声 / z、热启动 / 再生，两组障碍：
  in： 和训练同分布的新障碍；
  ood：比训练时更大的障碍（半宽 0.30–0.38、半高 0.20–0.25）。

用法
  python random_obstacle.py --plot_demos                    # 先看看示范长什么样
  python random_obstacle.py --all                           # 种子 0–4，NCA / DP × 看见 / 看不见
  python random_obstacle.py --all --aggregate_only
输出：runs/random/<模型>_<see|blind>_seed<s>/（模型、metrics.json、日志）和 runs/random/summary.md。
"""
import argparse
import json
import os
import subprocess
import sys
import time
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

import action_nca as A
import diffusion_baseline as DB

HERE = os.path.dirname(os.path.abspath(__file__))
KEYS = ("cx", "cy", "hw", "hh")
RANGES = {"in": dict(cx=(-0.15, 0.15), cy=(-0.3, 0.3), hw=(0.12, 0.30), hh=(0.08, 0.20)),
          "ood": dict(cx=(-0.15, 0.15), cy=(-0.3, 0.3), hw=(0.30, 0.38), hh=(0.20, 0.25))}
SETTINGS = {"nca": [("每次新 z", "cold_fresh"), ("z 固定", "cold_fixed"), ("再生", "warm")],
            "dp": [("每次新噪声", "fresh"), ("固定噪声", "fixed"), ("热启动", "warm")]}


# ----------------------------------------------------------------------------- 任务

def sample_rects(n, rng, which="in"):
    r = RANGES[which]
    return np.stack([rng.uniform(*r[k], n) for k in KEYS], 1).astype(np.float32)


def smoothstep(w):
    w = np.clip(w, 0.0, 1.0)
    return w * w * (3.0 - 2.0 * w)


def make_demos(rects, rng):
    """每个障碍一条左绕、一条右绕的示范。返回 demos (2n, DEMO_LEN + 1, 2) 和对应的障碍参数 (2n, 4)。
    横向偏移的形状：障碍下方一段平滑地偏出去，障碍高度范围内保持偏出，过了障碍再平滑地收回。"""
    u = np.linspace(0.0, 1.0, A.DEMO_LEN + 1)
    demos, cond = [], []
    for cx, cy, hw, hh in rects:
        for side in (-1.0, 1.0):
            while True:
                x0 = A.START[0] + rng.normal(0.0, 0.02)
                y0 = A.START[1] + rng.normal(0.0, 0.02)
                y = y0 + (A.GOAL[1] - y0) * u
                amp = side * cx + hw + rng.uniform(0.12, 0.30)          # 偏到障碍外缘之外 0.12–0.3
                lo, hi = cy - hh - 0.03, cy + hh + 0.03                 # 在这个高度区间里必须已经偏出去
                a0 = max(lo - rng.uniform(0.2, 0.35), y0 + 0.05)
                b1 = min(hi + rng.uniform(0.2, 0.35), A.GOAL[1] - 0.05)
                prof = smoothstep((y - a0) / (lo - a0)) * (1.0 - smoothstep((y - hi) / (b1 - hi)))
                x = x0 * (1.0 - u) + A.GOAL[0] * u + side * amp * prof
                p = np.stack([x, y], 1).astype(np.float32)
                r = np.tile(np.array([[cx, cy, hw, hh]], np.float32), (len(p) - 1, 1))
                if not A.seg_hits_rects(p[:-1], p[1:], r).any():
                    break
            demos.append(p)
            cond.append([cx, cy, hw, hh])
    return np.stack(demos), np.array(cond, np.float32)


# ----------------------------------------------------------------------------- 训练和评估（单个任务）

def nca_args(a):
    return Namespace(iters=a.nca_iters, batch=32, k=8, z_dim=4, lr=2e-3, pool=1024, obs_noise=0.01,
                     cold_min=24, cold_max=40, warm_min=6, warm_max=14, prec_weight=1.0, prec_samples=2,
                     prec_radius=0.04, push_frac=0.0, log_every=250, seed=a.seed)


def dp_args(a):
    return Namespace(method="ddim", iters=a.dp_iters, batch=256, lr=2e-3, lr_warmup=300, ema=0.999,
                     dims=(8, 16, 32), kernel=3, t_dim=16, groups=2, obs_noise=0.01, log_every=500, seed=a.seed)


def run_job(a):
    """训练一个模型并评估。a.model ∈ {nca, dp}，a.see 决定策略是否看见障碍。"""
    out = a.out
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    demos, cond = make_demos(sample_rects(a.n_obstacles, rng), rng)
    c = cond if a.see else None
    t0 = time.time()
    if a.model == "nca":
        torch.set_num_threads(1)
        model = A.train("latent", demos, nca_args(a), torch.device("cpu"), cond=c)
        torch.save(dict(state=model.state_dict(), obs_dim=6 if a.see else 2), os.path.join(out, "nca_latent.pt"))
        model.eval()
        planner = lambda st: DB.NCAPlanner(model, "latent", st)
    else:
        dev = torch.device(a.device)
        policy = DB.train_dp(demos, dp_args(a), dev, cond=c).cpu().eval()
        torch.save(dict(state=policy.state_dict(), lo=policy.lo, hi=policy.hi, cond=bool(a.see),
                        cond_lo=getattr(policy, "clo", None), cond_hi=getattr(policy, "chi", None),
                        dims=(8, 16, 32), kernel=3, t_dim=16, groups=2), os.path.join(out, "dp_policy.pt"))
        torch.set_num_threads(1)
        planner = lambda st: DB.DPPlanner(policy, st, 10, "leading", 50, 3)
    train_min = (time.time() - t0) / 60
    res = dict(train_minutes=train_min, n_demos=len(demos))
    for which, off in (("in", 1000), ("ood", 2000)):
        rects = sample_rects(a.episodes, np.random.default_rng(a.seed + off), which)
        for name, st in SETTINGS[a.model]:
            m = DB.rollout(planner(st), a.episodes, a.max_steps, "cpu", a.seed + 1, rects=rects, see=a.see)[0]
            res[f"{which}/{st}"] = m
            print(f"{which:3s} {name:6s} success {m['success']:.3f} collision {m['collision']:.3f} "
                  f"timeout {m['timeout']:.3f} flips {m['plan_flips']:.2f} kink {m['max_kink_x100']:.2f}", flush=True)
    with open(os.path.join(out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)


# ----------------------------------------------------------------------------- 画示范

def plot_demos(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    rng = np.random.default_rng(a.seed)
    rects = sample_rects(12, rng)
    demos, cond = make_demos(rects, rng)
    fig, axs = plt.subplots(2, 6, figsize=(15, 5.6))
    for i, ax in enumerate(axs.flat):
        cx, cy, hw, hh = rects[i]
        ax.add_patch(Rectangle((cx - hw, cy - hh), 2 * hw, 2 * hh, fc="#d9d8d3", ec="none"))
        for j in (2 * i, 2 * i + 1):
            ax.plot(demos[j, :, 0], demos[j, :, 1], "-o", ms=2, lw=1, color="#2a78d6" if j % 2 == 0 else "#eb6834")
        ax.set(xlim=(-1, 1), ylim=(-0.95, 0.95), aspect="equal", xticks=[], yticks=[])
    fig.suptitle("12 random obstacles, one left and one right demo each")   # 服务器上的 matplotlib 找不到中文字体
    path = os.path.join(a.runs, "demos.png")
    os.makedirs(a.runs, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    print(f"saved {path}")


# ----------------------------------------------------------------------------- 调度和汇总

def run_all(a):
    jobs = [(m, see, s) for s in a.seeds for m in ("nca", "dp") for see in (True, False)]

    def launch(job):
        m, see, s = job
        d = os.path.join(a.runs, f"{m}_{'see' if see else 'blind'}_seed{s}")
        if os.path.exists(os.path.join(d, "metrics.json")):
            return
        os.makedirs(d, exist_ok=True)
        cmd = [sys.executable, os.path.abspath(__file__), "--job", m, "--seed", str(s), "--out", d,
               "--nca_iters", str(a.nca_iters), "--dp_iters", str(a.dp_iters), "--episodes", str(a.episodes),
               "--n_obstacles", str(a.n_obstacles), "--device", f"cuda:{s % 2}" if torch.cuda.is_available() else "cpu"]
        if not see:
            cmd.append("--blind")
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONIOENCODING="utf-8")
        t0 = time.time()
        with open(os.path.join(d, "log.txt"), "w", encoding="utf-8") as f:
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=HERE).returncode
        print(f"{'完成' if rc == 0 else '失败'} {os.path.basename(d)}，{(time.time() - t0) / 60:.1f} 分钟", flush=True)

    with ThreadPoolExecutor(a.jobs) as ex:
        list(ex.map(launch, jobs))


def fmt(v, pct):
    v = np.asarray(v, float)
    return f"{100 * v.mean():.1f}% ± {100 * v.std():.1f}%" if pct else f"{v.mean():.2f} ± {v.std():.2f}"


def aggregate(a):
    M = {}
    for m in ("nca", "dp"):
        for see in ("see", "blind"):
            for s in a.seeds:
                f = os.path.join(a.runs, f"{m}_{see}_seed{s}", "metrics.json")
                if os.path.exists(f):
                    M[(m, see, s)] = json.load(open(f, encoding="utf-8"))
    L = [f"# 障碍随机化（种子 {', '.join(map(str, a.seeds))}；每个种子 {a.n_obstacles} 个障碍 × 左右两条示范；"
         f"每种方式 {a.episodes} 回合，评估种子 = 种子 + 1）\n",
         f"NCA：隐变量 + 精度损失权重 1，{a.nca_iters} 轮；DP：小 DP，DDIM 10 步，{a.dp_iters} 轮。"
         "in = 和训练同分布的新障碍；ood = 更大的障碍（半宽 0.30–0.38、半高 0.20–0.25）。"
         "“看见”= 观测包含障碍参数；“看不见”= 只有位置。单元格是均值 ± 标准差。\n"]
    for which in ("in", "ood"):
        L += [f"## {which}\n", "| 方法 | 障碍 | 执行方式 | 种子数 | 成功率 | 碰撞率 | 超时 | 计划翻转 | 粗糙度 | 最大拐折 | 成功回合左/右 |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for m, mn in (("nca", "NCA"), ("dp", "DP")):
            for see, sn in (("see", "看见"), ("blind", "看不见")):
                for name, st in SETTINGS[m]:
                    r = [M[(m, see, s)][f"{which}/{st}"] for s in a.seeds if (m, see, s) in M]
                    if not r:
                        continue
                    L.append(f"| {mn} | {sn} | {name} | {len(r)} | {fmt([x['success'] for x in r], True)} | "
                             f"{fmt([x['collision'] for x in r], True)} | {fmt([x['timeout'] for x in r], True)} | "
                             f"{fmt([x['plan_flips'] for x in r], False)} | {fmt([x['roughness_x100'] for x in r], False)} | "
                             f"{fmt([x['max_kink_x100'] for x in r], False)} | "
                             f"{sum(x['left'] for x in r)}/{sum(x['right'] for x in r)} |")
        L.append("")
    tm = {k: np.mean([M[x]["train_minutes"] for x in M if x[:2] == k]) for k in
          {x[:2] for x in M}}
    L.append("训练时间（分钟，均值）：" + "，".join(f"{m} {see} {v:.1f}" for (m, see), v in sorted(tm.items())))
    text = "\n".join(L) + "\n"
    with open(os.path.join(a.runs, "summary.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="跑所有种子和组合（每个组合一个子进程）")
    ap.add_argument("--aggregate_only", action="store_true")
    ap.add_argument("--plot_demos", action="store_true")
    ap.add_argument("--job", choices=["nca", "dp"], help="只跑一个组合（--all 内部调用）")
    ap.add_argument("--blind", action="store_true", help="策略看不见障碍（只给位置）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--n_obstacles", type=int, default=500, help="每个种子的训练障碍数（示范数 = 2 倍）")
    ap.add_argument("--nca_iters", type=int, default=3000)
    ap.add_argument("--dp_iters", type=int, default=6000)
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--max_steps", type=int, default=72)
    ap.add_argument("--device", default="cpu", help="DP 训练用的设备")
    ap.add_argument("--jobs", type=int, default=20)
    ap.add_argument("--out", default=None)
    ap.add_argument("--runs", default=os.path.join(HERE, "runs", "random"))
    a = ap.parse_args()
    if a.plot_demos:
        plot_demos(a)
    elif a.job:
        a.model, a.see = a.job, not a.blind
        a.out = a.out or os.path.join(a.runs, f"{a.job}_{'blind' if a.blind else 'see'}_seed{a.seed}")
        run_job(a)
    elif a.all:
        if not a.aggregate_only:
            run_all(a)
        aggregate(a)


if __name__ == "__main__":
    main()
