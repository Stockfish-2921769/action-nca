#!/usr/bin/env python3
"""
推理时可微避障引导（能量型 NCA 的第 1 层，不重新训练）。

代价 C(plan) = Σ relu(margin − sdf(p))²，p 取“当前位置 → 计划各航点”折线上的插值点，
sdf 是到矩形障碍的有符号距离。引导就是沿 −η∇C 推动作通道。两种放法、同样的梯度步数和步长：
  interleave：NCA 每长一步就推一次，修正之后由 NCA 继续生长、往邻近细胞传播；
  post：      先照常长完，再连续推同样多次，中间没有 NCA 参与（逐航点修补）。
两者的差别就是“生长机制有没有帮上忙”。

模型用 precision_loss.py 训练好的 runs/precision/w<权重>_seed<s>/nca_latent.pt，
评估和 action_nca.py 一样（同一个 run_episodes，评估种子 = 种子 + 1）。

用法
  python guidance.py                                  # 种子 0–4，精度损失权重 0 和 1 的模型
  python guidance.py --etas 0.1 --modes interleave    # 只看一个配置
  python guidance.py --aggregate_only
输出：runs/guidance/results.json 和 runs/guidance/summary.md。
"""
import argparse
import json
import os
import time
from argparse import Namespace
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

import action_nca as A

HERE = os.path.dirname(os.path.abspath(__file__))
SETTINGS = [("每次新 z", "cold_fresh"), ("z 固定", "cold_fixed"), ("再生", "warm")]


box_sdf = A.box_sdf   # 到矩形障碍的有符号距离（外正内负）


def obstacle_cost(plan, obs, margin, n=4):
    """每条计划的避障代价。plan: (B, T_P, 2)，obs: (B, 2)。返回 (B,)。"""
    P = torch.cat([obs[:, None], plan], 1)
    ts = torch.linspace(0.0, 1.0, n + 1, device=plan.device)[:-1].view(1, 1, -1, 1)
    pts = P[:, :-1, None] + ts * (P[:, 1:, None] - P[:, :-1, None])
    return (torch.relu(margin - box_sdf(pts)) ** 2).flatten(1).sum(1)


class GuidedNCA:
    """包住训练好的 ActionNCA，只替换 grow()：在生长过程中（或之后）加避障梯度。"""

    def __init__(self, model, eta, margin, mode):
        self.m, self.eta, self.margin, self.mode = model, eta, margin, mode
        self.blank = model.blank

    def push(self, s, obs):
        with torch.enable_grad():
            a = s[:, :2].detach().requires_grad_(True)
            c = obstacle_cost(a.transpose(1, 2), obs, self.margin).sum()
            g, = torch.autograd.grad(c, a)
        return torch.cat([s[:, :2] - self.eta * g, s[:, 2:]], 1)

    def grow(self, s, obs, z, steps, frames=None):
        if self.eta == 0:
            return self.m.grow(s, obs, z, steps)
        seed, emb = self.m.seed(obs, z), self.m.obs_net(obs)
        for _ in range(steps):
            s = self.m.step(s, seed, emb)
            if self.mode == "interleave":
                s = self.push(s, obs)
        if self.mode == "post":
            for _ in range(steps):
                s = self.push(s, obs)
        return s


def evaluate(job):
    torch.set_num_threads(1)
    w, seed, mode, eta, setting, a = job
    model = A.ActionNCA(z_dim=a["z_dim"])
    path = os.path.join(a["models"], f"w{w:g}_seed{seed}", "nca_latent.pt")
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    ev = Namespace(episodes=a["episodes"], z_dim=a["z_dim"], max_steps=72,
                   cold_steps=a["cold_steps"], warm_steps=a["warm_steps"])
    t0 = time.time()
    m = A.run_episodes(GuidedNCA(model, eta, a["margin"], mode), "latent", setting, ev, "cpu", seed=seed + 1)[0]
    m["seconds"] = time.time() - t0
    return dict(w=w, seed=seed, mode=mode, eta=eta, setting=setting, **m)


def fmt(v, pct):
    v = np.asarray(v, float)
    if pct:
        return f"{100 * v.mean():.1f}% ± {100 * v.std():.1f}%"
    return f"{v.mean():.2f} ± {v.std():.2f}"


def aggregate(args, rows):
    L = [f"# 推理时可微避障引导（种子 {', '.join(map(str, args.seeds))}，每种方式 {args.episodes} 回合，"
         f"评估种子 = 种子 + 1，margin {args.margin}）\n",
         "模型来自 precision_loss.py（训练 1500 轮）。单元格是 5 个种子的均值 ± 标准差。"
         "η 是每个梯度步的步长；interleave 和 post 的梯度步数相同（冷启动 "
         f"{args.cold_steps} 次，再生 {args.warm_steps} 次）。\n"]
    configs = [("—", 0.0)] + [(mo, e) for mo in args.modes for e in args.etas if e > 0]
    for w in args.weights:
        L += [f"## 精度损失权重 {w:g} 的模型\n",
              "| 执行方式 | 引导 | η | 成功率 | 碰撞率 | 超时 | 计划翻转 | 粗糙度 | 碰撞下降的种子 |",
              "|---|---|---|---|---|---|---|---|---|"]
        for name, setting in SETTINGS:
            base = {r["seed"]: r["collision"] for r in rows
                    if r["w"] == w and r["setting"] == setting and r["eta"] == 0}
            for mo, e in configs:
                r = sorted([r for r in rows if r["w"] == w and r["setting"] == setting and r["eta"] == e
                            and (e == 0 or r["mode"] == mo)], key=lambda r: r["seed"])
                if not r:
                    continue
                better = "" if e == 0 else f"{sum(x['collision'] < base[x['seed']] for x in r)}/{len(r)}"
                L.append(f"| {name} | {mo} | {e:g} | {fmt([x['success'] for x in r], True)} | "
                         f"{fmt([x['collision'] for x in r], True)} | {fmt([x['timeout'] for x in r], True)} | "
                         f"{fmt([x['plan_flips'] for x in r], False)} | {fmt([x['roughness_x100'] for x in r], False)} | "
                         f"{better} |")
        L.append("")
    text = "\n".join(L)
    with open(os.path.join(args.runs, "summary.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--weights", type=float, nargs="+", default=[0.0, 1.0], help="用哪些精度损失权重训练的模型")
    ap.add_argument("--etas", type=float, nargs="+", default=[0.0, 0.03, 0.1, 0.3])
    ap.add_argument("--modes", nargs="+", default=["interleave", "post"])
    ap.add_argument("--margin", type=float, default=0.03, help="离障碍多近开始有代价")
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--cold_steps", type=int, default=32)
    ap.add_argument("--warm_steps", type=int, default=10)
    ap.add_argument("--z_dim", type=int, default=4)
    ap.add_argument("--models", default=os.path.join(HERE, "runs", "precision"))
    ap.add_argument("--runs", default=os.path.join(HERE, "runs", "guidance"))
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--aggregate_only", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.runs, exist_ok=True)
    out = os.path.join(args.runs, "results.json")
    if args.aggregate_only:
        rows = json.load(open(out, encoding="utf-8"))
    else:
        a = dict(vars(args))
        jobs = [(w, s, mo, e, st, a) for w in args.weights for s in args.seeds for _, st in SETTINGS
                for mo, e in [("—", 0.0)] * (0.0 in args.etas)
                + [(mo, e) for mo in args.modes for e in args.etas if e > 0]]
        t0 = time.time()
        with ProcessPoolExecutor(args.jobs) as ex:
            rows = list(ex.map(evaluate, jobs))
        print(f"{len(jobs)} 次评估，{(time.time() - t0) / 60:.1f} 分钟")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=1, ensure_ascii=False)
    aggregate(args, rows)


if __name__ == "__main__":
    main()
