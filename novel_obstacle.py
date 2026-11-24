#!/usr/bin/env python3
"""
新障碍实验：测试时在绕行路线上放训练时没见过的圆形障碍，策略看不到它，只能通过可微代价（引导）知道。
比较 NCA 和小 Diffusion Policy，在同样的避障代价、同样的梯度步数和步长下：
  none：      不引导（看策略自己会撞多少）；
  interleave：生成过程中推（NCA 每长一步推一次；DP 把同样多次推动分到各个去噪步，推的是 x0 预测）；
  post：      生成完再推同样多次，生成器不参与（逐航点修补）。
梯度步数：冷启动 / 每次新噪声 / 固定噪声 32 次，再生 / 热启动 10 次（NCA 和 DP 相同）。

场景（每个回合各自采样，评估种子 + 100 决定障碍位置）：
  left：  固定一个圆（中心 (−0.6, 0)，半径 0.1）挡在左绕路线正中；
  big：   随机一侧的路线正中放一个半径 0.16 的圆（比 --push_frac 训练里的圆都大）；
  random2：左右两条路线上各放一个圆（半径 0.08），中心取在一条随机示范式路线上（再加一点扰动），
          所以无论选哪边都要绕开。

模型：NCA 用 runs/precision/w1_seed<s>/nca_latent.pt（精度损失权重 1，目前最好的 NCA）；
DP 用 runs/novel/dp_seed<s>/dp_policy.pt（multi_seed.py 里的小 DP 配置，6000 轮）。
避障代价同 guidance.py：Σ relu(margin − sdf)²，sdf 取矩形障碍和所有圆的距离，margin 0.03。

用法
  python novel_obstacle.py
  python novel_obstacle.py --aggregate_only
输出：runs/novel/results.json 和 runs/novel/summary.md。
"""
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

import action_nca as A
import diffusion_baseline as DB

HERE = os.path.dirname(os.path.abspath(__file__))
COLD_PUSH, WARM_PUSH = 32, 10
METHODS = [("NCA，每次新 z", "nca", "cold_fresh"), ("NCA，z 固定", "nca", "cold_fixed"), ("NCA，再生", "nca", "warm"),
           ("DP，每次新噪声", "dp", "fresh"), ("DP，固定噪声", "dp", "fixed"), ("DP，热启动", "dp", "warm")]


# ----------------------------------------------------------------------------- 场景和代价

def make_scene(name, E, rng):
    """每个回合的额外圆形障碍，(E, K, 3)：圆心 x、y 和半径。"""
    if name == "left":
        return np.tile(np.array([[[-0.6, 0.0, 0.1]]], np.float32), (E, 1, 1))
    if name == "random2":
        out = []
        for side in (-1.0, 1.0):
            amp = rng.uniform(0.5, 0.7, E)
            u = rng.uniform(0.35, 0.65, E)
            w = np.clip((u - 0.2) / 0.6, 0.0, 1.0)
            x = side * amp * 0.5 * (1.0 - np.cos(2.0 * np.pi * w)) + rng.normal(0.0, 0.03, E)
            y = A.START[1] + (A.GOAL[1] - A.START[1]) * u + rng.normal(0.0, 0.03, E)
            out.append(np.stack([x, y, np.full(E, 0.08)], 1))
        return np.stack(out, 1).astype(np.float32)
    if name == "big":         # 比推动训练里的圆（半径 ≤ 0.12）大：随机一侧放一个半径 0.16 的圆
        side = np.where(rng.random(E) < 0.5, -1.0, 1.0)
        y = rng.uniform(-0.1, 0.1, E)
        x = side * rng.uniform(0.6, 0.7, E)
        return np.stack([x, y, np.full(E, 0.16)], 1)[:, None].astype(np.float32)
    raise ValueError(name)


def circle_hits(circles):
    """返回 extra_hit(a, b)：线段 a→b 上取 12 个点，有点落在任何圆内就算碰撞。"""
    c = circles

    def hit(a, b, n=12):
        ts = np.linspace(0.0, 1.0, n)[None, :, None]
        p = a[:, None] + ts * (b - a)[:, None]                                    # (E, n, 2)
        d = np.linalg.norm(p[:, :, None] - c[:, None, :, :2], axis=-1)            # (E, n, K)
        return (d < c[:, None, :, 2]).any((1, 2))
    return hit


class Pusher:
    """沿 −η∇C 推计划（世界坐标，(E, T_P, 2)）。C 包括矩形障碍和本回合的圆。
    proj = "normal" 时只推垂直于路径方向的分量（圆在正前方时，完整梯度几乎全是“向后”的，会顶住机器人）。"""

    def __init__(self, circles, eta, margin, proj="full"):
        self.c, self.eta, self.margin, self.proj = torch.from_numpy(circles), eta, margin, proj

    def cost(self, plan, obs):
        return A.plan_cost(plan, obs, self.c, self.margin).sum()

    def __call__(self, plan, obs, times=1):
        for _ in range(times):
            with torch.enable_grad():
                a = plan.detach().requires_grad_(True)
                g, = torch.autograd.grad(self.cost(a, obs), a)
            if self.proj == "normal":
                g = A.path_normal(g, plan, obs)
            plan = plan - self.eta * g
        return plan


def spread(total, n):
    """把 total 次推动尽量平均地分到 n 个生成步。"""
    return [round((i + 1) * total / n) - round(i * total / n) for i in range(n)]


# ----------------------------------------------------------------------------- 带引导的生成器

class GuidedNCA:
    """替换 NCAPlanner 里的 model：grow() 时按模式加推动。"""

    def __init__(self, model, pusher, mode):
        self.m, self.p, self.mode, self.blank = model, pusher, mode, model.blank

    def grow(self, s, obs, z, steps):
        if self.mode == "none":
            return self.m.grow(s, obs, z, steps)
        pushes = spread(COLD_PUSH if steps > 10 else WARM_PUSH, steps)
        seed, emb = self.m.seed(obs, z), self.m.obs_net(obs)
        for k in range(steps):
            s = self.m.step(s, seed, emb)
            if self.mode == "interleave" and pushes[k]:
                s = torch.cat([self.p(A.plan_of(s), obs, pushes[k]).transpose(1, 2), s[:, 2:]], 1)
        if self.mode == "post":
            s = torch.cat([self.p(A.plan_of(s), obs, sum(pushes)).transpose(1, 2), s[:, 2:]], 1)
        return s


class GuidedDP(DB.DPPlanner):
    def __init__(self, policy, mode, pusher, guide):
        super().__init__(policy, mode, 10, "leading", 50, 3)
        self.pusher, self.guide = pusher, guide

    def denoise(self, x, obs, ts, total):
        P, c = self.p, self.p.norm(obs)
        pushes = spread(total, len(ts))
        for i, t in enumerate(ts):
            tt = torch.full((x.shape[0],), t, dtype=torch.long)
            eps = P.net(x, tt, c)
            a = P.ac[t]
            a_prev = P.ac[ts[i + 1]] if i + 1 < len(ts) else torch.ones(())
            x0 = ((x - (1 - a).sqrt() * eps) / a.sqrt()).clamp(-1, 1)
            if self.guide == "interleave" and pushes[i]:      # 推 x0 预测；推完不再裁剪，否则会抹掉推到数据范围外的修正
                x0 = P.norm(self.pusher(P.unnorm(x0.transpose(1, 2)), obs, pushes[i])).transpose(1, 2)
            eps = (x - a.sqrt() * x0) / (1 - a).sqrt()
            x = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps
        plan = P.unnorm(x.transpose(1, 2))
        if self.guide == "post":
            plan = self.pusher(plan, obs, total)
        return plan

    def plan(self, obs):
        if self.mode == "warm" and self.prev is not None:
            prev = torch.cat([self.prev[:, A.T_A:], self.prev[:, -1:].expand(-1, A.T_A, -1)], dim=1)
            x = self.p.noised(prev, self.warm_ts, torch.randn(self.E, 2, A.T_P))
            ts, total = self.warm_ts, WARM_PUSH
        else:
            x = self.noise if self.mode == "fixed" else torch.randn(self.E, 2, A.T_P)
            ts, total = self.ts, COLD_PUSH
        self.calls += len(ts)
        with torch.no_grad():
            self.prev = self.denoise(x, obs, ts, total if self.guide != "none" else 0)
        return self.prev


def load_dp(path):
    ck = torch.load(path, map_location="cpu")
    p = DB.POLICIES[ck.get("method", "ddim")](ck["lo"], ck["hi"], dims=tuple(ck["dims"]), k=ck["kernel"],
                                               t_dim=ck.get("t_dim", 64), groups=ck.get("groups", 8))
    p.load_state_dict(ck["state"])
    return p.eval()


def evaluate(job):
    torch.set_num_threads(1)
    method, setting, scene, guide, eta, seed, a = job
    circles = make_scene(scene, a["episodes"], np.random.default_rng(seed + 100))
    pusher = Pusher(circles, eta, a["margin"], a["proj"])
    if method == "nca":
        m = A.ActionNCA()
        m.load_state_dict(torch.load(os.path.join(a["nca_models"], f"{a['nca_tag']}_seed{seed}", "nca_latent.pt"), map_location="cpu"))
        planner = DB.NCAPlanner(GuidedNCA(m.eval(), pusher, guide), "latent", setting)
    else:
        planner = GuidedDP(load_dp(os.path.join(a["dp_models"], f"dp_seed{seed}", "dp_policy.pt")), setting, pusher, guide)
    t0 = time.time()
    r = DB.rollout(planner, a["episodes"], 72, "cpu", seed + 1, extra_hit=circle_hits(circles))[0]
    r["seconds"] = time.time() - t0
    return dict(method=method, setting=setting, scene=scene, guide=guide, eta=eta, seed=seed, proj=a["proj"], **r)


# ----------------------------------------------------------------------------- 汇总

def fmt(v, pct):
    v = np.asarray(v, float)
    return f"{100 * v.mean():.1f}% ± {100 * v.std():.1f}%" if pct else f"{v.mean():.2f} ± {v.std():.2f}"


def aggregate(args, rows):
    L = [f"# 新障碍实验（种子 {', '.join(map(str, args.seeds))}，每种方式 {args.episodes} 回合，评估种子 = 种子 + 1，"
         f"障碍位置种子 = 种子 + 100，margin {args.margin}，推动方向 {args.proj}）\n",
         "NCA：精度损失权重 1 的隐变量模型（训练 1500 轮）；DP：小 DP（6000 轮，DDIM 10 步，热启动 t=50、3 步）。"
         f"引导的梯度步数：冷启动 {COLD_PUSH} 次，再生 / 热启动 {WARM_PUSH} 次。单元格是 5 个种子的均值 ± 标准差。\n"]
    for scene in args.scenes:
        L += [f"## 场景 {scene}\n",
              "| 方法 | 引导 | η | 成功率 | 碰撞率 | 超时 | 计划翻转 | 粗糙度 | 最大拐折 | 成功回合左/右（合计） | 毫秒/次规划 |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for name, method, setting in METHODS:
            for guide, eta in [("none", 0.0)] + [(g, e) for g in args.guides for e in args.etas]:
                r = [x for x in rows if (x["method"], x["setting"], x["scene"], x["guide"]) == (method, setting, scene, guide)
                     and (guide == "none" or x["eta"] == eta)]
                if not r:
                    continue
                L.append(f"| {name} | {guide} | {eta:g} | {fmt([x['success'] for x in r], True)} | "
                         f"{fmt([x['collision'] for x in r], True)} | {fmt([x['timeout'] for x in r], True)} | "
                         f"{fmt([x['plan_flips'] for x in r], False)} | {fmt([x['roughness_x100'] for x in r], False)} | "
                         f"{fmt([x['max_kink_x100'] for x in r], False)} | "
                         f"{sum(x['left'] for x in r)}/{sum(x['right'] for x in r)} | "
                         f"{np.mean([x['ms_per_plan'] for x in r]):.1f} |")
        L.append("")
    text = "\n".join(L)
    with open(os.path.join(args.runs, "summary.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--scenes", nargs="+", default=["left", "random2"])
    ap.add_argument("--guides", nargs="+", default=["interleave", "post"])
    ap.add_argument("--etas", type=float, nargs="+", default=[0.1, 0.3, 1.0])
    ap.add_argument("--methods", nargs="+", default=["nca", "dp"])
    ap.add_argument("--margin", type=float, default=0.03)
    ap.add_argument("--proj", choices=["full", "normal"], default="full",
                    help="full：完整梯度；normal：只推垂直于路径方向的分量")
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--nca_models", default=os.path.join(HERE, "runs", "precision"))
    ap.add_argument("--nca_tag", default="w1", help="NCA 模型目录名的前缀：<nca_models>/<tag>_seed<s>/")
    ap.add_argument("--dp_models", default=os.path.join(HERE, "runs", "novel"))
    ap.add_argument("--runs", default=os.path.join(HERE, "runs", "novel"))
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--aggregate_only", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.runs, exist_ok=True)
    out = os.path.join(args.runs, "results.json")
    if args.aggregate_only:
        rows = json.load(open(out, encoding="utf-8"))
    else:
        a = dict(vars(args))
        jobs = [(m, st, sc, g, e, s, a) for _, m, st in METHODS if m in args.methods for sc in args.scenes
                for g, e in [("none", 0.0)] + [(g, e) for g in args.guides for e in args.etas] for s in args.seeds]
        t0 = time.time()
        with ProcessPoolExecutor(args.jobs) as ex:
            rows = list(ex.map(evaluate, jobs))
        print(f"{len(jobs)} 次评估，{(time.time() - t0) / 60:.1f} 分钟")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=1, ensure_ascii=False)
    aggregate(args, rows)


if __name__ == "__main__":
    main()
