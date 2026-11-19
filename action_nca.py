#!/usr/bin/env python3
"""
Action NCA：用一维神经元胞自动机“长出”机器人动作序列（最小原型）

思路
  * 动作块里的每个未来时间步是一个细胞，16 步动作 = 一条 16 个细胞的一维细胞带。
  * 细胞状态 = 2 个动作通道（目标位置 x, y）+ 若干隐藏通道。
  * 细胞带左边界是固定的“种子细胞”，代表“现在”：写入机器人当前位置和低维隐变量 z。
  * 感知：沿时间轴的固定卷积核（恒等、一阶差分、二阶差分，半径 1 和 2）。
  * 更新：所有细胞共享的小 MLP 输出状态增量，每步只有随机一半的细胞更新。
  * 观测（这里就是当前位置）编码后广播给所有细胞；每个细胞另有一个固定的时间位置通道。
  * 不从噪声出发：冷启动时所有细胞都停在当前位置（“原地不动”的计划），再长成轨迹。
  * 随机性只来自种子里的 z（推理时 z ~ N(0, I)）。训练用 min-of-K（IMLE 式）：
    同一个观测试 K 个 z，只用离示范最近的那个计算损失，避免把左绕和右绕平均掉。
  * 滚动时域执行 = 再生：执行完 T_A 步后切掉已执行的细胞、整体前移、末尾补空白细胞，
    在新观测下只用少量步数把尾巴重新长出来。用 Growing NCA 的样本池训练这种续长。

玩具任务
  二维点机器人从下方起点绕过中间的障碍到达上方目标。示范一半左绕、一半右绕，
  而且开头一段是直线：在这段路上，只看位置分辨不出该往哪边绕。

用法
  pip install torch numpy matplotlib
  python action_nca.py                    # 训练确定性和隐变量两个模型，评估并画图
  python action_nca.py --iters 3000       # 训练更久
  python action_nca.py --device cuda      # 用 GPU
输出在 --out 目录（默认 ./nca_out）：metrics.json、results.png 和两个模型的权重。
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

START = np.array([0.0, -0.85], dtype=np.float32)
GOAL = np.array([0.0, 0.85], dtype=np.float32)
BOX_HW, BOX_HH = 0.22, 0.15   # 障碍物：以原点为中心的矩形，半宽和半高
DEMO_LEN = 48                 # 每条示范的步数
T_P = 16                      # 每次预测的步数 = 细胞数
T_A = 4                       # 每次执行的步数


# ----------------------------------------------------------------------------- 数据

def make_demos(n, rng):
    """一半左绕、一半右绕；先走一段直线，再从侧面绕过障碍。返回 (n, DEMO_LEN + 1, 2)。"""
    u = np.linspace(0.0, 1.0, DEMO_LEN + 1)
    out = []
    for i in range(n):
        side = -1.0 if i % 2 == 0 else 1.0
        x0 = START[0] + rng.normal(0.0, 0.02)
        y0 = START[1] + rng.normal(0.0, 0.02)
        amp = rng.uniform(0.5, 0.7)
        ua, ub = rng.uniform(0.15, 0.25), rng.uniform(0.75, 0.85)
        w = np.clip((u - ua) / (ub - ua), 0.0, 1.0)
        bump = 0.5 * (1.0 - np.cos(2.0 * np.pi * w))   # 升余弦：绕行段之外为 0，正中为 1
        x = x0 * (1.0 - u) + GOAL[0] * u + side * amp * bump
        y = y0 + (GOAL[1] - y0) * u
        out.append(np.stack([x, y], axis=1))
    return np.stack(out).astype(np.float32)


def chunk_targets(D, d, t):
    """示范 d 在时刻 t 之后的 T_P 个航点，越过末尾的用终点补齐。D: (N, L+1, 2)，d 和 t 是长整型张量。"""
    idx = torch.clamp(t[:, None] + 1 + torch.arange(T_P, device=D.device)[None], max=DEMO_LEN)
    return D[d[:, None], idx]


def seg_hits_box(a, b, n=12):
    """线段 a→b 是否穿过障碍物（沿线段取点检查）。a, b: (E, 2)。"""
    ts = np.linspace(0.0, 1.0, n)[None, :, None]
    p = a[:, None] + ts * (b - a)[:, None]
    return ((np.abs(p[..., 0]) < BOX_HW) & (np.abs(p[..., 1]) < BOX_HH)).any(1)


# ----------------------------------------------------------------------------- 模型

class ActionNCA(nn.Module):
    """一维动作 NCA。状态 s 的形状是 (B, C, T_P)，C = 2 个动作通道 + hidden 个隐藏通道。"""

    def __init__(self, hidden=22, z_dim=4, emb=16, width=64, fire_rate=0.5):
        super().__init__()
        self.C, self.z_dim, self.fire_rate = 2 + hidden, z_dim, fire_rate
        k = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0],     # 恒等：自身状态
                          [0.0, -0.5, 0.0, 0.5, 0.0],    # 一阶差分，半径 1（约等于速度）
                          [0.0, 1.0, -2.0, 1.0, 0.0],    # 二阶差分，半径 1（约等于加速度）
                          [-0.5, 0.0, 0.0, 0.0, 0.5],    # 一阶差分，半径 2
                          [1.0, 0.0, -2.0, 0.0, 1.0]])   # 二阶差分，半径 2
        self.register_buffer("kernel", k.repeat(self.C, 1).unsqueeze(1))          # (5C, 1, 5)
        self.register_buffer("pos", ((torch.arange(T_P) + 1.0) / T_P).view(1, 1, T_P))
        self.obs_net = nn.Sequential(nn.Linear(2, emb), nn.ReLU(), nn.Linear(emb, emb))
        self.update = nn.Sequential(
            nn.Conv1d(5 * self.C + 1 + emb, width, 1), nn.ReLU(),
            nn.Conv1d(width, self.C, 1, bias=False))
        nn.init.zeros_(self.update[-1].weight)   # 初始行为是“什么都不做”

    def seed(self, obs, z):
        """种子细胞 = 现在：动作通道写当前位置，隐藏通道写 z 和一个标记位。"""
        s = obs.new_zeros(obs.shape[0], self.C, 1)
        s[:, :2, 0] = obs
        s[:, 2:2 + self.z_dim, 0] = z
        s[:, 2 + self.z_dim, 0] = 1.0
        return s

    def blank(self, obs):
        """冷启动：所有细胞先停在当前位置，隐藏通道全为 0。"""
        s = obs.new_zeros(obs.shape[0], self.C, T_P)
        s[:, :2, :] = obs[:, :, None]
        return s

    def step(self, s, seed, emb):
        B = s.shape[0]
        x = torch.cat([seed, seed, s, s[:, :, -1:], s[:, :, -1:]], dim=2)   # 左边界是种子，右边界复制
        h = torch.cat([F.conv1d(x, self.kernel, groups=self.C),
                       self.pos.expand(B, 1, T_P),
                       emb[:, :, None].expand(B, emb.shape[1], T_P)], dim=1)
        ds = self.update(h)
        if self.fire_rate < 1.0:                                              # 随机异步更新
            ds = ds * (torch.rand(B, 1, T_P, device=s.device) < self.fire_rate).float()
        return s + ds

    def grow(self, s, obs, z, steps, frames=None):
        seed, emb = self.seed(obs, z), self.obs_net(obs)
        for _ in range(steps):
            s = self.step(s, seed, emb)
            if frames is not None:
                frames.append(plan_of(s).detach().cpu().numpy())
        return s


def shift(s, k=T_A):
    """再生：切掉已执行的 k 个细胞，整体前移，末尾补 k 个空白细胞（停在最后一个航点）。"""
    tail = s.new_zeros(s.shape[0], s.shape[1], k)
    tail[:, :2, :] = s[:, :2, -1:]
    return torch.cat([s[:, :, k:], tail], dim=2)


def plan_of(s):
    return s[:, :2, :].transpose(1, 2)   # (B, T_P, 2)


# ----------------------------------------------------------------------------- 训练

def train(kind, demos, args, dev):
    torch.manual_seed(args.seed)
    model = ActionNCA(z_dim=args.z_dim).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.iters, eta_min=args.lr * 0.05)
    K = args.k if kind == "latent" else 1
    D = torch.from_numpy(demos).to(dev)
    N, B, Zd, P = len(demos), args.batch, args.z_dim, args.pool
    last_t = DEMO_LEN - 1 - T_A                     # 池里的计划至少还能再往前走一段
    pool_s = torch.zeros(P, model.C, T_P, device=dev)
    pool_z = torch.zeros(P, Zd, device=dev)
    pool_d = torch.zeros(P, dtype=torch.long, device=dev)
    pool_t = torch.zeros(P, dtype=torch.long, device=dev)
    pool_ok = torch.zeros(P, dtype=torch.bool, device=dev)
    t0 = time.time()
    for it in range(1, args.iters + 1):
        # 冷启动：从“原地不动”的计划长出整段动作
        d = torch.randint(0, N, (B,), device=dev)
        t = torch.randint(0, DEMO_LEN, (B,), device=dev)
        obs = D[d, t] + args.obs_noise * torch.randn(B, 2, device=dev)
        tgt = chunk_targets(D, d, t)
        steps = int(torch.randint(args.cold_min, args.cold_max + 1, ()).item())
        if K > 1:
            with torch.no_grad():   # min-of-K：先试 K 个 z，挑出离示范最近的那个
                zk = torch.randn(B * K, Zd, device=dev)
                ok = obs.repeat_interleave(K, 0)
                sk = model.grow(model.blank(ok), ok, zk, steps)
                err = ((plan_of(sk) - tgt.repeat_interleave(K, 0)) ** 2).sum((1, 2)).view(B, K)
                z = zk.view(B, K, Zd)[torch.arange(B, device=dev), err.argmin(1)]
        else:
            z = torch.zeros(B, Zd, device=dev)
        s = model.grow(model.blank(obs), obs, z, steps)
        loss_c = ((plan_of(s) - tgt) ** 2).sum((1, 2)).mean()

        # 热启动：从样本池取执行到一半的计划，切掉、前移、补尾巴，再长几步
        loss_w = torch.zeros((), device=dev)
        live = torch.nonzero(pool_ok).squeeze(1)
        if live.numel() >= B:
            pi = live[torch.randperm(live.numel(), device=dev)[:B]]
            pd, pt = pool_d[pi], pool_t[pi] + T_A
            wobs = D[pd, pt] + args.obs_noise * torch.randn(B, 2, device=dev)
            wsteps = int(torch.randint(args.warm_min, args.warm_max + 1, ()).item())
            ws = model.grow(shift(pool_s[pi]), wobs, pool_z[pi], wsteps)
            loss_w = ((plan_of(ws) - chunk_targets(D, pd, pt)) ** 2).sum((1, 2)).mean()
            pool_s[pi], pool_t[pi], pool_ok[pi] = ws.detach(), pt, pt <= last_t

        loss = loss_c + loss_w
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        # 冷启动的结果放进样本池，优先占用已经走完的位置
        keep = torch.nonzero(t <= last_t).squeeze(1)
        if keep.numel():
            free = torch.nonzero(~pool_ok).squeeze(1)
            free = free[torch.randperm(free.numel(), device=dev)]
            if free.numel() < keep.numel():
                used = torch.nonzero(pool_ok).squeeze(1)
                used = used[torch.randperm(used.numel(), device=dev)[:keep.numel() - free.numel()]]
                free = torch.cat([free, used])
            slots = free[:keep.numel()]
            pool_s[slots], pool_z[slots] = s.detach()[keep], z[keep]
            pool_d[slots], pool_t[slots], pool_ok[slots] = d[keep], t[keep], True

        if it == 1 or it % args.log_every == 0:
            print(f"[{kind}] it {it:5d}  cold {loss_c.item():.4f}  warm {loss_w.item():.4f}"
                  f"  {time.time() - t0:.0f}s", flush=True)
    return model


# ----------------------------------------------------------------------------- 评估

@torch.no_grad()
def run_episodes(model, kind, setting, args, dev, seed):
    """闭环执行。setting：cold_fresh（每次冷启动、换新 z）、cold_fixed（冷启动、z 不变）、warm（再生续长）。"""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    E, Zd = args.episodes, args.z_dim
    pos = (START + rng.normal(0.0, 0.02, (E, 2))).astype(np.float32)
    z = torch.randn(E, Zd, device=dev) if kind == "latent" else torch.zeros(E, Zd, device=dev)
    alive = np.ones(E, bool)
    status = np.zeros(E, np.int64)        # 0 超时，1 成功，2 碰撞
    length = np.zeros(E, np.int64)
    paths, intents = [pos.copy()], [[] for _ in range(E)]
    s, nca_steps, n_plans = None, 0, 0
    for _ in range(math.ceil(args.max_steps / T_A)):
        obs = torch.from_numpy(pos).to(dev)
        if setting == "warm" and s is not None:
            s = model.grow(shift(s), obs, z, args.warm_steps)
            nca_steps += args.warm_steps
        else:
            if setting == "cold_fresh" and kind == "latent":
                z = torch.randn(E, Zd, device=dev)
            s = model.grow(model.blank(obs), obs, z, args.cold_steps)
            nca_steps += args.cold_steps
        n_plans += 1
        plan = plan_of(s).cpu().numpy()
        intent = plan[:, -1, 0] - pos[:, 0]          # 这一段计划打算往哪边偏
        for e in np.flatnonzero(alive & (pos[:, 1] < -0.4)):
            if abs(intent[e]) > 0.03:
                intents[e].append(np.sign(intent[e]))
        for k in range(T_A):
            new = np.where(alive[:, None], plan[:, k], pos).astype(np.float32)
            hit = alive & seg_hits_box(pos, new)
            status[hit] = 2
            length += alive
            alive &= ~hit
            pos = new
            done = alive & (np.linalg.norm(pos - GOAL, axis=1) < 0.06)
            status[done] = 1
            alive &= ~done
            paths.append(pos.copy())
        if not alive.any():
            break
    paths = np.stack(paths, 1)
    flips = [int(np.sum(np.diff(v) != 0)) if len(v) > 1 else 0 for v in intents]
    side = np.zeros(E)
    rough = []
    for e in range(E):
        p = paths[e, :length[e] + 1]
        m = np.abs(p[:, 1]) < BOX_HH
        if m.any():
            side[e] = np.sign(p[m, 0].mean())
        if status[e] == 1 and len(p) > 2:
            rough.append(np.linalg.norm(p[2:] - 2 * p[1:-1] + p[:-2], axis=1).mean())
    ok = status == 1
    metrics = dict(success=float(ok.mean()), collision=float((status == 2).mean()),
                   timeout=float((status == 0).mean()),
                   left=int((ok & (side < 0)).sum()), right=int((ok & (side > 0)).sum()),
                   plan_flips=float(np.mean(flips)),
                   roughness_x100=float(np.mean(rough) * 100) if rough else float("nan"),
                   nca_steps_per_plan=nca_steps / n_plans)
    return metrics, paths, status, length


@torch.no_grad()
def first_plan_offsets(model, kind, args, dev, n=2000):
    """在起点处用不同的 z 冷启动，返回每段计划最后一个航点的横向偏移。"""
    obs = torch.from_numpy(START[None]).to(dev).repeat(n, 1)
    z = torch.randn(n, args.z_dim, device=dev) if kind == "latent" else torch.zeros(n, args.z_dim, device=dev)
    s = model.grow(model.blank(obs), obs, z, args.cold_steps)
    return plan_of(s)[:, -1, 0].cpu().numpy(), z


@torch.no_grad()
def growth_frames(model, args, dev, z):
    obs = torch.from_numpy(START[None]).to(dev).repeat(z.shape[0], 1)
    s = model.blank(obs)
    frames = [plan_of(s).cpu().numpy()]
    model.grow(s, obs, z, args.cold_steps, frames=frames)
    return np.stack(frames)   # (steps + 1, n, T_P, 2)


# ----------------------------------------------------------------------------- 画图

def plot(results, demos, frames, offsets, det_offset, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def scene(ax, title, lim=((-1, 1), (-1, 1))):
        ax.add_patch(plt.Rectangle((-BOX_HW, -BOX_HH), 2 * BOX_HW, 2 * BOX_HH, color="0.7", zorder=1))
        ax.plot(*START, "o", color="0.2", ms=5, zorder=3)
        ax.plot(*GOAL, "*", color="0.2", ms=11, zorder=3)
        ax.set_xlim(*lim[0]); ax.set_ylim(*lim[1]); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=10)

    color = {0: "#BA7517", 1: "#1D9E75", 2: "#E24B4A"}   # 超时 / 成功 / 碰撞
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for ax, (name, (m, paths, status, length)) in zip(axes.flat[:4], results.items()):
        scene(ax, f"{name}\nsuccess {m['success']:.0%}  collision {m['collision']:.0%}  "
                  f"L/R {m['left']}/{m['right']}  plan flips {m['plan_flips']:.2f}")
        for demo in demos[:24]:
            ax.plot(demo[:, 0], demo[:, 1], color="0.88", lw=0.8, zorder=0)
        for e in range(min(40, len(paths))):
            p = paths[e, :length[e] + 1]
            ax.plot(p[:, 0], p[:, 1], color=color[int(status[e])], lw=1.1, alpha=0.75, zorder=2)

    ax = axes.flat[4]
    scene(ax, "one plan growing from the seed (two z's)\nlighter = earlier NCA step",
          lim=((-0.55, 0.55), (-0.95, -0.1)))
    n_frames = frames.shape[0]
    for j, f in enumerate(np.linspace(0, n_frames - 1, 7).astype(int)):
        a = 0.15 + 0.85 * j / 6
        for i, c in enumerate(("#1D9E75", "#7F77DD")):
            ax.plot(frames[f, i, :, 0], frames[f, i, :, 1], "-o", color=c, alpha=a, ms=2.5, lw=1.2)

    ax = axes.flat[5]
    ax.hist(offsets, bins=60, range=(-0.5, 0.5), color="#1D9E75", alpha=0.85, label="latent NCA, z ~ N(0, I)")
    ax.axvline(det_offset, color="#E24B4A", ls="--", lw=2, label="deterministic NCA")
    ax.set_xlabel("lateral offset of the first plan's last waypoint")
    ax.set_title("first plan at the start: one mode or two?", fontsize=10)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ----------------------------------------------------------------------------- 主程序

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--k", type=int, default=8, help="min-of-K 里每个观测试几个 z")
    ap.add_argument("--z_dim", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--pool", type=int, default=1024)
    ap.add_argument("--obs_noise", type=float, default=0.01)
    ap.add_argument("--cold_min", type=int, default=24)
    ap.add_argument("--cold_max", type=int, default=40)
    ap.add_argument("--warm_min", type=int, default=6)
    ap.add_argument("--warm_max", type=int, default=14)
    ap.add_argument("--cold_steps", type=int, default=32, help="评估时冷启动的 NCA 步数")
    ap.add_argument("--warm_steps", type=int, default=10, help="评估时再生续长的 NCA 步数")
    ap.add_argument("--n_demos", type=int, default=200)
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--max_steps", type=int, default=72)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="nca_out")
    args = ap.parse_args()

    dev = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)
    demos = make_demos(args.n_demos, np.random.default_rng(args.seed))

    models = {}
    for kind in ("det", "latent"):
        models[kind] = train(kind, demos, args, dev)
        torch.save(models[kind].state_dict(), os.path.join(args.out, f"nca_{kind}.pt"))

    settings = [("deterministic, cold start", "det", "cold_fixed"),
                ("latent z, cold start, new z each plan", "latent", "cold_fresh"),
                ("latent z, cold start, same z", "latent", "cold_fixed"),
                ("latent z, warm start (regrow)", "latent", "warm")]
    results = {}
    print()
    for name, kind, setting in settings:
        results[name] = run_episodes(models[kind], kind, setting, args, dev, seed=args.seed + 1)
        print(f"{name:40s} " + "  ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}"
                                          for k, v in results[name][0].items()))

    offsets, zs = first_plan_offsets(models["latent"], "latent", args, dev)
    det_offset = float(first_plan_offsets(models["det"], "det", args, dev, n=64)[0].mean())
    undecided = float(np.mean(np.abs(offsets) < 0.05))
    print(f"\nfirst plan at start: latent left {np.mean(offsets < -0.05):.0%}, right {np.mean(offsets > 0.05):.0%},"
          f" undecided {undecided:.0%}; deterministic offset {det_offset:+.3f}")
    pick = torch.stack([zs[int(np.argmin(offsets))], zs[int(np.argmax(offsets))]])
    frames = growth_frames(models["latent"], args, dev, pick)

    summary = {name: r[0] for name, r in results.items()}
    summary["first_plan_at_start"] = dict(left=float(np.mean(offsets < -0.05)), right=float(np.mean(offsets > 0.05)),
                                          undecided=undecided, deterministic_offset=det_offset)
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    plot(results, demos, frames, offsets, det_offset, os.path.join(args.out, "results.png"))
    print(f"saved to {args.out}/")


if __name__ == "__main__":
    main()
