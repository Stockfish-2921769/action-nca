#!/usr/bin/env python3
"""
Diffusion Policy 基线：在 action_nca.py 的同一个玩具任务上，用小型 1D 卷积 U-Net 生成动作块。
生成方法二选一（--method）：ddim（扩散 + DDIM，原版 Diffusion Policy）或 flow（流匹配 + Euler，π0 一类 VLA 的动作头做法）。

和 NCA 保持一致的部分
  * 同一份示范（make_demos，同一个随机种子）、同样的观测（只有当前位置，加 0.01 的观测噪声）、
    同样的目标（当前时刻之后 T_P = 16 个航点的绝对位置），每次执行 T_A = 4 步。
  * 闭环评估的起点、成功/碰撞判定和所有指标都和 action_nca.run_episodes 相同
    （rollout() 是它的通用版本，脚本会用 NCA 模型核对两者给出的数字完全一致）。

Diffusion Policy 的做法（Chi 等，2023）
  * 动作和观测按示范数据的最小/最大值归一化到 [-1, 1]。
  * 网络：条件 1D U-Net（每层两个残差块，卷积 + GroupNorm + Mish，FiLM 注入
    “扩散时间步嵌入 + 观测”），训练时 100 个扩散步、squaredcos_cap_v2 噪声表、预测噪声、MSE。
  * 推理：DDIM（eta = 0），默认 10 步，x0 裁剪到 [-1, 1]；参数用 EMA 权重。
  * 只用 1 帧观测（原版是 2 帧），这样和 NCA 的输入完全一样。

流匹配（--method flow，rectified flow）
  * 网络、归一化、训练流程和 EMA 都和 DDIM 版完全相同，只换目标：x_τ = (1 − τ)·噪声 + τ·动作，
    τ ~ U(0, 1)，网络预测速度 v = 动作 − 噪声；τ 乘以 100 后送进同一个正弦时间嵌入。
  * 推理：从 τ = 0 的纯噪声出发，用 Euler 法等间隔积分到 τ = 1（--ddim_steps 此时就是 Euler 步数）。
  * 热启动：把上一段计划插值到 τ = 1 − warm_t / 100（默认 0.5，信噪比和 DDIM 在第 50 步时接近），
    再积分 --warm_steps 步。

评估的执行方式
  * ddim_fresh：每次规划都重新采噪声（原版 Diffusion Policy 的做法）。
  * ddim_fixed：每个回合开头采一份噪声，之后每次规划都用它（对应 NCA 的 cold_fixed）。
  * ddim_warm：热启动（对应 NCA 的再生）。上一段计划切掉已执行的 T_A 步、前移、用最后一个
    航点补尾巴，加噪到中间的扩散步 --warm_t（SDEdit 式），再用 --warm_steps 步 DDIM 去噪。
    第一次规划仍然从纯噪声出发。

用法
  python diffusion_baseline.py                       # 训练、评估、和 nca_out/ 里的 NCA 模型对比
  python diffusion_baseline.py --device mps --train_only     # 在 Apple GPU 上训练（约 8 分钟），评估另跑
  python diffusion_baseline.py --load dp_out/dp_policy.pt   # 只评估（默认在 CPU 上）
  # 和 NCA 每步算力相当的小模型（每次调用约 0.19 M 乘加）：通道少时 GroupNorm 必须减到 2 组，否则训练不起来
  python diffusion_baseline.py --dims 8 16 32 --kernel 3 --t_dim 16 --groups 2 --lr 2e-3 --out dp_out/small_g2_lr2e-3
  # 流匹配版本：加 --method flow，其余参数相同
  python diffusion_baseline.py --method flow --device mps --train_only --out dp_out/flow
输出在 --out 目录（默认 ./dp_out）：metrics.json、compare.md、results_dp.png、dp_policy.pt。
"""
import argparse
import copy
import json
import math
import os
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import action_nca as A

T_P, T_A = A.T_P, A.T_A


# ----------------------------------------------------------------------------- 网络

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        f = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / (half - 1))
        a = t.float()[:, None] * f[None]
        return torch.cat([a.sin(), a.cos()], dim=-1)


class Conv1dBlock(nn.Sequential):
    def __init__(self, cin, cout, k, groups):
        super().__init__(nn.Conv1d(cin, cout, k, padding=k // 2), nn.GroupNorm(groups, cout), nn.Mish())


class CondResBlock(nn.Module):
    """两层卷积；条件向量经 FiLM 给出第一层输出每个通道的缩放和平移。"""

    def __init__(self, cin, cout, cond_dim, k, groups):
        super().__init__()
        self.b1, self.b2 = Conv1dBlock(cin, cout, k, groups), Conv1dBlock(cout, cout, k, groups)
        self.film = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, 2 * cout))
        self.skip = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, c):
        scale, bias = self.film(c).unsqueeze(-1).chunk(2, dim=1)
        h = self.b2(scale * self.b1(x) + bias)
        return h + self.skip(x)


class CondUnet1D(nn.Module):
    """Diffusion Policy 的 ConditionalUnet1D 的缩小版。输入 (B, 2, T_P)，输出同形状的噪声预测。"""

    def __init__(self, act_dim=2, obs_dim=2, dims=(32, 64, 128), k=5, t_dim=64, groups=8):
        super().__init__()
        self.t_emb = nn.Sequential(SinusoidalPosEmb(t_dim), nn.Linear(t_dim, 4 * t_dim), nn.Mish(),
                                   nn.Linear(4 * t_dim, t_dim))
        cd = t_dim + obs_dim
        pairs = list(zip([act_dim, *dims[:-1]], dims))
        self.down = nn.ModuleList()
        for i, (a, b) in enumerate(pairs):            # 最后一层不下采样：16 → 8 → 4
            self.down.append(nn.ModuleList([CondResBlock(a, b, cd, k, groups), CondResBlock(b, b, cd, k, groups),
                                            nn.Conv1d(b, b, 3, 2, 1) if i < len(pairs) - 1 else nn.Identity()]))
        m = dims[-1]
        self.mid = nn.ModuleList([CondResBlock(m, m, cd, k, groups), CondResBlock(m, m, cd, k, groups)])
        self.up = nn.ModuleList()
        for a, b in reversed(pairs[1:]):              # 和下采样层的输出拼接（U-Net 跳连）
            self.up.append(nn.ModuleList([CondResBlock(2 * b, a, cd, k, groups), CondResBlock(a, a, cd, k, groups),
                                          nn.ConvTranspose1d(a, a, 4, 2, 1)]))
        self.final = nn.Sequential(Conv1dBlock(dims[0], dims[0], k, groups), nn.Conv1d(dims[0], act_dim, 1))

    def forward(self, x, t, obs):
        c = torch.cat([self.t_emb(t), obs], dim=-1)
        hs = []
        for r1, r2, down in self.down:
            x = r2(r1(x, c), c)
            hs.append(x)
            x = down(x)
        for r in self.mid:
            x = r(x, c)
        for r1, r2, up in self.up:
            x = up(r2(r1(torch.cat([x, hs.pop()], dim=1), c), c))
        return self.final(x)


def cosine_alphas_cumprod(n, s=0.008, max_beta=0.999):
    """diffusers 的 squaredcos_cap_v2 噪声表。"""
    f = lambda u: math.cos((u + s) / (1 + s) * math.pi / 2) ** 2
    betas = torch.tensor([min(1 - f((i + 1) / n) / f(i / n), max_beta) for i in range(n)])
    return torch.cumprod(1.0 - betas, dim=0)


def ddim_timesteps(n_train, steps, spacing):
    """leading 和原版 Diffusion Policy 用的 diffusers 默认一致（10 步：90, 80, …, 0）；
    trailing 从最后一步开始（99, 89, …, 9），步数很少时更合理。"""
    if spacing == "leading":
        return list(range(0, n_train, n_train // steps))[:steps][::-1]
    return [int(round(v)) - 1 for v in np.arange(n_train, 0, -n_train / steps)]


class DiffusionPolicy(nn.Module):
    def __init__(self, lo, hi, dims=(32, 64, 128), k=5, t_dim=64, groups=8, n_train=100):
        super().__init__()
        self.net = CondUnet1D(dims=dims, k=k, t_dim=t_dim, groups=groups)
        self.n_train = n_train
        self.register_buffer("lo", torch.as_tensor(lo, dtype=torch.float32))
        self.register_buffer("hi", torch.as_tensor(hi, dtype=torch.float32))
        self.register_buffer("ac", cosine_alphas_cumprod(n_train))

    def norm(self, x):
        return 2 * (x - self.lo) / (self.hi - self.lo) - 1

    def unnorm(self, x):
        return (x + 1) / 2 * (self.hi - self.lo) + self.lo

    def loss(self, obs, tgt):
        x0 = self.norm(tgt).transpose(1, 2)                  # (B, 2, T_P)
        t = torch.randint(0, self.n_train, (x0.shape[0],), device=x0.device)
        eps = torch.randn_like(x0)
        a = self.ac[t].view(-1, 1, 1)
        return F.mse_loss(self.net(a.sqrt() * x0 + (1 - a).sqrt() * eps, t, self.norm(obs)), eps)

    @torch.no_grad()
    def denoise(self, x, obs, ts, frames=None):
        """DDIM（eta = 0）沿给定的时间步从 x 去噪。x 在归一化空间，形状 (B, 2, T_P)；返回 (B, T_P, 2) 的计划。"""
        c = self.norm(obs)
        for i, t in enumerate(ts):
            tt = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
            eps = self.net(x, tt, c)
            a = self.ac[t]
            a_prev = self.ac[ts[i + 1]] if i + 1 < len(ts) else torch.ones((), device=x.device)
            x0 = ((x - (1 - a).sqrt() * eps) / a.sqrt()).clamp(-1, 1)
            eps = (x - a.sqrt() * x0) / (1 - a).sqrt()      # 裁剪后按 x0 重新算噪声（同 diffusers）
            x = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps
            if frames is not None:
                frames.append(self.unnorm(x0.transpose(1, 2)).cpu().numpy())
        return self.unnorm(x.transpose(1, 2))

    def schedule(self, steps, spacing="leading"):
        """冷启动的去噪时间表；列表长度 = 网络调用次数。"""
        return ddim_timesteps(self.n_train, steps, spacing)

    def warm_schedule(self, warm_t, warm_steps):
        return [int(round(v)) for v in np.linspace(warm_t, 0, warm_steps + 1)[:-1]]

    def noised(self, plan, sched, eps):
        """把一段计划（B, T_P, 2）加噪到时间表的起点（扩散步 sched[0]）。"""
        a = self.ac[sched[0]]
        return a.sqrt() * self.norm(plan).transpose(1, 2) + (1 - a).sqrt() * eps


class FlowPolicy(DiffusionPolicy):
    """流匹配（rectified flow）。x_τ = (1 − τ)·噪声 + τ·动作，网络预测速度 v = 动作 − 噪声；
    推理从 τ = 0 的纯噪声出发，Euler 法积分到 τ = 1。网络和归一化与 DDIM 版相同（ac 缓冲区不用）。"""

    def loss(self, obs, tgt):
        x1 = self.norm(tgt).transpose(1, 2)                  # (B, 2, T_P)
        x0 = torch.randn_like(x1)
        tau = torch.rand(x1.shape[0], device=x1.device)
        xt = (1 - tau.view(-1, 1, 1)) * x0 + tau.view(-1, 1, 1) * x1
        return F.mse_loss(self.net(xt, tau * self.n_train, self.norm(obs)), x1 - x0)

    def schedule(self, steps, spacing=None):
        """等间隔 Euler 步：[(τ_0, τ_1), (τ_1, τ_2), …]，长度 = 网络调用次数。"""
        g = np.linspace(0.0, 1.0, steps + 1)
        return list(zip(g[:-1], g[1:]))

    def warm_schedule(self, warm_t, warm_steps):
        g = np.linspace(1.0 - warm_t / self.n_train, 1.0, warm_steps + 1)
        return list(zip(g[:-1], g[1:]))

    @torch.no_grad()
    def denoise(self, x, obs, sched, frames=None):
        c = self.norm(obs)
        for a, b in sched:
            v = self.net(x, torch.full((x.shape[0],), a * self.n_train, device=x.device), c)
            if frames is not None:                           # 当前对终点的一步预测，对应 DDIM 的 x0
                frames.append(self.unnorm((x + (1 - a) * v).clamp(-1, 1).transpose(1, 2)).cpu().numpy())
            x = x + (b - a) * v
        return self.unnorm(x.clamp(-1, 1).transpose(1, 2))

    def noised(self, plan, sched, eps):
        tau = sched[0][0]
        return (1 - tau) * eps + tau * self.norm(plan).transpose(1, 2)


POLICIES = {"ddim": DiffusionPolicy, "flow": FlowPolicy}


# ----------------------------------------------------------------------------- 训练

def train_dp(demos, args, dev):
    torch.manual_seed(args.seed)
    D = torch.from_numpy(demos).to(dev)
    flat = D.reshape(-1, 2)
    policy = POLICIES[args.method](flat.min(0).values, flat.max(0).values, dims=args.dims, k=args.kernel,
                             t_dim=args.t_dim, groups=args.groups).to(dev)
    ema = copy.deepcopy(policy).requires_grad_(False)
    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=1e-6)
    warm = min(args.lr_warmup, args.iters // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: (i + 1) / warm if i < warm else 0.5 * (1 + math.cos(math.pi * (i - warm) / (args.iters - warm))))
    N, B = len(demos), args.batch
    t0, run = time.time(), 0.0
    for it in range(1, args.iters + 1):
        d = torch.randint(0, N, (B,), device=dev)
        t = torch.randint(0, A.DEMO_LEN, (B,), device=dev)
        obs = D[d, t] + args.obs_noise * torch.randn(B, 2, device=dev)
        loss = policy.loss(obs, A.chunk_targets(D, d, t))
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        sched.step()
        decay = min(args.ema, (1 + it) / (10 + it))           # EMA 预热
        with torch.no_grad():
            for pe, p in zip(ema.parameters(), policy.parameters()):
                pe.lerp_(p, 1 - decay)
        run = loss.item() if it == 1 else 0.98 * run + 0.02 * loss.item()
        if it == 1 or it % args.log_every == 0:
            print(f"[dp] it {it:5d}  loss {run:.4f}  {time.time() - t0:.0f}s", flush=True)
    return ema


# ----------------------------------------------------------------------------- 规划器

class DPPlanner:
    """mode：fresh（每次换新噪声）、fixed（每回合一份噪声）、warm（上一段计划加噪后少步去噪）。"""

    def __init__(self, policy, mode, steps, spacing, warm_t=50, warm_steps=3):
        self.p, self.mode = policy, mode
        self.ts = policy.schedule(steps, spacing)
        self.warm_ts = policy.warm_schedule(warm_t, warm_steps)

    def reset(self, E, dev):
        self.E, self.dev, self.prev, self.calls = E, dev, None, 0
        self.noise = torch.randn(E, 2, T_P, device=dev) if self.mode == "fixed" else None

    def plan(self, obs):
        if self.mode == "warm" and self.prev is not None:
            prev = torch.cat([self.prev[:, T_A:], self.prev[:, -1:].expand(-1, T_A, -1)], dim=1)
            x = self.p.noised(prev, self.warm_ts, torch.randn(self.E, 2, T_P, device=self.dev))
            ts = self.warm_ts
        else:
            x = self.noise if self.mode == "fixed" else torch.randn(self.E, 2, T_P, device=self.dev)
            ts = self.ts
        self.calls += len(ts)
        self.prev = self.p.denoise(x, obs, ts)
        return self.prev


class NCAPlanner:
    """把 action_nca.run_episodes 里的规划逻辑原样搬过来（随机数的消耗顺序也相同）。"""

    def __init__(self, model, kind, setting, cold_steps=32, warm_steps=10, z_dim=4):
        self.m, self.kind, self.setting = model, kind, setting
        self.cold_steps, self.warm_steps, self.z_dim = cold_steps, warm_steps, z_dim

    def reset(self, E, dev):
        self.E, self.dev, self.s, self.calls = E, dev, None, 0
        self.z = (torch.randn(E, self.z_dim, device=dev) if self.kind == "latent"
                  else torch.zeros(E, self.z_dim, device=dev))

    def plan(self, obs):
        if self.setting == "warm" and self.s is not None:
            self.s = self.m.grow(A.shift(self.s), obs, self.z, self.warm_steps)
            self.calls += self.warm_steps
        else:
            if self.setting == "cold_fresh" and self.kind == "latent":
                self.z = torch.randn(self.E, self.z_dim, device=self.dev)
            self.s = self.m.grow(self.m.blank(obs), obs, self.z, self.cold_steps)
            self.calls += self.cold_steps
        return A.plan_of(self.s)


# ----------------------------------------------------------------------------- 评估

@torch.no_grad()
def rollout(planner, episodes, max_steps, dev, seed):
    """闭环执行，指标和 action_nca.run_episodes 完全相同；网络调用次数按规划器统计。"""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    E = episodes
    pos = (A.START + rng.normal(0.0, 0.02, (E, 2))).astype(np.float32)
    planner.reset(E, dev)
    alive = np.ones(E, bool)
    status = np.zeros(E, np.int64)        # 0 超时，1 成功，2 碰撞
    length = np.zeros(E, np.int64)
    paths, intents = [pos.copy()], [[] for _ in range(E)]
    n_plans, plan_time = 0, 0.0
    for _ in range(math.ceil(max_steps / T_A)):
        t0 = time.perf_counter()
        plan = planner.plan(torch.from_numpy(pos).to(dev)).cpu().numpy()
        plan_time += time.perf_counter() - t0
        n_plans += 1
        intent = plan[:, -1, 0] - pos[:, 0]
        for e in np.flatnonzero(alive & (pos[:, 1] < -0.4)):
            if abs(intent[e]) > 0.03:
                intents[e].append(np.sign(intent[e]))
        for k in range(T_A):
            new = np.where(alive[:, None], plan[:, k], pos).astype(np.float32)
            hit = alive & A.seg_hits_box(pos, new)
            status[hit] = 2
            length += alive
            alive &= ~hit
            pos = new
            done = alive & (np.linalg.norm(pos - A.GOAL, axis=1) < 0.06)
            status[done] = 1
            alive &= ~done
            paths.append(pos.copy())
        if not alive.any():
            break
    paths = np.stack(paths, 1)
    flips = [int(np.sum(np.diff(v) != 0)) if len(v) > 1 else 0 for v in intents]
    side = np.zeros(E)
    rough, kink = [], []
    for e in range(E):
        p = paths[e, :length[e] + 1]
        m = np.abs(p[:, 1]) < A.BOX_HH
        if m.any():
            side[e] = np.sign(p[m, 0].mean())
        if status[e] == 1 and len(p) > 2:
            dd = np.linalg.norm(p[2:] - 2 * p[1:-1] + p[:-2], axis=1)
            rough.append(dd.mean())
            kink.append(dd.max())          # 偶发的尖刺（计划衔接处的跳变）会被均值摊平，单独记最大值
    ok = status == 1
    metrics = dict(success=float(ok.mean()), collision=float((status == 2).mean()),
                   timeout=float((status == 0).mean()),
                   left=int((ok & (side < 0)).sum()), right=int((ok & (side > 0)).sum()),
                   plan_flips=float(np.mean(flips)),
                   roughness_x100=float(np.mean(rough) * 100) if rough else float("nan"),
                   max_kink_x100=float(np.mean(kink) * 100) if kink else float("nan"),
                   calls_per_plan=planner.calls / n_plans,
                   ms_per_plan=1000 * plan_time / n_plans)
    return metrics, paths, status, length


def count_macs(module, fn):
    """用前向钩子统计一次调用里 Conv1d / ConvTranspose1d / Linear 的乘加次数（batch = 1）。"""
    total = [0]

    def hook(m, inp, out):
        if isinstance(m, nn.Conv1d):
            total[0] += out.numel() * m.in_channels // m.groups * m.kernel_size[0]
        elif isinstance(m, nn.ConvTranspose1d):
            total[0] += inp[0].numel() * m.out_channels // m.groups * m.kernel_size[0]
        elif isinstance(m, nn.Linear):
            total[0] += out.numel() * m.in_features
    hs = [m.register_forward_hook(hook) for m in module.modules()
          if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear))]
    with torch.no_grad():
        fn()
    for h in hs:
        h.remove()
    return total[0]


def dp_macs_per_call(policy, dev):
    x = torch.zeros(1, 2, T_P, device=dev)
    return count_macs(policy.net, lambda: policy.net(x, torch.zeros(1, dtype=torch.long, device=dev),
                                                     torch.zeros(1, 2, device=dev)))


def nca_macs_per_step(model, dev):
    """一步 NCA 更新：共享 MLP（钩子统计）+ 固定核感知（F.conv1d，手算）。观测编码每次规划只算一次，忽略。"""
    obs = torch.zeros(1, 2, device=dev)
    seed, emb = model.seed(obs, torch.zeros(1, model.z_dim, device=dev)), model.obs_net(obs)
    mlp = count_macs(model.update, lambda: model.step(model.blank(obs), seed, emb))
    return mlp + model.kernel.shape[0] * model.kernel.shape[2] * T_P


@torch.no_grad()
def dp_first_plan(policy, ts, dev, n=2000, seed=0):
    torch.manual_seed(seed)
    obs = torch.from_numpy(A.START[None]).to(dev).repeat(n, 1)
    noise = torch.randn(n, 2, T_P, device=dev)
    return policy.denoise(noise, obs, ts)[:, -1, 0].cpu().numpy(), noise


def n_params(m):
    return sum(p.numel() for p in m.parameters())


# ----------------------------------------------------------------------------- 画图

def plot(results, demos, frames, offsets, nca_offsets, sweep, path, label="Diffusion Policy (DDIM)"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def scene(ax, title, lim=((-1, 1), (-1, 1))):
        ax.add_patch(plt.Rectangle((-A.BOX_HW, -A.BOX_HH), 2 * A.BOX_HW, 2 * A.BOX_HH, color="0.7", zorder=1))
        ax.plot(*A.START, "o", color="0.2", ms=5, zorder=3)
        ax.plot(*A.GOAL, "*", color="0.2", ms=11, zorder=3)
        ax.set_xlim(*lim[0]); ax.set_ylim(*lim[1]); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=10)

    color = {0: "#BA7517", 1: "#1D9E75", 2: "#E24B4A"}   # 超时 / 成功 / 碰撞
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for ax, (name, (m, paths, status, length)) in zip(axes.flat[:3], results.items()):
        scene(ax, f"{name}\nsuccess {m['success']:.0%}  collision {m['collision']:.0%}  "
                  f"L/R {m['left']}/{m['right']}  plan flips {m['plan_flips']:.2f}")
        for demo in demos[:24]:
            ax.plot(demo[:, 0], demo[:, 1], color="0.88", lw=0.8, zorder=0)
        for e in range(min(40, len(paths))):
            p = paths[e, :length[e] + 1]
            ax.plot(p[:, 0], p[:, 1], color=color[int(status[e])], lw=1.1, alpha=0.75, zorder=2)

    ax = axes.flat[3]
    scene(ax, f"{label}: predicted final plan at each step (two noises)\nlighter = earlier step",
          lim=((-0.55, 0.55), (-0.95, -0.1)))
    for j in range(frames.shape[0]):
        a = 0.15 + 0.85 * j / max(1, frames.shape[0] - 1)
        for i, c in enumerate(("#1D9E75", "#7F77DD")):
            ax.plot(frames[j, i, :, 0], frames[j, i, :, 1], "-o", color=c, alpha=a, ms=2.5, lw=1.2)

    ax = axes.flat[4]
    ax.hist(offsets, bins=60, range=(-0.5, 0.5), color="#378ADD", alpha=0.7, label=label)
    if nca_offsets is not None:
        ax.hist(nca_offsets, bins=60, range=(-0.5, 0.5), color="#1D9E75", alpha=0.5, label="latent NCA")
    ax.set_xlabel("lateral offset of the first plan's last waypoint")
    ax.set_title("first plan at the start", fontsize=10)
    ax.legend(fontsize=9)

    ax = axes.flat[5]
    for name, pts in sweep.items():
        pts = sorted(pts)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "-o", ms=4, label=name)
    ax.set_xscale("log")
    ax.set_xlabel("network calls per plan (log)")
    ax.set_ylabel("success rate")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.set_title("success vs. compute per plan", fontsize=10)
    if sweep:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ----------------------------------------------------------------------------- 主程序

def fmt(m):
    return "  ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in m.items())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", choices=["ddim", "flow"], default="ddim", help="扩散 + DDIM，或流匹配 + Euler")
    ap.add_argument("--iters", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--lr_warmup", type=int, default=300)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--dims", type=int, nargs="+", default=[32, 64, 128], help="U-Net 每层通道数")
    ap.add_argument("--kernel", type=int, default=5)
    ap.add_argument("--t_dim", type=int, default=64, help="扩散时间步嵌入的维度")
    ap.add_argument("--groups", type=int, default=8, help="GroupNorm 的组数；通道很少时应调小")
    ap.add_argument("--obs_noise", type=float, default=0.01)
    ap.add_argument("--ddim_steps", type=int, default=10, help="冷启动的采样步数（flow 时是 Euler 步数）")
    ap.add_argument("--spacing", choices=["leading", "trailing"], default="leading")
    ap.add_argument("--warm_t", type=int, default=50, help="热启动时把上一段计划加噪到第几个扩散步（共 100）")
    ap.add_argument("--warm_steps", type=int, default=3, help="热启动时的 DDIM 步数")
    ap.add_argument("--n_demos", type=int, default=200)
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--max_steps", type=int, default=72)
    ap.add_argument("--log_every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu", help="训练用的设备（cpu / cuda / mps）")
    ap.add_argument("--eval_device", default="cpu",
                    help="评估用的设备；默认 cpu，和 action_nca.py 的评估一致（随机数、耗时可比）")
    ap.add_argument("--train_only", action="store_true")
    ap.add_argument("--load", default=None, help="加载已训练的 dp_policy.pt，跳过训练")
    ap.add_argument("--nca_dir", default="nca_out", help="有 nca_det.pt / nca_latent.pt 时一起评估对比")
    ap.add_argument("--no_sweep", action="store_true")
    ap.add_argument("--out", default="dp_out")
    args = ap.parse_args()
    args.dims = tuple(args.dims)

    torch.backends.nnpack.set_flags(False)    # macOS 上 NNPACK 的一维卷积比默认实现慢约 4 倍
    os.makedirs(args.out, exist_ok=True)
    demos = A.make_demos(args.n_demos, np.random.default_rng(args.seed))
    dev = torch.device(args.eval_device)
    if args.load:
        ck = torch.load(args.load, map_location="cpu")
        args.iters, args.batch = ck.get("iters", args.iters), ck.get("batch", args.batch)
        args.dims, args.kernel, args.t_dim = tuple(ck["dims"]), ck["kernel"], ck.get("t_dim", 64)
        args.groups, args.lr = ck.get("groups", 8), ck.get("lr", args.lr)
        args.method = ck.get("method", "ddim")
        policy = POLICIES[args.method](ck["lo"], ck["hi"], dims=args.dims, k=args.kernel, t_dim=args.t_dim, groups=args.groups)
        policy.load_state_dict(ck["state"])
    else:
        policy = train_dp(demos, args, torch.device(args.device)).cpu()
        torch.save(dict(state=policy.state_dict(), lo=policy.lo, hi=policy.hi, dims=args.dims, kernel=args.kernel, t_dim=args.t_dim,
                        groups=args.groups, lr=args.lr, method=args.method,
                        iters=args.iters, batch=args.batch),
                   os.path.join(args.out, "dp_policy.pt"))
        if args.train_only:
            return
    policy = policy.to(dev).eval()
    eval_seed = args.seed + 1                 # 和 action_nca.py 评估时用的种子一致
    E, MS = args.episodes, args.max_steps
    dp_macs = dp_macs_per_call(policy, dev)
    tag, solver = ("DP", "DDIM") if args.method == "ddim" else ("FM", "Euler")

    # ---- Diffusion Policy 的三种执行方式
    print()
    dp_settings = [(f"{tag}, {solver} {args.ddim_steps}, new noise each plan", "fresh"),
                   (f"{tag}, {solver} {args.ddim_steps}, same noise", "fixed"),
                   (f"{tag}, warm start (t={args.warm_t}, {args.warm_steps} steps)", "warm")]
    results, rows = {}, []
    for name, mode in dp_settings:
        pl = DPPlanner(policy, mode, args.ddim_steps, args.spacing, args.warm_t, args.warm_steps)
        results[name] = rollout(pl, E, MS, dev, eval_seed)
        m = results[name][0]
        m["macs_per_plan"] = m["calls_per_plan"] * dp_macs
        rows.append((name, m))
        print(f"{name:45s} {fmt(m)}", flush=True)

    # ---- 同一套评估代码下的 NCA（如果 nca_out/ 里有模型）
    nca, nca_offsets = {}, None
    for kind in ("det", "latent"):
        f = os.path.join(args.nca_dir, f"nca_{kind}.pt")
        if os.path.exists(f):
            nca[kind] = A.ActionNCA().to(dev)
            nca[kind].load_state_dict(torch.load(f, map_location=dev))
    nca_macs = nca_macs_per_step(nca["latent"], dev) if nca else None
    if nca:
        for name, kind, setting in [("NCA, deterministic, cold start", "det", "cold_fixed"),
                                    ("NCA, latent z, cold start, new z each plan", "latent", "cold_fresh"),
                                    ("NCA, latent z, cold start, same z", "latent", "cold_fixed"),
                                    ("NCA, latent z, warm start (regrow)", "latent", "warm")]:
            if kind not in nca:
                continue
            m = rollout(NCAPlanner(nca[kind], kind, setting), E, MS, dev, eval_seed)[0]
            m["macs_per_plan"] = m["calls_per_plan"] * nca_macs
            rows.append((name, m))
            print(f"{name:45s} {fmt(m)}", flush=True)
        if "latent" in nca:   # 核对：通用 rollout 和 action_nca.run_episodes 给出的数字应完全一致
            ref = A.run_episodes(nca["latent"], "latent", "warm",
                                 SimpleNamespace(episodes=E, z_dim=4, max_steps=MS, cold_steps=32, warm_steps=10),
                                 dev, eval_seed)[0]
            mine = rows[-1][1]
            same = all(abs(ref[k] - mine[k]) < 1e-9 for k in ("success", "collision", "plan_flips", "roughness_x100"))
            print(f"\n核对 rollout 与 action_nca.run_episodes（latent warm）：{'一致' if same else '不一致！'}")
            nca_offsets = A.first_plan_offsets(nca["latent"], "latent",
                                               SimpleNamespace(z_dim=4, cold_steps=32), dev)[0]

    # ---- 起点处第一段计划的分布
    ts = policy.schedule(args.ddim_steps, args.spacing)
    offsets, noise = dp_first_plan(policy, ts, dev)
    first = dict(left=float(np.mean(offsets < -0.05)), right=float(np.mean(offsets > 0.05)),
                 undecided=float(np.mean(np.abs(offsets) <= 0.05)))
    print(f"\nfirst plan at start ({tag}): left {first['left']:.1%}, right {first['right']:.1%},"
          f" undecided {first['undecided']:.1%}")
    frames = []
    pick = torch.stack([noise[int(np.argmin(offsets))], noise[int(np.argmax(offsets))]])
    policy.denoise(pick, torch.from_numpy(A.START[None]).to(dev).repeat(2, 1), ts, frames=frames)

    # ---- 步数扫描：成功率随每次规划的网络调用次数变化
    sweep = {}
    if not args.no_sweep:
        print("\n步数扫描：")
        for sp in (("leading", "trailing") if args.method == "ddim" else ("uniform",)):
            key = f"{tag} fresh noise, {sp}"
            sweep[key] = []
            for st in (1, 2, 3, 5, 10, 20):
                m = rollout(DPPlanner(policy, "fresh", st, sp), E, MS, dev, eval_seed)[0]
                sweep[key].append((m["calls_per_plan"], m["success"]))
                print(f"  {key:28s} ddim_steps={st:2d}  success={m['success']:.3f}  collision={m['collision']:.3f}"
                      f"  flips={m['plan_flips']:.2f}  roughness={m['roughness_x100']:.2f}", flush=True)
        key = f"{tag} warm start (t={args.warm_t})"
        sweep[key] = []
        for ws in (1, 2, 3, 5):
            m = rollout(DPPlanner(policy, "warm", args.ddim_steps, args.spacing, args.warm_t, ws), E, MS, dev, eval_seed)[0]
            sweep[key].append((m["calls_per_plan"], m["success"]))
            print(f"  {key:28s} warm_steps={ws}  calls/plan={m['calls_per_plan']:.1f}  success={m['success']:.3f}"
                  f"  collision={m['collision']:.3f}  flips={m['plan_flips']:.2f}"
                  f"  roughness={m['roughness_x100']:.2f}", flush=True)
        if "latent" in nca:
            key = "NCA regrow"
            sweep[key] = []
            for ws in (2, 4, 6, 10, 14):
                m = rollout(NCAPlanner(nca["latent"], "latent", "warm", warm_steps=ws), E, MS, dev, eval_seed)[0]
                sweep[key].append((m["calls_per_plan"], m["success"]))
                print(f"  {key:28s} warm_steps={ws:2d}  calls/plan={m['calls_per_plan']:.1f}"
                      f"  success={m['success']:.3f}  roughness={m['roughness_x100']:.2f}", flush=True)

    # ---- 保存
    summary = dict(config=dict(vars(args), dp_params=n_params(policy.net), dp_macs_per_call=dp_macs,
                               nca_params=n_params(nca["latent"]) if nca else None, nca_macs_per_step=nca_macs,
                               eval_seed=eval_seed),
                   settings={name: m for name, m in rows}, first_plan_at_start=first,
                   sweep={k: [list(p) for p in v] for k, v in sweep.items()})
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    lines = [f"训练种子 {args.seed}，评估种子 {eval_seed}，每种方式 {E} 回合。"
             f"{tag}（{'扩散 + DDIM' if args.method == 'ddim' else '流匹配 + Euler'}）：{args.iters} 轮，batch {args.batch}，U-Net {args.dims}（卷积核 {args.kernel}，时间嵌入 {args.t_dim}，GroupNorm {args.groups} 组，学习率 {args.lr:g}），{n_params(policy.net):,} 参数，"
             f"每次调用 {dp_macs / 1e6:.2f} M 乘加。"
             + (f"NCA：{n_params(nca['latent']):,} 参数，每步 {nca_macs / 1e3:.0f} K 乘加。" if nca else ""), "",
             "| 执行方式 | 成功率 | 碰撞率 | 成功回合中左/右 | 每回合计划翻转 | 路径粗糙度 | 最大拐折 | 每次规划的网络调用 | 每次规划的乘加 (M) | 每次规划耗时 (ms, 200 回合一批) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for name, m in rows:
        lines.append(f"| {name} | {m['success']:.1%} | {m['collision']:.1%} | {m['left']} / {m['right']} | "
                     f"{m['plan_flips']:.3f} | {m['roughness_x100']:.2f} | {m['max_kink_x100']:.1f} | {m['calls_per_plan']:.1f} | "
                     f"{m['macs_per_plan'] / 1e6:.2f} | {m['ms_per_plan']:.1f} |")
    with open(os.path.join(args.out, "compare.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    plot(results, demos, np.stack(frames), offsets, nca_offsets, sweep, os.path.join(args.out, "results_dp.png"),
         label=f"{tag} ({solver})")
    print(f"\nsaved to {args.out}/")


if __name__ == "__main__":
    main()
