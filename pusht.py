#!/usr/bin/env python3
"""
Push-T（状态版）上的 Action NCA 与 Diffusion Policy。

数据：Diffusion Policy 原始的 pusht_cchi_v7_replay.zarr（206 条人类示范，25650 步），放在 data/pusht/：
  curl -O https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip && unzip pusht.zip -d data/
环境：gym_pusht（pip install gym-pusht），推杆由 PD 控制（有惯性），动作 = 推杆的目标位置（像素，0–512）。

观测（12 维）：当前帧和上一帧各 6 维 = 推杆 x、y，T 形块 x、y，sin θ，cos θ；位置都归一化到 [−1, 1]（x / 256 − 1）。
当前帧放在最前面，所以 obs[:, :2] 就是推杆当前位置（NCA 的种子细胞写这两维）。
动作块：从当前时刻起 16 步（越过示范末尾用最后一个动作补齐），执行前 8 步再重新规划（和 Diffusion Policy 相同）。

模型（都在 GPU 上训练）：
  NCA：隐变量 NCA，min-of-K（K = 8）+ 精度损失 + 样本池再生训练（和 action_nca.train 同一套做法）；
  DP： 条件 1D U-Net + DDIM（diffusion_baseline.DiffusionPolicy），EMA。
评估（CPU）：测试种子 100000 起，每种方式 --episodes 个回合，每回合最多 300 步（和 Diffusion Policy 相同）。
  得分 = 每回合覆盖率奖励的最大值（clip(覆盖率 / 0.95, 0, 1)）的平均；成功 = 覆盖率曾超过 0.95。
  动作粗糙度 = 执行的目标位置的二阶差分（像素）；推杆粗糙度同理；
  衔接跳变 = 重新规划时新计划第一步和旧计划在同一时刻的动作之差（像素），衡量相邻动作块是否连贯。

用法
  python pusht.py --job nca --seed 0                 # 训练一个模型并评估
  python pusht.py --job dp --seed 0 --dp_dims 64 128 256
  python pusht.py --all                              # 种子 0–2，NCA / DP / 小 DP，汇总
  python pusht.py --all --aggregate_only
  python pusht.py --job nca --seed 0 --extra runs/pusht_tune/dagger/se_global_seed{seed}.npz   # 加上 DAgger 样本训练
  python pusht.py --job nca --seed 0 --extra runs/pusht_tune/dart/dp_seed{seed}.npz --scale_iters 1   # DART 样本，训练轮数随数据量放大
输出：runs/pusht/<模型>_seed<s>/（模型、metrics.json、日志）和 runs/pusht/summary.md。
"""
import argparse
import copy
import json
import math
import multiprocessing as mp
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
import torch
import torch.nn as nn

import action_nca as A
import diffusion_baseline as DB

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "pusht", "pusht_cchi_v7_replay.zarr")
T_P, T_A, MAX_STEPS, TEST_SEED = A.T_P, 8, 300, 100000
SETTINGS = {"nca": [("每次新 z", "fresh"), ("z 固定", "fixed"), ("再生", "warm")],
            "dp": [("每次新噪声", "fresh"), ("固定噪声", "fixed"), ("热启动", "warm")]}


# ----------------------------------------------------------------------------- 数据

def feat(state):
    """(…, 5) 推杆 x、y，块 x、y、θ → (…, 6)，位置归一化到 [−1, 1]。"""
    state = np.asarray(state, np.float32)
    return np.concatenate([state[..., :4] / 256.0 - 1.0, np.sin(state[..., 4:5]), np.cos(state[..., 4:5])], -1)


def load_data(path=DATA):
    """返回 O (N, 12) 观测、Y (N, T_P, 2) 归一化动作块、NXT (N,) 执行 T_A 步之后的样本下标（越过回合末尾为 −1）。"""
    import zarr
    z = zarr.open(path, "r")
    state, action, ends = z["data/state"][:], z["data/action"][:], z["meta/episode_ends"][:]
    f, act = feat(state), action.astype(np.float32) / 256.0 - 1.0
    O, Y, NXT = [], [], []
    start = 0
    for end in ends:
        for i in range(start, end):
            O.append(np.concatenate([f[i], f[max(i - 1, start)]]))
            Y.append(act[np.minimum(np.arange(i, i + T_P), end - 1)])
            NXT.append(i + T_A if i + T_A < end else -1)
        start = end
    return np.array(O, np.float32), np.array(Y, np.float32), np.array(NXT, np.int64)


# ----------------------------------------------------------------------------- 训练

def train_nca(O, Y, NXT, a, dev):
    """和 action_nca.train 同一套：冷启动 min-of-K + 精度损失 + 样本池再生，只是样本换成 Push-T 的 (观测, 动作块)。"""
    torch.manual_seed(a.seed)
    O, Y, NXT = (torch.as_tensor(x, device=dev) for x in (O, Y, NXT))
    N, B, K, Zd, P = len(O), a.batch, a.k, a.z_dim, a.pool
    model = A.ActionNCA(hidden=a.nca_hidden, z_dim=Zd, emb=a.nca_emb, width=a.nca_width, fire_rate=a.fire_rate,
                        obs_dim=O.shape[1], z_emb=bool(a.z_emb), global_ch=bool(a.nca_global),
                        seed_pad=bool(a.seed_pad), noise_start=a.noise_start).to(dev)
    ema = copy.deepcopy(model).requires_grad_(False) if a.nca_ema > 0 else None
    wt = torch.ones(T_P, 1, device=dev)                                # 每个时刻的误差权重：前 T_A 步是真正会执行的
    wt[:T_A] = a.exec_weight
    sq = lambda p, t: ((p - t) ** 2 * wt).sum((1, 2))
    rough = lambda p: ((p[:, 2:] - 2 * p[:, 1:-1] + p[:, :-2]) ** 2).sum((1, 2)).mean()   # 计划的二阶差分（平滑项）
    opt = torch.optim.Adam(model.parameters(), lr=a.nca_lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.nca_iters, eta_min=a.nca_lr * 0.05)
    lib_rel = (Y - O[:, None, :2]).reshape(N, -1)                      # 精度损失用的“相对当前位置”的动作块库
    pool_s = torch.zeros(P, model.C, T_P, device=dev)
    pool_z = torch.zeros(P, Zd, device=dev)
    pool_i = torch.zeros(P, dtype=torch.long, device=dev)
    pool_ok = torch.zeros(P, dtype=torch.bool, device=dev)
    pool_n = torch.zeros(P, device=dev)                                # 每个样本已经连续再生了几次（诊断用）
    noise = lambda o: torch.cat([o[:, :4] + a.obs_noise * torch.randn_like(o[:, :4]), o[:, 4:]], 1)
    if a.regrow_jitter > 0:                                            # 每个样本所在示范回合的结尾（不含），再生抖动时用
        import zarr
        ends = zarr.open(DATA, "r")["meta/episode_ends"][:]
        END = torch.as_tensor(np.repeat(ends, np.diff(np.concatenate([[0], ends]))), device=dev)
    t0 = time.time()
    for it in range(1, a.nca_iters + 1):
        idx = torch.randint(0, N, (B,), device=dev)
        obs, tgt = noise(O[idx]), Y[idx]
        steps = int(torch.randint(a.cold_min, a.cold_max + 1, ()).item())
        with torch.no_grad():                                          # min-of-K
            zk = torch.randn(B * K, Zd, device=dev)
            ok = obs.repeat_interleave(K, 0)
            ek = torch.randn(B * K, 2, T_P, device=dev) if a.noise_start > 0 else None   # 从噪声出发：起点也是随机的
            sk = model.grow(model.blank(ok, ek), ok, zk, steps)
            err = sq(A.plan_of(sk), tgt.repeat_interleave(K, 0)).view(B, K)
            pick = err.argmin(1)
            z = zk.view(B, K, Zd)[torch.arange(B, device=dev), pick]
            e = ek.view(B, K, 2, T_P)[torch.arange(B, device=dev), pick] if ek is not None else None   # 选中的 z 配原来的噪声
        s = model.grow(model.blank(obs, e), obs, z, steps)
        loss_c = sq(A.plan_of(s), tgt).mean()

        loss_p = torch.zeros((), device=dev)
        if a.prec_weight > 0:                                          # 精度损失：随机 z 的计划要贴近附近状态的某个动作块
            M = a.prec_samples
            pobs = obs.repeat_interleave(M, 0)
            zr = torch.randn(B * M, Zd, device=dev)
            ps = model.grow(model.blank(pobs), pobs, zr, steps)
            rel = (A.plan_of(ps) - pobs[:, None, :2]).reshape(B * M, -1)
            near = torch.cdist(obs, O) < a.prec_radius
            near[torch.arange(B, device=dev), idx] = True
            e2 = torch.cdist(rel, lib_rel) ** 2
            loss_p = e2.masked_fill(~near.repeat_interleave(M, 0), float("inf")).min(1).values.mean()

        loss_w = torch.zeros((), device=dev)
        live = torch.nonzero(pool_ok).squeeze(1)
        if live.numel() >= B:                                          # 再生：切掉执行过的 8 个细胞、前移、补尾巴，续长
            pi = live[torch.randperm(live.numel(), device=dev)[:B]]
            ni = NXT[pool_i[pi]]
            if a.regrow_jitter > 0:                                    # 新观测取 8 ± jitter 步后：目标不再等于照抄旧计划，逼再生按观测纠正
                d = torch.randint(-a.regrow_jitter, a.regrow_jitter + 1, ni.shape, device=dev)
                ni = torch.minimum(pool_i[pi] + T_A + d, END[pool_i[pi]] - 1)
            wsteps = int(torch.randint(a.warm_min, a.warm_max + 1, ()).item())
            prev = A.shift(pool_s[pi], T_A)
            if a.regrow_noise > 0:                                     # 扰动旧计划的动作通道，让再生学会纠正和新观测不符的旧计划
                prev = torch.cat([prev[:, :2] + a.regrow_noise * torch.randn_like(prev[:, :2]), prev[:, 2:]], 1)
            ws = model.grow(prev, noise(O[ni]), pool_z[pi], wsteps)
            loss_w = sq(A.plan_of(ws), Y[ni]).mean()
            pool_s[pi], pool_i[pi], pool_ok[pi] = ws.detach(), ni, NXT[ni] >= 0
            pool_n[pi] += 1

        loss = loss_c + loss_w + a.prec_weight * loss_p
        if a.smooth_weight > 0:
            loss = loss + a.smooth_weight * (rough(A.plan_of(s)) + (rough(A.plan_of(ws)) if live.numel() >= B else 0.0))
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if ema is not None:
            decay = min(a.nca_ema, (1 + it) / (10 + it))
            with torch.no_grad():
                for pe, p in zip(ema.parameters(), model.parameters()):
                    pe.lerp_(p, 1 - decay)

        cand_s, cand_z, cand_i = s.detach(), z, idx                    # 冷启动的结果放进样本池
        if a.pool_random and a.prec_weight > 0:                        # 也放随机 z 长出的计划（评估时 z 是随机的，不是 min-of-K 选出来的）
            cand_s, cand_z = torch.cat([cand_s, ps.detach()]), torch.cat([cand_z, zr])
            cand_i = torch.cat([cand_i, idx.repeat_interleave(a.prec_samples, 0)])
        keep = torch.nonzero(NXT[cand_i] >= 0).squeeze(1)
        if a.pool_inject is not None and keep.numel() > a.pool_inject:  # 每轮只换进一部分，样本能连续再生更多次（评估时一回合约 30 次）
            keep = keep[torch.randperm(keep.numel(), device=dev)[:a.pool_inject]]
        if keep.numel():
            free = torch.nonzero(~pool_ok).squeeze(1)
            free = free[torch.randperm(free.numel(), device=dev)]
            if free.numel() < keep.numel():
                used = torch.nonzero(pool_ok).squeeze(1)
                used = used[torch.randperm(used.numel(), device=dev)[:keep.numel() - free.numel()]]
                free = torch.cat([free, used])
            slots = free[:keep.numel()]
            pool_s[slots], pool_z[slots], pool_i[slots], pool_ok[slots] = cand_s[keep], cand_z[keep], cand_i[keep], True
            pool_n[slots] = 0
        if it == 1 or it % a.log_every == 0:
            chain = pool_n[pool_ok].mean().item() if pool_ok.any() else 0.0
            print(f"[nca] it {it:6d}  cold {loss_c.item():.4f}  warm {loss_w.item():.4f}  prec {loss_p.item():.4f}"
                  f"  chain {chain:.1f}  {time.time() - t0:.0f}s", flush=True)
    return ema if ema is not None else model


def train_dp(O, Y, a, dev):
    torch.manual_seed(a.seed)
    O, Y = torch.as_tensor(O, device=dev), torch.as_tensor(Y, device=dev)
    N, B, cd = len(O), a.batch, O.shape[1] - 2
    one = torch.ones(2)
    policy = DB.DiffusionPolicy(-one, one, dims=tuple(a.dp_dims), k=a.dp_kernel, t_dim=a.dp_t_dim, groups=a.dp_groups,
                                cond_lo=-torch.ones(cd), cond_hi=torch.ones(cd)).to(dev)   # 数据已归一化到 [−1, 1]
    ema = copy.deepcopy(policy).requires_grad_(False)
    opt = torch.optim.AdamW(policy.parameters(), lr=a.dp_lr, weight_decay=1e-6)
    warm = 500
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: (i + 1) / warm if i < warm else 0.5 * (1 + math.cos(math.pi * (i - warm) / (a.dp_iters - warm))))
    t0, run = time.time(), 0.0
    for it in range(1, a.dp_iters + 1):
        idx = torch.randint(0, N, (B,), device=dev)
        obs = O[idx]
        obs = torch.cat([obs[:, :4] + a.obs_noise * torch.randn_like(obs[:, :4]), obs[:, 4:]], 1)
        loss = policy.loss(obs, Y[idx])
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        sched.step()
        decay = min(0.999, (1 + it) / (10 + it))
        with torch.no_grad():
            for pe, p in zip(ema.parameters(), policy.parameters()):
                pe.lerp_(p, 1 - decay)
        run = loss.item() if it == 1 else 0.98 * run + 0.02 * loss.item()
        if it == 1 or it % a.log_every == 0:
            print(f"[dp] it {it:6d}  loss {run:.4f}  {time.time() - t0:.0f}s", flush=True)
    return ema


# ----------------------------------------------------------------------------- 规划器（动作块长 16，执行 8 步）

class NCAPlan:
    def __init__(self, model, mode, z_dim, cold_steps, warm_steps, refresh=0, z_scale=1.0):
        self.m, self.mode, self.zd, self.cs, self.ws = model, mode, z_dim, cold_steps, warm_steps
        self.refresh = refresh                                           # 再生时每 refresh 次规划冷启动一次（0 = 从不）
        self.zs = z_scale                                                # z ~ N(0, z_scale²)：小于 1 时偏向常见的模态

    def reset(self, E):
        self.E, self.s, self.calls, self.n = E, None, 0, 0
        self.z = self.zs * torch.randn(E, self.zd)
        self.eps = torch.randn(E, 2, T_P) if self.m.noise_start > 0 else None   # 从噪声出发时 fixed 也固定起点的噪声

    @torch.no_grad()
    def plan(self, obs):
        if self.mode == "warm" and self.s is not None and not (self.refresh and self.n % self.refresh == 0):
            self.s = self.m.grow(A.shift(self.s, T_A), obs, self.z, self.ws)
            self.calls += self.ws
        else:
            if self.mode == "fresh":
                self.z = self.zs * torch.randn(self.E, self.zd)
            self.s = self.m.grow(self.m.blank(obs, self.eps if self.mode == "fixed" else None), obs, self.z, self.cs)
            self.calls += self.cs
        self.n += 1
        return A.plan_of(self.s)


class DPPlan:
    def __init__(self, policy, mode, steps=10, warm_t=50, warm_steps=3):
        self.p, self.mode = policy, mode
        self.ts, self.warm_ts = policy.schedule(steps, "leading"), policy.warm_schedule(warm_t, warm_steps)

    def reset(self, E):
        self.E, self.prev, self.calls = E, None, 0
        self.noise = torch.randn(E, 2, T_P)

    @torch.no_grad()
    def plan(self, obs):
        if self.mode == "warm" and self.prev is not None:
            prev = torch.cat([self.prev[:, T_A:], self.prev[:, -1:].expand(-1, T_A, -1)], 1)
            x, ts = self.p.noised(prev, self.warm_ts, torch.randn(self.E, 2, T_P)), self.warm_ts
        else:
            x = self.noise if self.mode == "fixed" else torch.randn(self.E, 2, T_P)
            ts = self.ts
        self.calls += len(ts)
        self.prev = self.p.denoise(x, obs, ts)
        return self.prev


# ----------------------------------------------------------------------------- 评估

def rollout(planner, episodes, seed, start=TEST_SEED):
    """E 个环境同步执行。返回指标和轨迹 dict：每个回合的推杆位置 agent、执行的动作 acts、块位姿 block（像素），
    以及 best（覆盖率奖励最大值）、success。"""
    import gym_pusht  # noqa: F401  注册环境
    import gymnasium as gym
    torch.manual_seed(seed)
    E = episodes
    envs = [gym.make("gym_pusht/PushT-v0", obs_type="state").unwrapped for _ in range(E)]
    states = np.stack([env.reset(seed=start + i)[0] for i, env in enumerate(envs)]).astype(np.float32)
    prev = states.copy()
    planner.reset(E)
    alive = np.ones(E, bool)
    best = np.zeros(E)
    success = np.zeros(E, bool)
    acts, agent, block = [[] for _ in range(E)], [[s[:2].copy()] for s in states], [[s[2:5].copy()] for s in states]
    jumps, prev_plan, plan_time, n_plans = [], None, 0.0, 0
    for t0 in range(0, MAX_STEPS, T_A):
        obs = torch.from_numpy(np.concatenate([feat(states), feat(prev)], 1))
        tic = time.perf_counter()
        plan = (planner.plan(obs).numpy() + 1.0) * 256.0                  # (E, 16, 2) 像素
        plan_time += time.perf_counter() - tic
        n_plans += 1
        if prev_plan is not None:                                        # 衔接跳变：新计划第一步 vs 旧计划同一时刻的动作
            jumps.extend(np.linalg.norm(plan[alive, 0] - prev_plan[alive, T_A], axis=1).tolist())
        prev_plan = plan
        for k in range(min(T_A, MAX_STEPS - t0)):
            prev = states.copy()
            for e in np.flatnonzero(alive):
                s, r, term, _, info = envs[e].step(plan[e, k])
                states[e] = s
                best[e] = max(best[e], r)
                acts[e].append(plan[e, k].copy())
                agent[e].append(s[:2].copy())
                block[e].append(s[2:5].copy())
                if term:
                    success[e], alive[e] = True, False
            if not alive.any():
                break
        if not alive.any():
            break
    rough = lambda seqs: float(np.mean([np.linalg.norm(np.diff(np.array(q), 2, axis=0), axis=1).mean()
                                        for q in seqs if len(q) > 2]))
    m = dict(score=float(best.mean()), success=float(success.mean()), action_rough=rough(acts),
             agent_rough=rough(agent), chunk_jump=float(np.mean(jumps)) if jumps else float("nan"),
             calls_per_plan=planner.calls / n_plans, ms_per_plan=1000 * plan_time / n_plans,
             steps=float(np.mean([len(q) for q in acts])))
    arr = lambda seqs: [np.array(q) for q in seqs]
    return m, dict(agent=arr(agent), acts=arr(acts), block=arr(block), best=best, success=success)


def make_planner(kind, path, mode, cold_steps=32, warm_steps=10, dp_steps=10, dp_warm_steps=3, fire_rate=None,
                 warm_refresh=0, z_scale=1.0):
    """从 nca.pt / dp.pt 建规划器（CPU）。fire_rate 不为 None 时覆盖 NCA 训练时的更新概率；
    warm_refresh > 0 时 NCA 再生每隔这么多次规划冷启动一次；z_scale 缩放 NCA 的随机 z。"""
    ck = torch.load(path, map_location="cpu")
    if kind == "nca":
        m = A.ActionNCA(hidden=ck["hidden"], z_dim=ck["z_dim"], emb=ck["emb"], width=ck["width"], obs_dim=ck["obs_dim"],
                        fire_rate=fire_rate if fire_rate is not None else ck.get("fire_rate", 0.5),
                        z_emb=ck.get("z_emb", False), global_ch=ck.get("global_ch", False),
                        seed_pad=ck.get("seed_pad", True), noise_start=ck.get("noise_start", 0.0))
        m.load_state_dict(ck["state"])
        return NCAPlan(m.eval(), mode, ck["z_dim"], cold_steps, warm_steps, warm_refresh, z_scale)
    cd, one = ck["cond_dim"], torch.ones(2)
    p = DB.DiffusionPolicy(-one, one, dims=tuple(ck["dims"]), k=ck["kernel"], t_dim=ck["t_dim"], groups=ck["groups"],
                           cond_lo=-torch.ones(cd), cond_hi=torch.ones(cd))
    p.load_state_dict(ck["state"])
    return DPPlan(p.eval(), mode, dp_steps, 50, dp_warm_steps)


def eval_setting(job):
    """在单独的进程里评估一种执行方式（gym 环境在 CPU 上）。"""
    kind, path, mode, a = job
    torch.set_num_threads(1)
    planner = make_planner(kind, path, mode, a["cold_steps"], a["warm_steps"], a["dp_steps"], a["dp_warm_steps"],
                           a.get("eval_fire_rate"), a.get("warm_refresh", 0), a.get("z_scale", 1.0))
    return mode, rollout(planner, a["episodes"], a["seed"] + 1)[0]


def run_job(a):
    out = a.out
    os.makedirs(out, exist_ok=True)
    O, Y, NXT = load_data()
    n_demo = len(O)
    for p in a.extra or []:                                            # DAgger：学生自己的状态 + 老师的动作块，和示范一起训练
        p = p.format(seed=a.seed)
        d = np.load(p if os.path.isabs(p) else os.path.join(HERE, p))
        O, Y = np.concatenate([O, d["O"]]), np.concatenate([Y, d["Y"]])
        NXT = np.concatenate([NXT, -np.ones(len(d["O"]), np.int64)])  # 没有“8 步之后”，不进再生的样本池
    if a.scale_iters and len(O) > n_demo:                              # 训练轮数随数据量放大：每个示范样本被看到的次数不变
        f = len(O) / n_demo
        sc = lambda n: int(round(n * f, -3)) if n >= 1000 else int(round(n * f))   # 冒烟测试时轮数很少，不取整到千
        a.nca_iters, a.dp_iters = sc(a.nca_iters), sc(a.dp_iters)
        print(f"scale_iters: {len(O)} samples ({f:.2f} x demos) -> nca_iters {a.nca_iters}, dp_iters {a.dp_iters}", flush=True)
    dev = torch.device(a.device)
    t0 = time.time()
    if a.job == "nca":
        model = train_nca(O, Y, NXT, a, dev).cpu().eval()
        path = os.path.join(out, "nca.pt")
        torch.save(dict(state=model.state_dict(), hidden=a.nca_hidden, z_dim=a.z_dim, emb=a.nca_emb, width=a.nca_width,
                        fire_rate=a.fire_rate, z_emb=bool(a.z_emb), global_ch=bool(a.nca_global),
                        seed_pad=bool(a.seed_pad), noise_start=a.noise_start, obs_dim=O.shape[1]), path)
    else:
        model = train_dp(O, Y, a, dev).cpu().eval()
        path = os.path.join(out, "dp.pt")
        torch.save(dict(state=model.state_dict(), dims=a.dp_dims, kernel=a.dp_kernel, t_dim=a.dp_t_dim,
                        groups=a.dp_groups, cond_dim=O.shape[1] - 2), path)
    res = dict(train_minutes=(time.time() - t0) / 60, params=sum(p.numel() for p in model.parameters()),
               n_samples=len(O), n_extra=len(O) - n_demo,
               args={k: v for k, v in vars(a).items() if isinstance(v, (int, float, str, list))})
    print(f"trained in {res['train_minutes']:.1f} min, {res['params']} params", flush=True)
    jobs = [(a.job, path, mode, vars(a)) for _, mode in SETTINGS[a.job]]
    with ProcessPoolExecutor(len(jobs), mp_context=mp.get_context("spawn")) as ex:   # 父进程已用过 CUDA，不能 fork
        for mode, m in ex.map(eval_setting, jobs):
            res[mode] = m
            print(f"{mode:6s} " + "  ".join(f"{k} {v:.3f}" for k, v in m.items()), flush=True)
    with open(os.path.join(out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)


# ----------------------------------------------------------------------------- 调度和汇总

# 名字 → (模型, 额外参数)。dp 约 410 万参数；dp_small 的参数量和 NCA（约 12 万）相近
VARIANTS = {"nca": ("nca", []), "dp": ("dp", []),
            "dp_small": ("dp", ["--dp_dims", "12", "24", "48", "--dp_kernel", "3", "--dp_t_dim", "32", "--dp_groups", "4"])}
LABELS = {"nca": "NCA", "dp": "DP", "dp_small": "小 DP"}


def run_all(a):
    jobs = [(m, s) for s in a.seeds for m in a.variants]

    def launch(job):
        m, s = job
        d = os.path.join(a.runs, f"{m}_seed{s}")
        if os.path.exists(os.path.join(d, "metrics.json")):
            return
        os.makedirs(d, exist_ok=True)
        cmd = [sys.executable, os.path.abspath(__file__), "--job", VARIANTS[m][0], "--seed", str(s), "--out", d,
               "--device", f"cuda:{(s + list(VARIANTS).index(m)) % 2}" if torch.cuda.is_available() else "cpu"]
        for k, v in vars(a).items():                                   # 其余参数原样传给子进程
            if v is not None and k not in {"all", "aggregate_only", "job", "seed", "seeds", "device", "jobs", "out", "runs",
                                           "variants"}:
                cmd += [f"--{k}", *map(str, v)] if isinstance(v, list) else [f"--{k}", str(v)]
        cmd += VARIANTS[m][1]                                          # 后出现的参数覆盖前面的
        env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONIOENCODING="utf-8")
        t0 = time.time()
        with open(os.path.join(d, "log.txt"), "w", encoding="utf-8") as f:
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=HERE).returncode
        print(f"{'完成' if rc == 0 else '失败'} {m}_seed{s}，{(time.time() - t0) / 60:.1f} 分钟", flush=True)

    with ThreadPoolExecutor(a.jobs) as ex:
        list(ex.map(launch, jobs))


def fmt(v, pct=False, nd=2):
    v = np.asarray(v, float)
    return f"{100 * v.mean():.1f}% ± {100 * v.std():.1f}%" if pct else f"{v.mean():.{nd}f} ± {v.std():.{nd}f}"


def aggregate(a):
    M = {}
    for m in a.variants:
        for s in a.seeds:
            f = os.path.join(a.runs, f"{m}_seed{s}", "metrics.json")
            if os.path.exists(f):
                M[(m, s)] = json.load(open(f, encoding="utf-8"))
    L = [f"# Push-T（状态版；种子 {', '.join(map(str, a.seeds))}；每种方式 {a.episodes} 个测试回合，测试种子 {TEST_SEED} 起）\n",
         "得分 = 每回合覆盖率奖励最大值的平均；成功 = 覆盖率曾超过 0.95。粗糙度、衔接跳变的单位是像素。单元格是均值 ± 标准差。\n",
         "| 方法 | 执行方式 | 种子数 | 得分 | 成功率 | 动作粗糙度 | 推杆粗糙度 | 衔接跳变 | 网络调用 / 次规划 | 毫秒 / 次规划 |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for m in a.variants:
        for name, st in SETTINGS[VARIANTS[m][0]]:
            r = [M[(m, s)][st] for s in a.seeds if (m, s) in M]
            if r:
                L.append(f"| {LABELS[m]} | {name} | {len(r)} | {fmt([x['score'] for x in r], nd=3)} | "
                         f"{fmt([x['success'] for x in r], True)} | {fmt([x['action_rough'] for x in r])} | "
                         f"{fmt([x['agent_rough'] for x in r])} | {fmt([x['chunk_jump'] for x in r])} | "
                         f"{np.mean([x['calls_per_plan'] for x in r]):.1f} | {np.mean([x['ms_per_plan'] for x in r]):.1f} |")
    L.append("")
    for m in a.variants:
        r = [M[(m, s)] for s in a.seeds if (m, s) in M]
        if r:
            L.append(f"- {LABELS[m]}：参数量 {r[0]['params']}，训练 {np.mean([x['train_minutes'] for x in r]):.1f} 分钟（GPU，均值）")
    text = "\n".join(L) + "\n"
    with open(os.path.join(a.runs, "summary.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--aggregate_only", action="store_true")
    ap.add_argument("--job", choices=["nca", "dp"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--obs_noise", type=float, default=0.0, help="位置观测上的高斯噪声（归一化单位）")
    ap.add_argument("--extra", nargs="+", default=None,
                    help="额外的（观测, 动作块）样本（.npz，含 O、Y；路径里的 {seed} 换成 --seed，相对路径从本文件所在目录算），"
                         "和示范合在一起训练（DAgger、DART，见 pusht_tune.py --stage dagger / dart）；这些样本不进 NCA 再生的样本池")
    ap.add_argument("--scale_iters", type=int, default=0,
                    help="1 = 有 --extra 时，NCA 和 DP 的训练轮数都乘以“样本总数 / 示范样本数”（取整到千），每个示范样本被看到的次数不变")
    # NCA
    ap.add_argument("--nca_iters", type=int, default=20000)
    ap.add_argument("--nca_lr", type=float, default=1e-3)
    ap.add_argument("--nca_hidden", type=int, default=62, help="隐藏通道数（加 2 个动作通道 = 细胞状态维数）")
    ap.add_argument("--nca_emb", type=int, default=64)
    ap.add_argument("--nca_width", type=int, default=256)
    ap.add_argument("--z_dim", type=int, default=4)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--pool", type=int, default=4096)
    ap.add_argument("--cold_min", type=int, default=24)
    ap.add_argument("--cold_max", type=int, default=40)
    ap.add_argument("--warm_min", type=int, default=6)
    ap.add_argument("--warm_max", type=int, default=14)
    ap.add_argument("--prec_weight", type=float, default=1.0)
    ap.add_argument("--prec_samples", type=int, default=2)
    ap.add_argument("--prec_radius", type=float, default=0.05)
    ap.add_argument("--cold_steps", type=int, default=32)
    ap.add_argument("--warm_steps", type=int, default=10)
    ap.add_argument("--fire_rate", type=float, default=0.5, help="每步每个细胞被更新的概率（1 = 同步更新）")
    ap.add_argument("--eval_fire_rate", type=float, default=None, help="评估时覆盖模型的 fire_rate")
    ap.add_argument("--nca_ema", type=float, default=0.0, help="NCA 参数的 EMA 衰减（0 = 不用，评估用 EMA 模型）")
    ap.add_argument("--smooth_weight", type=float, default=0.0, help="平滑项权重：冷启动和再生计划的二阶差分平方和")
    ap.add_argument("--regrow_noise", type=float, default=0.0, help="再生训练时旧计划动作通道上的高斯噪声（归一化单位）")
    ap.add_argument("--pool_inject", type=int, default=None,
                    help="每轮换进样本池的冷启动样本数（默认 = 全部，样本平均只再生 1–2 次；调小让样本连续再生更多次）")
    ap.add_argument("--pool_random", type=int, default=0, help="1 = 精度损失里随机 z 长出的计划也放进样本池")
    ap.add_argument("--regrow_jitter", type=int, default=0,
                    help="再生训练时新观测取 T_A ± jitter 步之后（同一示范回合内），目标随之改变（0 = 正好 T_A 步）")
    ap.add_argument("--z_emb", type=int, default=0, help="1 = z 也进观测编码、广播到每个细胞（默认只在种子细胞里）")
    ap.add_argument("--exec_weight", type=float, default=1.0, help="动作块前 T_A 步（会执行的部分）误差的权重")
    ap.add_argument("--nca_global", type=int, default=0, help="1 = 每个细胞的感知加上所有细胞状态的均值")
    ap.add_argument("--seed_pad", type=int, default=1, help="0 = 左边界复制第一个细胞，不用种子细胞（应配合 --z_emb 1）")
    ap.add_argument("--noise_start", type=float, default=0.0,
                    help="> 0 时冷启动从噪声出发：动作通道 = 当前位置 + N(0, σ²)（归一化坐标，1 ≈ 256 像素），训练和评估都用")
    ap.add_argument("--warm_refresh", type=int, default=0, help="评估再生时每隔这么多次规划冷启动一次（0 = 从不）")
    ap.add_argument("--z_scale", type=float, default=1.0, help="评估时 NCA 的 z ~ N(0, z_scale²)")
    # DP
    ap.add_argument("--dp_iters", type=int, default=30000)
    ap.add_argument("--dp_lr", type=float, default=1e-3)
    ap.add_argument("--dp_dims", type=int, nargs="+", default=[64, 128, 256])
    ap.add_argument("--dp_kernel", type=int, default=5)
    ap.add_argument("--dp_t_dim", type=int, default=64)
    ap.add_argument("--dp_groups", type=int, default=8)
    ap.add_argument("--dp_steps", type=int, default=10)
    ap.add_argument("--dp_warm_steps", type=int, default=3)
    # 评估和调度
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--log_every", type=int, default=1000)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--out", default=None)
    ap.add_argument("--runs", default=os.path.join(HERE, "runs", "pusht"))
    a = ap.parse_args()
    if a.job:
        a.out = a.out or os.path.join(a.runs, f"{a.job}_seed{a.seed}")
        run_job(a)
    elif a.all:
        if not a.aggregate_only:
            run_all(a)
        aggregate(a)


if __name__ == "__main__":
    main()
