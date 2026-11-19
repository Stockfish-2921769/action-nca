#!/usr/bin/env python3
"""扫描隐变量 NCA 的生长步数：冷启动（z 固定）的步数，以及热启动再生时每次续长的步数。

用法：python sweep_steps.py --model nca_out/nca_latent.pt
"""
import argparse

import torch

import action_nca as A


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="nca_out/nca_latent.pt")
    ap.add_argument("--z_dim", type=int, default=4)
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--max_steps", type=int, default=72)
    ap.add_argument("--seed", type=int, default=1, help="和 action_nca.py 评估时用的种子一致")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    dev = torch.device(args.device)
    model = A.ActionNCA(z_dim=args.z_dim).to(dev)
    model.load_state_dict(torch.load(args.model, map_location=dev))

    def run(setting, cold_steps, warm_steps):
        args.cold_steps, args.warm_steps = cold_steps, warm_steps
        return A.run_episodes(model, "latent", setting, args, dev, seed=args.seed)[0]

    print("冷启动、z 固定：")
    for cs in (8, 16, 24, 32, 40):
        r = run("cold_fixed", cs, 10)
        print(f"  cold_steps={cs:2d}  success={r['success']:.3f}  collision={r['collision']:.3f}"
              f"  flips={r['plan_flips']:.2f}  roughness={r['roughness_x100']:.2f}", flush=True)
    print("热启动再生（第一次冷启动 32 步）：")
    for ws in (2, 4, 6, 10, 14, 20):
        r = run("warm", 32, ws)
        print(f"  warm_steps={ws:2d}  steps/plan={r['nca_steps_per_plan']:.1f}  success={r['success']:.3f}"
              f"  collision={r['collision']:.3f}  L/R={r['left']}/{r['right']}  flips={r['plan_flips']:.2f}"
              f"  roughness={r['roughness_x100']:.2f}", flush=True)


if __name__ == "__main__":
    main()
