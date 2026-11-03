"""第 0 步：环境自检（纯 CPU，30 秒，不花钱）。

目的：在烧 GPU 之前，先确认"环境口径"是对的。判据：
  1) 回合上限必须是 200（不是 500）；
  2) 随机策略 150 回合的回报之和应落在 3000~3400（论文 3079；本机实测 3277）。
若对不上，后面的所有数字都没有可比性，先修环境。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loro.envs import effective_cap, make_env, random_baseline  # noqa: E402
from loro.metrics import PAPER  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import gymnasium as gym
    print(f"gymnasium {gym.__version__}")

    env = make_env(seed=args.seed)
    cap = effective_cap(env)
    print(f"env            : {env.spec.id if env.spec else 'wrapped'}")
    print(f"max ep len     : {cap}   (期望 200)")
    print(f"obs / action   : {env.observation_space} / {env.action_space}")

    res = random_baseline(n_episodes=args.episodes, seed=args.seed)
    print(f"random {args.episodes:>3d} eps : sum={res['sum_return']:.0f}  "
          f"mean ep len={res['mean_length']:.2f}")
    print(f"论文 random    : {PAPER['random']}")

    ok_cap = cap == 200
    ok_rand = 2800 <= res["sum_return"] <= 3800
    print(f"\n[{'PASS' if ok_cap else 'FAIL'}] 回合上限 = 200")
    print(f"[{'PASS' if ok_rand else 'FAIL'}] 随机基线在 2800~3800（论文 3079；"
          f"本机 12 个 seed 实测 3300±111）")
    return 0 if (ok_cap and ok_rand) else 1


if __name__ == "__main__":
    raise SystemExit(main())
