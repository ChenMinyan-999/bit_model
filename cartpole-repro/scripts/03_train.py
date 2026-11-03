"""第 3 步：DDQN 离线预训练 + 在线微调，产出 CartPole 学习曲线。

这一份是把作者仓库的 pretrain_from_llm.py + online_main.py 合并、参数化后的等价实现，
d3rlpy 的调用方式（DoubleDQNConfig / fit / fit_online / ReplayBuffer / ConstantEpsilonGreedy）
与原文逐行对应，不要随意改动。

变体：
  online          纯在线 DDQN，buffer 空，150 个评估点            -> 论文 On-policy
  loro            只用 D 预训练，buffer 空，140 个评估点           -> 论文 LORO（附录 A.1.2 的"丢弃"）
  mix             D 直接灌进 buffer 不预训练，150 个评估点          -> 论文"仅混合"
  loro-keep       预训练 + D 留在 buffer（原文没做，用来判定 §5.2 的"过拟合"解释）
  另外：把 --data 指向随机策略/在线策略采的数据，就是论文图 4 的两组消融。

用法：
  python scripts/03_train.py --variant loro --data artifacts/cartpole_7b_neps10.npz \
      --pretrain-steps 3000 --runs 5 --out results/loro_7b_3k
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loro.data import load_any, to_mdp_dataset  # noqa: E402
from loro.envs import H, make_env  # noqa: E402
from loro.metrics import summarize  # noqa: E402

# (是否预训练, 是否把 D 灌进 buffer, 评估点个数)
VARIANTS = {
    "online": (False, False, 150),
    "loro": (True, False, 140),
    "mix": (False, True, 150),
    "loro-keep": (True, True, 140),
}


def evaluate(algo, eval_env, seed: int) -> float:
    """跑 1 个贪心回合（论文口径：每个训练块后只评 1 次）。"""
    obs, _ = eval_env.reset(seed=seed)
    total, done = 0.0, False
    while not done:
        obs_in = np.expand_dims(obs, axis=0)
        action = int(algo.predict(obs_in)[0])
        obs, reward, terminated, truncated, _ = eval_env.step(action)
        total += float(reward)
        done = bool(terminated or truncated)
    return total


def train_once(args, ds, run_idx: int) -> dict:
    import d3rlpy
    import torch

    if args.threads > 0:
        # 小 MLP + 小 batch，线程数开太多反而极慢（线程调度开销 > 计算量）
        torch.set_num_threads(args.threads)

    pretrain, prefill, n_eval_default = VARIANTS[args.variant]
    n_eval = args.n_eval or n_eval_default
    seed = args.seed + run_idx
    d3rlpy.seed(seed)
    np.random.seed(seed)

    env = make_env(seed=seed)
    eval_env = make_env(seed=seed + 10_000)
    try:
        d3rlpy.envs.seed_env(env, seed)
    except Exception as exc:  # 老版本 d3rlpy 没有这个 helper
        print(f"[warn] d3rlpy.envs.seed_env 不可用：{exc}")

    tu = args.pretrain_target_update if pretrain else args.online_target_update
    config = d3rlpy.algos.DoubleDQNConfig(
        batch_size=args.batch_size,
        learning_rate=args.lr,
        gamma=args.gamma,
        target_update_interval=tu,
    )
    algo = config.create(device=args.device)
    explorer = d3rlpy.algos.ConstantEpsilonGreedy(args.eps)
    buffer = d3rlpy.dataset.ReplayBuffer(
        buffer=d3rlpy.dataset.FIFOBuffer(limit=args.buffer_size), env=env
    )

    mdp = None
    if ds is not None and (prefill or pretrain):
        # 注意：buffer.append_episode 要的是 d3rlpy 的 Episode 对象，
        # 不是 loro.data.Episode，所以统一从 MDPDataset 里取。
        mdp = to_mdp_dataset(ds)
    if mdp is not None and prefill:
        # 只有 mix / loro-keep 才把 D 灌进在线 buffer；loro（=论文算法 1 附录 A.1.2 的"丢弃"）不灌
        for ep in mdp.episodes:
            buffer.append_episode(ep)
    if os.environ.get("DBG"):
        print(f"[dbg] variant={args.variant} prefill={prefill} pretrain={pretrain} "
              f"buffer={buffer.transition_count}", flush=True)

    if pretrain:
        algo.fit(mdp, n_steps=args.pretrain_steps,
                 n_steps_per_epoch=min(200, args.pretrain_steps))
    if os.environ.get("DBG"):
        print(f"[dbg] after fit buffer={buffer.transition_count}", flush=True)

    curve = []
    t0 = time.time()
    exp_name = f"cartpole_{args.variant}_{args.label}_{run_idx}"
    for step in range(n_eval):
        kwargs = {"experiment_name": exp_name} if args.log else {}
        algo.fit_online(env=env, buffer=buffer, explorer=explorer,
                        n_steps=H, **kwargs)
        curve.append(evaluate(algo, eval_env, seed=seed * 1000 + step))

    env.close()
    eval_env.close()
    return {
        "variant": args.variant,
        "label": args.label,
        "run": run_idx,
        "seed": seed,
        "n_eval": n_eval,
        "pretrain": pretrain,
        "prefill": prefill,
        "pretrain_steps": args.pretrain_steps if pretrain else 0,
        "target_update_interval": tu,
        "curve": curve,
        "summary": summarize(curve),
        "wall_seconds": round(time.time() - t0, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(VARIANTS), default="loro")
    ap.add_argument("--data", default=None, help=".npz 或作者的 MDPDataset .pkl")
    ap.add_argument("--n-episodes", type=int, default=10, help="用 D 的前几个回合（论文 τ=10）")
    ap.add_argument("--pretrain-steps", type=int, default=3000)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42069)
    ap.add_argument("--label", default="llm")
    ap.add_argument("--device", default="cuda:0", help="cuda:0 或 cpu")
    ap.add_argument("--threads", type=int, default=0,
                    help="torch CPU 线程数；0=用默认（建议 CPU 上设 1~4，默认值经常慢 10 倍）")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--buffer-size", type=int, default=100_000)
    ap.add_argument("--eps", type=float, default=0.1)
    ap.add_argument("--pretrain-target-update", type=int, default=200,
                    help="原文预训练时把 target update 从 1000 改成了 200")
    ap.add_argument("--online-target-update", type=int, default=1000)
    ap.add_argument("--no-log", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--log", action="store_true",
                    help="写 d3rlpy_logs（每 200 步存一个模型，很占磁盘，默认关闭）")
    ap.add_argument("--n-eval", type=int, default=None, help="覆盖评估点数（调试用）")
    ap.add_argument("--out", default="results/run")
    args = ap.parse_args()

    pretrain, prefill, n_eval_default = VARIANTS[args.variant]
    n_eval = args.n_eval or n_eval_default
    if (pretrain or prefill) and not args.data:
        raise SystemExit(f"variant={args.variant} 需要 --data")

    ds = None
    if args.data:
        ds = load_any(args.data, n_episodes=args.n_episodes)
        rets = [e.total_return for e in ds.episodes]
        print(f"[data] {len(ds.episodes)} episodes, {len(ds)} transitions, "
              f"mean return {np.mean(rets):.1f} (n_eps={len(rets)})")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = []
    for i in range(args.runs):
        rec = train_once(args, ds, i)
        runs.append(rec)
        s = rec["summary"]
        print(f"[run {i}] seed={rec['seed']} cum={s['cum_reward']:.0f} "
              f"mean={s['mean_reward']:.1f} last20={s['last20_mean']:.1f} "
              f"({rec['wall_seconds']}s)")
        (out / f"run_{i}.json").write_text(
            json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")

    sums = np.array([r["summary"]["cum_reward"] for r in runs])
    agg = {
        "variant": args.variant,
        "label": args.label,
        "n_runs": len(runs),
        "cum_per_run": sums.tolist(),
        "cum_mean": float(sums.mean()),
        "cum_se": float(sums.std(ddof=1) / np.sqrt(len(sums))) if len(sums) > 1 else 0.0,
        "first10_mean": float(np.mean([r["summary"]["first10_mean"] for r in runs])),
        "n_eval": n_eval,
    }
    (out / "agg.json").write_text(json.dumps(agg, indent=2, ensure_ascii=False),
                                  encoding="utf-8")
    print(f"\n[{args.variant}] {n_eval} evals: "
          f"mean cum = {agg['cum_mean']:.0f} ± {agg['cum_se']:.0f}")
    print(f"  per-run: {[int(x) for x in sums]}")
    print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
