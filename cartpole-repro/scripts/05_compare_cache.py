"""第 5 步：把作者公开的曲线缓存复算一遍，核对论文数字。

用途：在你跑自己的实验之前，先确认"论文的 CartPole 数字到底是什么口径"。
这些 .pkl 里存的是 dict：{run_key: [每个评估点的回报, ...]}，纯 float 列表，
不需要 d3rlpy 就能读。
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loro.metrics import CACHE_MEANS, PAPER  # noqa: E402

# 缓存里的 key 前缀 -> 论文中的说法
LABELS = {
    "online": "On-policy（纯在线 DDQN）",
    "pretrain_7b_1000": "LORO 7B / 1k 步",
    "pretrain_7b_3000": "LORO 7B / 3k 步",
    "pretrain_32b_1000": "LORO 32B / 1k 步",
    "pretrain_32b_3000": "LORO 32B / 3k 步",
    "finetune_7b": "仅混合 7B",
    "finetune_32b": "仅混合 32B",
}


def load_cache(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--plot", action="store_true", help="画 7B 1k/3k 的 5 条原始曲线")
    ap.add_argument("--out", default="report")
    args = ap.parse_args()

    root = Path(args.artifacts)
    files = sorted(root.glob("cache_CartPole_*.pkl"))
    if not files:
        raise SystemExit(f"{root} 下没有 cache_CartPole_*.pkl，先跑 scripts/01_fetch_artifacts.py")

    for path in files:
        d = load_cache(path)
        groups: dict[str, list[float]] = {}
        for k, v in d.items():
            g = k.rsplit("_", 1)[0]
            groups.setdefault(g, []).append(float(np.sum(v)))
        print(f"\n=== {path.name}  ({len(d)} 条曲线) ===")
        print(f"{'key':<22}{'n':>3}{'mean':>10}{'se':>8}   per-run")
        for g, sums in groups.items():
            sums_arr = np.array(sums)
            se = sums_arr.std(ddof=1) / np.sqrt(len(sums_arr)) if len(sums_arr) > 1 else 0
            print(f"{g:<22}{len(sums):>3}{sums_arr.mean():>10.0f}{se:>8.0f}   "
                  f"{[int(x) for x in sums]}")

    print("\n=== 论文报告的 CartPole 数值 vs 缓存复算 ===")
    for key, paper in PAPER.items():
        cache = CACHE_MEANS.get(key)
        tail = f"  缓存 {cache:.0f}" if cache else ""
        print(f"{key:<24}{paper:>8}{tail}")
    print("\n注：LORO 的曲线只有 140 个评估点（少了前 10 个 LLM 回合），"
          "纯在线是 150 个；直接比 cum_reward 会额外差 7%，"
          "所以还要看 frac_of_max（sum / (n*200)）。")

    if args.plot:
        _plot(Path(args.artifacts) / "cache_CartPole_Neps_10.pkl", Path(args.out))
    return 0


def _plot(cache_path: Path, out: Path) -> None:
    """把 7B 在 1k / 3k 预训练步下的 5 条曲线画出来——一眼看到"崩"是什么样子。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n[skip] 没装 matplotlib")
        return

    d = load_cache(cache_path)
    out.mkdir(parents=True, exist_ok=True)
    groups = [("pretrain_7b_1000", "7B / 1000 pretrain steps"),
              ("pretrain_7b_3000", "7B / 3000 pretrain steps"),
              ("online", "online DDQN (baseline)")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True, dpi=150)
    for ax, (g, title) in zip(axes, groups):
        runs = [np.asarray(v, dtype=float) for k, v in d.items()
                if k.rsplit("_", 1)[0] == g]
        for i, c in enumerate(runs):
            ax.plot(np.arange(len(c)) * 200, c, lw=0.8, alpha=0.6, label=f"run {i}")
        m = np.mean(runs, axis=0)
        ax.plot(np.arange(len(m)) * 200, m, lw=2, color="black", label="mean")
        ax.axhline(200, ls="--", lw=0.8, color="grey")
        ax.set_title(f"{title}\nsum={sum(float(c.sum()) for c in runs)/len(runs):.0f}",
                     fontsize=9)
        ax.set_xlabel("env steps")
        ax.set_ylim(0, 210)
    axes[0].set_ylabel("greedy episode return")
    axes[0].legend(fontsize=6, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "author_cache_7b.png")
    print(f"-> {out / 'author_cache_7b.png'}")


if __name__ == "__main__":
    raise SystemExit(main())
