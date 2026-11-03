"""曲线指标与论文参考值。"""

from __future__ import annotations

import numpy as np

H = 200
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")

# 论文（arXiv:2505.10861v1）CartPole 数值，口径 = 全部评估回合回报之和
PAPER = {
    "random": 3079,
    "qwen7b_policy": 5140,
    "qwen32b_policy": 6840,
    "online": 24411,
    "loro_1k_7b": 6705,
    "loro_3k_7b": 27078,
    "loro_32b_1k": 25477,
    "mix_1k_7b": 23144,
    "pretrain_random_data": 25351,
    "pretrain_online_data": 22881,
}

# 作者仓库 data/cache_CartPole_Neps_10.pkl 里 5 条曲线的均值（我们用脚本复算过）
CACHE_MEANS = {
    "online": 24411.0,
    "pretrain_7b_1000": 6362.8,
    "pretrain_7b_3000": 26736.0,
    "pretrain_32b_1000": 25021.6,
    "pretrain_32b_3000": 18066.8,
    "finetune_7b(mix)": 24737.0,
    "finetune_32b(mix)": 24847.4,
}


def summarize(curve, cap: int = H) -> dict:
    """把一条评估曲线（每点 = 一个贪心回合的回报）压成若干指标。"""
    c = np.asarray(curve, dtype=float)
    n = len(c)
    return {
        "n_points": n,
        "cum_reward": float(c.sum()),          # 论文口径
        "mean_reward": float(c.mean()),
        "max_possible": float(n * cap),
        "frac_of_max": float(c.mean() / cap),  # 归一化，跨不同 n_points 可比
        "last20_mean": float(c[-20:].mean()) if n >= 20 else float(c.mean()),
        "first10_mean": float(c[:10].mean()) if n >= 10 else float(c.mean()),
        "auc_norm": float(_trapz(c / cap) / max(n - 1, 1)) if n > 1 else float(c.mean() / cap),
    }


def aggregate(curve_list) -> dict:
    """多条独立 run 的均值 ± 标准误（论文就是报 SE）。"""
    sums = np.array([float(np.sum(c)) for c in curve_list])
    return {
        "cum_mean": float(sums.mean()),
        "cum_se": float(sums.std(ddof=1) / np.sqrt(len(sums))) if len(sums) > 1 else 0.0,
        "cum_per_run": [float(x) for x in sums],
        "n_runs": len(sums),
    }


def episodes_to_threshold(curve, fraction: float = 0.9, cap: int = H) -> int | None:
    """首次（此后一直）达到 fraction*cap 的评估点下标。"""
    c = np.asarray(curve, dtype=float)
    target = fraction * cap
    ok = c >= target
    for i in range(len(c)):
        if ok[i:].all():
            return i
    return None
