"""CartPole 环境构造与评估工具。

关键点（决定数字能否对上）：
  * 回合上限必须是 200，而不是 CartPole-v1 默认的 500；
  * 原论文用的是 CartPole-v0（gymnasium 0.29.1 里 v0 的 spec.max_episode_steps 正好是 200）；
  * 评估是"每个训练块之后跑 1 个贪心回合"，不是 10 回合平均。
"""

from __future__ import annotations

import warnings

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import TimeLimit

H = 200  # 每回合步数上限；论文原文 max_episode_len = 200


def make_env(seed: int | None = None, cap: int = H) -> gym.Env:
    """构造带 200 步上限的 CartPole；v0 不存在时自动退回 v1 + TimeLimit(200)。"""
    with warnings.catch_warnings():
        # CartPole-v0 在 0.29 里被标记为 deprecated，但它就是论文用的那个（上限 200）
        warnings.simplefilter("ignore")
        try:
            env = gym.make("CartPole-v0")
        except gym.error.Error:
            env = gym.make("CartPole-v1")
    spec_cap = getattr(getattr(env, "spec", None), "max_episode_steps", None)
    if spec_cap != cap:
        env = TimeLimit(env, max_episode_steps=cap)
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
    return env


def effective_cap(env: gym.Env) -> int:
    """读取实际的回合步数上限（用于自检）。"""
    cur = env
    while True:
        if isinstance(cur, TimeLimit):
            return int(cur._max_episode_steps)
        inner = getattr(cur, "env", None)
        if inner is None:
            return int(getattr(getattr(cur, "spec", None), "max_episode_steps", -1))
        cur = inner


def greedy_episode(env: gym.Env, act_fn, seed: int | None = None):
    """跑一个完整回合，返回 (总回报, 步数)。act_fn: obs -> int。"""
    obs, _ = env.reset(seed=seed) if seed is not None else env.reset()
    total, steps, done = 0.0, 0, False
    while not done:
        action = int(act_fn(obs))
        obs, reward, terminated, truncated, _ = env.step(action)
        total += float(reward)
        steps += 1
        done = bool(terminated or truncated)
    return total, steps


def random_baseline(n_episodes: int = 150, seed: int = 0) -> dict:
    """随机策略基线：论文报告值 3079（150 回合）。本机实测约 3300±80。"""
    env = make_env(seed=seed)
    rng = np.random.default_rng(seed)
    returns = []
    for _ in range(n_episodes):
        obs, _ = env.reset(seed=int(rng.integers(2**31 - 1)))
        total, done = 0.0, False
        while not done:
            obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
            total += float(reward)
            done = bool(terminated or truncated)
        returns.append(total)
    env.close()
    return {
        "mean_return": float(np.mean(returns)),
        "sum_return": float(np.sum(returns)),
        "mean_length": float(np.mean(returns)),  # CartPole 每步奖励 +1，回报=步数
        "n_episodes": n_episodes,
    }
