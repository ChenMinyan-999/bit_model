"""轨迹数据容器与磁盘格式。

统一用 .npz 存（不依赖 d3rlpy 版本），同时支持读入作者仓库里 pickled 的 MDPDataset。
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Episode:
    observations: np.ndarray  # (T, obs_dim)
    actions: np.ndarray       # (T,)
    rewards: np.ndarray       # (T,)
    terminals: np.ndarray = field(default=None)  # (T,) 0/1

    def __post_init__(self):
        self.observations = np.asarray(self.observations, dtype=np.float32)
        self.actions = np.asarray(self.actions, dtype=np.int64)
        self.rewards = np.asarray(self.rewards, dtype=np.float32)
        if self.terminals is None:
            self.terminals = np.zeros(len(self.actions), dtype=np.float32)
            self.terminals[-1] = 1.0
        self.terminals = np.asarray(self.terminals, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.actions)

    @property
    def total_return(self) -> float:
        return float(self.rewards.sum())


@dataclass
class TransitionDataset:
    episodes: list[Episode]

    def __len__(self) -> int:
        return sum(len(e) for e in self.episodes)

    def concat(self):
        obs = np.concatenate([e.observations for e in self.episodes])
        act = np.concatenate([e.actions for e in self.episodes])
        rew = np.concatenate([e.rewards for e in self.episodes])
        ter = np.concatenate([e.terminals for e in self.episodes])
        return obs, act, rew, ter

    def first(self, n: int) -> "TransitionDataset":
        return TransitionDataset(self.episodes[:n])


def save_npz(path: str | Path, ds: TransitionDataset) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    obs, act, rew, ter = ds.concat()
    lengths = np.array([len(e) for e in ds.episodes], dtype=np.int64)
    np.savez_compressed(path, observations=obs, actions=act, rewards=rew,
                        terminals=ter, lengths=lengths)


def load_npz(path: str | Path) -> TransitionDataset:
    z = np.load(path)
    obs, act, rew, ter = z["observations"], z["actions"], z["rewards"], z["terminals"]
    lengths = z["lengths"]
    eps, i = [], 0
    for L in lengths:
        L = int(L)
        eps.append(Episode(obs[i:i + L], act[i:i + L], rew[i:i + L], ter[i:i + L]))
        i += L
    return TransitionDataset(eps)


def load_any(path: str | Path, n_episodes: int | None = None) -> TransitionDataset:
    """支持 .npz，以及作者仓库 data/*.pkl 里的 d3rlpy MDPDataset。"""
    path = Path(path)
    if path.suffix == ".npz":
        ds = load_npz(path)
    else:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        ds = from_mdp_dataset(obj)
    return ds.first(n_episodes) if n_episodes else ds


def from_mdp_dataset(mdp) -> TransitionDataset:
    """d3rlpy MDPDataset -> TransitionDataset。

    注意：作者仓库里的 pkl 是用旧版 d3rlpy 序列化的，Episode 上没有 `transitions`
    属性（只有 observations / actions / rewards / terminated）。所以这里只用这三个
    数组，并把"最后一步"当作终止——这正是作者 `get_new_dataset` 的做法：
        terminals = [0] * (len(rewards) - 1) + [1]
    这样既兼容旧 pkl，也不改变论文的语义。
    """
    eps = []
    for ep in mdp.episodes:
        obs = np.asarray(ep.observations, dtype=np.float32)
        act = np.asarray(ep.actions, dtype=np.int64).reshape(-1)
        rew = np.asarray(ep.rewards, dtype=np.float32).reshape(-1)
        n = len(rew)
        ter = np.zeros(n, dtype=np.float32)
        if n:
            ter[-1] = 1.0
        eps.append(Episode(obs, act, rew, ter))
    return TransitionDataset(eps)


def to_mdp_dataset(ds: TransitionDataset):
    """TransitionDataset -> d3rlpy.dataset.MDPDataset（延迟导入 d3rlpy）。"""
    import d3rlpy

    obs, act, rew, ter = ds.concat()
    if act.ndim == 2 and act.shape[1] == 1:
        act = act.reshape(-1)
    return d3rlpy.dataset.MDPDataset(
        observations=obs, actions=act, rewards=rew, terminals=ter.astype(bool)
    )
