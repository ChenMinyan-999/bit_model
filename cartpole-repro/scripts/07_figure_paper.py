"""把服务器跑出来的 results/ 画成 2×2 对比图（论文图 2/3/附录 A.3/数据质量）。

    python scripts/07_figure_paper.py --results server_results/results --out report/fig_repro.png
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(root: Path, tag: str):
    runs = [json.loads(Path(p).read_text(encoding="utf-8"))
            for p in sorted(glob.glob(str(root / tag / "run_*.json")))]
    return [np.asarray(r["curve"], dtype=float) for r in runs]


def panel(ax, root: Path, specs, title, x0=0):
    for tag, label, color in specs:
        curves = load(root, tag)
        if not curves:
            continue
        n = min(len(c) for c in curves)
        m = np.array([c[:n] for c in curves])
        mean = m.mean(0)
        se = m.std(0, ddof=1) / np.sqrt(len(m)) if len(m) > 1 else np.zeros(n)
        x = x0 + np.arange(n) * 200
        ax.plot(x, mean, lw=1.6, color=color, label=f"{label}  Σ={mean.sum():.0f}")
        ax.fill_between(x, mean - se, mean + se, color=color, alpha=0.18)
    ax.axhline(200, ls="--", lw=0.8, color="grey")
    ax.set_title(title, fontsize=9)
    ax.set_ylim(0, 210)
    ax.set_xlabel("environment steps", fontsize=7)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, loc="lower right")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="server_results/results")
    ap.add_argument("--out", default="report/fig_repro.png")
    args = ap.parse_args()
    root = Path(args.results)

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.2), dpi=150)
    panel(axes[0, 0], root,
          [("online", "online DDQN", "tab:blue"),
           ("loro_3k", "LORO (author 7B data, 3k)", "tab:orange")],
          "(a) LORO vs pure online  [paper Fig.2]")
    panel(axes[0, 1], root,
          [("mix", "mix only, no pretrain", "tab:green"),
           ("loro_3k", "LORO (pretrain, drop D)", "tab:orange")],
          "(b) pretrain vs mix  [paper Fig.3]")
    panel(axes[1, 0], root,
          [("loro_1k", "LORO 1k pretrain steps", "tab:purple"),
           ("loro_3k", "LORO 3k pretrain steps", "tab:orange")],
          "(c) pretrain steps  [paper App.A.3]")
    panel(axes[1, 1], root,
          [("loro_3k", "author Qwen-7B data", "tab:orange"),
           ("loro_self_3k", "our Qwen-7B data", "tab:red")],
          "(d) data quality  [paper Fig.4]")

    for ax in (axes[0, 0], axes[1, 0], axes[1, 1]):
        ax.set_ylabel("greedy return (cap 200)", fontsize=7)
    fig.suptitle("CartPole reproduction on RTX PRO 6000  (5 seeds, mean ± SE)", fontsize=10)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print("->", args.out)


if __name__ == "__main__":
    main()
