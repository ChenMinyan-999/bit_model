"""第 4 步：汇总 results/ 下的所有 run，画图并与论文数字对照。

用法：
  python scripts/04_report.py --results results --out report
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loro.metrics import PAPER, aggregate  # noqa: E402


def load_runs(results: Path):
    """results/<tag>/run_*.json -> {tag: [run dict, ...]}"""
    out = {}
    for d in sorted(p for p in results.iterdir() if p.is_dir()):
        runs = [json.loads(p.read_text(encoding="utf-8"))
                for p in sorted(d.glob("run_*.json"))]
        if runs:
            out[d.name] = runs
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="report")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--min-evals", type=int, default=20,
                    help="少于这么多评估点的 run 视为 smoke，默认不进表和主图")
    args = ap.parse_args()

    runs_by_tag = load_runs(Path(args.results))
    runs_by_tag = {k: [r for r in v if r["n_eval"] >= args.min_evals]
                   for k, v in runs_by_tag.items()}
    runs_by_tag = {k: v for k, v in runs_by_tag.items() if v}
    if not runs_by_tag:
        raise SystemExit("results/ 下没有 run_*.json，先跑 03_train.py")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"{'tag':<24}{'runs':>5}{'evals':>7}{'cum_mean':>10}{'±se':>7}"
          f"{'frac_max':>10}{'last20':>9}")
    rows = []
    for tag, runs in runs_by_tag.items():
        agg = aggregate([r["curve"] for r in runs])
        n_eval = runs[0]["n_eval"]
        frac = agg["cum_mean"] / (n_eval * 200)
        last20 = float(np.mean([r["summary"]["last20_mean"] for r in runs]))
        print(f"{tag:<24}{agg['n_runs']:>5}{n_eval:>7}{agg['cum_mean']:>10.0f}"
              f"{agg['cum_se']:>7.0f}{frac:>9.1%}{last20:>9.1f}")
        rows.append({"tag": tag, **agg, "n_eval": n_eval, "frac_max": frac,
                     "last20_mean": last20})

    (out / "summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False),
                                      encoding="utf-8")
    print(f"\n-> {out / 'summary.json'}")

    print("\n论文参考值（cum_reward 口径）：")
    for k, v in PAPER.items():
        print(f"  {k:<24}{v:>8}")

    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("\n[skip] 没装 matplotlib，跳过画图")
            return 0

        fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
        for tag, runs in runs_by_tag.items():
            curves = np.array([r["curve"] for r in runs], dtype=float)
            m = curves.mean(axis=0)
            se = curves.std(axis=0, ddof=1) / np.sqrt(len(curves)) if len(curves) > 1 else None
            x = np.arange(len(m)) * 200  # 横轴 = 环境步数
            ax.plot(x, m, label=f"{tag} (n={len(runs)})")
            if se is not None:
                ax.fill_between(x, m - se, m + se, alpha=0.2)
        ax.axhline(200, ls="--", lw=1, color="grey")
        ax.set_xlabel("environment steps")
        ax.set_ylabel("greedy episode return (cap 200)")
        ax.set_ylim(0, 210)
        ax.legend(fontsize=7)
        ax.set_title("CartPole: LORO reproduction")
        fig.tight_layout()
        fig.savefig(out / "curves.png")
        print(f"-> {out / 'curves.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
