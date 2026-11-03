"""第 1 步（可选但强烈建议）：把作者公开的 CartPole 数据/曲线拉到本地。

为什么先做这一步：
  * 作者仓库里有 30 回合的 Qwen-7B/32B LLM 轨迹（data/*.pkl）和 5 条曲线缓存（data/cache_*.pkl）；
  * 用它们可以在"零 GPU"的情况下把 RL 半边跑通、把 24411 / 26736 这两个数字对上；
  * 等这套管线验证过了，再去花 GPU 时间自己采 LLM 数据。
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import urllib.request
from pathlib import Path

REPO = "duongnhatthang/LORO-legacy"
API = f"https://api.github.com/repos/{REPO}"

CARTPOLE_FILES = [
    "data/CartPole_Qwen2.5-7B-Instruct_Neps_30_20250410211529.pkl",
    "data/CartPole_Qwen2.5-32B-Instruct_Neps_30_20250412032827.pkl",
    "data/cache_CartPole_Neps_10.pkl",
    "data/cache_CartPole_Neps_20.pkl",
    "data/cache_CartPole_Neps_30.pkl",
    "data/cache_CartPole_on_policy_pretrain_exp.pkl",
    "data/cache_CartPole_on_policy_pretrain_exp_rand.pkl",
]

MODEL_FILES = [
    "models/CartPole_ddqn_pretrain_7b_1000_steps.pkl",
    "models/CartPole_ddqn_pretrain_7b_3000_steps.pkl",
    "models/CartPole_ddqn_pretrain_32b_1000_steps.pkl",
    "models/CartPole_ddqn_pretrain_32b_3000_steps.pkl",
]


def _get(url: str) -> dict:
    last = None
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "loro-repro"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read())
        except Exception as exc:  # 网络抖动 / IncompleteRead，重试
            last = exc
            print(f"  retry {attempt + 1}/5: {exc}")
    raise RuntimeError(f"下载失败：{url}") from last


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--with-models", action="store_true",
                    help="同时下载作者的预训练模型（1MB×4，依赖 d3rlpy 版本，可能反序列化失败）")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tree = _get(f"{API}/git/trees/main?recursive=1")["tree"]
    sha = {t["path"]: t["sha"] for t in tree if t["type"] == "blob"}

    wanted = CARTPOLE_FILES + (MODEL_FILES if args.with_models else [])
    for path in wanted:
        if path not in sha:
            print(f"[skip] {path} 不在仓库里（可能改名）")
            continue
        dest = out / Path(path).name
        if dest.exists() and dest.stat().st_size > 0:
            print(f"[have] {dest}  ({dest.stat().st_size/1024:.0f} KB)")
            continue
        blob = _get(f"{API}/git/blobs/{sha[path]}")
        raw = base64.b64decode(blob["content"])
        dest.write_bytes(raw)
        print(f"[get ] {dest}  ({len(raw)/1024:.0f} KB)")

    print(f"\n完成：{out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
