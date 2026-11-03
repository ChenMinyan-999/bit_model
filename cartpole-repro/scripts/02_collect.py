"""第 2 步：用 LLM 采集 τ=10 个 CartPole 回合，产出离线数据集。

两种后端：
  --backend mock   用 PD 控制器假装 π_LLM（纯 CPU，几秒钟）。用来先验证整条管线。
  --backend vllm   真的调 Qwen2.5-7B-Instruct（需要 GPU + vllm；国产卡见 --dtype）。
  --backend api    调 OpenAI 兼容接口（DashScope / vLLM server / 任何兼容网关），最省事。
  --backend random 均匀随机动作，用来复现论文图 4 的"随机数据预训练"对照组。

实现细节刻意贴近原文（见手册 §3/§7）：
  * 每一步都是独立的单轮对话：system + 当前观测 + "Think step by step."，
    不带历史（原文 TranslationAgent.act 每步重建 messages），所以可以并行。
  * 解码：temperature=0.9, top_p=0.6, top_k 关闭, max_new_tokens=2000。
  * 动作：取回复里最后一个数字，1=左(0)、2=右(1)，失败回退到"左"。
  * --obs-shift next 复刻原文的 off-by-one（存 s_{t+1} 与 (a_t, r_t) 配对）；
    current 才是标准 MDP 转移。默认 next，为了和论文的数字可比。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loro import prompts  # noqa: E402
from loro.data import Episode, TransitionDataset, save_npz  # noqa: E402
from loro.envs import make_env  # noqa: E402


def pd_controller(obs, rng=None, quality: float = 0.6) -> int:
    """一个"能撑一会儿但远不到 200"的次优策略，替代 LLM 做 dry-run。

    quality=1.0 时是满血 PD 控制器（~200 步）；0.6 时平均 ~110 步，
    正好落在"覆盖够、质量不够"的区间——也就是论文里 LLM 的典型水位。
    """
    if rng is not None and rng.random() > quality:
        return int(rng.integers(2))
    _, _, theta, theta_dot = obs
    return 1 if (theta + 0.5 * theta_dot) > 0 else 0


def run_rollouts(act_batch, n_episodes: int, parallel: int, seed: int,
                 obs_shift: str, cap: int = 200):
    """并行跑 n_episodes 个回合。act_batch: (list[obs]) -> (list[action], n_parse_fail)。"""
    envs = [make_env(seed=seed + i) for i in range(parallel)]
    obs = [e.reset(seed=seed + i)[0] for i, e in enumerate(envs)]
    rec = [[] for _ in range(parallel)]
    episodes, parse_fail = [], 0
    t0 = time.time()

    while len(episodes) < n_episodes:
        idx = [i for i in range(parallel) if obs[i] is not None]
        prompts_in = [prompts.user_message(obs[i]) for i in idx]
        actions, fails = act_batch(prompts_in)
        parse_fail += fails
        for i, a in zip(idx, actions):
            prev = obs[i]
            nobs, r, term, trunc, _ = envs[i].step(int(a))
            done = bool(term or trunc)
            stored = nobs if obs_shift == "next" else prev
            rec[i].append((np.asarray(stored, dtype=np.float32), int(a), float(r), done))
            if done:
                arr = rec[i]
                episodes.append(Episode(
                    observations=np.stack([x[0] for x in arr]),
                    actions=np.array([x[1] for x in arr]),
                    rewards=np.array([x[2] for x in arr]),
                    terminals=np.array([1.0 if x[3] else 0.0 for x in arr]),
                ))
                rec[i] = []
                obs[i] = envs[i].reset()[0]
            else:
                obs[i] = nobs
    dt = time.time() - t0
    episodes = episodes[:n_episodes]
    for e in envs:
        e.close()
    return episodes, parse_fail, dt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["mock", "vllm", "api", "random"], default="mock")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--episodes", type=int, default=10, help="原文 τ=10")
    ap.add_argument("--parallel", type=int, default=8, help="同时跑几个环境（vLLM/API 自动并发）")
    ap.add_argument("--max-tokens", type=int, default=2000)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.6)
    ap.add_argument("--dtype", default="bfloat16",
                    help="vllm 后端的数据类型；摩尔线程 MUSA 卡只有 fp16，传 float16")
    ap.add_argument("--api-base", default="https://dashscope.aliyuncs.com/compatible-mode/v1",
                    help="api 后端的 OpenAI 兼容基址（结尾不要带 /chat/completions）")
    ap.add_argument("--api-key-env", default="DASHSCOPE_API_KEY")
    ap.add_argument("--seed", type=int, default=42069, help="原文 seed=42069")
    ap.add_argument("--obs-shift", choices=["next", "current"], default="next")
    ap.add_argument("--mock-quality", type=float, default=0.6,
                    help="仅 mock 后端：走 PD 控制的概率，1.0=满血控制器")
    ap.add_argument("--out", default="data/cartpole_llm_neps10.npz")
    ap.add_argument("--meta-out", default=None)
    args = ap.parse_args()

    if args.backend == "mock":
        rng = np.random.default_rng(args.seed)
        episodes, parse_fail, dt = _run_mock(
            args,
            lambda obs_list: ([pd_controller(o, rng, args.mock_quality) for o in obs_list], 0),
        )
    elif args.backend == "random":
        # 均匀随机动作：论文图 4 的"用随机策略数据预训练"那一组
        rng = np.random.default_rng(args.seed)
        episodes, parse_fail, dt = _run_mock(
            args, lambda obs_list: ([int(rng.integers(2)) for _ in obs_list], 0)
        )
    elif args.backend == "api":
        # OpenAI 兼容接口：DashScope / vLLM server / 任何网关。
        # 适合"卡是国产的、或者根本不想装 vLLM"的场景。
        key = os.environ.get(args.api_key_env, "")
        if not key:
            raise SystemExit(f"环境变量 {args.api_key_env} 为空；"
                             f"export {args.api_key_env}=sk-xxx 后重试")
        url = args.api_base.rstrip("/") + "/chat/completions"

        def one(text: str) -> str:
            payload = {
                "model": args.model,
                "messages": [
                    {"role": "system", "content": prompts.SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_tokens": args.max_tokens,
            }
            body = json.dumps(payload).encode("utf-8")
            for attempt in range(4):
                try:
                    req = urllib.request.Request(url, data=body, headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {key}",
                    })
                    with urllib.request.urlopen(req, timeout=120) as r:
                        data = json.loads(r.read())
                    return data["choices"][0]["message"]["content"]
                except Exception as exc:  # 429 / 超时 / 偶发 5xx
                    if attempt == 3:
                        print(f"[api] 放弃: {exc}")
                        return ""
                    time.sleep(2 * (attempt + 1))
            return ""

        def act_batch(texts):
            with ThreadPoolExecutor(max_workers=max(1, len(texts))) as ex:
                outs = list(ex.map(one, texts))
            acts, fails = [], 0
            for t in outs:
                if (not any(c.isdigit() for c in t)
                        and "left" not in t.lower()
                        and "right" not in t.lower()):
                    fails += 1
                acts.append(prompts.parse_action(t))
            return acts, fails

        episodes, parse_fail, dt = run_rollouts(
            act_batch, args.episodes, args.parallel, args.seed, args.obs_shift)
    else:
        from vllm import LLM, SamplingParams

        print(f"[vllm] loading {args.model} (dtype={args.dtype}) ...")
        llm = LLM(model=args.model, dtype=args.dtype, max_model_len=4096,
                  gpu_memory_utilization=0.90, trust_remote_code=True)
        try:
            tokenizer = llm.get_tokenizer()
        except AttributeError:  # 老版本 vLLM（如 MUSA 0.4.2）没有这个方法
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        sp = SamplingParams(temperature=args.temperature, top_p=args.top_p,
                            top_k=-1, max_tokens=args.max_tokens)

        def act_batch(texts):
            prompts_list = [
                tokenizer.apply_chat_template(
                    [{"role": "system", "content": prompts.SYSTEM_PROMPT},
                     {"role": "user", "content": t}],
                    tokenize=False, add_generation_prompt=True,
                )
                for t in texts
            ]
            outs = llm.generate(prompts_list, sp)
            acts, fails = [], 0
            for o in outs:
                text = o.outputs[0].text
                if (not any(c.isdigit() for c in text)
                        and "left" not in text.lower()
                        and "right" not in text.lower()):
                    fails += 1
                acts.append(prompts.parse_action(text))
            return acts, fails

        episodes, parse_fail, dt = run_rollouts(
            act_batch, args.episodes, args.parallel, args.seed, args.obs_shift)

    ds = TransitionDataset(episodes)
    save_npz(args.out, ds)
    rets = [e.total_return for e in episodes]
    meta = {
        "backend": args.backend,
        "model": {"vllm": args.model, "api": args.model, "mock": "pd-controller",
                  "random": "uniform-random"}[args.backend],
        "episodes": len(episodes),
        "transitions": len(ds),
        "obs_shift": args.obs_shift,
        "seed": args.seed,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "episode_returns": rets,
        "mean_return": float(np.mean(rets)),
        "max_return": float(np.max(rets)),
        "parse_fail": parse_fail,
        "wall_seconds": round(dt, 1),
    }
    meta_path = Path(args.meta_out or str(Path(args.out).with_suffix(".meta.json")))
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[done] {len(episodes)} episodes, {len(ds)} transitions, "
          f"mean return {meta['mean_return']:.1f}, max {meta['max_return']:.0f}, "
          f"parse_fail {parse_fail}, {dt:.0f}s")
    print(f"       -> {args.out}\n       -> {meta_path}")
    return 0


def _run_mock(args, act_batch_real):
    """mock 版本：与 run_rollouts 同构，但直接把 obs 交给 PD 控制器。"""
    envs = [make_env(seed=args.seed + i) for i in range(args.parallel)]
    obs = [e.reset(seed=args.seed + i)[0] for i, e in enumerate(envs)]
    rec = [[] for _ in range(args.parallel)]
    episodes, t0 = [], time.time()
    while len(episodes) < args.episodes:
        idx = [i for i in range(args.parallel) if obs[i] is not None]
        actions, _ = act_batch_real([obs[i] for i in idx])
        for i, a in zip(idx, actions):
            prev = obs[i]
            nobs, r, term, trunc, _ = envs[i].step(int(a))
            done = bool(term or trunc)
            stored = nobs if args.obs_shift == "next" else prev
            rec[i].append((np.asarray(stored, dtype=np.float32), int(a), float(r), done))
            if done:
                arr = rec[i]
                episodes.append(Episode(
                    observations=np.stack([x[0] for x in arr]),
                    actions=np.array([x[1] for x in arr]),
                    rewards=np.array([x[2] for x in arr]),
                    terminals=np.array([1.0 if x[3] else 0.0 for x in arr]),
                ))
                rec[i] = []
                obs[i] = envs[i].reset()[0]
            else:
                obs[i] = nobs
    dt = time.time() - t0
    for e in envs:
        e.close()
    return episodes[: args.episodes], 0, dt


if __name__ == "__main__":
    raise SystemExit(main())
