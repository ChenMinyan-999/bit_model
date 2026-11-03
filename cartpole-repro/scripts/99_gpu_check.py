"""第 0.5 步（可选）：上 GPU 前先确认这张卡 + 这套 torch 真的能算。

新卡（尤其是 Blackwell / sm_120）最常见的坑不是显存不够，而是装到的 torch
没有对应架构的 kernel，报 "no kernel image is available for execution on the device"。
这个脚本 30 秒给出结论。

    python scripts/99_gpu_check.py
    python scripts/99_gpu_check.py --vllm Qwen/Qwen2.5-7B-Instruct   # 顺带试一下真模型
"""

from __future__ import annotations

import argparse
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vllm", default=None, help="可选：给个模型路径，做一次 16 token 的生成")
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    try:
        import torch
    except ImportError:
        print("没有 torch：先 pip install torch（RL 环境用 CPU 轮子就够）")
        return 1

    print(f"python      : {sys.version.split()[0]}")
    print(f"torch       : {torch.__version__}")
    print(f"cuda build  : {torch.version.cuda}")
    if not torch.cuda.is_available():
        print("cuda avail  : False（CPU-only 也能跑 RL，但采 LLM 数据需要 GPU）")
        return 0

    n = torch.cuda.device_count()
    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        cap = torch.cuda.get_device_capability(i)
        print(f"gpu[{i}]      : {props.name}  {props.total_memory / 1024**3:.0f} GB  sm_{cap[0]}{cap[1]}")

    # 真正的判据：能不能在这张卡上跑一个 kernel
    try:
        a = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(10):
            c = a @ a
        torch.cuda.synchronize()
        dt = (time.time() - t0) / 10
        print(f"matmul 2048² : OK  {dt * 1e3:.1f} ms/次  ({2 * 2048**3 / dt / 1e12:.1f} TFLOP/s fp16)")
    except Exception as exc:
        print(f"matmul 失败  : {exc}")
        print("\n→ 这是典型的架构不匹配。Blackwell(sm_120) 需要 CUDA 12.8+ 的 torch，")
        print("  pip install torch --index-url https://download.pytorch.org/whl/cu128")
        return 1

    if args.vllm:
        print(f"\n[vllm] 试加载 {args.vllm} ...")
        try:
            from vllm import LLM, SamplingParams
            import vllm
            print(f"vllm        : {vllm.__version__}")
            llm = LLM(model=args.vllm, dtype=args.dtype, max_model_len=2048,
                      gpu_memory_utilization=0.85, trust_remote_code=True)
            out = llm.generate(["The capital of France is"], SamplingParams(max_tokens=16))
            print("generate ok :", repr(out[0].outputs[0].text))
        except Exception as exc:
            print(f"vllm 失败   : {type(exc).__name__}: {exc}")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
