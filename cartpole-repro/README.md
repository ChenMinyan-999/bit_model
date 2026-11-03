# CartPole 复现包（arXiv:2505.10861v1）

配套手册：[`../CartPole_复现执行手册.md`](../CartPole_复现执行手册.md)。这里只有能直接跑的代码。

## 这套东西是什么

论文 LORO 在 CartPole 上的实验，拆成可独立验证的 5 步：

```
00_sanity    环境自检（纯 CPU，30 秒）——先确认回合上限=200、随机基线 3100~3500
01_fetch     拉作者公开的 LLM 轨迹与曲线缓存（可选，但强烈建议）
02_collect   用 Qwen2.5-7B 采 10 个回合；或用 mock 策略假装 LLM
03_train     DDQN 离线预训练 + 在线微调，输出学习曲线
04_report    汇总/画图/和论文数字对照
05_compare   把作者的缓存曲线复算一遍，核对论文口径
```

## 验证状态（2026-11-02 在 RTX PRO 6000 上全流程跑通）

| 文件 | 状态 |
|---|---|
| `loro/envs.py`、`loro/prompts.py`、`loro/data.py`、`loro/metrics.py` | 已跑通（本机 py3.13 + 服务器 py3.10） |
| `scripts/00_sanity.py` | 已跑通，随机基线 3300±111（论文 3079） |
| `scripts/01_fetch_artifacts.py` | 已跑通，产物在 `artifacts/` |
| `scripts/02_collect.py --backend mock` / `random` | 已跑通 |
| `scripts/02_collect.py --backend vllm` | **已在服务器跑通**：vllm 0.10.2 + Qwen2.5-7B，10 回合 213 秒 |
| `scripts/02_collect.py --backend api` | 已用本地假服务器验证过 HTTP 通路 |
| `scripts/03_train.py` | **已在服务器跑通**：5 变体 × 5~10 seed，d3rlpy 2.6.0 |
| `scripts/04_report.py` | 已跑通（`--min-evals` 自动剔除 smoke） |
| `scripts/05_compare_cache.py --plot` | 已跑通，出图 `report/author_cache_7b.png` |
| `scripts/07_figure_paper.py` | 已跑通，出图 `report/fig_repro.png` |
| `scripts/99_gpu_check.py` | 无 torch 时的降级路径已验证 |

服务器跑出来的全部结果（含每条 run 的曲线 JSON）在 `server_results/`，结论见
[`../CartPole_复现结果.md`](../CartPole_复现结果.md)。

## 服务器上第一次跑（10 分钟，不烧 GPU）

```bash
conda create -n rl python=3.10 -y && conda activate rl
pip install torch --index-url https://download.pytorch.org/whl/cpu   # RL 用 CPU 版即可
pip install -r requirements-rl.txt

python scripts/99_gpu_check.py              # 可选：30 秒确认显卡能被 torch 认出来
python scripts/00_sanity.py                 # 两行必须 PASS
python scripts/01_fetch_artifacts.py        # 拿到作者数据，约 1 MB
python scripts/02_collect.py --backend mock --episodes 10 --out data/mock_neps10.npz

# smoke：把训练管线跑 3 个评估点，确认 d3rlpy API 对得上
python scripts/03_train.py --variant online --runs 1 --n-eval 3 --device cpu --out results/smoke
```

`smoke` 能跑完，才说明 d3rlpy 的版本/接口没问题。若这里就报错，先解决再花钱。

## 正式实验

```bash
# A. 零 GPU 路线：用作者的 Qwen-7B 轨迹，直接对齐论文数字
python scripts/03_train.py --variant online --runs 5 --device cpu --out results/online
python scripts/03_train.py --variant loro \
    --data artifacts/CartPole_Qwen2.5-7B-Instruct_Neps_30_20250410211529.pkl \
    --pretrain-steps 3000 --runs 5 --device cpu --out results/loro_3k
python scripts/04_report.py --results results --out report

# B. 自采 LLM 数据（GPU，1 小时内）
python scripts/02_collect.py --backend vllm --model Qwen/Qwen2.5-7B-Instruct \
    --episodes 10 --parallel 8 --out data/cartpole_llm_neps10.npz
bash scripts/run_matrix.sh data/cartpole_llm_neps10.npz 5

# C. 不想装 vLLM（或卡是国产的，如 MTT S4000）：走 OpenAI 兼容接口
export DASHSCOPE_API_KEY=sk-xxx
python scripts/02_collect.py --backend api --model qwen2.5-7b-instruct \
    --episodes 10 --parallel 8 --out data/cartpole_llm_neps10.npz
#   国产卡上跑 vLLM 时记得 --dtype float16（MUSA 只有 fp16）
```

## 消融（对应论文图 3 / 图 4）

```bash
# "仅混合、不预训练"：--variant mix（run_matrix 已包含）
# "用随机策略数据预训练"：
python scripts/02_collect.py --backend random --episodes 10 --out data/cartpole_random_neps10.npz
python scripts/03_train.py --variant loro-keep --data data/cartpole_random_neps10.npz \
    --pretrain-steps 3000 --runs 5 --device cpu --out results/pretrain_random
# "用在线 DDQN 自己采的数据预训练"：先用 --variant online 跑 1 个 run 存档，
#   再用 d3rlpy 的 buffer 导出（或直接复用 data/mock 数据），此路较绕，课程报告可略。
```

## 如果 `03_train.py` 在你的 d3rlpy 版本上跑不通

它不是官方代码，是按官方调用逐行重写的。真出问题时，两个选择：

1. **改回官方脚本**：按手册 §7 的 6 处改动直接跑 `LORO-legacy` 的 `pretrain_from_llm.py` / `online_main.py`；
2. **把报错发我**：大概率是 d3rlpy 版本差异（`fit_online` 的参数名、`ReplayBuffer` 的构造方式），
   改一行的事。

## 两个必须知道的坑

1. **回合上限**：必须 200。用了 `CartPole-v1` 而没套 `TimeLimit(200)`，所有数字都会翻倍。
2. **评估口径**：论文的"累积回报"= 150 个评估点（每 200 训练步评 1 个贪心回合）的回报**之和**，
   LORO 只有 140 个点。直接比总和会让 LORO 白吃 7% 的亏，报告里请同时给 `sum/(n*200)`。
