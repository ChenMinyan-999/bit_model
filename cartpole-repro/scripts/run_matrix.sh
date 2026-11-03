#!/usr/bin/env bash
# CartPole 复现矩阵：一条命令跑完"纯在线 / LORO / 仅混合 / LORO-KEEP"
# 用法：bash scripts/run_matrix.sh data/cartpole_llm_neps10.npz 5
set -euo pipefail

DATA=${1:-data/cartpole_llm_neps10.npz}
RUNS=${2:-5}
PY=${PY:-python}

echo "== online (150 evals) =="
$PY scripts/03_train.py --variant online --runs "$RUNS" --label llm --out results/online

for STEPS in 1000 3000; do
  echo "== LORO 7B / ${STEPS} steps (140 evals) =="
  $PY scripts/03_train.py --variant loro --data "$DATA" --pretrain-steps "$STEPS" \
      --runs "$RUNS" --label llm --out "results/loro_${STEPS}"
done

echo "== mix (no pretrain, prefill) =="
$PY scripts/03_train.py --variant mix --data "$DATA" --runs "$RUNS" \
    --label llm --out results/mix

echo "== LORO-KEEP (pretrain + prefill) =="
$PY scripts/03_train.py --variant loro-keep --data "$DATA" --pretrain-steps 3000 \
    --runs "$RUNS" --label llm --out results/loro_keep_3000

echo "== report =="
$PY scripts/04_report.py --results results --out report
