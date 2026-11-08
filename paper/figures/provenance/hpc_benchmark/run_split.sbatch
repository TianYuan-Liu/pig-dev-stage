#!/bin/bash
#SBATCH --job-name=pigsplit
#SBATCH --output=/storage/gge/Tian/pig_ml_benchmark/logs_split/sp_%A_%a.out
#SBATCH --error=/storage/gge/Tian/pig_ml_benchmark/logs_split/sp_%A_%a.err
#SBATCH --qos=short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=06:00:00

set -euo pipefail
WS=/storage/gge/Tian/pig_ml_benchmark
PY=/storage/gge/Tian/conda_envs/pigml/bin/python
: "${TASKLIST:?set TASKLIST}"
: "${SPLIT_OUT:?set SPLIT_OUT}"

export BENCH_WS=$WS BENCH_SCHEME=6stage
# threads follow the allocation: --cpus-per-task=1 => deterministic 1-thread fan-out;
# >1 => BLAS multi-thread (much faster for SVD-bound models; BA identical to displayed precision)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$TASKLIST")
read -r MODEL TISSUE CV SEED FOLD <<< "$line"
echo "task=$SLURM_ARRAY_TASK_ID -> $MODEL $TISSUE $CV $SEED f$FOLD host=$(hostname) $(date)"
$PY $WS/code/benchmark_split.py "$MODEL" "$TISSUE" "$CV" "$SEED" "$FOLD" "$SPLIT_OUT"
echo "done $(date)"
