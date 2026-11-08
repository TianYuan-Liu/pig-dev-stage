#!/bin/bash
#SBATCH --job-name=pigml_b6
#SBATCH --output=/storage/gge/Tian/pig_ml_benchmark/logs_6stage/bench_%A_%a.out
#SBATCH --error=/storage/gge/Tian/pig_ml_benchmark/logs_6stage/bench_%A_%a.err
#SBATCH --qos=short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=23:30:00
#SBATCH --array=0-20
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=liut33@cardiff.ac.uk

set -euo pipefail
WS=/storage/gge/Tian/pig_ml_benchmark
PYTHON=/storage/gge/Tian/conda_envs/pigml/bin/python
MODELS=$WS/code/MODELS.txt

export BENCH_WS=$WS
export BENCH_SCHEME=6stage
export BENCH_RESULTS=results_6stage
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

MODEL=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MODELS")
echo "task=$SLURM_ARRAY_TASK_ID model=$MODEL scheme=6stage host=$(hostname) start=$(date)"

$PYTHON $WS/code/benchmark_one_model.py "$MODEL"

echo "done model=$MODEL end=$(date)"
