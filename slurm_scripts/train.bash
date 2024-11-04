#!/bin/bash
#SBATCH -n 6
#SBATCH -N 1
#SBATCH --mem=64G
#SBATCH -t 48:00:00

### modify seed below
#SBATCH --array=41-42
#SBATCH --partition=3090-gcondo
#SBATCH --gres=gpu:1

# Use '%A' for array-job ID, '%J' for job ID and '%a' for task ID
#SBATCH -e sbatch_out/arrayjob-train-%A/%a.err
#SBATCH -o sbatch_out/arrayjob-train-%A/%a.out


source ~/.bashrc
conda activate ssl

# i=`expr $SLURM_ARRAY_TASK_ID`
# SEED=`expr $i % 5`
# ALGO_TYPE=`expr $i / 5`

SEED=$SLURM_ARRAY_TASK_ID

echo "SEED: $SEED"
echo "SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_JOB_ID"

RUN_NAME="${SLURM_ARRAY_JOB_ID}"

python -W ignore models/policy_head/policy_head.py --seed $SEED
