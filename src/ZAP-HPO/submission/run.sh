#!/bin/bash

#SBATCH -p bosch_cpu-cascadelake
#SBATCH --job-name ZAP-HPO
#SBATCH -o logs/%A-%a.%x.o
#SBATCH -e logs/%A-%a.%x.e

#SBATCH --cpus-per-task=8
#SBATCH --mem=48000

#SBATCH -a 1-25

source /home/ozturk/anaconda3/bin/activate zap_hpo

ARGS_FILE=submission/run.args
TASK_SPECIFIC_ARGS=$(sed "${SLURM_ARRAY_TASK_ID}q;d" $ARGS_FILE)
echo $TASK_SPECIFIC_ARGS

python runner.py $TASK_SPECIFIC_ARGS