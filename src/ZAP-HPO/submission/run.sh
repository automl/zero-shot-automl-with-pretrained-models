#!/bin/bash

#SBATCH -p bosch_cpu-cascadelake
#SBATCH --job-name ZAP-HPO
#SBATCH -o logs/%A-%a.%x.o
#SBATCH -e logs/%A-%a.%x.e

#SBATCH --cpus-per-task=1

#SBATCH -a 16-20

source /home/ozturk/anaconda3/bin/activate autodl_benchmark

ARGS_FILE=submission/loo.args
TASK_SPECIFIC_ARGS=$(sed "${SLURM_ARRAY_TASK_ID}q;d" $ARGS_FILE)
echo $TASK_SPECIFIC_ARGS

python runner.py --max_epoch 5 $TASK_SPECIFIC_ARGS