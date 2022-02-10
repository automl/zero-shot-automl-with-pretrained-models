#!/bin/bash

#SBATCH -p mlhiwidlc_gpu-rtx2080
#SBATCH --job-name eval_ZAP
#SBATCH -o logs/%A-%a.%x.o
#SBATCH -e logs/%A-%a.%x.e

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task 1

#SBATCH -a 1-826875

source /home/ozturk/anaconda3/bin/activate autodl
pwd

ARGS_FILE=submission/per_icgen_augmentation_x_configs.args
TASK_SPECIFIC_ARGS=$(sed "${SLURM_ARRAY_TASK_ID}q;d" $ARGS_FILE)

echo $TASK_SPECIFIC_ARGS

python src/competition/run_local_test.py --experiment_group_dir "../../data/per_icgen_augmentation_x_configs_evaluations" $TASK_SPECIFIC_ARGS

