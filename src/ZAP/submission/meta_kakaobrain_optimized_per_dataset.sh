#!/bin/bash

#SBATCH -p mlhiwidlc_gpu-rtx2080
#SBATCH --job-name hpo_ZAP
#SBATCH -o logs/%A-%a.%x.o
#SBATCH -e logs/%A-%a.%x.e

#SBATCH --mail-user=ozturk@informatik.uni-freiburg.de
#SBATCH --mail-type=END,FAIL

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task 8

#SBATCH -a 1-525
 
source /home/ozturk/anaconda3/bin/activate autodl

ARGS_FILE=submission/per_icgen_augmentation_hpo.args
TASK_SPECIFIC_ARGS=$(sed "${SLURM_ARRAY_TASK_ID}q;d" $ARGS_FILE)
echo $TASK_SPECIFIC_ARGS

python src/hpo/optimize.py --experiment_group_dir "../../../data/per_icgen_augmentation_hpo" --job_id $SLURM_ARRAY_JOB_ID  $TASK_SPECIFIC_ARGS
