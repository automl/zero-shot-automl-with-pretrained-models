#!/bin/bash

#SBATCH -p bosch_cpu-cascadelake
#SBATCH --job-name af_dry
#SBATCH -o logs/%A-%a.%x.o
#SBATCH -e logs/%A-%a.%x.e

#SBATCH --cpus-per-task 32
#SBATCH --mem 192000

#SBATCH -a 1-1

source /home/ozturk/anaconda3/bin/activate autodl
pwd

ARGS_FILE=submission/AutoFolio_args/ZAP-AS.args
TASK_SPECIFIC_ARGS=$(sed "${SLURM_ARRAY_TASK_ID}q;d" $ARGS_FILE)

echo $TASK_SPECIFIC_ARGS

python AutoFolioPipeline.py --tune $TASK_SPECIFIC_ARGS
