#!/bin/bash

#SBATCH -p bosch_cpu-cascadelake
#SBATCH --job-name af_ZAP-AS
#SBATCH -o logs/%A-%a.%x.o
#SBATCH -e logs/%A-%a.%x.e

#SBATCH --cpus-per-task=1
#SBATCH --mem=6000

#SBATCH -a 1-35

source /home/ozturk/anaconda3/bin/activate autodl

ARGS_FILE=submission/AutoFolio_args/ZAP-AS.args
TASK_SPECIFIC_ARGS=$(sed "${SLURM_ARRAY_TASK_ID}q;d" $ARGS_FILE)
echo $TASK_SPECIFIC_ARGS

python AutoFolioPipeline.py --tune --autofolio_model_path "../../data/models/AutoFolio_models/ZAP" $TASK_SPECIFIC_ARGS
