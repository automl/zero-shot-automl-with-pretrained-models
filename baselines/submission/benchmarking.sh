#!/bin/bash

#SBATCH -p mlhiwidlc_gpu-rtx2080
#SBATCH --job-name dlb_0
#SBATCH -o logs/%A-%a.%x.o
#SBATCH -e logs/%A-%a.%x.e

#SBATCH -a 1-10500

source /home/ozturk/anaconda3/bin/activate autodl_benchmark
pwd

ARGS_FILE=submission/benchmarking_batch_0.args
TASK_SPECIFIC_ARGS=$(sed "${SLURM_ARRAY_TASK_ID}q;d" $ARGS_FILE)
echo $TASK_SPECIFIC_ARGS

python run_local_test.py $TASK_SPECIFIC_ARGS
