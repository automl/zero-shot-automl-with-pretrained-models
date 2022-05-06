#!/bin/bash

#SBATCH --job-name ZAP-Create-Data
#SBATCH -o logs/%A-%a.%x.o
#SBATCH -e logs/%A-%a.%x.e
#SBATCH --mem=0
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=dipti.sengupta@students.uni-freiburg.de


cd $(ws_find zap_ws)
python3 -m venv zap_env --python=python3.7
source zap_env/bin/activate

cd zero-shot-automl-with-pretrained-models/
# pip install -r requirements.txt

FILE=meta_dataset.zip
if [ ! -f "$FILE" ]; then
    echo "$FILE does not exist. Will download and unzip here"
    wget -O meta_dataset.zip https://bit.ly/3NnhP8W
    tar -zxvf meta_dataset.zip
fi

echp "Run creation of csv"
python3 src/ZAP-HPO/create_data_m.py

deactivate

