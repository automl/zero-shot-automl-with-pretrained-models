# Zero-Shot AutoML with Pretrained Models
<img src="https://raw.githubusercontent.com/automl/zero-shot-automl-with-pretrained-models/master/overview.png" width="400"/>

Given a new dataset D and a low compute budget, how should we choose a pre-trained model to fine-tune to D, and set the fine-tuning hyperparameters without risking overfitting, particularly if D is small? Here, we extend automated machine learning (AutoML) to best make these choices. Our domain-independent meta-learning approach learns a zero-shot surrogate model which, at test time, allows to select the right deep learning (DL) pipeline (including the pre-trained model and fine-tuning hyperparameters) for a new dataset D given only trivial meta-features describing D such as image resolution or the number of classes. To train this zero-shot model, we collect performance data for many DL pipelines on a large collection of datasets and meta-train on this data to minimize a pairwise ranking objective. We evaluate our approach under the strict time limit on the vision track of the ChaLearn AutoDL challenge benchmark, clearly outperforming all challenge contenders.

Paper link: tba

# Download Models and Meta-Dataset
Our data can be downloaded under the following links:
* Meta-Dataset: https://bit.ly/3B1zvl0
* Models: https://bit.ly/3BhIAGB

# Installation

Create and activate a Python 3.7 environment and run

'''
pip install -r requirements.txt
'''

# Documentation

## 1. ZAP-AS

In order to execute the experiments below please download and decompress the necessary *meta-dataset* files under './data' folder. 

## 2.1. Train a single model

To train an AutoFolio model over ZAP run

```
python AutoFolioPipeline.py --tune \
  --perf_path PATH_TO_COST_MATRIX
  --feat_path PATH_TO_META_FEATURES
  --cv_csv PATH_TO_INNER_CV_FOLDS
  --autofolio_model_path PATH_TO_RESULTING_MODEL(S)
  --exp_suffix MODEL_SAVENAME
```

e.g

```
python AutoFolioPipeline.py --tune \
  --perf_path ../data/meta_dataset/perf_matrix.csv 
  --feat_path ../data/meta_dataset/meta_features.csv
  --cv_csv ../data/meta_dataset/inner_CV_folds.csv
  --autofolio_model_path ../../data/AutoFolio_models/ZAP
  --exp_suffix ZAP_single
```

This tunes AutoFolio hyperparameters 5-fold cross validation given in the `PERF_MAT_FOLDER/inner_CV_folds.csv` file.

## 2.2. Train a model per outer-CV

To train AutoFolio models per core test-dataset, one needs to prepare a seperate performance matrix and meta features for each core dataset. `create_outer_CV_files.py` script creates these files as well as array job arguments for Meta. Under the `submission` folder one may find example bash scripts for different specifications of training.

```
python create_outer_CV_files.py
```

then simply run

```
sbatch submission/af_outer_CV_simple.sh
```

## 2. ZAP-HPO

## 3. ZAP Benchmark
