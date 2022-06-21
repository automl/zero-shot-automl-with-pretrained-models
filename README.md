# Zero-Shot AutoML with Pretrained Models
<img src="https://github.com/automl/zero-shot-automl-with-pretrained-models/blob/main/overview.png?raw=true" width="400"/>

Given a new dataset D and a low compute budget, how should we choose a pre-trained model to fine-tune to D, and set the fine-tuning hyperparameters without risking overfitting, particularly if D is small? Here, we extend automated machine learning (AutoML) to best make these choices. Our domain-independent meta-learning approach learns a zero-shot surrogate model which, at test time, allows to select the right deep learning (DL) pipeline (including the pre-trained model and fine-tuning hyperparameters) for a new dataset D given only trivial meta-features describing D such as image resolution or the number of classes. To train this zero-shot model, we collect performance data for many DL pipelines on a large collection of datasets and meta-train on this data to minimize a pairwise ranking objective. We evaluate our approach under the strict time limit on the vision track of the ChaLearn AutoDL challenge benchmark, clearly outperforming all challenge contenders.

Paper link: tba

# Download Models and Meta-Dataset
Our data can be downloaded under the following links:
* Datasets (~196G): https://bit.ly/3B1zvl0
* Models (~700M): https://bit.ly/3BhIAGB
* Meta-dataset (~2.5M): https://bit.ly/3NnhP8W 

and should be placed under `.data/datasets`, `.data/models`, `.data/meta_dataset`. 

The *Meta-dataset* download consists of the pipeline configuration files, preextracted meta-features and the performance matrix. One can directly download this to skip costly meta-dataset acquisition procedure.

# Installation

Create and activate a Python 3.7 environment and run

```
pip install -r requirements.txt
```

# Documentation

For the meta-dataset preparation steps please refer to [ZAP](src/ZAP/README.md). One may download the outputs of this step (see above: *Meta-dataset*) and skip directly to ZAP-AS or ZAP-HPO.

## 1. ZAP-AS

In order to execute the experiments below please download and decompress the necessary *meta-dataset* files under './data' folder. Then tto prepare the inner CV folds that is required for tuning the AutoFolio model can be prepared by simply running:

```
python -m src.avalable_datasets
```

### 2.1. Train a single model

To train an AutoFolio model over ZAP run

```
python -m AutoFolioPipeline.py --tune \
  --perf_path PATH_TO_COST_MATRIX \
  --feat_path PATH_TO_META_FEATURES \
  --cv_csv PATH_TO_INNER_CV_FOLDS \
  --autofolio_model_path PATH_TO_RESULTING_MODEL(S) \
  --exp_suffix MODEL_SAVENAME \
```

e.g

```
python -m AutoFolioPipeline --tune \
  --perf_path ../../data/meta_dataset/perf_matrix.csv \
  --feat_path ../../data/meta_dataset/meta_features.csv \
  --cv_csv ../../data/meta_dataset/inner_CV_folds.csv \
  --autofolio_model_path ../../data/models/AutoFolio_models/ZAP \
  --exp_suffix ZAP_single
```

This tunes AutoFolio hyperparameters 5-fold cross validation given in the `PERF_MAT_FOLDER/inner_CV_folds.csv` file.

### 2.2. Train a model per outer-CV

To train AutoFolio models per core test-dataset, one needs to prepare a seperate performance matrix and meta features for each core dataset. `create_outer_CV_files.py` script creates these files as well as array job arguments for Meta. Under the `submission` folder one may find example bash scripts for different specifications of training. The outputted files will be the same version of inputs except the meta-test rows are dropped e.g input shape: 525x525 -> output shape: 510x525.

```
python create_outer_CV_files.py
```

then to tune (default wallclock time: 22 hours) and train the AutoFolio models simply run

```
sbatch submission/af_outer_CV_simple.sh
```

## 2. ZAP-HPO

# ZAP Benchmark

Mention ZAP-AS and HPO submissions are here 
This benchmark contains the 525 ZAP datasets. 

The solutions in submission format are under `./baselines` including the ZAP-AS and ZAP-HPO submissions.

### 1. Create the array job arguments and run baselines

The `baselines/` folder contains all the baselines and variations of our approach.

First

```
cd baselines/
```

and

```
python submission/create_benchmarking_args.py
```

will create arguments for all solutions provided, shuffles and splits them into batches. Then one can submit batch by batch via

```
sbatch submission/benchmarking.sh
```

by only changing `ARGS_FILE` parameter inside the script.

### 2. Collect and plot the results

The above procedure will provide *raw* AutoDL outputs, which includes a `scoring.txt` file for each run. To read these files(time-consuming), save in a nice json format for later use, and also plot the experiment results on the paper run 

```
python analysis/get_benchmarking_results.py
```

One can switch off `parse_and_collect_results()` method inside the script above after the first execution and switch on if only there is a modification. This script also provides box plots and ranking results in the paper.

For the cost matrix heat map and meta-feature scatter plot run

```
python analysis/meta_dataset/meta_dataset_analysis.py
```

This will provide these two plots under the `analysis/meta_dataset/`.
