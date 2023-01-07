# Zero-Shot AutoML with Pretrained Models
<img src="https://github.com/automl/zero-shot-automl-with-pretrained-models/blob/main/overview.png?raw=true" width="400"/>

Given a new dataset D and a low compute budget, how should we choose a pre-trained model to fine-tune to D, and set the fine-tuning hyperparameters without risking overfitting, particularly if D is small? Here, we extend automated machine learning (AutoML) to best make these choices. Our domain-independent meta-learning approach learns a zero-shot surrogate model which, at test time, allows to select the right deep learning (DL) pipeline (including the pre-trained model and fine-tuning hyperparameters) for a new dataset D given only trivial meta-features describing D such as image resolution or the number of classes. To train this zero-shot model, we collect performance data for many DL pipelines on a large collection of datasets and meta-train on this data to minimize a pairwise ranking objective. We evaluate our approach under the strict time limit on the vision track of the ChaLearn AutoDL challenge benchmark, clearly outperforming all challenge contenders.

Paper: https://arxiv.org/abs/2206.08476

# Download Models and Meta-Dataset
Our data can be downloaded under the following links:
* Datasets (~196G): https://bit.ly/3B1zvl0
* Models (~1.8G): https://bit.ly/3BhIAGB
* Meta-dataset (~30M): https://bit.ly/3NnhP8W 

Create a data folder `mkdir data`.
Extract and place the folders as `.data/datasets`, `.data/models`, `.data/meta_dataset`. 

The *Meta-dataset* download consists of the pipeline configuration files, preextracted meta-features and the performance(cost) matrix (also the performance matrix of final accuracies can be found here). The input files exclusive to ZAP-HPO, namely `cls_names.pkl`, `data_m.csv` can be found here. One can directly download this to skip costly meta-dataset acquisition procedure.

Alternatively one can find a single dataset (cifar10) augmentation under the link below in order to check the format, test the setup etc.

* 0/cifar10 (35M): ...

# Installation

Create and activate a Python 3.7 environment and run

```
pip install -r requirements.txt
```

# Documentation

For the meta-dataset preparation steps please refer to [ZAP](src/ZAP/README.md). One may download the outputs of this step (see above: *Meta-dataset*) and skip directly to ZAP-AS or ZAP-HPO to prepare the meta-models. The instructions below have been prepared for running on the default meta-dataset. For the custom meta-datasets, some of the steps may require modifications in code/arguments (especially in ZAP-HPO) for a successful execution.

## 1. ZAP-AS

In order to execute the experiments below please download and decompress the necessary *meta-dataset* files under './data' folder. Then to prepare the inner CV folds that is required for tuning the AutoFolio model can be prepared by simply running:

```
python available_datasets.py
```

### 1.1. Train a single model

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

This tunes AutoFolio hyperparameters using 5-fold cross validation where the folds are provided in the `PERF_MAT_FOLDER/inner_CV_folds.csv` file.

### 1.2. Train a model per outer-CV

To train AutoFolio models per core test-dataset, one needs to prepare a seperate performance matrix and meta features for each core dataset. `create_outer_CV_files.py` script creates these files as well as array job arguments for Meta. Under the `submission` folder one may find example bash scripts for different specifications of training. The outputted files will be the same version of inputs except the meta-test rows are dropped e.g input shape: 525x525 -> output shape: 510x525.

```
python create_outer_CV_files.py
```

then to tune (default wallclock time: 22 hours) and train the AutoFolio models simply run

```
sbatch submission/af_outer_CV_simple.sh
```

## 2. ZAP-HPO

This section provides the information to train a ZAP-HPO surrogate. Same as the previous section (ZAP-AS), this one also requires a meta-dataset. 

### 2.2. Train a single-model

To train a surrogate with its default hyperparameters

```
python runner.py --config_path default_config.yaml --cv FOLD_NUM
```

The default run argument (`--split_type`) is set to train a single model. The cross-validation fold number [1,5] should be provided. This validates the surrogate only on the specified fold. One needs to execute the command above for each of the folds, i.e, 5 times for 5-fold cross-validation.

### 2.2. Train a model per outer-CV

Similar to the above, but one core-dataset is left out.

```
python runner.py --config_path default_config.yaml --split_type loo --loo OUTER_FOLD_DATASET_NAME --cv FOLD_NUM
```

The `OUTER_FOLD_DATASET_NAME` is the name of a core-dataset, e.g. cifar10.

### 2.3. More on variations and ablations

One can run the variations of ZAP-HPO and also the ablations (introducing sparsity to the meta-dataset or omitting the meta-features) by setting specific arguments for `runner.py`. Please refer to the code or run the command below for the complete set of arguments.

```
python runner.py --help
```

# ZAP Benchmark

The `baselines/` folder contains all the baselines and variations of our approach. The solutions labelled as "CRC" are the ones that we used on getting the results on the paper. For convenience and ease of use, the ZAP-HPO code is modified and it is still work in progress. A solution using the most recent code is also (one with no CRC label) provided here.

This benchmark contains the 525 ZAP datasets. Necessary downloads for this section are the datasets, the meta-models (if above section is skipped or to reproduce CRC results), and the pipeline configurations.

The solutions in submission format are under `./baselines` including the ZAP-AS and ZAP-HPO submissions.

The top-3 winner solutions of the latest AutoDL competition are not provided here. One can refer to the competition webpage ([link](https://autodl.chalearn.org/)) and winners' respective repositories. We provide link to the top-3 solutions' repositories here for completion:

- [DeepWisdom](https://github.com/DeepWisdom/AutoDL)
- [DeepBlueAI](https://github.com/DeepBlueAI/AutoDL)
- [PASA-NJU](https://github.com/HazzaCheng/AutoDL2019)

### 0. Running a single solution on a single dataset

In order to run a solution (e.g. ZAP-HPO) on a dataset (e.g. 0-cifar10) one can run the command below

```
python run_local_test.py \
  --code_dir SOLUTION_DIR_NAME \
  --dataset_dir PATH/TO/DATASET_DIR \
  --result_dir PATH/TO/RESULT_DIR
```

e.g.

```
python run_local_test.py \
  --code_dir ZAP-HPO_submission \
  --dataset_dir ../data/datasets/0/cifar10 \
  --result_dir ../data/results/ZAP-HPO_0-cifar10_result
```

### 1. Create the array job arguments and run baselines

```
python submission/create_benchmarking_args.py
```

will create arguments for all solutions provided, shuffles and splits them into batches. Then one can submit batch by batch via

```
sbatch submission/benchmarking.sh
```

by only changing `ARGS_FILE` parameter inside the script.

P.S: (The reason for batching the arguments) Array jobs' length can be at most 30000 using SLURM scheduler. Benchmarking all the solutions on all datasets for 10 repetitions (each solution-dataset pair) exceeds this constraint. 

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

# Folder Structure
```
  .
  ├── analysis          # scripts to visualize the meta-dataset, collect and visualize the benchmarking runs
  ├── baselines         # self-sufficient solutions and scripts to run solutions (locally/on cluster) on datasets
  ├── data             
  │   ├── datasets      # image datasets in AutoDL format
  │   ├── meta-dataset  # cost matrix, meta features, DL pipeline configs
  │   └── models        # trained ZAP-AS and ZAP-HPO models 
  ├── src            
  │   ├── ZAP           # meta-dataset creation (sampling pipelines, creating cost matrix, meta-feature extraction)
  │   ├── ZAP-AS        # algorithm selection
  │   └── ZAP-HPO       # zero-shot HPO
  └── ...
```
