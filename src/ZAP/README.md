# Meta-dataset Creation

Consists of necessary steps to prepare the meta-dataset

## 0. Environment
Create and activate a Python 3.7 environment and run
```bash
bash install/requirements_gcc.sh
pip install -r install/requirements.txt
bash install/requirements_torch_cuda100.sh
bash install/install_winner_cv.sh
bash install/install_just.sh        # Optional command runner
bash install/install_precommit.sh   # Developer dependency
```

## 1. Offline Preparation

### 1.1. Creation of the pipeline candidates

To perform BOHB over ZAP datasets (on Meta):

1. Create array job argument file `submission/ARGS_JOBFILE_PATH` via 
  ```
  python submission/create_hpo_args.py --args_savepath ARGS_JOBFILE_PATH
  ```
  
  e.g
  
  ```
  python submission/create_hpo_args.py --args_savepath per_icgen_augmentation_hpo.args
  ```

2. Submit the job on cluster via 
  ```
  sbatch submission/meta_kakaobrain_optimized_per_dataset.sh
  ``` 
Change the array job arguments filename to *submission/ARGS_JOBFILE_PATH* and (optionally) specify directory for the BOHB result files via the argument ```--experiment_group_dir BOHB_RES_DIR```. 

3. Fetch the incumbent (best) pipelines via 
  ```
  python src/hpo/incumbents_to_config.py --output_dir CONFIG_OUTPUT_DIR --experiment_group_dir BOHB_RES_DIR 
  ```
  
  e.g
  
  ```
  python src/hpo/incumbents_to_config.py --output_dir ../../data/configs/kakaobrain_optimized_per_icgen_augmentation \
                                         --experiment_group_dir ../../data/per_icgen_augmentation_hpo 
  ```

The directory `../../data/configs/configs/kakaobrain_optimized_per_icgen_augmentation` contains a `.yaml` file for each incumbent pipeline.

### 1.2. Creation of the performance matrix
  
To evaluate each ZAP pipeline-dataset pair (on Meta):

1. Create array job arguments
  ```
  python submission/create_datasets_x_configs_args.py --configs_path CONFIG_OUTPUT_DIR --command_file_name ARGS_JOBFILE_PATH
  ```
  
  e.g
  
  ```
  python submission/create_datasets_x_configs_args.py --configs_path ../../data/kakaobrain_optimized_per_icgen_augmentation \
                                                      --command_file_name per_icgen_augmentation_x_configs.args
  ```

2. Run the jobs on Meta
  ```
  sbatch submission/meta_kakaobrain_datasets_x_configs.sh
  ``` 

Change the array job arguments filename to *submission/ARGS_JOBFILE_PATH* and (optionally) specify directory for the AutoDL ingestion and scoring output files via the argument ```--experiment_group EVALUATION_RES_DIR```.

3. Parse the evaluation results and create the performance matrix via 
  ```
  python src/hpo/performance_matrix_from_evaluation.py --experiment_group EVALUATION_RES_DIR --output_savedir META_DATASET_OUTPUT_DIR
  ```
  
  e.g
  
  ```
  python src/hpo/performance_matrix_from_evaluation.py --experiment_group "../../data/per_icgen_augmentation_x_configs_evaluations" \
                                                       --output_savedir ../../data/meta_dataset
  ```

This results in a performance matrix `.csv` and `.pkl` files under `META_DATASET_OUTPUT_DIR`.

ADD: Perf matrix heatmap

## 1.3. Precomputing meta-features

To extract the simple meta-features of ZAP datasets, run the command below:

```
python -m src.meta_features.precompute_meta_features --dataset_dir DATASETS_PARENT_PATH  --output_savedir META_DATASET_OUTPUT_DIR
```
