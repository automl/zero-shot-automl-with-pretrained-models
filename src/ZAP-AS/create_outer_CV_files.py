import os
import sys
import pandas as pd

sys.path.append("../")

from available_datasets import all_datasets

n_augmentations = 15

def split_perf_mat(meta_dataset_main_path):

    perf_df = pd.read_csv(os.path.join(meta_dataset_main_path, "perf_matrix.csv"), index_col = 0)
    fold_df = pd.read_csv(os.path.join(meta_dataset_main_path, "inner_CV_folds.csv"), index_col = 0)

    for idx, dataset_name in enumerate(all_datasets):
        
        perf_fold_path = os.path.join(meta_dataset_main_path, "perf_matrix_per_outer_CV_fold", dataset_name)
        os.makedirs(perf_fold_path, exist_ok = True)
        
        test_augmentations = [str(n)+'-'+dataset_name for n in range(n_augmentations)]

        training_perf_df = perf_df.drop(test_augmentations, 0) # from rows
        test_perf_df = perf_df.loc[test_augmentations]
        training_fold_df = fold_df.drop(test_augmentations, 0)

        training_perf_df.to_csv(os.path.join(perf_fold_path, 'perf_matrix.csv'))
        test_perf_df.to_csv(os.path.join(perf_fold_path, 'perf_matrix_test.csv'))
        training_fold_df.to_csv(os.path.join(perf_fold_path, 'inner_CV_folds.csv'))

def split_meta_features(meta_dataset_main_path):
    
    meta_features_path = os.path.join(meta_dataset_main_path, "meta_features.csv")

    mf_df = pd.read_csv(meta_features_path, index_col = 0)

    for idx, dataset_name in enumerate(all_datasets):
        mf_fold_path = os.path.join(meta_dataset_main_path, "meta_features_per_outer_CV_fold", dataset_name)
        os.makedirs(mf_fold_path, exist_ok = True)

        test_augmentations = [str(n)+'-'+dataset_name for n in range(n_augmentations)]

        training_mf_df = mf_df.drop(test_augmentations)
        test_mf_df = mf_df.loc[test_augmentations]

        training_mf_df.to_csv(os.path.join(mf_fold_path, 'meta_features.csv'))
        test_mf_df.to_csv(os.path.join(mf_fold_path, 'meta_features_test.csv'))

def create_AutoFolio_cv_run_args(meta_dataset_main_path, args_savepath, exp_suffix):
    
    f = open(args_savepath, "w+")
    for dataset_name in all_datasets:
        core_perf_path = os.path.join(meta_dataset_main_path, "perf_matrix_per_outer_CV_fold", dataset_name, "perf_matrix.csv")
        core_feat_path = os.path.join(meta_dataset_main_path, "meta_features_per_outer_CV_fold", dataset_name, "meta_features.csv")
        cv_csv = os.path.join(meta_dataset_main_path, "perf_matrix_per_outer_CV_fold", dataset_name, "inner_CV_folds.csv")
        
        cmd = " ".join(["--perf_path", core_perf_path, 
                        "--feat_path", core_feat_path,
                        "--exp_suffix", f"{exp_suffix}/{dataset_name}",
                        "--cv_csv", cv_csv])
        f.write(cmd)
        f.write('\n')
    f.close()



if __name__ == "__main__":


    meta_dataset_main_path = "../../data/meta_dataset"
    args_savepath = "submission/ZAP-AS.args"
    exp_suffix = "ZAP"
    split_perf_mat(meta_dataset_main_path)
    split_meta_features(meta_dataset_main_path)
    create_AutoFolio_cv_run_args(meta_dataset_main_path, args_savepath, exp_suffix)

    
    