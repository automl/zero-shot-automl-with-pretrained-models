import os
import yaml
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np


font = {'size': 72}
mpl.rc('font', **font)


def perf_matrix_heatmap(perf_matrix_path, savepath = "analysis/meta_dataset/perf_matrix_heatmap.png"):
    
    perf_df = pd.read_csv(perf_matrix_path, index_col = 0)

    perf_df = perf_df.reindex(perf_df.mean().sort_values(ascending = False).index, axis=1)
    perf_df = perf_df.reindex(perf_df.mean(1).sort_values(ascending= False).index, axis=0)

    plt.figure(figsize=(50, 30))
    ax = sns.heatmap(perf_df.values, linewidth=0, xticklabels = 50, yticklabels = 50, cmap = "rocket")
    plt.xlabel("Pipelines", size=108)
    plt.ylabel("Datasets", size=108)
    plt.xticks(size=72, rotation = 45)
    plt.yticks(size=72, rotation = 45)
    plt.tight_layout()
    plt.savefig(savepath)
    plt.clf()


def get_single_best(perf_matrix_path, configs_path, verbose = False):

    perf_df = pd.read_csv(perf_matrix_path, index_col = 0)

    mean_performances = np.mean(perf_df.values, axis = 0)
    generalist = np.argmax(mean_performances)
    pred_config_name = perf_df.columns.values[generalist]
    aug, config_name = pred_config_name.split('-')
    single_best_path = os.path.join(configs_path, str(aug), config_name+".yaml")

    with open(single_best_path) as stream:
        model_config = yaml.safe_load(stream)

    config = model_config['autocv']

    if verbose:
        print(f"Single best {pred_config_name}")

    return config


def get_datasetwise_alc(perf_matrix_path):

    perf_df = pd.read_csv(perf_matrix_path, index_col = 0)

    per_dataset_alc_dict = dict()
    for idx, sample in perf_df.iterrows():
        per_dataset_alc_dict[idx] = sample.values.mean()

    return per_dataset_alc_dict


def get_single_best_per_outer_cv_fold(perf_matrix_folds_path, configs_path, verbose = False):
    
    single_best_per_test_fold = dict()
    for test_dataset_name in os.listdir(perf_matrix_folds_path):
        perf_matrix_path = os.path.join(perf_matrix_folds_path, perf_matrix_folder, "perf_matrix.csv")
        config = get_single_best(perf_matrix_path, configs_path, verbose)
        single_best_per_test_fold[test_dataset_name] = config
    
    return single_best_per_test_fold


def simple_meta_feat_dist(meta_features_path):

    meta_features_df = pd.read_csv(meta_features_path, index_col = 0)

    num_classes = meta_features_df.num_classes.values
    num_channels = meta_features_df.num_channels.values
    num_train = meta_features_df.num_train.values
    res = meta_features_df.resolution_0.values

    rs, cs = np.unique(res, return_counts = True)

    for r, c in zip(rs, cs):
        print(f"{r}: {c}")

    rs, cs = np.unique(num_channels, return_counts = True)

    for r, c in zip(rs, cs):
        print(f"{r}: {c}")


def simple_meta_feat_scatterplot(meta_features_path, autodl_meta_features = None, savepath = "analysis/meta_dataset/meta_features_scatterplot.png"):
    
    meta_features_df = pd.read_csv(meta_features_path, index_col = 0)
    num_classes = meta_features_df.num_classes.values
    num_train = meta_features_df.num_train.values
    plt.figure(figsize=(10, 10))
    plt.scatter(np.log2(num_classes), np.log10(num_train), color = "b", alpha = 0.5)

    if autodl_meta_features is not None:
        public_num_classes, public_num_train, \
        feedback_num_classes, feedback_num_train, \
        final_num_classes, final_num_train = autodl_meta_features

        plt.scatter(np.log2(public_num_classes), np.log10(public_num_train), s = 400.0, color = "g", marker = ".", alpha = 0.8)
        plt.scatter(np.log2(feedback_num_classes), np.log10(feedback_num_train), s = 400.0, color = "r", marker = ".", alpha = 0.8)
        plt.scatter(np.log2(final_num_classes), np.log10(final_num_train), s = 400.0, color = "r", marker = "*", alpha = 0.8)
    
    plt.xlabel("# of Classes (log2-scale)", size=24)
    plt.ylabel("# of Images (log10-scale)", size=24)
    plt.grid()
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.tight_layout()
    plt.savefig()
    plt.clf(savepath)

if __name__ == "__main__":
    
    perf_matrix_path = "../data/meta_dataset/perf_matrix.csv"
    perf_mat_folds_path = "../data/meta_dataset/perf_matrix_per_outer_CV_fold"
    configs_path = "../data/configs/kakaobrain_optimized_per_icgen_augmentation"

    # Plot the performance matrix heatmap and save
    perf_matrix_heatmap(perf_matrix_path)

    # Get the single-best pipeline of the whole meta-dataset
    single_best_config = get_single_best(perf_matrix_path, configs_path)
    single_best_per_test_fold = get_single_best_per_outer_cv_fold(perf_matrix_path, configs_path)

    # Get the mean alc score for each dataset
    per_dataset_alc_dict = get_datasetwise_alc(perf_matrix_path)

    meta_features_path = "../data/meta_dataset/meta_features.csv"
    
    # Hardcoded meta-features of AutoDL benchmark datasets divided according to challenge phases
    autodl_public_datasets = ["Munster", "Chucky", "Pedro", "Decal", "Hammer", "City"]
    public_num_classes = [10, 100, 26, 11, 7, 10]
    public_num_train = [60000, 48061, 80095, 634, 8050, 48060]
    public_res = [28, 32, -1, -1, 400, 32]

    autodl_feedback_datasets = ["Ukulele", "Caucase", "Beatriz", "Saturn", "Hippocrate","Apollon", "Freddy"]
    feedback_num_classes = [3, 257, 15, 3, 2, 100, 2]
    feedback_num_train = [6979, 24518, 4406,324000, 175917,6077,546055]
    feedback_res = [-1, -1, 350, 28, 96,-1,-1]

    autodl_final_datasets = ["Loukoum", "Tim",  "Ideal", "Ray", "Cucumber"]
    final_num_classes = [ 3, 200, 45, 7, 100]
    final_num_train = [ 27938, 80000,  25231, 4492, 18366]
    final_res = [-1, 32, 256, 976, -1]

    autodl_meta_features = [public_num_classes, public_num_train, feedback_num_classes, feedback_num_train, final_num_classes, final_num_train]
    
    simple_meta_feat_scatterplot(meta_features_path, autodl_meta_features)
    simple_meta_feat_dist(meta_features_path)
