import os
import sys
sys.path.append(os.getcwd())
sys.path.append("../")

from pathlib import Path
import numpy as np
import pandas as pd

from available_datasets import all_datasets

def get_scores_dataset_x_configs(dataset_eval_dir, n_augmentations, metric = "alc"):
    config_names = []
    avg_config_scores = []
    failed_jobs = []
    for n in range(n_augmentations):
        augmentation_dir = Path(dataset_eval_dir, str(n))
        paths_list = sorted(augmentation_dir.glob("*"))
        all_config_paths = [path for path in paths_list if not path.is_file()]

        n_repeat = len(set([int(str(c.absolute())[-1]) for c in all_config_paths]))
        
        for config_path in all_config_paths: 
            mean_config_name = str(n)+"-"+config_path.name.rsplit("_", maxsplit=1)[0]
            if mean_config_name not in config_names:
                config_names.append(mean_config_name)

        # Splits all config paths [Chuck_0, Chuck_1, ..., Hammer_0, Hammer_1]
        # Into according sublists [[Chuck_0, Chuck_1],..., [Hammer_0, Hammer1]]
        config_sublists = [all_config_paths[x:x + n_repeat] for x in range(0, len(all_config_paths), n_repeat)]

        for i, config_path_sublist in enumerate(config_sublists):
            config_scores = []
            for n, config_path in enumerate(config_path_sublist):
                score_path = config_path / "score" / "scores.txt"
  
                if metric == "alc":
                    # get alc
                    score = float(score_path.read_text().split(" ")[1].split("\n")[0])
                    config_scores.append(score)
                elif metric == "acc":
                    # get accuracy
                    score_info = score_path.read_text().split("\n")
                    score = score_info[4].split(": ")[-1][1:-1].split(", ")[-1]
                    config_scores.append(float(score))

            if not config_scores:
                avg_config_scores.append(0.0)
            else:
                avg_config_scores.append(np.mean(config_scores))

                #print("Finalized %s-%s config %s. Score mean: %f, var: %f " \
                   # % (augmentation_dir.parent.parent.name, augmentation_dir.parent.name, config_names[i], np.mean(config_scores), np.var(config_scores)))

    assert len(avg_config_scores) == len(config_names), \
        "something went wrong, number of configs != scores"

    return avg_config_scores, config_names, failed_jobs


def create_df_perf_matrix(experiment_group_dir, n_augmentations, existing_df=None):
    all_failed_jobs = []
    for n in range(n_augmentations):
        
        print(f"Parsing augmentation {n}")

        augmentation_dir = Path(experiment_group_dir, str(n))
        
        for i, dataset_eval_dir in enumerate(sorted(augmentation_dir.iterdir())):

            if dataset_eval_dir.is_dir():  # iterdir also yields files
                avg_config_scores, config_names, failed_jobs = get_scores_dataset_x_configs(dataset_eval_dir, n_augmentations)

                if i == 0 and n == 0:
                    indices = config_names.copy()
                    df = pd.DataFrame(columns=config_names, index=indices)

                df.loc[str(n)+"-"+dataset_eval_dir.name] = avg_config_scores

                all_failed_jobs += failed_jobs

    if existing_df is not None:
        df = pd.concat([existing_df, df], axis=1)

    print(f"Number of failed jobs: {len(all_failed_jobs)}")
    f = open("submission/evaluation_failed_jobs.args", "w+")
    f.write("\n".join(all_failed_jobs))
    f.close()
    
    return df

def transform_to_long_matrix(df, n_samples):
    """
    Transform a dataframe of shape (n_algorithms, n_datasets) to
    shape (n_algorithms * n_samples, n_datasets) by simply copying the rows n_sample times.
    This is required s.t. the perf matrix complies with the shape of the feature matrix.
    """
    new_df = pd.DataFrame(columns=df.columns)
    for index, row in df.iterrows():
        for i in range(n_samples):
            new_index = index + "_{}".format(i)
            new_df.loc[new_index] = row

    train_dataset_names = [d + "_{}".format(i) for d in train_datasets for i in range(n_samples)]
    valid_dataset_names = [d + "_{}".format(i) for d in val_datasets for i in range(n_samples)]
    new_df_train = new_df.loc[new_df.index.isin(train_dataset_names)]
    new_df_valid = new_df.loc[new_df.index.isin(valid_dataset_names)]

    return new_df, new_df_train, new_df_valid


def export_df(
    df,
    experiment_group_dir,
    df_train=None,
    df_valid=None,
    file_name="perf_matrix",
    export_path=None
):
    train_file_name = file_name + "_train.pkl"
    valid_file_name = file_name + "_valid.pkl"
    file_name = file_name + ".pkl"

    if export_path is None:
        export_path = experiment_group_dir / "perf_matrix"

    export_path.mkdir(parents=True, exist_ok=True)

    df.to_pickle(path=export_path / file_name)
    df.to_csv(path_or_buf=(export_path / file_name).with_suffix(".csv"), float_format="%.5f")

    if df_train is not None:
        df_train.to_pickle(path=export_path / train_file_name)
        df_train.to_csv(
            path_or_buf=(export_path / train_file_name).with_suffix(".csv"), float_format="%.5f"
        )

    if df_valid is not None:
        df_valid.to_pickle(path=export_path / valid_file_name)
        df_valid.to_csv(
            path_or_buf=(export_path / valid_file_name).with_suffix(".csv"), float_format="%.5f"
        )

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--experiment_group", required=True, type=Path, help="Evaluation results dir")
    parser.add_argument("--output_savedir", type=str, default = '../../data/meta_dataset')
    parser.add_argument("--n_augmentations", type=int, default = 15)
    args = parser.parse_args()

    df = create_df_perf_matrix(args.experiment_group, args.n_augmentations)
    
    export_df(
        df=df,
        experiment_group_dir=args.experiment_group,
        df_train=None,
        df_valid=None,
        file_name="perf_matrix",
        export_path= Path(args.output_savedir)
    )
  