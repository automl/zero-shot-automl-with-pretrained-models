import os
import pandas as pd
import tqdm
import time
from config2vector import get_config_response, create_searchspace


def load_df():
    data_path = os.path.join('../../data', "data_m.csv")  # '../../data'
    data_path = 'data_m_gen.csv'
    data = pd.read_csv(data_path, header=0)

    for col in data:
        print(len(data[col].unique()))
    print("done")


# construct-> based on the meta features, per meta feature line in the meta-features.csv,
# merge metafeatures, perf matrix and config yaml.
def construct_csv(meta_feat, perf_mat, incumbent_list, save_as):
    # converting to dicts for easier handling
    meta_dict = [value for value in meta_feat.to_dict('index').values()]
    data_csv = []
    # Can it be faster??
    for features in tqdm.tqdm(meta_dict[:1]):
        # accuracy for testing on features['dataset] by incumbent eg  accuracy.loc[[incumbent]]
        for incumbent in incumbent_list:
            new_row = {**features, 'incumbent_of': incumbent}
            accuracy_list = perf_mat.loc[[new_row['dataset']]]
            accuracy = accuracy_list[incumbent].values.item()
            # -1 because rank sets the lowest to 1 and data_m.csv uses 0. False so highest score is ranked 1st
            ranks = accuracy_list.rank(axis=1, ascending=False, method='first') - 1
            # value returns as a series
            new_row['accuracy'] = accuracy
            # open the relevant yaml from the incumbent, (replace to get paths right)
            yaml_path = os.path.join(args.metadata_path, 'configs/kakaobrain_optimized_per_icgen_augmentation',
                                     incumbent.replace('-', '/') + '.yaml')
            hyper_params = create_searchspace(yaml_path)
            new_row = {**new_row, **hyper_params, 'ranks': int(ranks[incumbent].values.item())}
            data_csv.append(new_row)

    data_df = pd.DataFrame(data_csv)
    data_df.set_index(['dataset'])
    data_df.to_csv(save_as, index=False)

    print('stop here')


if __name__ == "__main__":
    start = time.time()
    import argparse

    parser = argparse.ArgumentParser(description='Generate data_m.csv')
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--metadata-path', type=str, default='meta_dataset', help='the path of meta_dataset directory')
    parser.add_argument('--save-path', type=str, default='data', help='save generated data_m.csv to this location')
    parser.add_argument('--save-file', type=str, default='data_m_gen.csv', help='name the generated data_m.csv')
    args = parser.parse_args()

    meta_features = pd.read_csv(os.path.join(os.getcwd(), args.metadata_path, 'meta_features.csv'))
    meta_features.columns.values[0] = "dataset"
    top_config = os.path.join(args.metadata_path, 'configs/kakaobrain_optimized_per_icgen_augmentation')
    all_yaml_paths, incumbent_of, perf_matrix = get_config_response(config_dir=top_config,
                                                                    response_dir=os.path.join(args.metadata_path,
                                                                                              'perf_matrix.csv'))
    construct_csv(meta_features, perf_matrix, sorted(incumbent_of), args.save_file)
    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
