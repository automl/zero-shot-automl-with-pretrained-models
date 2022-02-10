import os
import json
import itertools

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
sns.set()

#font = {'weight': 'bold', 'size': 24}
#mpl.rc('font', **font)

def list_strip(list_str):
    list_str = list_str[1:-1]
    return [float(elem.strip()) for elem in list_str.split(',')]

def parse_n_collect_results(savepath = "./benchmark_results/results.json"):

    with open(savepath, "r") as f:
        soln_score_dict = json.load(f)
    
    #soln_score_dict = dict()

    for soln_id, soln_path in zip(SOLUTIONS, SOLUTION_PATHS):
        print(f"Solution: {soln_id}")

        if soln_id not in ['ekrem_generalist_old']:
            continue

        data_score_dict = dict()
        for d_id in ALL_DATASET_IDS:
            rep_score_dict = {"score": [],  "duration": [], "timestamps": [], "nauc": [], "accuracy": []}
            for i in range(N_REPETITIONS):
                score_path = os.path.join(soln_path, '-'.join([d_id, str(i)]), "score", "scores.txt")
                
                try:
                    f = open(score_path, 'r+')
                    info = [line.strip() for line in f.readlines()]
                    f.close()
                    values = [field.split(':')[-1].strip() for field in info]
                    values = values[:2]+[list_strip(values[i]) for i in range(2,5)]
                except:
                    values = [0.0, 0.0, [0.0], [0.0], [0.0]]
                    print(f"Solution: {soln_id} on {d_id} failed!")


                score_dict = {'score': float(values[0]), 
                              'duration': float(values[1]), 
                              'timestamps': values[2], 
                              'nauc': values[3], 
                              'accuracy': values[4]}

                score_path = os.path.join(soln_path, '-'.join([d_id, str(i)]), "score")
                if not os.path.isdir(score_path):
                    os.makedirs(score_path)
                
                score_path = os.path.join(score_path, "scores.json")      
                with open(score_path, 'w') as f:
                    json.dump(score_dict, f)

                rep_score_dict["score"].append(score_dict["score"])
                rep_score_dict["duration"].append(score_dict["duration"])
                rep_score_dict["timestamps"].append(score_dict["timestamps"])
                rep_score_dict["nauc"].append(score_dict["nauc"])
                rep_score_dict["accuracy"].append(score_dict["accuracy"])

            data_score_dict[d_id] = rep_score_dict

        soln_score_dict[soln_id] = data_score_dict

    with open(savepath, "w") as f:
        json.dump(soln_score_dict, f)

def get_incumbent(_list):
    current_incumbent = 0
    incumbent_list = []
    for e in _list:
        if e > current_incumbent:
            current_incumbent = e
        incumbent_list.append(current_incumbent)
    return incumbent_list

def load_n_report_results(loadpath = "./benchmark_results/results.json", reportpath = "./benchmark_results/results_summary.txt"):
    with open(loadpath, "r") as f:
        soln_score_dict = json.load(f)

    soln_summary = open(reportpath, "w+")
    soln_summary.write("\t".join(["Name", "Mean", "Std Dev", "NAUC", "Acc"])+"\n")
    soln_scores = dict(); soln_naucs = dict(); soln_accs = dict(); soln_scores_per_dataset = dict(); soln_std_per_dataset = dict() 
    unfolded_scores = dict(); unfolded_naucs = dict()
    soln_nauc_per_dataset = dict(); soln_acc_per_dataset = dict()

    soln_evolutions = dict()
    for solution, data_score_dict in soln_score_dict.items():

        print(f"Solution: {solution}")
        
        report_duration = 0
        dataset_score = list()
        dataset_duration = list()
        dataset_nauc = list()
        dataset_acc = list()
        dataset_evolution = list()
        for d_id, rep_score_dict in data_score_dict.items():
            score = rep_score_dict["score"]
            duration = rep_score_dict["duration"]
            timestamps = rep_score_dict["timestamps"]
            nauc = [s[-1] for s in rep_score_dict["nauc"]]
            acc = [s[-1] for s in rep_score_dict["accuracy"]]

            rep_evolution = []
            for ts, nauc_ts, acc_ts in zip(timestamps, rep_score_dict["nauc"], rep_score_dict["accuracy"]):
                run_evolution = np.zeros(1200)
                for t, nauc_t, acc_t in zip(ts, nauc_ts, acc_ts):
                    run_evolution[int(t)] = nauc_t
                run_evolution = get_incumbent(run_evolution)
                rep_evolution.append(run_evolution)

            dataset_score.append(score)
            dataset_duration.append(duration)
            dataset_nauc.append(nauc)
            dataset_acc.append(acc)
            report_duration += np.mean(duration)
            dataset_evolution.append(rep_evolution)

        dataset_score = np.stack(dataset_score)
        dataset_duration = np.stack(dataset_duration)
        dataset_nauc = np.stack(dataset_nauc)
        dataset_acc = np.stack(dataset_acc)
        dataset_evolution = np.stack(dataset_evolution)

        solution_score = np.mean(dataset_score, axis = 0)
        per_dataset_score = np.mean(dataset_score, axis = 1)
        per_dataset_std = np.std(dataset_score, axis = 1)
        solution_mean, solution_std = np.mean(solution_score), np.std(solution_score)
        solution_evolution = np.mean(dataset_evolution, axis = 0)

        soln_nauc = np.mean(dataset_nauc, axis = 0)
        per_dataset_nauc = np.mean(dataset_nauc, axis = 1)
        soln_acc = np.mean(dataset_acc, axis = 0)
        per_dataset_acc = np.mean(dataset_acc,  axis = 1)

        print(f"Solution {solution.upper()} mean score: {solution_mean} -/+ {solution_std}")
        soln_summary.write("\t".join([solution.upper(), str(round(solution_mean, 4)), str(round(solution_std, 4)), str(round(np.mean(soln_nauc), 4)), str(round(np.mean(soln_acc), 4))]))
        soln_summary.write("\n")

        
        soln_scores[solution] = solution_score
        soln_naucs[solution] = soln_nauc
        soln_accs[solution] = soln_acc
        soln_scores_per_dataset[solution] = per_dataset_score
        soln_std_per_dataset[solution] = per_dataset_std
        soln_nauc_per_dataset[solution] = per_dataset_nauc
        soln_acc_per_dataset[solution] = per_dataset_acc

        unfolded_scores[solution] = dataset_score
        unfolded_naucs[solution] = dataset_nauc

        soln_evolutions[solution] = solution_evolution

        print(f"Average time taken for a run: {round(report_duration/5250, 2)} sec")

    soln_summary.close()

    return soln_scores, soln_naucs, soln_accs, soln_scores_per_dataset, soln_std_per_dataset, soln_nauc_per_dataset, soln_acc_per_dataset, unfolded_scores, unfolded_naucs, soln_evolutions

def benchmarking_boxplot(soln_scores, solutions, solution_labels, savepath = "./benchmark_results/benchmarking_boxplot.png", figsize = (6, 9), xlabel = "ALC score"):

    ordered_soln_scores = []
    for solution in solutions:
        ordered_soln_scores.append(soln_scores[solution])

    plt.figure(figsize=figsize)
    sns.boxplot(ordered_soln_scores, solution_labels, orient = "h", palette = "Set2")
    plt.xlabel(xlabel, size=24, weight = 'bold')
    plt.xticks(size=24, rotation = 45)
    plt.yticks(size=24, rotation = 0)

    plt.tight_layout()
    plt.savefig(savepath)

def benchmarking_violinplot(soln_scores, solutions, solution_labels, savepath = "./benchmark_results/benchmarking_violinplot.png", figsize = (6, 9)):

    ordered_soln_scores = []
    for solution in solutions:
        ordered_soln_scores.append(soln_scores[solution])

    ordered_soln_scores = pd.DataFrame(ordered_soln_scores, columns = np.arange(525))
    ordered_soln_scores['solution'] = solution_labels
    ordered_soln_scores = pd.melt(ordered_soln_scores, id_vars="solution", var_name="repetition", value_name="mean_per_dataset")

    plt.figure(figsize=figsize)
    sns.violinplot(x = "mean_per_dataset", y = "solution", data = ordered_soln_scores) # , inner = "point", cut = 0
    plt.xlabel("ALC score", size=24, weight = 'bold')
    plt.xticks(size=24, rotation = 45)
    plt.yticks(size=24, rotation = 0)
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(savepath)

def datasetwise_comparison_scatterplot(soln_scores_per_dataset, soln_std_per_dataset, savepath = "./benchmark_results/comparison_scatterplots", bar_width = 0.7):

    solutions = COMPETITIVE_SOLUTIONS+STRONG_BASELINES+WINNER_BASELINES
    labels = COMPETITIVE_LABELS+STRONG_LABELS +WINNER_LABELS
    from_solutions = COMPETITIVE_SOLUTIONS+STRONG_BASELINES
    from_labels = COMPETITIVE_LABELS+STRONG_LABELS 
    to_solutions = WINNER_BASELINES
    to_labels = WINNER_LABELS
    
    if os.path.exists(savepath):
        os.system(f"rm -rf {savepath}/*")

    _skip = len(to_labels)
    _rest = len(from_labels)
    our_comparison_idx = [i for i in range(0, _skip*_rest, _skip)]

    os.makedirs(savepath, exist_ok = True)

    comparison_scores = []
    sig_comparison_scores = []
    comparison_labels = []
    for (_, (soln_id_x, label_x)), (_, (soln_id_y, label_y)) in itertools.combinations(enumerate(zip(solutions, labels)), 2):

        print(label_x, label_y)

        scores_x = soln_scores_per_dataset[soln_id_x]
        scores_y = soln_scores_per_dataset[soln_id_y]
        std_x = soln_std_per_dataset[soln_id_x]
        std_y = soln_std_per_dataset[soln_id_y]
        comparison = [x > y for x, y in zip(scores_x, scores_y)]
        x_better = sum(comparison)
        y_better = 525-sum(comparison)

        x_sig_better = sum([x-xp > y+yp for x, y, xp, yp in zip(scores_x, scores_y, std_x, std_y)])
        y_sig_better = sum([y-yp > x+xp for x, y, xp, yp in zip(scores_x, scores_y, std_x, std_y)])

        scores = np.log(1-np.concatenate([scores_x, scores_y]))
        mean = scores.mean()
        std = scores.std()
        scores = (scores-mean)/std
        _min = min(scores)
        _max = max(scores)
        scores_x = scores[:len(scores_x)]
        scores_y = scores[len(scores_x):]
        
        if (label_x in to_labels) or (label_y in to_labels):
            comparison_scores.append([x_better, y_better])
            sig_comparison_scores.append([x_sig_better, y_sig_better])
            comparison_labels.append(f"A.{label_x}\nB.{label_y}")
        
        print(f"{soln_id_x}:{x_better} and {soln_id_y}:{y_better}")

        plt.figure(figsize=(15, 15))
        plt.scatter(-scores_x, -scores_y, c = "blue", marker = ".", alpha = 0.6)
        plt.plot([-_min, -_max], [-_min, -_max], c = "red")
        plt.xlabel(f"{label_x} (Score)")
        plt.ylabel(f"{label_y} (Score)")
        plt.title(f"Datasetwise comparison between {label_x}:{x_better} and {label_y}:{y_better}")
        plt.tight_layout()
        plt.savefig(os.path.join(savepath, f"{soln_id_x}-{soln_id_y}.png"))
        plt.clf()

    comparison_scores = np.stack(comparison_scores)
    comparison_labels = np.stack(comparison_labels)
    
    for i, s in enumerate(from_labels):
        fig, ax = plt.subplots()
        fig.set_size_inches(7, 5)

        p1 = ax.bar(comparison_labels[i*_skip:(i+1)*_skip], comparison_scores[:,1][i*_skip:(i+1)*_skip], bar_width, label='B')
        p2 = ax.bar(comparison_labels[i*_skip:(i+1)*_skip], comparison_scores[:,0][i*_skip:(i+1)*_skip], bar_width, bottom=comparison_scores[:,1][i*_skip:(i+1)*_skip], label='A')
        ax.axhline(262.5, color = "red")
        ax.set_ylabel('Number of better solutions')
        ax.set_title(f'{s} by pairwise dataset performances')
        ax.bar_label(p1, label_type='center')
        ax.bar_label(p2, label_type='center')
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(savepath, f"pairwise_performance_bar_chart_{s}.png")) 

    '''
    fig, ax = plt.subplots()
    fig.set_size_inches(35, 10)

    p1 = ax.bar(comparison_labels[our_comparison_idx], comparison_scores[:,1][our_comparison_idx], bar_width, label='B')
    p2 = ax.bar(comparison_labels[our_comparison_idx], comparison_scores[:,0][our_comparison_idx], bar_width, bottom=comparison_scores[:,1][our_comparison_idx], label='A')
    ax.axhline(262.5, color = "red")
    ax.set_ylabel('Number of better solutions')
    ax.set_title('Solutions by pairwise dataset performances')
    ax.bar_label(p1, label_type='center')
    ax.bar_label(p2, label_type='center')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(savepath, "pairwise_performance_bar_chart.png")) 
    '''
    sig_comparison_scores = np.stack(sig_comparison_scores)

    for i, s in enumerate(from_labels):
        fig, ax = plt.subplots()
        fig.set_size_inches(7, 5)

        p1 = ax.bar(comparison_labels[i*_skip:(i+1)*_skip], sig_comparison_scores[:,1][i*_skip:(i+1)*_skip], bar_width, label='B')
        p2 = ax.bar(comparison_labels[i*_skip:(i+1)*_skip], sig_comparison_scores[:,0][i*_skip:(i+1)*_skip], bar_width, bottom=sig_comparison_scores[:,1][i*_skip:(i+1)*_skip], label='A')
        ax.set_ylabel('Number of better solutions')
        ax.set_title(f'{s} by significant pairwise dataset performances')
        ax.bar_label(p1, label_type='center')
        ax.bar_label(p2, label_type='center')
        ax.legend()
        plt.xticks(fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(savepath, f"significant_pairwise_performance_bar_chart_{s}.png")) 

    '''
    fig, ax = plt.subplots()
    fig.set_size_inches(35, 10)

    p1 = ax.bar(comparison_labels[our_comparison_idx], sig_comparison_scores[:,1][our_comparison_idx], bar_width, label='B')
    p2 = ax.bar(comparison_labels[our_comparison_idx], sig_comparison_scores[:,0][our_comparison_idx], bar_width, bottom=sig_comparison_scores[:,1][our_comparison_idx], label='A')
    ax.set_ylabel('Number of better solutions')
    ax.set_title('Solutions by significant pairwise dataset performances')
    ax.bar_label(p1, label_type='center')
    ax.bar_label(p2, label_type='center')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(savepath, "significant_pairwise_performance_bar_chart.png"))
    '''

def ranking_old(soln_scores_per_dataset, soln_nauc_per_dataset, soln_acc_per_dataset, main_solutions, main_solution_labels, savepath = "./benchmark_results/ranking_summary.txt"):
    f = open(savepath, "w+")
    for soln, lbl in zip(main_solutions, main_solution_labels):
        solutions = [soln]+WINNER_BASELINES
        solution_labels = [lbl]+WINNER_LABELS

        soln_ranking_per_dataset = []
        for idx, dataset_id in enumerate(ALL_DATASET_IDS):
            soln_scores = [1-soln_scores_per_dataset[soln][idx] for soln in solutions]
            sorted_solution_idx = np.argsort(soln_scores)
            rankings = np.argsort(sorted_solution_idx)+1
            soln_ranking_per_dataset.append(rankings)

        soln_nauc_ranking_per_dataset = []
        for idx, dataset_id in enumerate(ALL_DATASET_IDS):
            soln_scores = [1-soln_nauc_per_dataset[soln][idx] for soln in solutions]
            sorted_solution_idx = np.argsort(soln_scores)
            rankings = np.argsort(sorted_solution_idx)+1
            soln_nauc_ranking_per_dataset.append(rankings)

        '''
        soln_acc_ranking_per_dataset = []
        for idx, dataset_id in enumerate(ALL_DATASET_IDS):
            soln_scores = [1-soln_acc_per_dataset[soln][idx] for soln in solutions]
            sorted_solution_idx = np.argsort(soln_scores)
            rankings = np.argsort(sorted_solution_idx)+1
            soln_acc_ranking_per_dataset.append(rankings)
        '''

        soln_ranking_per_dataset = np.stack(soln_ranking_per_dataset)
        soln_rankings = soln_ranking_per_dataset.mean(0)

        soln_nauc_ranking_per_dataset = np.stack(soln_nauc_ranking_per_dataset)
        soln_nauc_rankings = soln_nauc_ranking_per_dataset.mean(0)

        #soln_acc_ranking_per_dataset = np.stack(soln_acc_ranking_per_dataset)
        #soln_acc_rankings = soln_acc_ranking_per_dataset.mean(0)

        
        for soln, rank_alc, rank_nauc in zip(solution_labels, soln_rankings, soln_nauc_rankings):
            f.write(f"Solution: {soln.upper()} rank ALC: {round(rank_alc, 2)} rank NAUC: {round(rank_nauc, 2)}")
            f.write('\n')
        f.write("---------------------------------------------------------------------------------------------------------")
        f.write("\n")
    
    f.close()

def ranking_all(soln_scores_per_dataset, soln_nauc_per_dataset, soln_acc_per_dataset, main_solutions, main_solution_labels, savepath = "./benchmark_results/ranking_summary.txt"):
    f = open(savepath, "w+")
    
    solutions = main_solutions+WINNER_BASELINES
    solution_labels = main_solution_labels+WINNER_LABELS

    soln_ranking_per_dataset = []
    for idx, dataset_id in enumerate(ALL_DATASET_IDS):
        soln_scores = [1-soln_scores_per_dataset[soln][idx] for soln in solutions]
        sorted_solution_idx = np.argsort(soln_scores)
        rankings = np.argsort(sorted_solution_idx)+1
        soln_ranking_per_dataset.append(rankings)

    soln_nauc_ranking_per_dataset = []
    for idx, dataset_id in enumerate(ALL_DATASET_IDS):
        soln_scores = [1-soln_nauc_per_dataset[soln][idx] for soln in solutions]
        sorted_solution_idx = np.argsort(soln_scores)
        rankings = np.argsort(sorted_solution_idx)+1
        soln_nauc_ranking_per_dataset.append(rankings)

    soln_acc_ranking_per_dataset = []
    for idx, dataset_id in enumerate(ALL_DATASET_IDS):
        soln_scores = [1-soln_acc_per_dataset[soln][idx] for soln in solutions]
        sorted_solution_idx = np.argsort(soln_scores)
        rankings = np.argsort(sorted_solution_idx)+1
        soln_acc_ranking_per_dataset.append(rankings)

    soln_ranking_per_dataset = np.stack(soln_ranking_per_dataset)
    soln_rankings = soln_ranking_per_dataset.mean(0)

    soln_nauc_ranking_per_dataset = np.stack(soln_nauc_ranking_per_dataset)
    soln_nauc_rankings = soln_nauc_ranking_per_dataset.mean(0)

    soln_acc_ranking_per_dataset = np.stack(soln_acc_ranking_per_dataset)
    soln_acc_rankings = soln_acc_ranking_per_dataset.mean(0)

    for soln, rank_alc, rank_nauc, rank_acc in zip(solution_labels, soln_rankings, soln_nauc_rankings, soln_acc_rankings):
        f.write(f"Solution: {soln.upper()} rank ALC: {round(rank_alc, 2)} rank NAUC: {round(rank_nauc, 2)} rank ACC: {round(rank_acc, 2)}")
        f.write('\n')
    f.write("---------------------------------------------------------------------------------------------------------")
    f.write("\n")
    
    f.close()

def ranking(unfolded_scores, unfolded_naucs, main_solutions, main_solution_labels, savepath = "./benchmark_results/ranking_summary.txt"):
    f = open(savepath, "w+")
    for soln, lbl in zip(main_solutions, main_solution_labels):
        solutions = [soln]+WINNER_BASELINES
        solution_labels = [lbl]+WINNER_LABELS

        soln_alc_per_dataset = []
        for idx, dataset_id in enumerate(ALL_DATASET_IDS):
            rankings = []
            for r in range(10):
                rep_scores = [1-unfolded_scores[soln][idx][r] for soln in solutions]
                sorted_rep_idx = np.argsort(rep_scores)
                rep_rankings = np.argsort(sorted_rep_idx)+1
                rankings.append(rep_rankings)

            soln_alc_per_dataset.append(rankings)

        soln_alc_per_dataset = np.stack(soln_alc_per_dataset)
        soln_alc_per_rep = soln_alc_per_dataset.mean(0)
        soln_alc_rankings = soln_alc_per_rep.mean(0)
        soln_alc_stds = soln_alc_per_rep.std(0)

        soln_nauc_ranking_per_dataset = []
        for idx, dataset_id in enumerate(ALL_DATASET_IDS):
            rankings = []
            for r in range(10):
                rep_scores = [1-unfolded_naucs[soln][idx][r] for soln in solutions]
                sorted_rep_idx = np.argsort(rep_scores)
                rep_rankings = np.argsort(sorted_rep_idx)+1
                rankings.append(rep_rankings)
            
            soln_nauc_ranking_per_dataset.append(rankings)

        soln_nauc_ranking_per_dataset = np.stack(soln_nauc_ranking_per_dataset)
        soln_nauc_per_rep = soln_nauc_ranking_per_dataset.mean(0)
        soln_nauc_rankings = soln_nauc_per_rep.mean(0)
        soln_nauc_stds = soln_nauc_per_rep.std(0)

        for soln, rank_alc, std_alc, rank_nauc, std_nauc in zip(solution_labels, soln_alc_rankings, soln_alc_stds, soln_nauc_rankings, soln_nauc_stds):
            f.write(f"Solution: {soln.upper()} rank ALC: {round(rank_alc, 2)}+/-{round(std_alc, 2)} rank NAUC: {round(rank_nauc, 2)}+/-{round(std_nauc, 2)}")
            f.write('\n')
        f.write("---------------------------------------------------------------------------------------------------------")
        f.write("\n")
    
    f.close()

def evolution_plot(soln_evolutions, solutions, solution_labels, savepath = "results/evolution_plot.png"):
    ordered_soln_evolutions = []
    for solution in solutions:
        ordered_soln_evolutions.append(soln_evolutions[solution].mean(0))

    plt.figure(figsize=(10, 10))
    for solution, label in zip(solutions, solution_labels):
        evolution_vector = soln_evolutions[solution].mean(0)[:60]
        evolution_error = soln_evolutions[solution].std(0)[:60]
        plt.step(np.arange(60), evolution_vector, label = solution)
        plt.fill_between(np.arange(60), evolution_vector-evolution_error, evolution_vector+evolution_error, step='pre', alpha = 0.2)
        #plt.errorbar(np.arange(60), evolution_vector, yerr = evolution_error, label = solution)
    plt.xlabel("Time (seconds)", size=24)
    plt.ylabel("ALC score", size = 24)
    plt.xticks(size=24)
    plt.yticks(size=24)
    plt.ylim([0.70, 1.])
    plt.legend(loc = "best")
    plt.tight_layout()
    plt.savefig(savepath)

if __name__ == '__main__':

    RUN_RESULTS_DIR = "../benchmark_run_results/"
    RESULTS_DIR = "./benchmark_results"

    # Solution folder names
    COMPETITIVE_SOLUTIONS = ['ZAP-AS', 'ZAP-HPO'] 
    RANDOM_BASELINES = ['Random-selection-I', 'Random-selection-II', 'Random-selection-III']
    WINNER_BASELINES = ['DeepWisdom', 'DeepBlueAI', 'PASA_NJU']
    STRONG_BASELINES = ['Single-best', 'Oracle']
    SPARSE_ABLATION_SOLUTIONS = ['ZAP-HPO-D25', 'ZAP-HPO-D50', 'ZAP-HPO-D75']
    
    # Labels for plots
    COMPETITIVE_LABELS = ['ZAP-AS','ZAP-HPO']
    RANDOM_LABELS = ['Random I', 'Random II', 'Random III'] 
    WINNER_LABELS = ['DeepWisdom', 'DeepBlueAI', 'PASA-NJU']
    STRONG_LABELS = ['Single-best', 'Oracle']
    SPARSE_ABLATION_LABELS = ['ZAP-HPO (75%)', 'ZAP-HPO (50%)', 'ZAP-HPO (25%)']

    SOLUTIONS = COMPETITIVE_SOLUTIONS+WINNER_BASELINES+SPARSE_ABLATION_SOLUTIONS+RANDOM_BASELINES+STRONG_BASELINES
    LABELS = COMPETITIVE_LABELS+WINNER_LABELS+SPARSE_ABLATION_LABELS+RANDOM_LABELS+STRONG_LABELS

    ALL_DATASETS = ['cifar100', 'cycle_gan_vangogh2photo', 'uc_merced', 'cifar10', 'cmaterdb_devanagari', 
                    'cmaterdb_bangla', 'mnist', 'horses_or_humans', 'kmnist', 'cycle_gan_horse2zebra', 'cycle_gan_facades', 
                    'cycle_gan_apple2orange', 'imagenet_resized_32x32', 'cycle_gan_maps', 'omniglot', 'imagenette', 'emnist_byclass', 
                    'svhn_cropped', 'colorectal_histology', 'coil100', 'stanford_dogs', 'rock_paper_scissors', 'tf_flowers', 
                    'cycle_gan_ukiyoe2photo', 'cassava', 'fashion_mnist', 'emnist_mnist', 'cmaterdb_telugu', 'malaria', 'eurosat_rgb', 
                    'emnist_balanced', 'cars196', 'cycle_gan_iphone2dslr_flower', 'cycle_gan_summer2winter_yosemite', 'cats_vs_dogs']


    SOLUTION_PATHS = [os.path.join(RUN_RESULTS_DIR, s) for s in SOLUTIONS]
    N_AUGMENTATIONS = 15
    ALL_DATASET_IDS = [str(n)+"-"+dataset_name for n in range(N_AUGMENTATIONS) for dataset_name in ALL_DATASETS]
    N_REPETITIONS = 10

    parse_n_collect_results()
    soln_scores, soln_naucs, soln_accs, soln_scores_per_dataset, soln_std_per_dataset, soln_nauc_per_dataset, soln_acc_per_dataset, unfolded_scores, unfolded_naucs, soln_evolutions = load_n_report_results()

    datasetwise_comparison_scatterplot(soln_scores_per_dataset, soln_std_per_dataset, bar_width = 0.5)
    
    solutions = COMPETITIVE_SOLUTIONS+WINNER_BASELINES
    solution_labels = COMPETITIVE_LABELS+WINNER_LABELS
    evolution_plot(soln_evolutions, solutions, solution_labels)

    
    solutions = RANDOM_BASELINES+WINNER_BASELINES+[STRONG_BASELINES[0]]+COMPETITIVE_SOLUTIONS+SPARSE_ABLATION_SOLUTIONS+[STRONG_BASELINES[1]]
    solution_labels = RANDOM_LABELS+WINNER_LABELS+[STRONG_LABELS[0]]+COMPETITIVE_LABELS+SPARSE_ABLATION_LABELS+[STRONG_LABELS[1]]
    benchmarking_boxplot(soln_scores, solutions, solution_labels, f"{RESULTS_DIR}/benchmarking_boxplot/all.png", (9, 18))
    benchmarking_boxplot(soln_naucs, solutions, solution_labels, f"{RESULTS_DIR}/benchmarking_boxplot/all_nauc.png", (9, 18), 'NAUC Score')
    benchmarking_boxplot(soln_accs, solutions, solution_labels, f"{RESULTS_DIR}/benchmarking_boxplot/all_acc.png", (9, 18), 'Multi-classification Accuracy')
    benchmarking_violinplot(soln_scores_per_dataset, solutions, solution_labels, f"{RESULTS_DIR}/benchmarking_violinplot/all.png")

    solutions = COMPETITIVE_SOLUTIONS
    solution_labels = COMPETITIVE_LABELS
    benchmarking_boxplot(soln_scores, solutions, solution_labels, f"{RESULTS_DIR}/benchmarking_boxplot/main.png", (9, 4))
    benchmarking_violinplot(soln_scores_per_dataset, solutions, solution_labels, f"{RESULTS_DIR}/benchmarking_violinplot/main.png")

    solutions = WINNER_BASELINES+RANDOM_BASELINES+STRONG_BASELINES
    solution_labels = WINNER_LABELS+RANDOM_LABELS+STRONG_LABELS
    benchmarking_boxplot(soln_scores, solutions, solution_labels, f"{RESULTS_DIR}/benchmarking_boxplot/only_baselines.png", (9, 12))
    benchmarking_violinplot(soln_scores_per_dataset, solutions, solution_labels, f"{RESULTS_DIR}/benchmarking_violinplot/only_baselines.png")

    solutions = WINNER_BASELINES+COMPETITIVE_SOLUTIONS
    solution_labels = WINNER_LABELS+COMPETITIVE_LABELS
    benchmarking_boxplot(soln_scores, solutions, solution_labels, f"{RESULTS_DIR}/benchmarking_boxplot/best.png", (9, 10))
    benchmarking_violinplot(soln_scores_per_dataset, solutions, solution_labels, f"{RESULTS_DIR}/benchmarking_violinplot/best.png")

    solutions = COMPETITIVE_SOLUTIONS+SPARSE_ABLATION_SOLUTIONS
    solution_labels = COMPETITIVE_LABELS+SPARSE_ABLATION_LABELS
    benchmarking_boxplot(soln_scores, solutions, solution_labels, f"{RESULTS_DIR}/benchmarking_boxplot/sparse.png", (9, 7))
    benchmarking_violinplot(soln_scores_per_dataset, solutions, solution_labels, f"{RESULTS_DIR}/benchmarking_violinplot/sparse.png")

    ####################################

    ranking(unfolded_scores, unfolded_naucs, COMPETITIVE_SOLUTIONS, COMPETITIVE_LABELS, f"{RESULTS_DIR}/ranking_summary_competitive.txt")

    ranking(unfolded_scores, unfolded_naucs, SPARSE_ABLATION_SOLUTIONS, SPARSE_ABLATION_LABELS, f"{RESULTS_DIR}/ranking_summary_sparse.txt")

    ranking(unfolded_scores, unfolded_naucs, STRONG_BASELINES, STRONG_LABELS, f"{RESULTS_DIR}/ranking_summary_strong.txt")
