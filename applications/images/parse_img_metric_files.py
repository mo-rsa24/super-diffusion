import os
import re
import ast
import numpy as np
import pandas as pd
from pathlib import Path
import json

with open("taskinfo.json") as f:
    TASKINFO = json.load(f)
print(TASKINFO)
TASKS = set([elem['dir_'] for elem in TASKINFO])
print("TASKS", TASKS)

def parse_tifa(methods, op):
    taskrank = {}
    def extract_final_dict(file_path):
        """Extracts the dictionary found after 'FINAL DICT' in a file."""
        dictionary = {}
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for i, line in enumerate(lines):
                if "FINAL DICT" in line:
                    next_line = lines[i + 1].strip()
                    dictionary = ast.literal_eval(next_line)
                    if isinstance(dictionary, dict):
                        return dictionary
    
        return {}
    
    def combine_dictionaries_from_directory(directory_path):
        """Loops over files in the directory and combines dictionaries found after 'FINAL DICT'."""
        combined_dict = {}
        np.random.seed(42)
        for root, _, files in os.walk(directory_path):
            for file in sorted(files):
                print(f"TASK:: {file}")
                if file not in taskrank:
                    taskrank[file] = {}
                file_path = os.path.join(root, file)
                file_dict = extract_final_dict(file_path)
                for method in methods: 
                    if method not in combined_dict:
                        combined_dict[method] = []
                    if method == "joint":
                        mod = ""
                        if "or" in op:
                            mod = "_or"
                        if op == "and":
                            results_ab = file_dict[f"sd_ab{mod}"]['min']
                            results_ba = file_dict[f"sd_ba{mod}"]['min']
                        elif op == "or_max":
                            results_ab = file_dict[f"sd_ab{mod}"]['max']
                            results_ba = file_dict[f"sd_ba{mod}"]['max']
                        elif op == "or_diff":
                            results_ab = np.abs(np.array(file_dict[f"sd_ab{mod}"]['max']) - np.array(file_dict[f"sd_ab{mod}"]['min']))
                            results_ba = np.abs(np.array(file_dict[f"sd_ba{mod}"]['max']) - np.array(file_dict[f"sd_ba{mod}"]['min']))
                        mean_ab = np.mean(results_ab)
                        mean_ba = np.mean(results_ba)
                        
                        if mean_ab >= mean_ba:
                            print("joint AB is better")
                            taskrank[file][method] = mean_ab
                            combined_dict[method].extend(results_ab)
                        else:
                            print("joint BA is better")
                            taskrank[file][method] = mean_ba
                            combined_dict[method].extend(results_ba)
                    elif method == "coin_flip":
                        if op == "or_max":
                            results_a = file_dict[f"sd_a"]['max']
                            results_b = file_dict[f"sd_b"]['max']
                        elif op == "or_diff":
                            results_a = np.abs(np.array(file_dict[f"sd_a"]['max']) - np.array(file_dict[f"sd_a"]['min']))
                            results_b = np.abs(np.array(file_dict[f"sd_b"]['max']) - np.array(file_dict[f"sd_b"]['min']))
                        coins = np.random.choice([0, 1], size=len(results_b))
                        task_results = [results_a[i] if coins[i] == 0 else results_b[i] for i in range(len(coins))]
                        combined_dict[method].extend(task_results)
                        taskrank[file][method] = np.mean(task_results)

                
                    else:
                        if op == "and":
                            combined_dict[method].extend(file_dict[method]['min'])
                            taskrank[file][method] = np.mean(file_dict[method]['min'])
                        elif op == "or_max":
                            combined_dict[method].extend(file_dict[method]['max'])
                            taskrank[file][method] = np.mean(file_dict[method]['max'])
                        elif op == "or_diff":
                            diff = np.abs(np.array(file_dict[method]['max']) - np.array(file_dict[method]['min']))
                            combined_dict[method].extend(diff)
                            taskrank[file][method] = np.mean(diff)

        return combined_dict
    
    if op == "and":
        directory_path = "log_min"
    else:
        directory_path = "log_tifa"
    combined_dict = combine_dictionaries_from_directory(directory_path)
    
    for method in methods: 
        print("METHOD:: {} | mean ± std:: {:.4f} ± {:.4f}".format(method, np.mean(combined_dict[method]), np.std(combined_dict[method])))
    counter_and = 0
    counter_avg = 0
    counter_joint = 0
    counter_none = 0
    for task in taskrank:
        if op == "and":
            if taskrank[task]['and'] > taskrank[task]['avg'] and taskrank[task]['and'] > taskrank[task]['joint']:
                counter_and += 1
            elif taskrank[task]['avg'] > taskrank[task]['and'] and taskrank[task]['avg'] > taskrank[task]['joint']:
                counter_avg += 1
            elif taskrank[task]['joint'] > taskrank[task]['and'] and taskrank[task]['joint'] > taskrank[task]['avg']:
                counter_joint += 1
            else:
                counter_none += 1
        elif "or" in op:
            if taskrank[task]['or'] > taskrank[task]['joint'] and taskrank[task]['or'] > taskrank[task]['coin_flip']:
                counter_and += 1
            elif taskrank[task]['joint'] > taskrank[task]['or'] and taskrank[task]['joint'] > taskrank[task]['coin_flip']:
                counter_joint += 1
            elif taskrank[task]['coin_flip'] > taskrank[task]['or'] and taskrank[task]['coin_flip'] > taskrank[task]['joint']:
                counter_avg += 1
            else:
                counter_none += 1
    if op == "and":
        print("GLOBAL RESULTS, tasks AND win {}".format(counter_and))
        print("GLOBAL RESULTS, tasks AVG win {}".format(counter_avg))
        print("GLOBAL RESULTS, tasks JOINT win {}".format(counter_joint))
        print("GLOBAL RESULTS, tasks NONE win {}".format(counter_none))
    elif "or" in op:
        print("GLOBAL RESULTS, tasks OR win {}".format(counter_and))
        print("GLOBAL RESULTS, tasks COIN FLIP win {}".format(counter_avg))
        print("GLOBAL RESULTS, tasks JOINT win {}".format(counter_joint))
        print("GLOBAL RESULTS, tasks NONE win {}".format(counter_none))


def parse_csv(path_, metric=None, op='and'):
    df = pd.read_csv(path_)
    if op == "and":
        col = f"min_{metric}"
        vals = df[col].values
    elif "or" in op:
        col_1 = f"{metric}_raw_score_1"
        vals_1 = np.array(df[col_1].values)

        col_2 = f"{metric}_raw_score_2"
        vals_2 = np.array(df[col_2].values)
        if op == "or_diff":
            vals = np.abs(vals_1 - vals_2)
        elif op == "or_max":
            vals = [max(a, b) for a, b in zip(vals_1, vals_2)]

    return vals

def get_csvs(rootdir):
    rootdir = Path(rootdir)
    csv_list = sorted(rootdir.glob('*.csv'))
    return csv_list

def parse_clip_or_ir(metric, methods, op="and"):
    taskrank = {}
    for method in methods:
        print("="*100)
        results = []
        assert len(TASKS) == 20
        np.random.seed(42)
        for task in sorted(TASKS):
            if method == "joint":
                mod = ""
                if "or" in op:
                    mod = "_or"
                dir_ = f"/projects/superdiff/saved_sd_results/metrics_sd_ab{mod}"
                csv_ab = Path(dir_) / f"metrics_sd_ab{mod}_{task}.csv"
                task_results_ab = parse_csv(csv_ab, metric, op)

                dir_ = f"/projects/superdiff/saved_sd_results/metrics_sd_ba{mod}"
                csv_ba = Path(dir_) / f"metrics_sd_ba{mod}_{task}.csv"
                task_results_ba = parse_csv(csv_ba, metric, op)
                if np.mean(task_results_ab) >= np.mean(task_results_ba):
                    print("AB CHOSEN")
                    task_results = task_results_ab
                else: 
                    print("BA CHOSEN")
                    task_results = task_results_ba
            elif method == "coin_flip":
                dir_ = f"/projects/superdiff/saved_sd_results/metrics_sd_a"
                csv_a = Path(dir_) / f"metrics_sd_a_{task}.csv"
                task_results_a = parse_csv(csv_a, metric, op)

                dir_ = f"/projects/superdiff/saved_sd_results/metrics_sd_b"
                csv_b = Path(dir_) / f"metrics_sd_b_{task}.csv"
                task_results_b = parse_csv(csv_b, metric, op)
                
                coins = np.random.choice([0, 1], size=len(task_results_b))
                task_results = [task_results_a[i] if coins[i] == 0 else task_results_b[i] for i in range(len(coins))]

            else:
                dir_ = f"/projects/superdiff/saved_sd_results/metrics_{method}"
                csv = Path(dir_) / f"metrics_{method}_{task}.csv"
                print("CURRENT TASK:", csv.name)
                task_results = parse_csv(csv, metric, op)
            print("METHOD:: {} | mean ± std:: {:.4f} ± {:.4f}".format(method, np.mean(task_results), np.std(task_results)))
            if task not in taskrank:
                taskrank[task] = {}
            taskrank[task][method] = np.mean(task_results)
            results.extend(task_results)
        assert len(results) == 400
        print("GLOBAL RESULTS, METHOD:: {} | mean ± std:: {:.4f} ± {:.4f}".format(method, np.mean(results), np.std(results)))

op="or_diff"
methods_and = ['and', 'avg', "joint"]
methods_or = ['or', "joint", "coin_flip"]

parse_clip_or_ir("clip", methods_or, op=op)
parse_clip_or_ir("ir", methods_or, op=op)
parse_tifa(methods_or, op)
