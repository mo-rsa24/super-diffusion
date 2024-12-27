# metrics script obtained from Alexander Tong
import pandas as pd
import shutil


import argparse
import numpy as np
import subprocess
import os
import itertools, re
import dataframe_image as dfi

from pathlib import Path
from tqdm import tqdm

MODEL_PATHS = {
        # output_model_name: path_to_generated_proteins
         "superposition_sde_500t_AND_logp0": "../superdiff/generated_proteins/superposition_sde_500t_AND_logp0",
        }

DESIGNABILITY_RMSD_THRESH = 2
TM_SCORE_THRESH = 0.5
MAXCLUSTER_PATH="/projects/superdiff/maxcluster64bit"

FOLDSEEK_PATH="/projects/superdiff/foldseek"
FOLDSEEK_PDB_DB_PATH="/projects/superdiff/foldseek_pdb_db/pdb"

RESULTS_DIR="../superdiff/results_check/"
SAVE_FILE="combined_data"
os.makedirs(RESULTS_DIR, exist_ok=True)

dssp_to_abc = {
    "I": "a",
    "H": "a",
    "G": "a",
    "E": "b",
    "B": "b",
    "S": "c",
    "T": "c",
    "C": "c",
    "": "c",
}


def compute_ss_ratios(pdb_paths, tmp_dir=None, paths=None):
    helix = []
    sheet = []
    coil = []
    tmp_dir = None
    for pdb in tqdm(pdb_paths):
        dssp_output = "dssp_temp.txt"
        if tmp_dir is not None:
            dssp_output = os.path.join(tmp_dir, dssp_output)
        try:
            # Run DSSP
            subprocess.run(["mkdssp", "-i", pdb, "-o", dssp_output])

            # Initialize variables
            sse_start = None
            sse_list = []
            with open(dssp_output, "r") as f:
                for i, line in enumerate(f):
                    # Find the SSE start line
                    if sse_start is None and line.startswith(
                        "  #  RESIDUE AA STRUCTURE"
                    ):
                        sse_start = i + 1
                        continue

                    # If we've reached SSE records
                    if sse_start is not None and len(line) > 0 and line[13] != "!":
                        sse_list.append(line[16])

            # Convert to numpy array
            sse = np.array(sse_list, dtype="U1")
            sse[sse == " "] = "C"

            os.remove(dssp_output)
            sse = np.array([dssp_to_abc[e] for e in sse], dtype="U1")
            helix_count = np.sum(sse == "a")
            sheet_count = np.sum(sse == "b")
            coil_count = np.sum(sse == "c")
            total_count = len(sse)
            helix.append(helix_count / total_count)
            sheet.append(sheet_count / total_count)
            coil.append(coil_count / total_count)

        except Exception as e:
            # raise e
            print(pdb)
            print(e)
            helix.append(-1)
            sheet.append(-1)
            coil.append(-1)

    return helix, sheet, coil

def compute_structural_novelty(
    df, result_file, tmp_dir, pdb_path="esm_path"
):

    df = df[df["rmsd"] < DESIGNABILITY_RMSD_THRESH]

    tmp_pdb_dir = os.path.join(tmp_dir, "tmp_pdbs")
    os.makedirs(tmp_pdb_dir, exist_ok=True)

    # Store mapping of temp file names to their original file names
    temp_to_original = {}
    for idx, pdb in enumerate(df[pdb_path].tolist()):
        if os.path.exists(pdb):
            # Prepend the index to the filename to ensure uniqueness
            temp_name = f"{idx}_{os.path.basename(pdb)}"
            temp_to_original[temp_name] = pdb
            shutil.copy(pdb, os.path.join(tmp_pdb_dir, temp_name))
        else:
            print(pdb)
            print(f"File {pdb} not found!")
    subprocess.run(
        [
            "foldseek",
            "easy-search",
            tmp_pdb_dir,
            FOLDSEEK_PDB_DB_PATH,
            result_file,
            tmp_dir,
            "--format-output",
            "query,target,alntmscore",
        ],
        stdout=subprocess.PIPE,
    )
    shutil.rmtree(tmp_pdb_dir)
    try:
        # Dictionary to store tmscores with original pdb as key
        scores_dict = {}
        with open(result_file, "r") as f:
            for line in f:
                columns = line[:-1].split("\t")
                temp_name = columns[0] + ".pdb"
                original_name = temp_to_original[temp_name]
                if original_name not in scores_dict:
                    scores_dict[original_name] = float(columns[2])
    except Exception as e:
        print(e)

    # Create a new dataframe from scores_dict
    scores_df = pd.DataFrame(
        list(scores_dict.items()), columns=[pdb_path, "novelty_tmscore"]
    )

    # Merge on the pdb_path
    df = df.merge(scores_df, on=pdb_path, how="left")
    return df



def run_maxcluster(
    pdb_list, cluster_threshold=0.8, clustering_method=3
):
    temp_pdb_file = "temp_pdb_list.txt"
    clustering_file = "temp_clustering.txt"

    with open(temp_pdb_file, "w") as f:
        for pdb in pdb_list:
            f.write(pdb + "\n")

    with open(clustering_file, "w") as fd:
        subprocess.run(
            [
                MAXCLUSTER_PATH,
                "-l",
                temp_pdb_file,
                "-C",
                str(clustering_method),
                "-Tm",
                str(cluster_threshold),
            ],
            stdout=fd,
        )

    with open(clustering_file, "r") as f:
        for line in f:
            if line.startswith("INFO  :") and "Clusters @ Threshold" in line:
                clusters = int(line.split()[2])
                os.remove(temp_pdb_file)
                os.remove(clustering_file)
                return clusters / len(pdb_list)

def pairwise_tm_score(pdb_list, tmp_dir):
    scores = []
    for pdb1, pdb2 in itertools.combinations(pdb_list, 2):
        result = subprocess.run(
            [MAXCLUSTER_PATH, pdb1, pdb2, "-in"], stdout=subprocess.PIPE
        )
        output = result.stdout.decode()

        # Extract the TM score from the MaxCluster output
        match = re.search(r"TM=(\d+\.\d+)", output)
        if match:
            tm_score = float(match.group(1))
            scores.append(tm_score)
        else:
            raise ValueError(
                f"Could not extract TM score from MaxCluster output for PDBs {pdb1} and {pdb2}"
            )
    return np.array(scores) if scores else np.array(0)

def average_pairwise_tm_score(pdb_list, tmp_dir):
    scores = pairwise_tm_score(pdb_list, tmp_dir)
    return scores.mean(), scores.std()

def compute_diversity_by_bin(
    df,
    cluster_threshold=0.8,
    clustering_method=3,
    tmp_dir=None,
):
    df = df[df["rmsd"] < DESIGNABILITY_RMSD_THRESH]

    lengths = df["seq_length"].unique()
    results = []

    for length in lengths:
        length_df = df[df["seq_length"] == length]
        maxcluster_diversity = run_maxcluster(
            length_df["pdb_path"].tolist(),
            cluster_threshold,
            clustering_method,
        )
        avg_tm_score, std_tm_score = average_pairwise_tm_score(
            length_df['pdb_path'].tolist(), tmp_dir
        )
        results.append((length, maxcluster_diversity, avg_tm_score, std_tm_score))

    result_df = pd.DataFrame(
        results,
        columns=[
            "seq_length",
            "MaxCluster Diversity",
            "Avg Pairwise TM-score",
            "STD Pairwise TM-score",
        ],
    )
    return result_df

def get_min_rmsd(df):
    df = df.drop(index=0)
    min_rmsd_index = df['rmsd'].idxmin()
    min_rmsd = df.loc[min_rmsd_index, 'rmsd']
    return min_rmsd_index, min_rmsd

def get_max_tmscore(df):
    df = df.drop(index=0)
    max_tm_index = df['tm_score'].idxmax()
    max_tm = df.loc[max_tm_index, 'tm_score']
    return max_tm_index, max_tm

def get_protein_data(model_path):
    data = {
        "esm_path": [],
        "sequence": [],
        "tm_score": [],
        "rmsd": [],
        "seq_length": [],
        "pdb_path": [],
    }
    directory_list_of_lengths = sorted(model_path.glob('*/self_consistency'))
    for length_dir in directory_list_of_lengths:
        print(f"Getting data from........... {length_dir}")
        csv_list = sorted(length_dir.glob('*/*.csv'))
        csv_list = [path_ for path_ in csv_list if int(str(path_).split("/")[-2].split("_")[-1]) < 50] # keep only samples 0-49
        assert len(csv_list) == 50, len(csv_list)
        for csv in tqdm(csv_list, colour='blue'):
            df = pd.read_csv(csv, index_col=0)
            min_rmsd_index, min_rmsd = get_min_rmsd(df)
            tm_score = df.loc[min_rmsd_index, 'tm_score']
            esm_path = df.loc[min_rmsd_index, 'sample_path']
            if "test" in str(model_path):
                esm_path = esm_path.replace("composition_attempt_scoreoutput_negll_2", "test")
            print(model_path, esm_path)
            sequence = df.loc[min_rmsd_index, 'sequence']
            seq_length = len(sequence)

            pdb_path =  csv.parent
            pdb_path = pdb_path / (pdb_path.stem +  ".pdb")
            pdb_path = str(pdb_path)

            data['rmsd'].append(min_rmsd)
            data['tm_score'].append(tm_score)
            data['esm_path'].append(esm_path)
            data['sequence'].append(sequence)
            data['seq_length'].append(seq_length)
            data['pdb_path'].append(pdb_path)
    data = pd.DataFrame(data)
    return data 

def compute_raw_scores():
    tmp_dir = "./tmp"
    all_data, all_diversities, all_novelties = [], [], []
    for model, model_path in MODEL_PATHS.items():
        model_path = Path(model_path)
        print(f"Loading from....... {model_path}")
        model_data = get_protein_data(model_path)
        all_data.append(model_data)

        # Diversity
        diversity = compute_diversity_by_bin(model_data)
        print("diversity done")
        print(diversity)

        # novelty
        novelty_result_file = os.path.join(tmp_dir, "novelty_alignment.m8")
        novelty = compute_structural_novelty(
            model_data,
            novelty_result_file,
            tmp_dir,
        )
        print("novelty done")

        model_data["model"] = model
        diversity["model"] = model
        novelty["model"] = model

        (model_data["alpha_ratio"],
         model_data["beta_ratio"],
         model_data["loop_ratio"]) = compute_ss_ratios(model_data["pdb_path"], tmp_dir=tmp_dir)

        all_data.append(model_data)
        all_diversities.append(diversity)
        all_novelties.append(novelty)

    shutil.rmtree(tmp_dir)


    data = pd.concat(all_data, ignore_index=True)
    div = pd.concat(all_diversities, ignore_index=True)
    nov = pd.concat(all_novelties, ignore_index=True)


    data = data[(data["seq_length"].isin(range(100, 301, 50)))]
    nov = nov[(nov["seq_length"].isin(range(100, 301, 50)))]
    div = div[(div["seq_length"].isin(range(100, 301, 50)))]

    data.to_csv(f"{RESULTS_DIR}/{SAVE_FILE}.csv")
    nov.to_csv(f"{RESULTS_DIR}/{SAVE_FILE}_with_novelty.csv")
    div.to_csv(f"{RESULTS_DIR}/{SAVE_FILE}_diversity.csv")

    return data, nov, div

def compute_weighted_avg(df, seq_counts):
    # Merge the counts into df2
    df = pd.merge(df, seq_counts, on=["seq_length", "model"])

    # Compute the weighted average
    df["weighted_MaxCluster Diversity"] = (
        df["MaxCluster Diversity"] * df["count"]
    )
    weighted_maxcluster_avg = (
        df.groupby("model")["weighted_MaxCluster Diversity"].sum()
        / df.groupby("model")["count"].sum()
    )

    weighted_maxcluster_avg.name = "Avg Maxcluster"

    df["weighted_Avg Pairwise TM-score"] = (
        df["Avg Pairwise TM-score"] * df["count"]
    )
    weighted_tm_score = (
        df.groupby("model")["weighted_Avg Pairwise TM-score"].sum()
        / df.groupby("model")["count"].sum()
    )

    weighted_tm_score = (
        df.groupby("model")["weighted_Avg Pairwise TM-score"].sum()
        / df.groupby("model")["count"].sum()
    )

    weighted_tm_score.name = "Avg TM"

    return weighted_maxcluster_avg, weighted_tm_score

def format_results(data_df, novelty_df, diversity_df, weighted_maxcluster_df, weighted_tm_score_df):

    novelty_mean_df = novelty_df.groupby(["model"])["novelty_tmscore"].mean()
    novelty_sem_df = novelty_df.groupby(["model"])["novelty_tmscore"].sem()
    novelty_mean_df.name = "Novelty TM"
    novelty_sem_df.name = "Novelty TM sem"

    agg_df = pd.concat(
        [novelty_mean_df, novelty_sem_df, weighted_maxcluster_df, weighted_tm_score_df], axis=1
    )

    merged_df = data_df.join(
        novelty_df.set_index("pdb_path").drop(
            columns=["model", "tm_score", "rmsd", "seq_length"]
        ),
        on="pdb_path",
        lsuffix="_designability",
        rsuffix="_novelty",
    )
    
    merged_df[f"rmsd < {DESIGNABILITY_RMSD_THRESH}"] = merged_df["rmsd"] < DESIGNABILITY_RMSD_THRESH
    merged_df["Novelty Frac"] = merged_df[f"rmsd < {DESIGNABILITY_RMSD_THRESH}"] & (merged_df["novelty_tmscore"] < 0.5)
    merged_df["Novelty Frac 0.3"] = merged_df[f"rmsd < {DESIGNABILITY_RMSD_THRESH}"] & (
        merged_df["novelty_tmscore"] < 0.3
    )

    grouped_df = (
        merged_df[
            [
                "seq_length",
                "rmsd",
                "rmsd < 2",
                "model",
                "novelty_tmscore",
                "Novelty Frac",
                "Novelty Frac 0.3",
                "beta_ratio",
            ]
        ][
            (merged_df["seq_length"] <= 300)
        ]
        .groupby(["model", "seq_length"])
        .head(50)
        .groupby(
            ["model"],
        )
    )

    arrs = [
        [
            "Designability",
            "Designability",
            "Novelty",
            "Novelty",
            "Novelty",
            "Diversity",
            "Diversity",
            "Diversity",
        ],
        [
            r"Fraction ($\uparrow$)",
            r"scRMSD ($\downarrow$)",
            r"Fraction < 0.5 ($\uparrow$)",
            r"Fraction < 0.3 ($\uparrow$)",
            r"avg. max TM ($\downarrow$)",
            r"$\beta$-frac ($\uparrow$)",
            r"Pairwise TM ($\downarrow$)",
            r"Max Cluster ($\uparrow$)",
        ],
    ]
    tuples = list(zip(*arrs))

    mean_df = (
        (
            (
                grouped_df.mean()
                .sort_values("rmsd < 2")
                .join(agg_df.drop(columns=["Avg Maxcluster", "Novelty TM sem"]))
                .round(3)
            )
            .drop(columns="Avg TM")
            .join(
                diversity_df.groupby("model")  # .drop(columns="model")
                .mean()[["MaxCluster Diversity", "Avg Pairwise TM-score"]]
                .round(3)
            )
        )[
            [
                "rmsd < 2",
                "rmsd",
                "Novelty Frac",
                "Novelty Frac 0.3",
                "novelty_tmscore",
                "beta_ratio",
                "Avg Pairwise TM-score",
                "MaxCluster Diversity",
            ]
        ]
        .T.set_index(pd.MultiIndex.from_tuples(tuples, names=["type", "metric"]))
        .T
    )
    

    sem_df = (
        (
            (
                grouped_df.sem()
                .sort_values("rmsd < 2")
                .join(agg_df.drop(columns=["Avg Maxcluster", "Novelty TM sem"]))
                .round(3)
            )
            .drop(columns="Avg TM")
            .join(
                diversity_df.groupby("model")  # .drop(columns="model")
                .sem()[["MaxCluster Diversity", "Avg Pairwise TM-score"]]
                .round(3)
            )
        )
        [
            [
                "rmsd < 2",
                "rmsd",
                "Novelty Frac",
                "Novelty Frac 0.3",
                "novelty_tmscore",
                "beta_ratio",
                "Avg Pairwise TM-score",
                "MaxCluster Diversity",
            ]
        ]
        .T.set_index(pd.MultiIndex.from_tuples(tuples, names=["type", "metric"]))
        .T
    )
    

    def highlight_fn(s):
        highlight_dict = {
            ("Designability", r"Fraction ($\uparrow$)"): "max",
            ("Designability", r"scRMSD ($\downarrow$)"): "min",
            ("Novelty", r"Fraction < 0.5 ($\uparrow$)"): "max",
            ("Novelty", r"Fraction < 0.3 ($\uparrow$)"): "max",
            ("Novelty", r"avg. max TM ($\downarrow$)"): "min",
            ("Diversity", r"$\beta$-frac ($\uparrow$)"): "max",
            ("Diversity", "Pairwise TM ($\\downarrow$)"): "min",
            ("Diversity", "Max Cluster ($\\uparrow$)"): "max",
        }
        s = mean_df[s.name]
        is_highlighted = np.zeros_like(s)
        if s.name in highlight_dict:
            if highlight_dict[s.name] == "max":
                is_highlighted = s == s.max()
            if highlight_dict[s.name] == "min":
                is_highlighted = s == s.min()
        return ["font-weight: bold" if v else "" for v in is_highlighted]
    
    final_df = mean_df.copy()
    final_df.to_csv(f"{RESULTS_DIR}/final_results.csv")
    for column in sem_df.columns:
        final_df[column] = (
            mean_df[column]
            .apply(lambda x: "${:0.3f}".format(x))
            .str.cat(sem_df[column].apply(lambda x: "{:0.2f}$".format(x)), sep=" \pm ")
        )
    print(final_df)
    for index, row in final_df.iterrows():
        if not os.path.exists(f"{RESULTS_DIR}/{index}"):
            os.makedirs(f"{RESULTS_DIR}/{index}")
        with open(f"{RESULTS_DIR}/{index}/final_results_w_std.txt", 'w') as f:
            f.write(" & ".join(row.values))
        print(" & ".join(row.values))

    final_df.to_csv(f"{RESULTS_DIR}/final_results_w_std.csv")
    final_df = final_df.style.apply(highlight_fn).format(precision=3)
    dfi.export(final_df, f'{RESULTS_DIR}/composing_kappas.png', use_mathjax=True)




def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    PRELOAD=False
    if PRELOAD:
        data_df = pd.read_csv(f"{RESULTS_DIR}/{SAVE_FILE}.csv")
        novelty_df = pd.read_csv(f"{RESULTS_DIR}/{SAVE_FILE}_with_novelty.csv")
        diversity_df = pd.read_csv(f"{RESULTS_DIR}/{SAVE_FILE}_diversity.csv")
    else:
        data_df, novelty_df, diversity_df = compute_raw_scores() 
    print("novelty_df", novelty_df)
    print("diversity_df", diversity_df)

    seq_counts = novelty_df.groupby(["seq_length", "model"]).size().reset_index(name="count")

    weighted_maxcluster_df, weighted_tm_score_df = compute_weighted_avg(diversity_df, seq_counts)

    format_results(data_df, novelty_df, diversity_df, weighted_maxcluster_df, weighted_tm_score_df)
    
if __name__ == "__main__":
    main()
