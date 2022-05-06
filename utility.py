import numpy as np
import pandas as pd
import time
import random
import os
from load_data import retrieve_cv_metric
import statsmodels.api as sm
from statsmodels.formula.api import ols

"""
File containing small utility functions
"""


def create_directory(dir_name):
    """
    creates the output directory if not already present
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def average_dicos(dicos):
    dico = dict()
    counter = dict()
    for key in dicos[0]:
        counter[key] = 0
        dico[key] = 0
        for dictio in dicos:
            if key in dictio:
                dico[key] += dictio[key]
                counter[key] += 1
        if counter[key] > 0:
            dico[key] = dico[key] / counter[key]

    return dico


def compute_bootstrap_distribution(n_bootstrap, n_subjects, scores_n_perm, n_single_perm):
    start_time = time.time()

    scores_bootstrap = [None] * n_bootstrap
    for i in range(n_bootstrap):
        dicos = [dict() for _ in range(n_subjects)]
        for j in range(n_subjects):
            dicos[j] = scores_n_perm[j][random.randint(0, n_single_perm - 1)]
        scores_bootstrap[i] = average_dicos(dicos)

    duration = time.time() - start_time
    print("Running models done in " + str(duration) + " seconds")
    return scores_bootstrap


def compute_p_val_bootstrap(df_bootstrap, df_group_results):
    pvals = dict()
    for modality in df_bootstrap:
        gv = df_group_results[modality][0]
        count = len([v for v in df_bootstrap[modality] if v > gv])
        pvals[modality] = ((count + 1) / (len(df_bootstrap[modality]) + 1)) * 4
    return pvals


def verbose_dataframe(df, subjects_ids, compare=False):
    column_names = ["Modality", "Region", "Score", "Score_mean_dev"]
    if compare:
        column_names = ["Analysis", "Score", "Score_mean_dev"]
    vb_df = pd.DataFrame(columns=column_names)
    for entry in df:
        keywords = entry.split('_')

        if "vis" in keywords:
            mod = "Vision"
        elif "aud" in keywords:
            mod = "Audition"
        else:
            mod = "Cross-modal"

        region = "V5 " if "V5" in keywords else "PT "
        region += "L" if "L" in keywords else "R"

        analysis = mod + " " + region

        n = np.sqrt(len(df[entry]) - df[entry].isnull().sum())
        avg = np.mean(df[entry])

        for i in subjects_ids:
            if df[entry][i]:
                new_entry = dict()
                if compare:
                    new_entry["Analysis"] = analysis
                else:
                    new_entry["Modality"] = mod
                    new_entry["Region"] = region
                new_entry["Score"] = df[entry][i]
                new_entry["Score_mean_dev"] = df[entry][i] / n + avg - avg / n
                vb_df = vb_df.append(new_entry, ignore_index=True)

    return vb_df


def cfm_string_to_matrix(cfm_string):
    if pd.isna(cfm_string): return [[np.nan]]
    cf = cfm_string.replace("[", "").replace("]", "").replace("\n", "")
    cf = cf.split('.')[:-1]
    cf = np.asarray(list(map(int, cf))).reshape(4, 4)
    return cf


def compute_group_confusion_matrix(df_cf_matrixes, subjects_ids):
    group_cf = dict()
    for modality in df_cf_matrixes:
        n_of_nans = df_cf_matrixes[modality].isnull().sum()
        gcf = np.zeros((4, 4, len(subjects_ids) - n_of_nans))
        l = 0
        for i, subj_id in enumerate(subjects_ids):
            cfm = cfm_string_to_matrix(df_cf_matrixes[modality][subj_id])
            if not pd.isna(cfm[0][0]):
                cfm = cfm / np.sum(cfm) * 400
                for j in range(4):
                    for k in range(4):
                        gcf[j][k][l] = cfm[j][k]
                l += 1
        group_cf[modality] = gcf
    return group_cf


def compute_anova(folder_base, folder_candidate):
    base_df = retrieve_cv_metric(folder_base, "accuracy")
    base_df = base_df[base_df.columns.drop(list(base_df.filter(regex='cross')))]
    base_df = base_df[base_df.columns.drop(list(base_df.filter(regex='vis_PT')))]
    base_df = verbose_dataframe(base_df, range(1, 24), compare=True)
    base_df["dataset"] = np.repeat(["base"], 23 * 6)
    base_df.dropna(inplace=True)
    base_df.drop("Score_mean_dev", axis=1, inplace=True)
    candidate_df = retrieve_cv_metric(folder_candidate, "accuracy")
    candidate_df = candidate_df[candidate_df.columns.drop(list(candidate_df.filter(regex='cross')))]
    candidate_df = candidate_df[candidate_df.columns.drop(list(candidate_df.filter(regex='vis_PT')))]
    candidate_df = verbose_dataframe(candidate_df, range(1, 24), compare=True)
    candidate_df["dataset"] = np.repeat(["candidate"], 23 * 6)
    candidate_df.dropna(inplace=True)
    candidate_df.drop("Score_mean_dev", axis=1, inplace=True)

    df = base_df.append(candidate_df)

    # Performing two-way ANOVA
    model = ols('Score ~ C(dataset) +C(Analysis):C(dataset)', data=df).fit()
    results = sm.stats.anova_lm(model, typ=2)

    print(results)


def compute_anova_demeaned(folder_base, folder_candidate):
    base_df = retrieve_cv_metric(folder_base, "accuracy")
    base_df = base_df[base_df.columns.drop(list(base_df.filter(regex='cross')))]
    base_df = base_df[base_df.columns.drop(list(base_df.filter(regex='vis_PT')))]

    candidate_df = retrieve_cv_metric(folder_candidate, "accuracy")
    candidate_df = candidate_df[candidate_df.columns.drop(list(candidate_df.filter(regex='cross')))]
    candidate_df = candidate_df[candidate_df.columns.drop(list(candidate_df.filter(regex='vis_PT')))]

    tmp_df = base_df.append(candidate_df)
    mean_vector = tmp_df.mean(axis=0)
    base_df = base_df.sub(mean_vector, axis=1)
    candidate_df = candidate_df.sub(mean_vector, axis=1)

    base_df = verbose_dataframe(base_df, range(1, 24), compare=True)
    base_df["dataset"] = np.repeat(["base"], 23 * 6)
    base_df.dropna(inplace=True)
    base_df.drop("Score_mean_dev", axis=1, inplace=True)

    candidate_df = verbose_dataframe(candidate_df, range(1, 24), compare=True)
    candidate_df["dataset"] = np.repeat(["candidate"], 23 * 6)
    candidate_df.dropna(inplace=True)
    candidate_df.drop("Score_mean_dev", axis=1, inplace=True)

    df = base_df.append(candidate_df)

    # Performing two-way ANOVA
    model = ols('Score ~ C(dataset)', data=df).fit()
    results = sm.stats.anova_lm(model, typ=2)

    print(results)
