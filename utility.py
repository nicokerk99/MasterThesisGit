import numpy as np
import pandas as pd
import time
import random
import os

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
        for dictio in dicos :
            if key in dictio :
                dico[key] += dictio[key]
                counter[key] += 1
        if counter[key] > 0 :
            dico[key] = dico[key]/counter[key]

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
        pvals[modality] = (count+1)/(len(df_bootstrap[modality])+1)
    return pvals


def verbose_dataframe(df, subjects_ids):
    column_names = ["Modality", "Region", "Score"]
    vb_df = pd.DataFrame(columns=column_names)
    for entry in df :
        keywords = entry.split('_')
        for i in subjects_ids:
            if df[entry][i]:
                new_entry = dict()
                if "vis" in keywords :
                    new_entry["Modality"] = "Vision"
                elif "aud" in keywords :
                    new_entry["Modality"] = "Audition"
                else:
                    new_entry["Modality"] = "Cross-modal"
                new_entry["Region"] = "V5 " if "V5" in keywords else "PT "
                new_entry["Region"] += "L" if "L" in keywords else "R"
                new_entry["Score"] = df[entry][i]
                vb_df = vb_df.append(new_entry, ignore_index=True)

    return vb_df


def cfm_string_to_matrix(cfm_string):
    if pd.isna(cfm_string) : return [[np.nan]]
    cf = cfm_string.replace("[","").replace("]","").replace("\n","")
    cf = cf.split('.')[:-1]
    cf = np.asarray(list(map(int, cf))).reshape(4,4)
    return cf


def compute_group_confusion_matrix(df_cf_matrixes, subjects_ids):
    group_cf = dict()
    for modality in df_cf_matrixes:
        number_of_nans = df_cf_matrixes[modality].isnull().sum()
        gcf = np.zeros((4, 4, len(subjects_ids) - number_of_nans))
        l = 0
        for i, subj_id in enumerate(subjects_ids):
            cfm = cfm_string_to_matrix(df_cf_matrixes[modality][subj_id])
            if not pd.isna(cfm[0][0]):
                cfm = cfm/np.sum(cfm)*100
                for j in range(4):
                    for k in range(4):
                        gcf[j][k][l] = cfm[j][k]
                l += 1
        group_cf[modality] = gcf
    return group_cf