from nilearn.image import load_img
from nilearn.plotting import plot_glass_brain
from masks import *
import pandas as pd
import numpy as np
from pathlib import Path


def get_maps(id_subjects, folder_name, is_from_mohamed=False):
    """ loads the t-maps and beta-maps for all the subjects of the experiments
     @:param : id_subjects is a list of subject id's (ints) from which you want to get the maps
     @:param : folder_name is the folder where the maps are stored"""

    t_maps = [None] * len(id_subjects)
    beta_maps = [None] * len(id_subjects)

    for i, identity in enumerate(id_subjects):
        ide = str(identity)
        if not is_from_mohamed:
            t_maps[i] = load_img(folder_name + "/sub" + ide + "_4D_t_maps_0" + ".nii")
            beta_maps[i] = load_img(folder_name + "/sub" + ide + "_4D_beta_0" + ".nii")
        else:
            t_maps[i] = load_img(folder_name + "/Sub" + ide + "/4D_t_maps_0" + ".nii")
            beta_maps[i] = load_img(folder_name + "/Sub" + ide + "/4D_beta_0" + ".nii")

    return t_maps, beta_maps


def get_masks(id_subjects, folder_name, plot=False):
    """ returns the masks for the 4 regions('V5_R', 'V5_L', 'PT_R', 'PT_L') in a dictionary by subject
    and a dictionary of booleans telling if the mask exists or not.
      @:param : id_subjects is a list of subject id's (ints) from which you want to get the maps
      @:param : folder_name is the folder where the maps are stored"""

    masks_names = ['V5_R', 'V5_L', 'PT_R', 'PT_L']
    masks = [dict() for _ in range(len(id_subjects))]
    masks_exist = [dict() for _ in range(len(id_subjects))]

    for i, identity in enumerate(id_subjects):
        ide = str(identity)
        if identity <= 9:
            ide = "0" + ide

        for name in masks_names:
            my_file = Path(folder_name + "/" + name + "_sub_" + ide + ".nii")
            if my_file.is_file():
                # file exists
                masks[i][name] = load_img(folder_name + "/" + name + "_sub_" + ide + ".nii")
                masks_exist[i][name] = True
                if plot:
                    plot_glass_brain(masks[i][name], title=name + " - subject " + ide)
            else:
                masks_exist[i][name] = False

    return masks, masks_exist


def load_full_data(subjects_ids, n_classes, nb_runs, maps_folder="brain_maps", masks_folder="masks",
                   is_from_mohamed=False):
    length_one_modality = n_classes*nb_runs
    maps_masked = [dict() for _ in subjects_ids]
    masks_exist = [dict() for _ in subjects_ids]
    for i, subj_id in enumerate(subjects_ids):
        t_maps, beta_maps = get_maps([subj_id], maps_folder, is_from_mohamed)
        masks, masks_present = get_masks([subj_id], folder_name=masks_folder, plot=False)
        maps = apply_mask_to_maps(t_maps, masks, masks_present)
        maps_masked[i]["vis"] = get_part_of_maps(maps, 0, length_one_modality,
                                                 masks_present)  # maps acquired for the vision experiment
        maps_masked[i]["aud"] = get_part_of_maps(maps, length_one_modality, 2 * length_one_modality,
                                                 masks_present)  # maps acquired for the audition experiment
        masks_exist[i] = masks_present[0]
        del t_maps
        del beta_maps
        del masks  # to relieve memory

    maps_masked = change_maps_masked_org(maps_masked, subjects_ids, n_classes, nb_runs)
    return maps_masked, masks_exist


def retrieve_cv_metric(out_directory, metric):
    """
    :param out_directory: directory from which retrieve the result
    :param metric: the metric we want to retrieve
    :return: a pandas dataframe with rows = subjects and columns = analyses,
    each value being the metric of the analysis for the corresponding subject
    """
    # joining dataframes in prevision of plotting
    cv_within_df = pd.read_csv(out_directory + metric + "_within.csv", index_col=0)
    cv_cross_df = pd.read_csv(out_directory + metric + "_cross.csv", index_col=0)
    for col in cv_cross_df.columns: cv_within_df[col] = cv_cross_df[col]

    return cv_within_df


def retrieve_cv_matrixes(out_directory):
    # joining dataframes in prevision of plotting
    cv_within_df = pd.read_csv(out_directory + "confusion_matrixes_within.csv", index_col=0)
    cv_cross_df = pd.read_csv(out_directory + "confusion_matrixes_cross.csv", index_col=0)
    for col in cv_cross_df.columns: cv_within_df[col] = cv_cross_df[col]

    return cv_within_df


def retrieve_bootstrap_metric(out_directory, metric):
    bootstrap_within_df = pd.read_csv(out_directory + metric + "_bootstraps_within.csv", index_col=0)
    bootstrap_cross_df = pd.read_csv(out_directory + metric + "_bootstraps_cross.csv", index_col=0)
    for col in bootstrap_cross_df.columns: bootstrap_within_df[col] = bootstrap_cross_df[col]

    return bootstrap_within_df


def retrieve_pvals(out_directory):
    df = pd.read_csv(out_directory + "estimated_pval_bootstrap.csv", index_col=0)
    return df.to_dict('records')[0]


def change_maps_masked_org(maps_masked, subjects_ids, n_classes, nb_runs):
    for i, subj_id in enumerate(subjects_ids):
        for stimuli in ["vis", "aud"]:
            dic = maps_masked[i][stimuli][0]
            for k in dic:
                dimension = dic[k].shape
                reorg = np.zeros(dimension)
                data = dic[k]
                for r in range(nb_runs):
                    for c in range(n_classes):
                        reorg[r*n_classes+c] = data[nb_runs*c+r]
                maps_masked[i][stimuli][0][k] = reorg
    return maps_masked


def change_confusion_matrixes_org(cfm, subjects_ids, model_names):
    new_cfm = {name:[dict() for _ in subjects_ids] for name in model_names}
    for i, subj_id in enumerate(subjects_ids):
        for modality in cfm[i]:
            for name in model_names:
                new_cfm[name][i][modality] = cfm[i][modality][name]
    return new_cfm


def change_cfm_bootstrap_org(cfm, subjects_ids, model_names, n_single_perm):
    new_cfm = {name:[[dict() for _ in range(n_single_perm)] for _ in subjects_ids] for name in model_names}
    for i, subj_id in enumerate(subjects_ids):
        for j in range(n_single_perm):
            for modality in cfm[i][j]:
                for name in model_names:
                    new_cfm[name][i][j][modality] = cfm[i][j][modality][name]
    return new_cfm

    
"""
def retrieve_bootstrap_matrixes(out_directory):
    bootstrap_within_df = pd.read_csv(out_directory + "bootstraps_within.csv", index_col=0)
    bootstrap_cross_df = pd.read_csv(out_directory + "bootstraps_cross.csv", index_col=0)
    for col in bootstrap_cross_df.columns: bootstrap_within_df[col] = bootstrap_cross_df[col]

    return bootstrap_within_df
"""