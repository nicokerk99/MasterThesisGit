from typing import Dict, Tuple
from nilearn.image import load_img
from nilearn.plotting import plot_glass_brain
from masks import *
import pandas as pd
import numpy as np
import numpy.typing as npt
from pathlib import Path
import nibabel


def get_maps(id_subjects: List[int], 
             folder_name: str, 
             is_from_mohamed: bool=False) -> Tuple[List[nibabel.nifti1.Nifti1Image], List[nibabel.nifti1.Nifti1Image]]:
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

    return t_maps, beta_maps  # type: ignore


def get_masks(id_subjects: List[int], 
              folder_name: str, 
              plot: bool=False, 
              is_from_mohamed: bool=False) -> Tuple[List[Dict[str, nibabel.nifti1.Nifti1Image]], List[Dict[str, bool]]]:
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
            underscore = "" if is_from_mohamed else "_"
            my_file = Path(folder_name + "/" + name + "_sub" + underscore + ide + ".nii")
            if my_file.is_file():
                # file exists
                masks[i][name] = load_img(folder_name + "/" + name + "_sub" + underscore + ide + ".nii")
                masks_exist[i][name] = True
                if plot:
                    plot_glass_brain(masks[i][name], title=name + " - subject " + ide)
            else:
                masks_exist[i][name] = False

    return masks, masks_exist


def load_full_data(subjects_ids: range, 
                   n_classes: int, 
                   nb_runs: int, 
                   maps_folder: str="brain_maps", 
                   masks_folder: str="masks",
                   is_from_mohamed: bool=False, 
                   use_t_maps: bool=True) -> Tuple[List[Dict[str, List[Dict[str, npt.NDArray[np.float64]]]]], List[Dict[str, bool]]]:
    length_one_modality = n_classes*nb_runs
    maps_masked = [dict() for _ in subjects_ids]
    masks_exist = [dict() for _ in subjects_ids]
    for i, subj_id in enumerate(subjects_ids):
        t_maps, beta_maps = get_maps([subj_id], maps_folder, is_from_mohamed)
        masks, masks_present = get_masks([subj_id], folder_name=masks_folder, plot=False, is_from_mohamed=is_from_mohamed)
        if use_t_maps :
            maps = apply_mask_to_maps(t_maps, masks, masks_present)
        else :
            maps = apply_mask_to_maps(beta_maps, masks, masks_present)
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


def retrieve_cv_metric(out_directory: str, 
                       metric: str, 
                       only_within: bool=False) -> pd.DataFrame:
    """
    :param out_directory: directory from which retrieve the result
    :param metric: the metric we want to retrieve
    :return: a pandas dataframe with rows = subjects and columns = analyses,
    each value being the metric of the analysis for the corresponding subject
    """
    # joining dataframes in prevision of plotting
    cv_within_df = pd.read_csv(out_directory + metric + "_within.csv", index_col=0)
    if only_within:
        return cv_within_df
    cv_cross_df = pd.read_csv(out_directory + metric + "_cross.csv", index_col=0)
    for col in cv_cross_df.columns: cv_within_df[col] = cv_cross_df[col]

    return cv_within_df


def retrieve_cv_matrixes(out_directory: str, 
                         only_within: bool=False) -> pd.DataFrame:
    # joining dataframes in prevision of plotting
    cv_within_df = pd.read_csv(out_directory + "confusion_matrixes_within.csv", index_col=0)
    if only_within:
        return cv_within_df
    cv_cross_df = pd.read_csv(out_directory + "confusion_matrixes_cross.csv", index_col=0)
    for col in cv_cross_df.columns: cv_within_df[col] = cv_cross_df[col]

    return cv_within_df


def retrieve_val_scores(out_directory: str, 
                        only_within: bool=False) -> pd.DataFrame:
    # joining dataframes in prevision of plotting
    cv_within_df = pd.read_csv(out_directory + "validation_scores_within.csv", index_col=0)
    if only_within:
        return cv_within_df
    cv_cross_df = pd.read_csv(out_directory + "validation_scores_cross.csv", index_col=0)
    for col in cv_cross_df.columns: cv_within_df[col] = cv_cross_df[col]

    return cv_within_df


def retrieve_bootstrap_metric(out_directory: str, 
                              metric: str, 
                              only_within: bool=False) -> pd.DataFrame:
    bootstrap_within_df = pd.read_csv(out_directory + metric + "_bootstraps_within.csv", index_col=0)
    if only_within:
        return bootstrap_within_df
    bootstrap_cross_df = pd.read_csv(out_directory + metric + "_bootstraps_cross.csv", index_col=0)
    for col in bootstrap_cross_df.columns: bootstrap_within_df[col] = bootstrap_cross_df[col]

    return bootstrap_within_df


def retrieve_masks_exist(out_directory: str) -> pd.DataFrame:
    bootstrap_within_df = pd.read_csv(out_directory + "masks_exist.csv", index_col=0)

    return bootstrap_within_df


def retrieve_pvals(out_directory: str, 
                   default_keys: List[str]=[]) -> Dict[str, float]:
    my_file = Path(out_directory + "estimated_pval_bootstrap.csv")
    if my_file.is_file():
        df = pd.read_csv(out_directory + "estimated_pval_bootstrap.csv", index_col=0)
        return df.to_dict('records')[0]
    else :
        print("No p-values found in directory : "+out_directory)
        return dict((key, 1) for key in default_keys)


def change_maps_masked_org(maps_masked: List[Dict[str, List[Dict[str, npt.NDArray[np.float64]]]]], 
                           subjects_ids: range, 
                           n_classes: int, 
                           nb_runs: int) -> List[Dict[str, List[Dict[str, npt.NDArray[np.float64]]]]]:
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


def change_confusion_matrixes_org(cfm: List[Dict[str, Dict[str, npt.NDArray[np.float64]]]],
                                  subjects_ids: range, 
                                  model_names: List[str]) -> Dict[str, List[Dict[str, npt.NDArray[np.float64]]]]:
    new_cfm = {name:[dict() for _ in subjects_ids] for name in model_names}
    for i, subj_id in enumerate(subjects_ids):
        for modality in cfm[i]:
            for name in model_names:
                new_cfm[name][i][modality] = cfm[i][modality][name]
    return new_cfm


def change_cfm_bootstrap_org(cfm: List[List[Dict[str, Dict[str, npt.NDArray[np.float64]]]]],
                             subjects_ids: range, 
                             model_names: List[str], 
                             n_single_perm: int) -> Dict[str, List[List[Dict[str, npt.NDArray[np.float64]]]]]:
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