from nilearn.image import load_img
from nilearn.plotting import plot_glass_brain
from masks import *
import pandas as pd
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
        if identity <= 9 :
            ide = "0" + ide

        for name in masks_names:
            my_file = Path(folder_name + "/" + name + "_sub" + ide + ".nii")
            if my_file.is_file():
                # file exists
                masks[i][name] = load_img(folder_name + "/" + name + "_sub" + ide + ".nii")
                masks_exist[i][name] = True
                if plot:
                    plot_glass_brain(masks[i][name], title=name + " - subject " + ide)
            else:
                masks_exist[i][name] = False

    return masks, masks_exist


def load_full_data(subjects_ids, length_one_modality, maps_folder="brain_maps", masks_folder="masks",
                   is_from_mohamed=False):
    maps_masked = [dict() for _ in subjects_ids]
    masks_exist = [dict() for _ in subjects_ids]
    for i, subj_id in enumerate(subjects_ids):
        t_maps, beta_maps = get_maps([subj_id], maps_folder, is_from_mohamed)
        masks, masks_present = get_masks([subj_id], folder_name=masks_folder, plot=False)
        maps = apply_mask_to_maps(t_maps, masks, masks_exist)
        maps_masked[i]["vis"] = get_part_of_maps(maps, 0, length_one_modality,
                                                 masks_exist)  # maps acquired for the vision experiment
        maps_masked[i]["aud"] = get_part_of_maps(maps, length_one_modality, 2 * length_one_modality,
                                                 masks_exist)  # maps acquired for the audition experiment
        masks_exist[i] = masks_present[0]
        del t_maps
        del beta_maps
        del masks  # to relieve memory

    return maps_masked, masks_exist


def retrieve_cv_scores(out_directory):
    # joining dataframes in prevision of plotting
    cv_within_df = pd.read_csv(out_directory+"group_scores_within.csv", index_col=0)
    cv_cross_df = pd.read_csv(out_directory+"group_scores_cross.csv", index_col=0)
    for col in cv_cross_df.columns: cv_within_df[col] = cv_cross_df[col]

    return cv_within_df


def retrieve_bootstrap_scores(out_directory):
    bootstrap_within_df = pd.read_csv(out_directory + "bootstraps_within.csv", index_col=0)
    bootstrap_cross_df = pd.read_csv(out_directory + "bootstraps_cross.csv", index_col=0)
    for col in bootstrap_cross_df.columns: bootstrap_within_df[col] = bootstrap_cross_df[col]

    return bootstrap_within_df
