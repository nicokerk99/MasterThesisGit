from nilearn.image import load_img
from nilearn.plotting import plot_glass_brain
from masks import *


def get_maps(id_subjects, folder_name):
    """ loads the t-maps and beta-maps for all the subjects of the experiments
     @:param : id_subjects is a list of subject id's (ints) from which you want to get the maps
     @:param : folder_name is the folder where the maps are stored"""

    t_maps = [None] * len(id_subjects)
    beta_maps = [None] * len(id_subjects)

    for i, identity in enumerate(id_subjects):
        ide = str(identity)
        t_maps[i] = load_img(folder_name + "/sub" + ide + "_4D_t_maps_0" + ".nii")
        beta_maps[i] = load_img(folder_name + "/sub" + ide + "_4D_beta_0" + ".nii")

    return t_maps, beta_maps


def get_masks(id_subjects, folder_name, plot=False):
    """ returns the masks for the 4 regions('V5_R', 'V5_L', 'PT_R', 'PT_L') in a dictionary by subject.
      @:param : id_subjects is a list of subject id's (ints) from which you want to get the maps
      @:param : folder_name is the folder where the maps are stored"""

    masks_names = ['V5_R', 'V5_L', 'PT_R', 'PT_L']
    masks = [dict() for _ in range(len(id_subjects))]

    for i, identity in enumerate(id_subjects):
        ide = str(identity)
        if identity <= 9: ide = "0" + ide

        masks[i] = dict()
        for name in masks_names:
            masks[i][name] = load_img(folder_name + "/" + name + "_sub" + ide + ".nii")
            if plot:
                plot_glass_brain(masks[i][name], title=name + " - subject " + ide)
    return masks


def load_full_data(subjects_ids, length_one_modality, maps_folder="brain_maps", masks_folder="masks"):
    maps_masked = [dict() for _ in subjects_ids]
    for i, subj_id in enumerate(subjects_ids):
        t_maps, beta_maps = get_maps([subj_id], folder_name=maps_folder)
        masks = get_masks([subj_id], folder_name=masks_folder, plot=False)
        maps = apply_mask_to_maps(t_maps, masks)
        maps_masked[i]["vis"] = get_part_of_maps(maps, 0,
                                                 length_one_modality)  # maps acquired for the vision experiment
        maps_masked[i]["aud"] = get_part_of_maps(maps, length_one_modality,
                                                 2 * length_one_modality)  # maps acquired for the audition experiment
        del t_maps
        del beta_maps
        del masks  # to relieve memory

    return maps_masked
