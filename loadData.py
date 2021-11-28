from nilearn.image import load_img
from nilearn.plotting import plot_glass_brain


def get_maps(n_subjects):
    t_maps = [None] * n_subjects
    beta_maps = [None] * n_subjects

    for i in range(n_subjects):
        t_maps[i] = load_img("brain_maps/sub" + str(i + 1) + "_4D_t_maps_0.nii")
        beta_maps[i] = load_img("brain_maps/sub" + str(i + 1) + "_4D_beta_0.nii")

    return t_maps, beta_maps


def get_masks(n_subjects, plot=False):
    masks_names = ['V5_R', 'V5_L', 'PT_R', 'PT_L']
    masks = [None] * n_subjects
    masks_full = [None] * n_subjects

    for i in range(n_subjects):
        masks_full[i] = load_img("masks/WHOLE_sub" + str(i + 1) + ".nii")
        masks[i] = dict()
        for name in masks_names:
            masks[i][name] = load_img("masks/ROI_sub" + str(i + 1) + "_" + name + ".nii")
            if plot :
                plot_glass_brain(masks[i][name], title=name + " - subject " + str(i + 1))
    return masks, masks_full
