from nilearn.image import load_img
from nilearn.plotting import plot_glass_brain


def get_maps(id_subjects, test = ""):
    """ loads the t-maps and beta-maps for all the subjects of the experiments
     @:param : id_subjects is a list of subject id's (ints) from which you want to get the maps"""

    t_maps = [None] * len(id_subjects)
    beta_maps = [None] * len(id_subjects)

    for i, identity in enumerate(id_subjects):
        ide = str(identity)
        t_maps[i] = load_img("brain_maps/sub" + ide + "_4D_t_maps_0" + test + ".nii")
        beta_maps[i] = load_img("brain_maps/sub" + ide + "_4D_beta_0" + test + ".nii")

    return t_maps, beta_maps


def get_masks(id_subjects, plot=False, test = ""):
    """ returns the masks for the 4 regions('V5_R', 'V5_L', 'PT_R', 'PT_L') in a dictionary by subject.
      @:param : id_subjects is a list of subject id's (ints) from which you want to get the maps"""

    masks_names = ['V5_R', 'V5_L', 'PT_R', 'PT_L']
    masks = [dict() for _ in range(len(id_subjects))]

    for i, identity in enumerate(id_subjects):
        ide = str(identity) 
        if identity <= 9 : ide = "0"+ide
        
        masks[i] = dict()
        for name in masks_names:
            masks[i][name] = load_img("masks/" + name + "_sub" + ide + test + ".nii")
            if plot :
                plot_glass_brain(masks[i][name], title=name + " - subject " + ide)
    return masks
