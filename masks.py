from nilearn.masking import apply_mask
from nilearn.image import index_img


def apply_mask_to_maps(maps, masks):
    """ returns a dictionary with maps where the masks were applied
     @maps : a list of maps (t-maps or beta-maps)
     @masks : a dictionary where the keys are the names of the masks, and the value the nifti mask
     """

    nb_subjects = len(maps)
    length = maps[0].shape[3]  # amount of 3D images
    maps_masked = [None] * nb_subjects

    for i, mask_dict in enumerate(masks):
        maps_masked[i] = dict()
        for mask in mask_dict:
            maps_masked[i][mask] = apply_mask(maps[i], mask_dict[mask])

    return maps_masked


def get_part_of_maps(maps, start_index, end_index):
    """ maps is a list (size : n_subjects) of dictionaries (keys are types of masks), which contain n_samples maps """
    maps_sliced = [None] * len(maps)
    for i in range(len(maps)):
        maps_sliced[i] = dict()
        for mask_type in maps[i]:
            maps_sliced[i][mask_type] = maps[i][mask_type][start_index:end_index,:]

    return maps_sliced
