from nilearn.masking import apply_mask
# from nilearn.image import index_img


def apply_mask_to_maps(maps, masks, masks_exist):
    """ returns a list (size n_subjects) of dictionaries with maps where the masks were applied
     @maps : a list (size n_subjects) of maps (t-maps or beta-maps)
     @masks : a list (size n_subjects) of dictionaries
            where the keys are the names of the masks, and the value the nifti mask """

    n_subjects = len(maps)
    maps_masked = [dict() for _ in range(n_subjects)]

    for i, mask_dict in enumerate(masks):
        for mask in mask_dict:
            if masks_exist[i][mask] :
                maps_masked[i][mask] = apply_mask(maps[i], mask_dict[mask])

    return maps_masked


def get_part_of_maps(maps, start_index, end_index, masks_exist):
    """  returns the maps where only keep samples from start_index to end_index
    @maps : list (size n_subjects) of dictionaries (keys are types of masks), which contain n_samples maps """
    maps_sliced = [dict() for _ in range(len(maps))]
    for i in range(len(maps)):
        for mask_type in maps[i]:
            if masks_exist[i][mask_type]:
                maps_sliced[i][mask_type] = maps[i][mask_type][start_index:end_index, :]

    return maps_sliced
