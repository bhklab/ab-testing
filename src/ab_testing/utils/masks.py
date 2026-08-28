import numpy as np
import SimpleITK as sitk


def get_max_area_slice(mask: np.ndarray | sitk.Image) -> tuple[np.ndarray|sitk.Image, int]: 
    '''  
    Find maximum area slice of a given ground truth segmentation. Will be the slice 
    to calculate RERECIST on. 

    Parameters
    ----------
    mask: np.ndarray | sitk.Image
        3D segmentation mask to find the maximum area slice in.

    Returns
    ----------
    max_area_slice: np.ndarray | sitk.Image
        2D slice with the largest masked area. Returned in same format as mask input.
    max_area_slice_idx: int 
        The index of the slice (z-axis) with the largest masked area.
    '''
    if isinstance(mask, sitk.Image):
        # Convert mask to array for calculations
        mask = sitk.GetArrayFromImage(mask)

    slice_sums = np.sum(mask, axis = (1, 2)) # Get the pixel area of each slice 
    max_area_slice_idx = np.argmax(slice_sums) # Get the slice index with the largest pixel area 
    max_area_slice = mask[max_area_slice_idx] # Get the slice out of the mask array


    if isinstance(mask, sitk.Image):
        max_area_slice = sitk.GetImageFromArray(max_area_slice)

    return max_area_slice, max_area_slice_idx