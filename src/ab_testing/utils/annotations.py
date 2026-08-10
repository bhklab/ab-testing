import numpy as np
from pathlib import Path
from skimage.draw import line
from skimage.measure import regionprops


def get_slice_properties(mask_slice: np.ndarray) -> tuple[float, float, float, float, float]:
    """Utility function for prompt generation function to get properties of a given mask slice."""
    try: 
        props = regionprops(mask_slice)[0]
        y_cent, x_cent = props.centroid
        orientation = props.orientation
        semi_maj_axis_len = props.axis_major_length / 2
    except Exception as e: 
         # Usually errors will arise here if there is an issue with region props calculation and the mask being too small to calculate anything from. 
         raise Exception(f'error {e} and sum of mask slice is {mask_slice.sum()}')

    return x_cent, y_cent, orientation, semi_maj_axis_len


def get_recist_pts(mask_slice: np.ndarray) -> np.ndarray:
    """Get the coordinates of the endpoints of the an automatically generated RECIST line in a mask array. 
    
    Parameters
    ----------
    mask_slice: np.ndarray[int, int]
        The ground truth segmentation at the largest area slice. Expects (z, x, y) format.
    
    Returns
    -------
    recist_pts: np.ndarray[float, float, float, float]
        Holds the information for the found RERECIST line in the form [x_r1, y_r1, x_r2, y_r2]
    """
    x_cent, y_cent, orientation, semi_maj_axis_len = get_slice_properties(mask_slice)

    x_r1 = x_cent - np.sin(orientation) * semi_maj_axis_len
    y_r1 = y_cent - np.cos(orientation) * semi_maj_axis_len

    x_r2 = x_cent + np.sin(orientation) * semi_maj_axis_len
    y_r2 = y_cent + np.cos(orientation) * semi_maj_axis_len

    recist_pts = np.array([x_r1, y_r1, x_r2, y_r2])
    
    return recist_pts


def get_line_from_recist(recist_pts: np.array, 
                         slice_idx: int, 
                         scan_size: np.array):
    '''
    From the RECIST measurement coordinates, generate a line connecting both coordinates on the correct slice and return an np.ndarray the same shape as the image.
    Output to be compatible with the ['recist'] array of the .npz files needed for MedSAM2-RECIST.

    Parameters
    ----------
    recist_pts: array
        A list of coordinates in [x1, y1, x2, y2] format that defines the RECIST measurement 
    slice_idx: int
        The slice that the measurement was taken on
    scan_size: np.array
        The x, y, and z size of the image in [z_space, x_space, y_space] format
    
    Returns
    ----------
    recist_arr: np.ndarray
        A binary array of the same shape as the image with the pixels of the line = 1
    '''
    # Generate an array in the same size as the image filled with all zeros 
    recist_arr = np.zeros(scan_size)
    
    # Round the coordinate values to their nearest integers 
    coords_round = np.rint(recist_pts).astype(int)

    # Draw line using coordinates 
    rr, cc = line(coords_round[0], coords_round[1], coords_round[2], coords_round[3])

    # Put line into the correct slice in the RECIST array of all zeros 
    recist_arr[slice_idx][cc, rr] = 1

    return recist_arr
