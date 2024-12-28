import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter
from skimage.morphology import disk, erosion

def preprocess_vclr(retina_rgb, mask):
    """
    Perform Vessel Central Light Reflex Removal on the green channel of the retina image.

    Parameters:
    retina_rgb (3D array): Original retina image (RGB).
    mask (2D array): Binary mask.

    Returns:
    retina_g_vclr (2D array): Green channel with light reflex removed.
    """
    # Ensure the mask is binary
    mask = erosion(mask, disk(1))

    # Extract green channel
    retina_g = retina_rgb[:, :, 1]

    # Perform morphological opening with disk-shaped structuring element
    struct_elem = disk(1)
    retina_g_open = cv2.morphologyEx(retina_g.astype(np.float64), cv2.MORPH_OPEN, struct_elem)

    # Background Homogenization
    # Mean filtering (3x3)
    mean_filtered_img = uniform_filter(retina_g_open, size=3)

    # Gaussian filtering (9x9, sigma=1.8)
    conv_img = gaussian_filter(mean_filtered_img, sigma=1.8)

    # Mean filtering (69x69)
    mean_filtered_img2 = uniform_filter(conv_img, size=69)
    mean_filtered_img2 = mean_filtered_img2 * mask

    # Subtract to homogenize the background
    isc = retina_g_open - mean_filtered_img2
    isc = isc - np.min(isc)
    isc = isc / np.max(isc)

    # Mask the result
    retina_g_vclr = isc * mask

    return retina_g_vclr