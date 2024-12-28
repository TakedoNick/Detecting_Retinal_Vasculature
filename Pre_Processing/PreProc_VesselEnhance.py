import cv2
from skimage.filters.rank import tophat
from skimage.morphology import disk

def preprocess_vessel_enhancement(I_h):
    """
    Enhance blood vessels using top-hat transformation.

    Parameters:
    I_h (2D array): Enhanced and normalized image from CLAHE.

    Returns:
    I_ve (2D array): Vessel-enhanced image.
    """
    I_complement = cv2.bitwise_not(I_h)  # Complement the image
    I_ve = tophat(I_complement, disk(8))  # Top-hat transformation
    return I_ve