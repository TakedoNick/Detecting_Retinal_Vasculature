from skimage.exposure import equalize_adapthist
from skimage.util import img_as_ubyte
import cv2

def preprocess_clahe(retina_g_vclr, mask):
    """
    Apply CLAHE to enhance contrast in the green channel and sharpen the image.

    Parameters:
    retina_g_vclr (2D array): Image after VCLR.
    mask (2D array): Binary mask.

    Returns:
    I_h (2D array): Enhanced and normalized image.
    """
    I_clahe = img_as_ubyte(equalize_adapthist(retina_g_vclr)) * mask
    I_sharp = cv2.GaussianBlur(I_clahe, (3, 3), 0) * mask
    I_h = cv2.normalize(I_sharp, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
    return I_h