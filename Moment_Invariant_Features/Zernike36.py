import numpy as np
from ZernikeMoments import get_zernike_moment

def zernike_features36(img):
    """
    Compute Zernike features (36-dimensional) for a given image.

    Parameters:
    img (2D array): Input image.

    Returns:
    numpy.ndarray: 36-dimensional Zernike feature vector.
    """
    Z_36 = np.zeros(36)  # Initialize the feature vector

    # Zernike moments for n=0 to 10, varying m
    n, m = 0, 0
    _, AOH, _ = get_zernike_moment(img, n, m)
    Z_36[0] = AOH

    n, m = 1, 1
    _, AOH, _ = get_zernike_moment(img, n, m)
    Z_36[1] = AOH

    n, m = 2, 0
    _, AOH, _ = get_zernike_moment(img, n, m)
    Z_36[2] = AOH

    n, m = 2, 2
    _, AOH, _ = get_zernike_moment(img, n, m)
    Z_36[3] = AOH

    n, m = 3, 1
    _, AOH, _ = get_zernike_moment(img, n, m)
    Z_36[4] = AOH

    n, m = 3, 3
    _, AOH, _ = get_zernike_moment(img, n, m)
    Z_36[5] = AOH

    # Compute features for n = 4 to 10
    start_index = 6
    for n in range(4, 11):
        Ztemp = []
        if n % 2 == 0:  # Even n
            m_values = range(0, n + 1, 2)
        else:  # Odd n
            m_values = range(1, n + 1, 2)

        for m in m_values:
            _, AOH, _ = get_zernike_moment(img, n, m)
            Ztemp.append(AOH)

        end_index = start_index + len(Ztemp)
        Z_36[start_index:end_index] = Ztemp
        start_index = end_index

    return Z_36