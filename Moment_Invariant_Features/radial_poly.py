import numpy as np
from math import factorial

def radial_poly(r, n, m):
    """
    Compute the radial polynomial for Zernike moments.
    
    Parameters:
    r (2D array): Radius values
    n (int): Radial order
    m (int): Azimuthal frequency
    
    Returns:
    rad (2D array): Radial polynomial values
    """
    rad = np.zeros_like(r)
    for s in range((n - abs(m)) // 2 + 1):
        c = ((-1)**s * factorial(n - s) /
             (factorial(s) *
              factorial((n + abs(m)) // 2 - s) *
              factorial((n - abs(m)) // 2 - s)))
        rad += c * r**(n - 2 * s)
    return rad