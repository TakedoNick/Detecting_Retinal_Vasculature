import numpy as np
from radial_poly import radial_poly

def get_zernike_moment(p, n, m):
    """
    Compute the Zernike moment of an image.
    
    Parameters:
    p (2D array): Input image
    n (int): Radial order
    m (int): Azimuthal frequency
    
    Returns:
    Z (complex): Zernike moment
    A (float): Magnitude of the moment
    Phi (float): Phase angle of the moment in degrees
    """
    # Dimensions of the input image
    N = p.shape[0]
    
    # Create coordinate grid
    x = np.arange(1, N+1)
    y = x
    X, Y = np.meshgrid(x, y)
    
    # Compute radius and angle
    R = np.sqrt((2*X - N - 1)**2 + (2*Y - N - 1)**2) / N
    Theta = np.arctan2((N - 1 - 2*Y + 2), (2*X - N + 1 - 2))
    R = np.where(R <= 1, R, 0)  # Set values outside the unit circle to 0
    
    # Compute the radial polynomial
    Rad = radial_poly(R, n, m)
    
    # Compute the product
    Product = p.astype(np.float64) * Rad * np.exp(-1j * m * Theta)
    Z = np.sum(Product)
    
    # Normalize the moment
    cnt = np.count_nonzero(R) + 1
    Z = (n + 1) * Z / cnt
    
    # Compute magnitude and phase
    A = np.abs(Z)
    Phi = np.angle(Z, deg=True)
    
    return Z, A, Phi