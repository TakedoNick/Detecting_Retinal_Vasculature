import numpy as np

def get_window(img, x, y, window_size):
    """
    Extract a square window of pixels around a center point (x, y) from the image.
    Handles boundary conditions by clipping the window to valid image bounds.

    Parameters:
    img (2D array): Input image
    x (int): Center row
    y (int): Center column
    window_size (int): Size of the square window

    Returns:
    2D array: Extracted window of pixels
    """
    dist = window_size // 2
    up = max(0, x - dist)
    down = min(img.shape[0], x + dist + 1)
    left = max(0, y - dist)
    right = min(img.shape[1], y + dist + 1)
       
    # Create window with bounds handling
    window = np.zeros((window_size, window_size), dtype=img.dtype)
    window_start_row = max(0, dist - x)
    window_start_col = max(0, dist - y)
    window[window_start_row:window_start_row + (down - up), 
           window_start_col:window_start_col + (right - left)] = img[up:down, left:right]
    return window