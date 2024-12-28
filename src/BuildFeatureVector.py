import os
import cv2
import numpy as np
from skimage.filters import gaussian

import pickle
from concurrent.futures import ProcessPoolExecutor
from ..Pre_Processing.getWindow import get_window
from ..Pre_Processing.PreProc_VCLR import preprocess_vclr
from ..Pre_Processing.PreProc_CLAHE import preprocess_clahe
from ..Pre_Processing.PreProc_VesselEnhance import preprocess_vessel_enhancement
from ..Moment_Invariant_Features.Zernike36 import zernike_features36

def save_features(filepath, features):
    with open(filepath, 'wb') as f:
        pickle.dump(features, f)

def process_single_image(file_name, img_base_path, img_gt_base_path, img_mask_base_path, op_dir):
    winsize_gray = 9
    winsize_zer = 17
    h = cv2.getGaussianKernel(winsize_zer, 1.7) @ cv2.getGaussianKernel(winsize_zer, 1.7).T

    img_path = os.path.join(img_base_path, file_name)
    gt_path = os.path.join(img_gt_base_path, file_name.replace('.tif', '').replace('training', 'manual1') + '.gif')
    mask_path = os.path.join(img_mask_base_path, file_name.replace('.tif', '').replace('training', 'training_mask') + '.gif')

    retina_rgb = cv2.imread(img_path)
    gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE) > 0
    retina_rgb_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 0

    rmask = retina_rgb_mask.astype(np.uint8)

    # Extracting the green channel
    retina_g = retina_rgb[:, :, 1]
    retina_g = retina_g * rmask

    # Vessel Central Light Reflex Removal
    retina_g_vclr = preprocess_vclr(retina_rgb, retina_g)

    # Background Homogenization
    I_h = preprocess_clahe(retina_g_vclr, rmask)

    # Vessel Enhancement
    I_ve = preprocess_vessel_enhancement(I_h)

    # Feature extraction
    rows, cols = I_h.shape
    features = []
    for r in range(winsize_zer // 2, rows - winsize_zer // 2):
        for c in range(winsize_zer // 2, cols - winsize_zer // 2):
            if rmask[r, c] == 1:
                pixel = I_h[r, c]
                window_pixels = get_window(I_h, r, c, winsize_gray)
                window_list = window_pixels.flatten()

                # Gray-level features
                f1 = pixel - np.min(window_list)
                f2 = np.max(window_list) - pixel
                f3 = pixel - np.mean(window_list)
                f4 = np.std(window_list)
                f5 = pixel

                slide_window = I_ve[r - winsize_zer // 2:r + winsize_zer // 2 + 1,
                                    c - winsize_zer // 2:c + winsize_zer // 2 + 1]

                if np.sum(slide_window) == 0:
                    z_36_g = np.zeros(36)
                else:
                    window_gauss_filter = slide_window.astype(np.float64) * h
                    z_36_g = zernike_features36(window_gauss_filter)

                features.append({
                    "grayfeatures": [f1, f2, f3, f4, f5],
                    "zernike": z_36_g,
                    "sumzernike": np.sum(z_36_g),
                    "meanzernike": np.mean(z_36_g),
                    "target": gt_img[r, c],
                    "x": r,
                    "y": c
                })
        print(f"{file_name} : Row {r}/{rows - winsize_zer // 2}")

    save_features(os.path.join(op_dir, file_name.replace('.tif', '.pkl')), features)

def process_images():
    img_base_path = r'E:\Alex\Nick\training\images\\'
    img_gt_base_path = r'E:\Alex\Nick\training\manual1\\'
    img_mask_base_path = r'E:\Alex\Nick\training\mask\\'
    op_dir = r'training\\'
    os.makedirs(op_dir, exist_ok=True)

    file_list = [f for f in os.listdir(img_base_path) if f.endswith('.tif')]

    # Parallel processing using ProcessPoolExecutor
    with ProcessPoolExecutor() as executor:
        executor.map(process_single_image, file_list, 
                     [img_base_path] * len(file_list), 
                     [img_gt_base_path] * len(file_list), 
                     [img_mask_base_path] * len(file_list), 
                     [op_dir] * len(file_list))

if __name__ == "__main__":
    process_images()