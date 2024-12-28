import os
import cv2
import numpy as np
import pickle

def process_feature_maps():
    img_mask_base_path = r'E:\AlexNoel\Nick\training\mask\\'
    feature_maps = []

    for imno in range(1, 21):
        # Load data
        features_file = os.path.join('training', f"{imno + 20}_training.pkl")
        with open(features_file, 'rb') as f:
            features = pickle.load(f)

        # Load mask
        retina_rgb_mask = cv2.imread(os.path.join(img_mask_base_path, f"{imno + 20}_training_mask.gif"), 0)
        mask = (retina_rgb_mask > 0).astype(np.uint8)
        height, width = mask.shape

        # Initialize arrays
        g = np.zeros((5, len(features)))
        z = np.zeros((36, len(features)))
        t = np.zeros(len(features))

        # Extract feature components
        for i, feature in enumerate(features):
            g[:, i] = feature["grayfeatures"]
            z[:, i] = feature["zernike"]
            t[i] = feature["target"]

        # Initialize feature maps
        GF = np.zeros((height, width, 5))
        zernikes = np.zeros((height, width, 36))
        t_img = np.zeros((height, width))
        k = 0

        # Map features to image
        for i in range(8, height - 8):  # Equivalent to 9:height-8 in MATLAB (0-based indexing)
            for j in range(8, width - 8):  # Equivalent to 9:width-8
                if mask[i, j] == 1:
                    GF[i, j, :] = g[:, k]
                    zernikes[i, j, :] = z[:, k]
                    t_img[i, j] = t[k]
                    k += 1

        # Store in feature maps
        feature_maps.append({
            "Gray": GF,
            "Target": t_img,
            "Zernike": zernikes,
            "SelectedZernike": zernikes[:, :, [2, 6, 12, 20, 30]]
        })

    # Save feature maps
    with open('AllFeatureMaps_train.pkl', 'wb') as f:
        pickle.dump(feature_maps, f)


process_feature_maps()