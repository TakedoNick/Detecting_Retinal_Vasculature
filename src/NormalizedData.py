import os
import numpy as np
import pickle

def normalize_features_training(data_dir, output_file):
    """
    Normalize features from training data

    Parameters:
    data_dir (str): Directory containing .pkl files with features.
    output_file (str): File path to save the normalized features and targets.

    Returns:
    None
    """
    P = []  # Feature array
    T = []  # Target array
    stats = []  # Store normalization stats (mean, std) for each file

    # Process all training files
    file_list = [f for f in os.listdir(data_dir) if f.endswith('.pkl')]
    for file_name in file_list:
        file_path = os.path.join(data_dir, file_name)
        with open(file_path, 'rb') as f:
            features = pickle.load(f)

        len_limit = len(features)
        train_nn = np.zeros((len_limit, 16))
        t = np.zeros(len_limit)

        for i, feature in enumerate(features):
            g1 = feature["grayfeatures"]
            z1 = feature["zernike"]
            train_nn[i, :] = np.concatenate((np.array(g1, dtype=float), np.array(z1, dtype=float)))
            t[i] = feature["target"]

        mean = np.mean(train_nn, axis=0)
        std = np.std(train_nn, axis=0)
        std[std == 0] = 1  # Prevent division by zero
        train_nn = (train_nn - mean) / std

        stats.append({"mean": mean, "std": std})  # Save stats for testing
        P.append(train_nn)
        T.append(t)

    P = np.vstack(P)
    T = np.hstack(T)

    with open(output_file, 'wb') as f:
        pickle.dump({"features": P, "targets": T, "stats": stats}, f)