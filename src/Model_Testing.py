import torch
import numpy as np
import os
import cv2
import pickle
from ..src.PatternNetCNN_Training import PatternNet
from ..src.Model_Eval import evaluate

def test_pattern_net(data_dir, mask_base_path, model_file, output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load trained model
    input_dim, hidden_dim, output_dim = 16, 16 * 2 + 1, 1 
    model = PatternNet(input_dim, hidden_dim, output_dim).to(device)
    model.load_state_dict(torch.load(model_file))
    model.eval()

    # Load test data files
    file_list = [f for f in os.listdir(data_dir) if f.endswith('.pkl')]
    os.makedirs(output_dir, exist_ok=True)

    for file_idx, file_name in enumerate(file_list):
        file_path = os.path.join(data_dir, file_name)
        with open(file_path, 'rb') as f:
            features = pickle.load(f)

        mask_path = os.path.join(mask_base_path, f"{file_idx + 1}_test_mask.gif")
        mask = (cv2.imread(mask_path, 0) > 0).astype(np.uint8)

        # Extract features
        all_feat = np.hstack((
            np.array([f["grayfeatures"] for f in features]),
            np.array([f["zernike"] for f in features])
        ))
        selected_zernikes = [3, 7, 13, 21, 31]
        test_nn = np.hstack((
            all_feat[:, :5],
            all_feat[:, selected_zernikes + 5],
            np.sum(all_feat[:, selected_zernikes + 5], axis=1, keepdims=True)
        ))
        t = np.array([f["target"] for f in features])

        # Normalize features using training stats
        mean = np.mean(test_nn, axis=0)
        std = np.std(test_nn, axis=0)
        std[std == 0] = 1
        test_nn = (test_nn - mean) / std

        # Convert to PyTorch tensors
        test_nn_tensor = torch.tensor(test_nn, dtype=torch.float32).to(device)

        # Make predictions
        with torch.no_grad():
            scores = model(test_nn_tensor).cpu().numpy()

        # Binary classification thresholding
        C = (scores >= 0.6).astype(np.uint8)

        # Generate evaluation values
        eval_values = evaluate(t, C)

        # Generate prediction image
        prob_pix_data = C
        generated_img = np.zeros_like(mask, dtype=float)
        k = 0
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i, j] == 1:
                    generated_img[i, j] = prob_pix_data[k]
                    k += 1

        # Apply top-hat filtering
        windowsize = 3
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (windowsize, windowsize))
        thimage = cv2.morphologyEx(generated_img, cv2.MORPH_TOPHAT, kernel)

        # Save outputs
        output_file = os.path.join(output_dir, f"op_{file_idx + 1}_filt{windowsize}.pkl")
        with open(output_file, 'wb') as f:
            pickle.dump({
                "C": C,
                "AC": t,
                "generatedImg": generated_img,
                "thimage": thimage,
                "evalValues": eval_values,
                "scores": scores
            }, f)