# Detecting Retinal Vasculature

This repository contains the implementation of a supervised blood vessel segmentation technique for digital fundus images using Zernike Moment-based features. The project introduces a novel approach to detect retinal vasculature with high accuracy, aiding in the diagnosis of ophthalmologic and cardiovascular disorders.

## Overview

Retinal vessel segmentation is crucial for diagnosing various diseases such as diabetic retinopathy, hypertension, and other cardiovascular conditions. This project combines advanced pre-processing techniques with a supervised artificial neural network (ANN) to accurately classify blood vessel pixels in fundus images.

## Methodology

The segmentation pipeline consists of three main stages:

![Methodology]{imgs/methodology.png "Methodology"}

### 1. Preprocessing
- **Input**: Fundus images.
- **Output**: Preprocessed images with enhanced vessel structures and suppressed background noise.

Steps:
- Extract the green channel, which offers maximum contrast for vessels.
- Apply morphological operations for light reflex removal.
- Perform contrast enhancement using CLAHE.
- Generate a vessel-enhanced image via top-hat filtering.

### 2. Feature Extraction
- **Gray-Level Features**:
  - Intensity differences within local windows (minimum, maximum, mean, and standard deviation).
- **Zernike Moment Features**:
  - 36 Zernike coefficients calculated from 17×17 windows.
  - Dominant coefficients selected based on maximum discriminability.

### 3. Classification
- Features are passed to an ANN:
  - Input Layer: 11 nodes (features).
  - Hidden Layers: 3 layers with 23 nodes each (tanh activation).
  - Output Layer: 2 nodes (softmax for vessel/background classification).

## Performance

| Metric                | DRIVE Dataset | STARE Dataset |
|-----------------------|---------------|---------------|
| **Accuracy**          | 94.5%        | 94.86%        |
| **Sensitivity**       | 69.94%       | 62.98%        |
| **Specificity**       | 98.11%       | 98.39%        |
| **AUC**              | 93.94%       | 95.07%        |

The proposed model achieves high accuracy while preserving thin blood vessels, a challenge for traditional methods.

## Dataset

The method is evaluated on two publicly available datasets:
- **[DRIVE](https://www.isi.uu.nl/Research/Databases/DRIVE/)**: 40 fundus images (20 for training, 20 for testing).
- **[STARE](http://cecas.clemson.edu/~ahoover/stare/)**: 20 fundus images for testing.

## Installation

Clone the repository and install the dependencies:
```bash
git clone https://github.com/TakedoNick/Detecting_Retinal_Vasculature.git
cd Detecting_Retinal_Vasculature
pip install -r requirements.txt
```

# Usage

## Training
```bash
python train.py --data_dir <path_to_training_data>
```

## Testing
```bash
python test.py --data_dir <path_to_testing_data> --output_dir <output_path>
```

### Visualization
```bash
python visualize_results.py --results_dir <path_to_results>
```

# References
References
•	Adapa D., Joseph Raj A. N., et al. “A supervised blood vessel segmentation technique for digital fundus images using Zernike Moment-based features.” PLOS ONE, 2020.
 
•	DRIVE Dataset

•	STARE Dataset
