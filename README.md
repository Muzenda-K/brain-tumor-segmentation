# Brain Tumor Segmentation using Multi-Output U-Net

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19.0-orange)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![License](https://img.shields.io/badge/License-MIT-green)](https://opensource.org/licenses/MIT)

## Table of Contents

1. [Project Overview](#project-overview)
2. [Model Architecture](#model-architecture)
3. [Installation](#installation)
4. [Dataset Preparation](#dataset-preparation)
5. [Training](#training)
6. [Evaluation](#evaluation)
7. [Visualization](#visualization)
8. [Performance Metrics](#performance-metrics)
9. [Customization](#customization)
10. [Troubleshooting](#troubleshooting)
11. [License](#license)

## Project Overview

This project implements a multi-task U-Net model for brain tumor segmentation from MRI scans. The model simultaneously predicts three tumor sub-regions:

- Whole Tumor (WT)
- Tumor Core (TC)
- Enhancing Tumor (ET)

The implementation uses TensorFlow/Keras with custom loss functions and visualization tools for model evaluation.

## Model Architecture

The model is based on a U-Net architecture with the following key components:

### Encoder Path

- 4 encoding blocks with [64, 128, 256, 512] filters
- Each block consists of:
  - Two 3×3 convolutional layers with BatchNorm and ReLU
  - Max pooling (2×2) for downsampling

### Bottleneck

- 1024 filters with 50% dropout for regularization

### Decoder Path

- 4 decoding blocks with [512, 256, 128, 64] filters
- Each block consists of:
  - Transposed convolution (2×2) for upsampling
  - Concatenation with skip connections
  - Two 3×3 convolutional layers with BatchNorm and ReLU

### Multi-Task Heads

- Three parallel output heads (1×1 conv + sigmoid)
  - WT head (whole tumor)
  - TC head (tumor core)
  - ET head (enhancing tumor)

## Installation

1. Clone this repository:

```bash
git clone https://github.com/yourusername/brain-tumor-segmentation.git
cd brain-tumor-segmentation
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset Preparation

The model expects input images with shape (240, 240, 4) where the channels are:

1. T1-weighted
2. T1-weighted with contrast
3. T2-weighted
4. FLAIR

### Preprocessing

1. Normalize each modality separately (zero mean, unit variance)
2. Resample all images to 1mm isotropic resolution
3. Register all modalities to a common space
4. Crop/pad to (240, 240) size

## Training

To train the model:

```python
from model import build_unet_multioutput
from losses import focal_tversky_loss

# Build model
model = build_unet_multioutput(input_shape=(240, 240, 4))

# Compile with multi-task losses
model.compile(optimizer='adam',
             loss={
                 'wt_head': focal_tversky_loss,
                 'tc_head': focal_tversky_loss,
                 'et_head': focal_tversky_loss
             },
             metrics={'wt_head': dice_coefficient,
                      'tc_head': dice_coefficient,
                      'et_head': dice_coefficient})

# Train
history = model.fit(train_dataset,
                    validation_data=val_dataset,
                    epochs=num_epochs,
                    callbacks=[...])
```

## Visualization

The package includes visualization utilities to compare predictions with ground truth:

### Overlay Types

1. **Ground Truth Overlay**:

   - WT: Red
   - TC: Green
   - ET: Blue

2. **Prediction Overlay**:

   - Same color scheme as ground truth

3. **Error Overlay**:
   - True Positives: Original colors
   - False Negatives (missed tumors): Yellow
   - False Positives (extra predictions): Magenta

Example visualization code:

```python
from visualization import overlay_mask, overlay_errors

# For one sample
flair = x_batch[i, ..., 3]  # FLAIR channel
gt_overlay = overlay_mask(flair, gt_wt, gt_tc, gt_et)
pred_overlay = overlay_mask(flair, pred_wt, pred_tc, pred_et)
error_overlay = overlay_errors(flair, gt_masks, pred_masks)
```

## Performance Metrics

Dice score:
| Region | Dice Score |
|--------|------------|
| WT | 0.8548 |
| TC | 0.7484 |
| ET | 0.7242 |

## Customization

### Model Parameters

Modify `build_unet_multioutput()` to change:

- Input shape
- Number of filters
- Dropout rate
- Depth of network

### Loss Functions

Available loss functions:

1. `focal_tversky_loss`: Focuses on hard examples
2. `dice_loss`: Standard Dice implementation
3. `binary_crossentropy`: Traditional BCE

### Visualization

Sample overlay screenshots:
Compare prediction vs ground truth: - TP (correct) regions keep their color - FP (predicted but not GT): Magenta - FN (GT but not predicted): Yellow

![FLAIR Image](images/flair.png)
![Ground Truth Overlay](images/gt_overlay.png)
![Prediction Overlay](images/pred_overlay.png)
![Error Overlay](images/error_overlay.png)

### Demo app

![Demo](demo.gif)

## Troubleshooting

1. **Out of Memory Errors**:

   - Reduce batch size
   - Use mixed precision training
   - Crop images to smaller size

2. **Poor Convergence**:

   - Check data normalization
   - Adjust learning rate
   - Try different loss weights

3. **NaN Losses**:
   - Add small epsilon (1e-7) to denominators
   - Clip predictions (e.g., to [1e-7, 1-1e-7])

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```
@misc{brain-tumor-segmentation-unet,
  author = {Muzenda K},
  title = {Multi-Output U-Net for Brain Tumor Segmentation},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Muzenda-K/brain-tumor-segmentation}}
}
```
