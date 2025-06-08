#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from dataprep_multitask import get_train_val_datasets
from utils.losses import dice_loss, dice_coefficient, focal_tversky_loss  # custom metrics
from models.unet_multitask import build_unet_multioutput
from glob import glob
from visualize import visualize_batch
import h5py

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 3=errors only
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

# Set seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Paths
DATA_DIR = "BraTS20/BraTS2020_training_data/content/data"
MODEL_SAVE_PATH = "models/unet_brats.keras"

# Hyperparameters
BATCH_SIZE = 8
EPOCHS = 30

# Load datasets
train_dataset, val_dataset = get_train_val_datasets(DATA_DIR, batch_size=BATCH_SIZE)


# ------------------- Model Setup -------------------
model = build_unet_multioutput(input_shape=(240, 240, 4))
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss={
        'wt_head': focal_tversky_loss,
        'tc_head': focal_tversky_loss,
        'et_head': focal_tversky_loss,
    },
    loss_weights={
        'wt_head': 1.0,
        'tc_head': 1.0,
        'et_head': 2.0,
    },
    metrics={
        'wt_head': dice_coefficient,
        'tc_head': dice_coefficient,
        'et_head': dice_coefficient,
    }
)

# ------------------- Callbacks -------------------
callbacks = [
    ModelCheckpoint("models/unet_multihead_brats.keras", save_best_only=True, monitor='val_et_head_dice_coefficient', mode='max'),
    ReduceLROnPlateau(monitor='val_et_head_dice_coefficient', factor=0.5, patience=4, min_lr=1e-6, verbose=1),
    EarlyStopping(monitor='val_et_head_dice_coefficient', mode='max', patience=8, restore_best_weights=True)
]

# ------------------- Training -------------------
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    steps_per_epoch=len(train_dataset) // BATCH_SIZE,
    validation_steps=len(val_dataset) // BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=callbacks
)


# In[4]:


import matplotlib.pyplot as plt

print("\nGenerating predictions and visualizing...\n")

# Get one batch from validation set
for x_batch, y_batch in val_dataset.take(1):
    preds = model.predict(x_batch)
    pred_masks = {
        'wt_head': (preds[0] > 0.5).astype(np.float32),
        'tc_head': (preds[1] > 0.5).astype(np.float32),
        'et_head': (preds[2] > 0.5).astype(np.float32),
    }

    def overlay_prediction(img, pred_masks):
        flair = img[:, :, 3]
        flair_norm = (flair - flair.min()) / (flair.max() - flair.min() + 1e-8)
        overlay = np.stack([flair_norm]*3, axis=-1)
        overlay[pred_masks['wt_head'][..., 0] == 1] = [1, 0, 0]  # Red
        overlay[pred_masks['tc_head'][..., 0] == 1] = [0, 1, 0]  # Green
        overlay[pred_masks['et_head'][..., 0] == 1] = [0, 0, 1]  # Blue
        return overlay

    for i in range(min(3, x_batch.shape[0])):
        image = x_batch[i].numpy()
        flair = image[:, :, 3]
        pred_overlay = overlay_prediction(image, {
            'wt_head': pred_masks['wt_head'][i],
            'tc_head': pred_masks['tc_head'][i],
            'et_head': pred_masks['et_head'][i],
        })

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(flair, cmap='gray')
        plt.title("FLAIR")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(pred_overlay)
        plt.title("Predicted Mask Overlay")
        plt.axis('off')
        plt.show()

    break


# In[5]:


import matplotlib.pyplot as plt

def overlay_mask(flair, wt, tc, et):
    """
    Build RGB overlay on FLAIR using WT (red), TC (green), ET (blue) masks.
    """
    flair_norm = (flair - flair.min()) / (flair.max() - flair.min() + 1e-8)
    overlay = np.stack([flair_norm]*3, axis=-1)
    overlay[wt[..., 0] == 1] = [1, 0, 0]  # Red
    overlay[tc[..., 0] == 1] = [0, 1, 0]  # Green
    overlay[et[..., 0] == 1] = [0, 0, 1]  # Blue
    return overlay


# Predict one batch
for x_batch, y_batch in val_dataset.take(1):
    preds = model.predict(x_batch)

    for i in range(3):  # Show first 3 samples
        img = x_batch[i].numpy()
        flair = img[:, :, 3]

        # Predicted masks (thresholded)
        wt_pred = (preds[0][i] > 0.5).astype(np.float32)
        tc_pred = (preds[1][i] > 0.5).astype(np.float32)
        et_pred = (preds[2][i] > 0.5).astype(np.float32)

        # Ground truth masks
        wt_true = y_batch['wt_head'][i].numpy()
        tc_true = y_batch['tc_head'][i].numpy()
        et_true = y_batch['et_head'][i].numpy()

        # Overlays
        gt_overlay = overlay_mask(flair, wt_true, tc_true, et_true)
        pred_overlay = overlay_mask(flair, wt_pred, tc_pred, et_pred)

        # Plot side-by-side
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(gt_overlay)
        plt.title("Ground Truth Overlay")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(pred_overlay)
        plt.title("Predicted Overlay")
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    break


# In[6]:


import matplotlib.pyplot as plt
import numpy as np

def overlay_mask(flair, wt, tc, et):
    """
    Build RGB overlay on FLAIR using WT (red), TC (green), ET (blue).
    """
    flair_norm = (flair - flair.min()) / (flair.max() - flair.min() + 1e-8)
    overlay = np.stack([flair_norm]*3, axis=-1)
    overlay[wt[..., 0] == 1] = [1, 0, 0]  # Red: WT
    overlay[tc[..., 0] == 1] = [0, 1, 0]  # Green: TC
    overlay[et[..., 0] == 1] = [0, 0, 1]  # Blue: ET
    return overlay

def overlay_errors(flair, gt, pred):
    """
    Compare prediction vs ground truth:
    - TP (correct) regions keep their color
    - FP (predicted but not GT): Magenta
    - FN (GT but not predicted): Yellow
    """
    flair_norm = (flair - flair.min()) / (flair.max() - flair.min() + 1e-8)
    overlay = np.stack([flair_norm]*3, axis=-1)

    for mask_name, color, idx in zip(['wt_head', 'tc_head', 'et_head'],
                                     [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                     range(3)):
        gt_mask = gt[mask_name][..., 0]
        pred_mask = pred[mask_name][..., 0]

        tp = np.logical_and(gt_mask == 1, pred_mask == 1)
        fn = np.logical_and(gt_mask == 1, pred_mask == 0)  # Missed
        fp = np.logical_and(gt_mask == 0, pred_mask == 1)  # Extra

        overlay[tp] = color            # Correct
        overlay[fn] = [1, 1, 0]        # Yellow for FN
        overlay[fp] = [1, 0, 1]        # Magenta for FP

    return overlay

# Run on one batch
for x_batch, y_batch in val_dataset.take(1):
    preds = model.predict(x_batch)

    for i in range(3):
        img = x_batch[i].numpy()
        flair = img[:, :, 3]

        # Threshold predictions
        pred_masks = {
            'wt_head': (preds[0][i] > 0.5).astype(np.float32),
            'tc_head': (preds[1][i] > 0.5).astype(np.float32),
            'et_head': (preds[2][i] > 0.5).astype(np.float32)
        }

        gt_masks = {
            'wt_head': y_batch['wt_head'][i].numpy(),
            'tc_head': y_batch['tc_head'][i].numpy(),
            'et_head': y_batch['et_head'][i].numpy()
        }

        # Overlays
        gt_overlay = overlay_mask(flair, gt_masks['wt_head'], gt_masks['tc_head'], gt_masks['et_head'])
        pred_overlay = overlay_mask(flair, pred_masks['wt_head'], pred_masks['tc_head'], pred_masks['et_head'])
        error_overlay = overlay_errors(flair, gt_masks, pred_masks)

        # Plot all 3
        plt.figure(figsize=(18, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(gt_overlay)
        plt.title("Ground Truth Overlay")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(pred_overlay)
        plt.title("Predicted Overlay")
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(error_overlay)
        plt.title("Error Overlay\nFN=Yellow | FP=Magenta")
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    break


# In[ ]:




