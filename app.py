import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

from models.unet_multitask import build_unet_multioutput

st.set_page_config(page_title="Brain Tumor Segmentation - BraTS20", layout="wide")

# --------- Load Model ---------
@st.cache_resource
def load_model():
    model_path = "models/unet_multihead_brats.keras"
    if not os.path.exists(model_path):
        st.error(f"Model weights file not found at {model_path}. Please check the path.")
        return None
    model = build_unet_multioutput(input_shape=(240, 240, 4))
    model.load_weights(model_path)
    return model

model = load_model()
if model is None:
    st.stop()

# --------- Utility Functions ---------

def overlay_mask(flair, wt, tc, et):
    flair_norm = (flair - flair.min()) / (flair.max() - flair.min() + 1e-8)
    if flair_norm.ndim == 2:
        flair_norm = np.expand_dims(flair_norm, axis=-1)
    overlay = np.repeat(flair_norm, 3, axis=-1)

    overlay[wt[..., 0].astype(bool)] = [1, 0, 0]  # Red: WT
    overlay[tc[..., 0].astype(bool)] = [0, 1, 0]  # Green: TC
    overlay[et[..., 0].astype(bool)] = [0, 0, 1]  # Blue: ET

    return overlay

def overlay_errors(flair, gt_masks, pred_masks):
    flair_norm = (flair - flair.min()) / (flair.max() - flair.min() + 1e-8)
    overlay = np.repeat(np.expand_dims(flair_norm, axis=-1), 3, axis=-1)

    for mask_name, color in zip(['wt', 'tc', 'et'], [[1, 0, 0], [0, 1, 0], [0, 0, 1]]):
        gt = gt_masks[mask_name][..., 0].astype(bool)
        pred = pred_masks[mask_name][..., 0].astype(bool)

        tp = np.logical_and(gt, pred)
        fn = np.logical_and(gt, ~pred)
        fp = np.logical_and(~gt, pred)

        overlay[tp] = color            # Correct
        overlay[fn] = [1, 1, 0]        # Yellow for FN
        overlay[fp] = [1, 0, 1]        # Magenta for FP

    return overlay

def load_h5_slice(file):
    try:
        with h5py.File(file, 'r') as f:
            if 'image' not in f or 'mask' not in f:
                raise ValueError("H5 file must contain 'image' and 'mask' datasets.")
            image = f['image'][()]
            mask = f['mask'][()]

        # Normalize image
        image = (image - np.mean(image, axis=(0, 1), keepdims=True)) / \
                (np.std(image, axis=(0, 1), keepdims=True) + 1e-6)

        # Handle different mask formats
        if mask.ndim == 3 and mask.shape[-1] == 3:
            ncr = mask[..., 0]
            ed = mask[..., 1]
            et = mask[..., 2]
            wt = ((ncr + ed + et) > 0).astype(np.float32)[..., np.newaxis]
            tc = ((ncr + et) > 0).astype(np.float32)[..., np.newaxis]
            et = (et > 0).astype(np.float32)[..., np.newaxis]
        else:
            mask = mask.astype(np.uint8)
            wt = (mask > 0).astype(np.float32)[..., np.newaxis]
            tc = np.isin(mask, [1, 4]).astype(np.float32)[..., np.newaxis]
            et = (mask == 4).astype(np.float32)[..., np.newaxis]

        flair = image[:, :, 3]  # FLAIR channel

        return image, flair, wt, tc, et

    except Exception as e:
        st.error(f"Error loading H5 file: {e}")
        return None, None, None, None, None

def preprocess_image(image):
    target_shape = (240, 240, 4)
    if image.shape != target_shape:
        st.warning(f"Resizing input image from {image.shape} to {target_shape}.")
        image = tf.image.resize(image, target_shape[:2], method=tf.image.ResizeMethod.BILINEAR)
        image = tf.cast(image, tf.float32)
    return image

# --------- Default Demo File Handling ---------

DEMO_FILE_PATH = "volume_101_slice_63.h5"

@st.cache_data
def get_demo_data():
    return load_h5_slice(DEMO_FILE_PATH)

# --------- Streamlit UI ---------

st.title("🧠 Brain Tumor Segmentation - BraTS2020")

st.markdown("### 🧪 Try it out:")
st.markdown("Use the demo below to see how the segmentation looks, or upload your own `.h5` file.")

# Upload section
uploaded_file = st.file_uploader("Upload a BraTS slice (.h5 file)", type=['h5'])

# Use uploaded file or fallback to demo
if uploaded_file is not None:
    file_to_use = uploaded_file
    source = "Uploaded File"
else:
    file_to_use = DEMO_FILE_PATH
    source = "Demo File"

# Load data
image, flair, wt, tc, et = load_h5_slice(file_to_use)
if image is None:
    st.error("Could not load selected file.")
else:
    image = preprocess_image(image)
    pred = model.predict(image[np.newaxis, ...])

    # Threshold predictions
    pred_masks = {
        'wt': (pred[0][0] > 0.5).astype(np.float32),
        'tc': (pred[1][0] > 0.5).astype(np.float32),
        'et': (pred[2][0] > 0.5).astype(np.float32)
    }

    gt_masks = {
        'wt': wt,
        'tc': tc,
        'et': et
    }

    # Overlays
    gt_overlay = overlay_mask(flair, wt, tc, et)
    pred_overlay = overlay_mask(flair, pred_masks['wt'], pred_masks['tc'], pred_masks['et'])
    error_overlay = overlay_errors(flair, gt_masks, pred_masks)

    # Display width
    IMAGE_DISPLAY_WIDTH = 300

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("**Original FLAIR Image**")
        if flair.ndim == 3 and flair.shape[-1] == 1:
            flair = flair.squeeze(axis=-1)
        flair_norm = (flair - flair.min()) / (flair.max() - flair.min() + 1e-8)
        flair_display = np.repeat(flair_norm[..., np.newaxis], 3, axis=-1)
        st.image(flair_display, clamp=True, width=IMAGE_DISPLAY_WIDTH)

    with col2:
        st.markdown(f"**Ground Truth Overlay ({source})**")
        st.image(gt_overlay, clamp=True, width=IMAGE_DISPLAY_WIDTH)

    with col3:
        st.markdown("**Predicted Mask Overlay**")
        st.image(pred_overlay, clamp=True, width=IMAGE_DISPLAY_WIDTH)

    with col4:
        st.markdown("**Error Mask Overlay**\nFN=Yellow | FP=Magenta")
        st.image(error_overlay, clamp=True, width=IMAGE_DISPLAY_WIDTH)