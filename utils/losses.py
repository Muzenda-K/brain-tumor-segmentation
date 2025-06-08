# utils/losses.py

import tensorflow as tf

def dice_coefficient(y_true, y_pred, smooth=1e-7):
    """
    Dice coefficient for evaluating segmentation performance.
    
    Args:
        y_true: Ground truth masks (batch_size, H, W, 3) - one-hot encoded
        y_pred: Predicted masks (batch_size, H, W, 3) - softmax probabilities
        smooth: Smoothing factor to avoid division by zero
    
    Returns:
        Dice coefficient score (float)
    """
    y_true_f = tf.reshape(y_true, [-1, 3])
    y_pred_f = tf.reshape(y_pred, [-1, 3])
    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=0)
    denom = tf.reduce_sum(y_true_f + y_pred_f, axis=0)
    return tf.reduce_mean((2. * intersection + smooth) / (denom + smooth))


def dice_loss(y_true, y_pred):
    """
    Dice loss function for model training.
    """
    return 1. - dice_coefficient(y_true, y_pred)


def tversky_coefficient(y_true, y_pred, alpha=0.7, beta=0.3, smooth=1e-7):
    """
    Generalized Dice Loss: Tversky Index
    
    Args:
        alpha: Weight for false positives
        beta: Weight for false negatives
    """
    y_true_f = tf.reshape(y_true, [-1, 3])
    y_pred_f = tf.reshape(y_pred, [-1, 3])

    true_pos = tf.reduce_sum(y_true_f * y_pred_f, axis=0)
    false_neg = tf.reduce_sum(y_true_f * (1. - y_pred_f), axis=0)
    false_pos = tf.reduce_sum((1. - y_true_f) * y_pred_f, axis=0)

    return tf.reduce_mean(
        (true_pos + smooth) / (true_pos + alpha * false_neg + beta * false_pos + smooth)
    )


def tversky_loss(y_true, y_pred):
    """
    Tversky loss function.
    """
    return 1. - tversky_coefficient(y_true, y_pred)


def focal_tversky_loss(y_true, y_pred, gamma=0.75):
    """
    Focal Tversky Loss: emphasizes hard-to-segment regions.
    """
    tv = tversky_coefficient(y_true, y_pred)
    return tf.pow(1. - tv, gamma)