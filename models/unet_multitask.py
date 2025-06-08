#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras import backend as K

# ------------------ U-Net Blocks ------------------
def conv_block(input_tensor, num_filters):
    x = layers.Conv2D(num_filters, 3, padding='same')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(num_filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x

def encoder_block(input_tensor, num_filters):
    x = conv_block(input_tensor, num_filters)
    p = layers.MaxPooling2D(2)(x)
    return x, p

def decoder_block(input_tensor, skip_tensor, num_filters):
    x = layers.Conv2DTranspose(num_filters, 2, strides=2, padding='same')(input_tensor)
    x = layers.Concatenate()([x, skip_tensor])
    x = conv_block(x, num_filters)
    return x

# ------------------ Multi-Output U-Net ------------------
def build_unet_multioutput(input_shape=(240, 240, 4)):
    inputs = Input(shape=input_shape)

    # Encoder
    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    # Bottleneck + Dropout
    b = conv_block(p4, 1024)
    b = layers.Dropout(0.5)(b)

    # Decoder
    d1 = decoder_block(b, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    # Multi-task Heads
    wt_out = layers.Conv2D(1, 1, activation='sigmoid', name='wt_head')(d4)  # Whole Tumor
    tc_out = layers.Conv2D(1, 1, activation='sigmoid', name='tc_head')(d4)  # Tumor Core
    et_out = layers.Conv2D(1, 1, activation='sigmoid', name='et_head')(d4)  # Enhancing Tumor

    model = Model(inputs=[inputs], outputs=[wt_out, tc_out, et_out], name="U-Net-MultiOutput")
    return model

# ------------------ Loss and Metrics ------------------
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def focal_tversky_loss(alpha=0.5, beta=0.7, gamma=1.33):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        tp = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
        fp = tf.reduce_sum((1 - y_true) * y_pred, axis=[1, 2, 3])
        fn = tf.reduce_sum(y_true * (1 - y_pred), axis=[1, 2, 3])

        tversky = (tp + 1e-7) / (tp + alpha * fp + beta * fn + 1e-7)
        return tf.reduce_mean(tf.pow((1 - tversky), gamma))
    return loss
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras import backend as K

# ------------------ U-Net Blocks ------------------
def conv_block(input_tensor, num_filters):
    x = layers.Conv2D(num_filters, 3, padding='same')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(num_filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x

def encoder_block(input_tensor, num_filters):
    x = conv_block(input_tensor, num_filters)
    p = layers.MaxPooling2D(2)(x)
    return x, p

def decoder_block(input_tensor, skip_tensor, num_filters):
    x = layers.Conv2DTranspose(num_filters, 2, strides=2, padding='same')(input_tensor)
    x = layers.Concatenate()([x, skip_tensor])
    x = conv_block(x, num_filters)
    return x

# ------------------ Multi-Output U-Net ------------------
def build_unet_multioutput(input_shape=(240, 240, 4)):
    inputs = Input(shape=input_shape)

    # Encoder
    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    # Bottleneck + Dropout
    b = conv_block(p4, 1024)
    b = layers.Dropout(0.5)(b)

    # Decoder
    d1 = decoder_block(b, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    # Multi-task Heads
    wt_out = layers.Conv2D(1, 1, activation='sigmoid', name='wt_head')(d4)  # Whole Tumor
    tc_out = layers.Conv2D(1, 1, activation='sigmoid', name='tc_head')(d4)  # Tumor Core
    et_out = layers.Conv2D(1, 1, activation='sigmoid', name='et_head')(d4)  # Enhancing Tumor

    model = Model(inputs=[inputs], outputs=[wt_out, tc_out, et_out], name="U-Net-MultiOutput")
    return model

