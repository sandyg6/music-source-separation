# model.py

import tensorflow as tf
from tensorflow.keras import layers, Model
from config import INPUT_SHAPE, NUM_STEMS

def unet_model(input_shape=INPUT_SHAPE, num_stems=NUM_STEMS):
    """Create a U-Net model for source separation."""
    inputs = layers.Input(shape=input_shape)

    # Encoding path (downsampling)
    conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    pool1 = layers.MaxPooling2D((2, 2))(conv1)

    conv2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    pool2 = layers.MaxPooling2D((2, 2))(conv2)

    conv3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    pool3 = layers.MaxPooling2D((2, 2))(conv3)

    conv4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(pool3)
    pool4 = layers.MaxPooling2D((2, 2))(conv4)

    # Bottleneck
    bottleneck = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(pool4)

    # Decoding path (upsampling)
    up1 = layers.Conv2DTranspose(512, (3, 3), strides=(2, 2), activation='relu', padding='same')(bottleneck)
    concat1 = layers.concatenate([up1, conv4])
    upconv1 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(concat1)

    up2 = layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), activation='relu', padding='same')(upconv1)
    concat2 = layers.concatenate([up2, conv3])
    upconv2 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(concat2)

    up3 = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), activation='relu', padding='same')(upconv2)
    concat3 = layers.concatenate([up3, conv2])
    upconv3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(concat3)

    up4 = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), activation='relu', padding='same')(upconv3)
    concat4 = layers.concatenate([up4, conv1])
    upconv4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(concat4)

    # Output layer (separated stems)
    outputs = layers.Conv2D(num_stems, (1, 1), activation='linear', padding='same')(upconv4)

    model = Model(inputs, outputs)
    return model
