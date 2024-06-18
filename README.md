# Image Enhancement and Denoising using U-Net Architecture
This project implements an image denoising algorithm using a U-Net architecture in TensorFlow. The model is trained to remove noise from grayscale images.

## Table of Contents
1. [Description](#description)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Model Architecture](#model-architecture)
5. [Training](#training)
6. [Evaluation](#evaluation)
7. [Results](#results)


## Description
This project aims to enhance the quality of images by removing noise using a convolutional neural network. The U-Net architecture is particularly effective for image-to-image tasks such as denoising.

## Installation
To get started with this project, clone the repository and install the required dependencies.

### Clone the repository
git clone https://github.com/madhu03111/Image-denoising.git

### Install dependencies
Ensure you have the following dependencies installed:
- numpy
- tensorflow
- pillow
- matplotlib

## Usage
To run the image denoising model, follow these steps:

1. Prepare your dataset: Place your noisy images in the `Train/low` directory and the corresponding clean images in the `Train/high` directory. Similarly, place your test images in the `Test/low` and `Test/high` directories.
2. Train the model:
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, losses, Model
from tensorflow.keras.metrics import MeanSquaredError, MeanAbsoluteError
from PIL import Image
import matplotlib.pyplot as plt

3. Evaluate the model:
```python
results = unet.evaluate(test_generator, steps=50)
```


## Model Architecture

The U-Net model consists of downsampling and upsampling blocks, enabling it to capture context and perform precise localization. The architecture is particularly suited for image-to-image translation tasks.

## Training

Train the model using the `fit` method with the provided data generators. Adjust the number of epochs and steps per epoch as necessary.
```python
history = unet.fit(train_generator,
                   steps_per_epoch=12,  
                   epochs=25, 
                   validation_data=test_generator,
                   validation_steps=6)
```


## Evaluation

Evaluate the model using the `evaluate` method and print the results.
```python
results = unet.evaluate(test_generator, steps=50)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test PSNR: {test_psnr:.2f} dB")
print(f"Test MSE: {test_mse:.4f}")
print(f"Test MAE: {test_mae:.4f}")
```
![image](https://github.com/madhu03111/Image-denoising/assets/149709601/c3866c3a-5518-42a3-8b53-24392da49e4e)


## Results

Plot the training and validation loss to visualize the model's performance over time.
```python
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
```





