# DCGAN Implementation in PyTorch

A simple implementation of Deep Convolutional Generative Adversarial Network (DCGAN) for generating images using PyTorch.

## Overview

This project implements a DCGAN that trains a generator and discriminator network to generate new images from random noise. The implementation includes:

- **Generator**: Transforms random noise vectors into realistic images using transposed convolutions
- **Discriminator**: Distinguishes between real and generated images using convolutional layers
- **Training Loop**: Alternates between training discriminator and generator with adversarial loss

## Architecture

### Generator
- Input: Random noise vector (100 dimensions)
- Architecture: 5 transpose convolutional layers with batch normalization and ReLU activation
- Output: Generated images (64x64 pixels)

### Discriminator  
- Input: Images (real or generated)
- Architecture: 5 convolutional layers with batch normalization and LeakyReLU activation
- Output: Single probability score (real vs fake)

## Configuration

```python
config = {
    'data_path': 'dataset/',
    'batch_size': 1024,
    'learning_rate': 2e-4,
    'channel': 1,           # Grayscale images
    'image_size': 64,
    'feature_map_size': 64,
    'input_noise_dim': 100,
    'num_epochs': 5
}
```

## Training Results

The model trains for 5 epochs and displays:
- Discriminator and Generator loss curves
- Generated image samples every epoch
- Training progress with loss metrics

## Requirements

- PyTorch
- torchvision 
- matplotlib
- numpy
- tqdm

## Usage

1. Prepare your dataset in the specified directory
2. Configure hyperparameters in the config dictionary
3. Run the training notebook
4. Monitor training progress and generated samples

