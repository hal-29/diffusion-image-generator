# Conditional Diffusion Model for MNIST Generation

## Overview

This project implements a conditional Denoising Diffusion Probabilistic Model (DDPM) for generating MNIST digits based on class labels. The model learns to gradually denoise images while being conditioned on digit labels (0-9), allowing for controllable generation of specific digits.

## Key Features

- **Conditional UNet Architecture**: The model incorporates class embeddings to guide the generation process
- **Flexible Diffusion Process**: Supports both linear and cosine noise schedules
- **DDIM Sampling**: Includes accelerated sampling with Denoising Diffusion Implicit Models
- **Class Conditioning**: Allows generation of specific MNIST digits by providing class labels

## Project Structure

```
diffusion/
├── models/
│   ├── unet.py          # UNet with residual blocks and attention
│   ├── attention.py     # Self-attention layer implementation
├── data/
│   ├── loaders.py       # MNIST data loading utilities
├── diffusion/
│   ├── forward.py       # Forward diffusion process
│   ├── sampling.py      # DDIM sampling implementation
├── utils/
│   ├── schedulers.py    # Noise scheduling functions
│   ├── time_embedding.py # Time step embeddings
│   ├── class_embedding.py # Class label embeddings
├── config.py            # Configuration parameters
├── train.py             # Training script
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/hal-29/conditional-diffusion.git
   cd conditional-diffusion
   ```

2. Install dependencies:
   ```bash
   pip install torch torchvision tqdm
   ```

## Usage

### Training

To train the conditional diffusion model on MNIST:
```bash
python scripts/train.py
```

### Sampling

After training, you can generate samples using the `DDIMSampler` class:

```python
from diffusion.models.unet import UNet
from diffusion.diffusion.forward import ForwardDiffusion
from diffusion.sampling import DDIMSampler

# Load trained model
model = UNet(time_emb_dim=128, class_emb_dim=64, num_classes=10)
model.load_state_dict(torch.load("checkpoints/unet_class_conditional.pth"))

# Initialize sampler
diffusion = ForwardDiffusion(timesteps=1000)
sampler = DDIMSampler(model, diffusion)

# Generate samples for class 3 (digit '3')
labels = torch.tensor([3]).repeat(10)  # Generate 10 samples of digit 3
samples = sampler.sample((10, 1, 32, 32), labels, num_steps=50)
```

## Key Components

### 1. Conditional UNet
- Residual blocks with time and class conditioning
- Self-attention layers for global context
- Skip connections between encoder and decoder

### 2. Diffusion Process
- Linear and cosine noise schedules
- Forward process for gradual noise addition
- Reverse process for learned denoising

### 3. Class Conditioning
- Embedding layer for class labels
- Projection into feature space
- Combined with time embeddings in each residual block

## Results

After training, you can:
1. Generate digits conditioned on specific classes
2. Visualize the denoising process step-by-step
3. Experiment with interpolation between classes

---
Copyright (c) July 2025. built by [Haileiyesus Mesafint](https://github.com/hal-29) as part of iCog-Labs AGI internship.