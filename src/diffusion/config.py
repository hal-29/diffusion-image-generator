import torch


class Config:
    # Diffusion
    TIMESTEPS = 1000
    IMG_SIZE = 32
    BETA_START = 0.0001
    BETA_END = 0.02

    # Training
    BATCH_SIZE = 128
    LR = 1e-4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


config = Config()
