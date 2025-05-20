import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusers import DDIMPipeline, DDIMScheduler, UNet2DModel
from PIL import Image
from torch import optim
from torchvision import transforms
from tqdm import tqdm

from miscellaneous import utilities

# --- Set device ---
device = utilities.get_device()
print(f"Device used: {device}.")

# --- Configuration ---
MODEL_PATH = "./model_weights/UNet_Attn_256_Large_L/"
BATCH_SIZE = 4
TIMESTEPS = 50

# --- Load model + scheduler ---
model = UNet2DModel.from_pretrained(os.path.join(MODEL_PATH, "unet")).to(device)
scheduler = DDIMScheduler.from_pretrained(os.path.join(MODEL_PATH, "scheduler"))

# --- Generation ---
model.eval()

# Initialization
x_t = torch.randn(
    (BATCH_SIZE, 1, model.config.sample_size, model.config.sample_size), device=device
)
scheduler.set_timesteps(TIMESTEPS, device=device)

# Reverse diffusion process
for t in scheduler.timesteps:
    with torch.no_grad():
        model_output = model(x_t, t).sample
    x_t = scheduler.step(
        model_output,
        t,
        x_t,
        eta=0.0,
    ).prev_sample

# Visualization
for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.imshow(x_t[i, 0].detach().cpu().numpy(), cmap="gray")
    plt.axis("off")
plt.show()
