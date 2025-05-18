import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusers import DDIMScheduler, UNet2DModel
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from IPPy import operators
from IPPy import utilities as IPutils
from miscellaneous import data, utilities

# --- Set device ---
device = utilities.get_device()
print(f"Device used: {device}.")

# --- Configuration ---
MODEL_PATH = "./model_weights/UNet_128/"
SAVING_PATH = "./results/UNet_128/"
SEED = None
TIMESTEPS = 200

STEP_SIZE = 10.0  # Weight of data consistency gradient

NOISE_LEVEL = 0.0
START_ANGLE, END_ANGLE = 0, 180
N_ANGLES = 60
DET_SIZE = 512

# --- Load model + scheduler ---
model = UNet2DModel.from_pretrained(os.path.join(MODEL_PATH, "unet")).to(device)
scheduler = DDIMScheduler.from_pretrained(os.path.join(MODEL_PATH, "scheduler"))

# --- Load data ---
test_data = data.MayoDataset(
    data_path="../data/Mayo/test",
    data_shape=model.config.sample_size,
)
x_true = test_data[0].unsqueeze(0).to(device)

# Define operator
K = operators.CTProjector(
    img_shape=x_true.shape[-2:],
    angles=np.linspace(np.deg2rad(START_ANGLE), np.deg2rad(END_ANGLE), N_ANGLES),
    geometry="parallel",
    det_size=DET_SIZE,
)

# Generate test problem
if SEED:
    torch.manual_seed(SEED)  # Set seed if required
y = K(x_true)
y_delta = y + IPutils.gaussian_noise(y, NOISE_LEVEL)

# Set optimizer and loss fn
loss_fn = torch.nn.MSELoss()

# Initialize metrics
psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

# Set scheduler
scheduler.set_timesteps(TIMESTEPS)

# Start from pure noise
x = torch.randn(x_true.shape, device=device)

# DPS loop
start_time = time.time()
psnr_vec = []
ssim_vec = []
for i, t in enumerate(scheduler.timesteps):
    x.requires_grad_(True)

    # Predict noise with the diffusion model
    with torch.no_grad():
        model_output = model(x, t).sample  # Output is predicted noise ε_theta

    # Estimate the denoised image x0
    x0 = scheduler.step(model_output, t, x).prev_sample

    # Convert x0 to [0, 1] for applying K
    x0_clamped = x0.clamp(-1, 1)  # Optional, just to be safe
    x0_in_01 = (x0_clamped + 1) / 2  # Scale to [0, 1]

    with torch.no_grad():
        psnr_vec.append(psnr_metric(x0_in_01, x_true).item())
        ssim_vec.append(ssim_metric(x0_in_01, x_true).item())

    # Compute data fidelity gradient: ∇_x ||K(x0) - y||^2
    loss = loss_fn(K(x0_in_01), y_delta)
    grad = torch.autograd.grad(loss, x, retain_graph=True)[0]

    # Update with data guidance (projected Langevin update)
    x = x - STEP_SIZE * grad
    x = x.detach()  # Detach for next step

    # Apply diffusion update (reinject noise)
    x = scheduler.step(model_output, t, x).prev_sample

    print(
        f"(Time {time.time() - start_time:0.2f}s) Step {i+1:03d} | Loss: {loss.item():.4f} | PSNR: {psnr_vec[-1]:.2f} | SSIM: {ssim_vec[-1]:.4f}"
    )

# Final reconstruction
reconstructed = x0.detach()
