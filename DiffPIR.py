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

LAMBDA_REG = 0.1  # Regularization parameter
SIGMA_N = 0.1  # Noise level

NOISE_LEVEL = 0.1
START_ANGLE, END_ANGLE = 0, 180
N_ANGLES = 60
DET_SIZE = 512

# --- Load model + scheduler ---
model = UNet2DModel.from_pretrained(os.path.join(MODEL_PATH, "unet")).to(device)
scheduler = DDIMScheduler.from_pretrained(os.path.join(MODEL_PATH, "scheduler"))
scheduler.set_timesteps(TIMESTEPS)

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

# Initialize x with zeros or any prior
x = torch.zeros(x_true.shape, device=device)

# Ensure x_true and y are on the correct device
x_true = x_true.to(device)
y_delta = y_delta.to(device)

# DiffPIR loop
start_time = time.time()
psnr_vec = []
ssim_vec = []
for i in range(TIMESTEPS):
    x.requires_grad_(True)

    # Data Fidelity Step
    loss = (1 / (2 * SIGMA_N**2)) * loss_fn(K(x), y)
    fidelity_grad = torch.autograd.grad(loss, x, retain_graph=True)[0]
    x = x - LAMBDA_REG * fidelity_grad
    x = x.detach()

    # Prior Step using diffusion model
    t = scheduler.timesteps[i]
    with torch.no_grad():
        noise_pred = model(x, t).sample
    x = scheduler.step(noise_pred, t, x).prev_sample
    x = x.detach()

    # Monitoring
    x_clamped = x.clamp(-1, 1)
    x_01 = (x_clamped + 1) / 2  # Rescale to [0, 1]

    with torch.no_grad():
        psnr_vec.append(psnr_metric(x_01, x_true).item())
        ssim_vec.append(ssim_metric(x_01, x_true).item())

    print(
        f"(Time {time.time() - start_time:0.2f}s) Step {i+1:03d} | Loss: {loss.item():.4f} | PSNR: {psnr_vec[-1]:.2f} | SSIM: {ssim_vec[-1]:.4f}"
    )
