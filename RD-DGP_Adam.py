import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusers import DDIMScheduler, UNet2DModel
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from IPPy import operators
from IPPy import utilities as IPutils
from IPPy.nn import losses
from miscellaneous import data, initializers, utilities

# --- Set device ---
device = utilities.get_device()
print(f"Device used: {device}.")

# --- Configuration ---
MODEL_PATH = "./model_weights/UNet_256/"
SAVING_PATH = "./results/UNet_256/"
GENERATION_TIMESTEPS = 50

NOISE_LEVEL = 0.0
START_ANGLE, END_ANGLE = 0, 180
N_ANGLES = 120
DET_SIZE = 256

RECONSTRUCTION_TIMESTEPS = 15
NUM_ITER = 0
STEP_SIZE = 1e-2
SEED = None

# --- Load model + scheduler ---
model = UNet2DModel.from_pretrained(os.path.join(MODEL_PATH, "unet")).to(device)
scheduler = DDIMScheduler.from_pretrained(os.path.join(MODEL_PATH, "scheduler"))


# --- Load data ---
test_data = data.MayoDataset(
    data_path="../data/Mayo/train",
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

# --- Run inversion process ---
z = initializers.RandomInitializer(model, scheduler)(seed=None)
z = initializers.InverseInitializer(model, scheduler)(x_true, num_timesteps=10)

# Set optimizer and loss fn
optimizer = torch.optim.AdamW([z], lr=STEP_SIZE, weight_decay=0)
loss_fn = losses.MixedLoss(
    (
        torch.nn.MSELoss(),
        torch.nn.SmoothL1Loss(),
        losses.SSIMLoss(),
        losses.FourierLoss(),
    ),
    weight_parameters=(1, 0, 0, 0),
)

# Initialize metrics
psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

# Optimize z such that KG(z) ≈ y_delta
scheduler.set_timesteps(RECONSTRUCTION_TIMESTEPS)

start_time = time.time()
psnr_vec = []
ssim_vec = []
for step in range(NUM_ITER):
    optimizer.zero_grad()

    # Reverse diffusion process (Generation)
    x = z
    for t in scheduler.timesteps:
        with torch.no_grad():
            noise_pred = model(x, t).sample
        x = scheduler.step(noise_pred, t, x).prev_sample
    # Normalize in [0, 1] range
    x = (x + 1.0) / 2.0

    # Compute loss and gradient update
    loss = loss_fn(K(x), y_delta)

    with torch.no_grad():
        psnr_vec.append(psnr_metric(x, x_true).item())
        ssim_vec.append(ssim_metric(x, x_true).item())
    loss.backward()
    optimizer.step()
    print(
        f"(Time {time.time() - start_time:0.2f}s) Step {step:03d} | Loss: {loss.item():.4f} | PSNR: {psnr_vec[-1]:.2f} | SSIM: {ssim_vec[-1]:.4f}"
    )

# --- Create last generation
scheduler.set_timesteps(GENERATION_TIMESTEPS)

x = z
for t in scheduler.timesteps:
    with torch.no_grad():
        noise_pred = model(x, t).sample
    x = scheduler.step(noise_pred, t, x).prev_sample
# Normalize in [0, 1] range
x_gen = (x + 1.0) / 2.0

# --- Saving ---
print(f"Saved results can be found in: {SAVING_PATH}...")
os.makedirs(SAVING_PATH, exist_ok=True)

# Metrics
torch.save(torch.tensor(psnr_vec), os.path.join(SAVING_PATH, "psnr.pth"))
torch.save(torch.tensor(ssim_vec), os.path.join(SAVING_PATH, "ssim.pth"))

# Images
plt.subplot(1, 3, 1)
plt.imshow(x_true.detach().cpu().numpy()[0, 0], cmap="gray")
plt.title("True")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(y_delta.detach().cpu().numpy()[0, 0], cmap="gray")
plt.title("Corrupted")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(x_gen.detach().cpu().numpy()[0, 0], cmap="gray")
plt.title(f"Reconstructed\n(SSIM {ssim_metric(x_gen, x_true).item():0.4f})")
plt.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(SAVING_PATH, "reconstruction.png"), dpi=400)
plt.close()
