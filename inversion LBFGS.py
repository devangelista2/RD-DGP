import os

import torch
from diffusers import DDIMScheduler, UNet2DModel
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from miscellaneous import data, utilities

# --- Set device ---
device = utilities.get_device()
print(f"Device used: {device}.")

# --- Configuration ---
MODEL_PATH = "./model_weights/UNet_256/"
BATCH_SIZE = 4
TIMESTEPS = 30

ITERATION_TIMESTEPS = 10
NUM_ITER = 200
STEP_SIZE = 1e-2

# --- Load model + scheduler ---
model = UNet2DModel.from_pretrained(os.path.join(MODEL_PATH, "unet")).to(device)
scheduler = DDIMScheduler.from_pretrained(os.path.join(MODEL_PATH, "scheduler"))


# --- Load data ---
test_data = data.MayoDataset(
    data_path="../data/Mayo/train",
    data_shape=model.config.sample_size,
)
x_true = test_data[0].unsqueeze(0).to(device)

# --- Run inversion process ---
z = torch.randn(x_true.shape, requires_grad=True, device=device)  # z_0

# Initialize metrics
psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

# Define optimizer
optimizer = torch.optim.LBFGS([z], lr=1.0, max_iter=1, history_size=10)
loss_fn = torch.nn.MSELoss()

# Step 4: Optimize z such that G(z) ≈ x_true
scheduler.set_timesteps(ITERATION_TIMESTEPS)


# Define closure for LBFGS
def closure():
    optimizer.zero_grad()
    x = z

    # Run full denoising process (G(z)) through scheduler
    for t in scheduler.timesteps:
        with torch.no_grad():
            noise_pred = model(x, t).sample
        x = scheduler.step(noise_pred, t, x).prev_sample

    x_rec = (x + 1.0) / 2.0

    loss = loss_fn(x_rec, x_true)
    loss.backward()

    # Optional monitoring
    with torch.no_grad():
        psnr = psnr_metric(x_rec, x_true)
        ssim = ssim_metric(x_rec, x_true)
        print(
            f"Loss: {loss.item():.4f} | PSNR: {psnr.item():.2f} | SSIM: {ssim.item():.4f}"
        )

    return loss


# --- Optimization loop ---
for step in range(NUM_ITER):  # You can adjust number of LBFGS outer iterations
    optimizer.step(closure)
