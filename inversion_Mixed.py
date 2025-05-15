import os

import torch
from diffusers import DDIMScheduler, UNet2DModel
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from miscellaneous import data, utilities

# --- Set device ---
device = utilities.get_device()
print(f"Device used: {device}.")

# --- Configuration ---
MODEL_PATH = "./model_weights/UNet_128/"
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
    data_path="../data/Mayo/test",
    data_shape=model.config.sample_size,
)
x_true = test_data[0].unsqueeze(0).to(device)

# --- Run inversion process ---
z = torch.randn(x_true.shape, requires_grad=True, device=device)  # z_0

# Set optimizer and loss fn
optimizer = torch.optim.AdamW([z], lr=STEP_SIZE, weight_decay=0)
loss_fn = torch.nn.MSELoss()

# Initialize metrics
psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

# Step 4: Optimize z such that G(z) ≈ x_true
scheduler.set_timesteps(ITERATION_TIMESTEPS)


def generative_forward(z):
    x = z
    for t in scheduler.timesteps:
        with torch.no_grad():
            noise_pred = model(x, t).sample
        x = scheduler.step(noise_pred, t, x).prev_sample
    return (x + 1.0) / 2.0


def compute_loss_and_metrics(x_gen, x_true):
    loss = loss_fn(x_gen, x_true)

    with torch.no_grad():
        psnr = psnr_metric(x_gen, x_true)
        ssim = ssim_metric(x_gen, x_true)
    return loss, psnr.item(), ssim.item()


# --- Stage 1: Adam Optimization ---
adam = torch.optim.Adam([z], lr=1e-1)
for step in range(50):  # Adjust number of Adam steps
    adam.zero_grad()
    x_gen = generative_forward(z)
    loss, psnr, ssim = compute_loss_and_metrics(x_gen, x_true)
    loss.backward()
    adam.step()
    print(
        f"[Adam {step:03d}] Loss: {loss.item():.4f} | PSNR: {psnr:.2f} | SSIM: {ssim:.4f}"
    )

# --- Stage 2: L-BFGS Fine-tuning ---
lbfgs = torch.optim.LBFGS(
    [z], lr=1e-2, max_iter=1, history_size=10, line_search_fn="strong_wolfe"
)


def closure():
    lbfgs.zero_grad()
    x_gen = generative_forward(z)
    loss, psnr, ssim = compute_loss_and_metrics(x_gen, x_true)
    loss.backward()
    print(
        f"[LBFGS {step}] Loss: {loss.item():.4f} | PSNR: {psnr:.2f} | SSIM: {ssim:.4f}"
    )
    return loss


for step in range(30):  # Number of outer LBFGS calls
    lbfgs.step(closure)
