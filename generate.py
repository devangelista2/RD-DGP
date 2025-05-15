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
MODEL_PATH = "./model_weights/UNet_128/"
BATCH_SIZE = 4
TIMESTEPS = 50

# --- Load model + scheduler ---
model = UNet2DModel.from_pretrained(os.path.join(MODEL_PATH, "unet")).to(device)
scheduler = DDIMScheduler.from_pretrained(os.path.join(MODEL_PATH, "scheduler"))

pipeline = DDIMPipeline(model, scheduler).to(device)


# --- Generation ---
def generate_images(model, scheduler, x=None, num_timesteps=50, batch_size=4, eta=0.0):
    """
    Replicates the core generation process of a DDIM-like pipeline.

    Args:
        model: The UNet model (diffusers library).
        scheduler: The DDIMScheduler or compatible (diffusers library).
        x (torch.Tensor, optional): Initial latents (batch, channels, H, W). Defaults to None (random noise).
        num_timesteps (int, optional): Number of denoising steps. Defaults to 50.
        batch_size (int, optional): Batch size if x is None. Defaults to 4.
        eta (float, optional): DDIM eta parameter (0.0 for DDIM, >0 for DDPM-like stochasticity). Defaults to 0.0.

    Returns:
        torch.Tensor: The generated image latents.
    """
    model.eval()
    device = model.device
    dtype = model.dtype

    if x is None:
        if isinstance(model.config.sample_size, int):
            height = width = model.config.sample_size
        else:
            height, width = model.config.sample_size
        shape = (batch_size, model.config.in_channels, height, width)
        latents = torch.randn(shape, device=device, dtype=dtype)
    else:
        if (
            x.shape[0] != batch_size and x.shape[0] != 1
        ):  # Allow single provided latent to be used for batch
            print(
                f"Warning: Provided 'x' has batch size {x.shape[0]}, not matching 'batch_size' parameter {batch_size} or 1. Using batch size from 'x'."
            )
        if x.shape[0] == 1 and batch_size > 1:
            latents = x.repeat(batch_size, 1, 1, 1).to(device, dtype=dtype)
        else:
            latents = x.to(device, dtype=dtype)

    scheduler.set_timesteps(num_timesteps, device=device)
    timesteps = scheduler.timesteps

    for t in timesteps:
        model_output = model(latents, t).sample
        scheduler_output = scheduler.step(
            model_output,
            t,
            latents,
            eta=eta,
        )
        latents = scheduler_output.prev_sample

    return latents


def invert_generation_process(
    model,
    scheduler,
    target_image_latents,  # This is 'x' in the formula ||G(z) - x||^2
    num_inference_timesteps,
    eta_ddim=0.0,
    num_optimization_steps=200,
    learning_rate=0.01,
    initial_z_guess=None,
    regularization_weight=0.0,  # Optional: L2 regularization on z
    device=None,
):
    """
    Solves the optimization problem: min_z || G(z) - target_image_latents ||_2^2
    to find the initial noise 'z' that generates 'target_image_latents'.

    Args:
        model: The UNet model (diffusers library).
        scheduler: The DDIMScheduler or compatible (diffusers library).
        target_image_latents (torch.Tensor): The target image in the latent space.
                                            Shape (batch_size, channels, H, W).
        num_inference_timesteps (int): Number of denoising steps for G(z) (generate_images).
        eta_ddim (float, optional): Eta parameter for G(z). Defaults to 0.0.
        num_optimization_steps (int, optional): Number of optimization steps. Defaults to 200.
        learning_rate (float, optional): Learning rate for AdamW. Defaults to 0.01.
        initial_z_guess (torch.Tensor, optional): An initial guess for z.
                                                 If None, z is initialized randomly.
                                                 Shape should match target_image_latents.
                                                 Defaults to None.
        regularization_weight (float, optional): Weight for L2 regularization on z.
                                                 Defaults to 0.0 (no regularization).
        device (torch.device or str, optional): Device to run the optimization on.
                                                If None, inferred from model. Defaults to None.

    Returns:
        torch.Tensor: The optimized initial noise 'z'.
    """
    if device is None:
        device = model.device
    else:
        model.to(device)  # Ensure model is on the specified device

    target_image_latents = target_image_latents.to(device, dtype=model.dtype)
    batch_size = target_image_latents.shape[0]
    noise_shape = target_image_latents.shape  # (batch_size, channels, H, W)

    if initial_z_guess is None:
        # Initialize z randomly
        z = torch.randn(
            noise_shape, device=device, dtype=model.dtype, requires_grad=True
        )
    else:
        z = (
            initial_z_guess.clone()
            .detach()
            .to(device, dtype=model.dtype)
            .requires_grad_(True)
        )

    optimizer = optim.AdamW([z], lr=learning_rate)
    loss_fn = torch.nn.MSELoss()

    print(f"Optimizing z for {num_optimization_steps} steps...")
    for step in tqdm(range(num_optimization_steps)):
        optimizer.zero_grad()

        # G(z): Generate image from current z
        # The generate_images function needs to be differentiable with respect to its input 'x' (which is 'z' here)
        generated_latents_g_z = generate_images(
            model,
            scheduler,
            x=z,  # Pass current z as the initial latent for generation
            num_timesteps=num_inference_timesteps,
            batch_size=batch_size,  # ensure batch_size in G matches z
            eta=eta_ddim,
        )

        # Calculate loss: || G(z) - target_image_latents ||_2^2
        reconstruction_loss = loss_fn(generated_latents_g_z, target_image_latents)

        total_loss = reconstruction_loss
        if regularization_weight > 0:
            l2_reg = torch.norm(z) ** 2
            total_loss += regularization_weight * l2_reg

        # Backward pass and optimization
        total_loss.backward()
        optimizer.step()

        if (
            step % (num_optimization_steps // 10) == 0
            or step == num_optimization_steps - 1
        ):
            print(
                f"Step {step}/{num_optimization_steps}, Loss: {total_loss.item():.4f} (Reconstruction: {reconstruction_loss.item():.4f})"
            )

    print("Optimization finished.")
    return z.detach()


x_true = (
    torch.tensor(
        np.array(Image.open("../data/Mayo/train/C002/0.png").convert("L")),
        device=device,
    )
    .unsqueeze(0)
    .unsqueeze(1)
)
x_true = transforms.Resize(128)(x_true)

x_rec = invert_generation_process(
    model,
    scheduler,
    x_true,
    num_inference_timesteps=10,
    eta_ddim=0.0,
    num_optimization_steps=200,
    learning_rate=1,
    regularization_weight=0,
    device=device,
)

# Visualize
plt.subplot(1, 2, 1)
plt.imshow(x_true[0, 0].cpu(), cmap="gray")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(
    generate_images(model, scheduler, x_rec, num_timesteps=20, batch_size=1, eta=0)[
        0, 0
    ]
    .detach()
    .cpu(),
    cmap="gray",
)
plt.axis("off")

plt.show()
