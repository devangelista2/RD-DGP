import os

import torch
from accelerate import Accelerator
from diffusers import (  # Or whatever model you are using
    DDIMPipeline,
    DDIMScheduler,
    UNet2DModel,
)

# 1. Initialize Accelerator (same as during training)
# Create accelerator
accelerator = Accelerator(
    gradient_accumulation_steps=1,
    mixed_precision="fp16",
)
device = accelerator.device

# 2. Define your model, optimizer, etc.
model = UNet2DModel(
    sample_size=256,
    in_channels=1,
    out_channels=1,
    layers_per_block=2,
    block_out_channels=tuple([128 * m for m in (1, 2, 4, 4, 8)]),
    down_block_types=(
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",
        "AttnDownBlock2D",
    ),
    up_block_types=(
        "AttnUpBlock2D",
        "AttnUpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ),
).to(device)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.0001,
)

# 3. Prepare them with the accelerator
model, optimizer = accelerator.prepare(model, optimizer)

# 4. Specify the checkpoint directory
checkpoint_dir = (
    "./checkpoint-60/"  # This should be the same path you used for save_state
)

# 5. Load the state
print(f"Loading state from {checkpoint_dir}...")
accelerator.load_state(checkpoint_dir)
print("State loaded successfully!")
# Now your model, optimizer, and (if applicable) scheduler
# should have their states restored from the checkpoint.
# You can continue training or use the model for inference.

# Example: Check if model weights are loaded (you'd normally just proceed)
# This is just for demonstration
# You can access the underlying model after preparation if needed for specific operations
unwrapped_model = accelerator.unwrap_model(model)

# --- 3. Create a scheduler (if you don't have one loaded) ---
# DDIMPipeline requires a scheduler. If you trained with DDPMScheduler,
# you'd typically initialize it with the same parameters.
# If you used a different scheduler, initialize that one.
scheduler = DDIMScheduler(
    num_train_timesteps=1000
)  # Default parameters are often fine for inference, adjust if needed

# --- 4. Create the Diffusers Pipeline ---
# The DDIMPipeline needs a UNet model and a scheduler.
# If you used a different type of pipeline (e.g., DDPMPipeline), initialize that one.
pipeline = DDIMPipeline(unet=unwrapped_model, scheduler=scheduler)
print("DDIMPipeline created.")

# --- 5. Save the Pipeline ---
# Specify the directory where you want to save your pipeline.
# By default, Diffusers saves models in .safetensors format if `safetensors` library is installed.
save_directory = "./model_weights/MI_UNet_256"
pipeline.save_pretrained(save_directory)
print(f"Pipeline saved to {save_directory}")
