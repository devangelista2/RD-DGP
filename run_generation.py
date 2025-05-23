import argparse
import logging
import os

import numpy as np
import torch
import yaml
from tqdm import tqdm

from miscellaneous import model_setup, utilities

# --- Initialization ---
CONFIG_NAME = "./configs/mayo_ct.yaml"

parser = argparse.ArgumentParser(description="Diffusion Model Sample Generation Script")
parser.add_argument(
    "--config",
    type=str,
    default=CONFIG_NAME,
    help="Path to YAML config file",
)
args = parser.parse_args()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- Load Configuration & Basic Setup ---
config = utilities.load_config(args.config)
saving_path = utilities.setup_paths(config)

# --- Setup Logger ---
# All subsequent log messages will go to console and the file in saving_path
main_logger = utilities.setup_logger(
    "Generation_Experiment", os.path.join(saving_path, "experiment.log")
)

main_logger.info(f"Starting experiment with configuration: {args.config}")
main_logger.info(f"Results will be saved in: {saving_path}")

device = utilities.get_device(config["device"], main_logger)

if config.get("seed") is not None:
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    main_logger.info(f"Seed set to: {config['seed']}")

# --- Load model + scheduler ---
model, scheduler = model_setup.load_model_and_scheduler(config, device, main_logger)

# --- Generation ---
num_channels = config["image_channels"]
batch_size = 4
eta_ddim = 0.0

img_size_to_use = config.get("image_size", model.config.sample_size)
if isinstance(img_size_to_use, int):
    img_size_to_use = (img_size_to_use, img_size_to_use)

main_logger.info(
    f"Starting generation: batch_size={batch_size}, steps={config['generation_timesteps']}, eta={eta_ddim}"
)
main_logger.info(f"Sample properties: channels={num_channels}, size={img_size_to_use}")

model.eval()  # Set model to evaluation mode

# Initial random noise
x_t = torch.randn(
    (batch_size, num_channels, img_size_to_use[0], img_size_to_use[1]),
    device=device,
)

scheduler.set_timesteps(config["generation_timesteps"], device=device)

main_logger.info("Starting reverse diffusion process...")
for t in tqdm(
    scheduler.timesteps, desc="Generating Samples"
):  # Added tqdm for progress
    with torch.no_grad():
        # Ensure t is on the same device as x_t if model requires it
        model_output = model(x_t, t.to(device)).sample

    x_t = scheduler.step(
        model_output,
        t,  # t is already on the correct device from scheduler.timesteps if device was passed to set_timesteps
        x_t,
        eta=eta_ddim,
    ).prev_sample

main_logger.info("Sample generation complete.")

# --- Saving ---
utilities.save_generated_images(x_t, saving_path, config, main_logger)

# Save the configuration used for this run
with open(os.path.join(saving_path, "generation_config_used.yaml"), "w") as f:
    yaml.dump(config, f)
main_logger.info("Configuration saved.")
main_logger.info("Sample Generation script finished successfully.")
