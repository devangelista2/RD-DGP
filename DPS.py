import argparse
import logging
import os

import numpy as np
import torch
import yaml

from IPPy import utilities as IPutils
from miscellaneous import model_setup, solvers, utilities

# --- Initialization ---
CONFIG_NAME = "DPS.yaml"

parser = argparse.ArgumentParser(description="Diffusion Posterior Sampling (DPS)")
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
config = utilities.load_config(f"experiments/{args.config}")
saving_path = utilities.setup_paths(config)

# --- Setup Logger ---
# All subsequent log messages will go to console and the file in saving_path
main_logger = utilities.setup_logger(
    "DPS_Experiment", os.path.join(saving_path, "dps_experiment.log")
)
main_logger.info(f"Starting experiment with configuration: {args.config}")
main_logger.info(f"Results will be saved in: {saving_path}")

device = utilities.get_device(config["device"], main_logger)

if config.get("seed") is not None:
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    main_logger.info(f"Seed set: {config['seed']}")

model, scheduler = model_setup.load_model_and_scheduler(config, device, main_logger)

img_size_to_use = config.get("image_size", model.config.sample_size)
if isinstance(img_size_to_use, int):
    img_size_to_use = (img_size_to_use, img_size_to_use)
main_logger.info(f"Target image size set to: {img_size_to_use}")

x_true = utilities.load_and_preprocess_image(
    config, img_size_to_use, device, main_logger
)
img_shape_hw = x_true.shape[-2:]
num_channels = config["image_channels"]

K = model_setup.get_operator(config, img_shape_hw, num_channels, device, main_logger)

main_logger.info("Generating test problem (corrupted data y_delta)...")
y = K(x_true)
y_delta = y + IPutils.gaussian_noise(y, config["noise_level"])
main_logger.info(f"Test problem generated. y_delta shape: {y_delta.shape}")

initial_x_t = torch.randn(x_true.shape, device=device)
main_logger.info(f"Initial x_t (noise) shape: {initial_x_t.shape}")

# --- Initialize and Run DPSSolver ---
solver = solvers.DPSSolver(
    model, scheduler, K, y_delta, x_true, config, device, main_logger
)
x_reconstructed, psnr_vec, ssim_vec, loss_vec = solver.run_algorithm(initial_x_t)

# --- Saving Results ---
main_logger.info(f"Saving final results to: {saving_path}")
torch.save(torch.tensor(psnr_vec), os.path.join(saving_path, "dps_psnr_history.pth"))
torch.save(torch.tensor(ssim_vec), os.path.join(saving_path, "dps_ssim_history.pth"))
torch.save(torch.tensor(loss_vec), os.path.join(saving_path, "dps_loss_history.pth"))
main_logger.info("Metrics history saved.")

with open(os.path.join(saving_path, "dps_config_used.yaml"), "w") as f:
    yaml.dump(config, f)
main_logger.info("Configuration saved.")

final_ssim_val = solver.ssim_metric(x_reconstructed, x_true).item()
utilities.plot_and_save_images(
    x_true,
    y_delta,
    x_reconstructed,
    final_ssim_val,
    saving_path,
    config,
    main_logger,
)

main_logger.info("DPS Experiment finished successfully.")
