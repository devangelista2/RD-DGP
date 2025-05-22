import argparse
import logging
import os

import numpy as np
import torch
import yaml

from IPPy import utilities as IPutils
from miscellaneous import model_setup, solvers, utilities

# --- Initialization ---
CONFIG_NAME = "DGP.yaml"

parser = argparse.ArgumentParser(description="DGP with Diffusion Models Inversion")
parser.add_argument(
    "--config",
    type=str,
    default=CONFIG_NAME,
    help="Path to the YAML configuration file (default: config.yaml)",
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
    "DGP_Experiment", os.path.join(saving_path, "experiment.log")
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

img_size_to_use = config.get("image_size", model.config.sample_size)
if isinstance(img_size_to_use, int):
    img_size_to_use = (img_size_to_use, img_size_to_use)
main_logger.info(f"Target image size set to: {img_size_to_use}")

# --- Load data ---
x_true = utilities.load_and_preprocess_image(
    config, img_size_to_use, device, main_logger
)
img_shape_hw = x_true.shape[-2:]
num_channels = config["image_channels"]

# --- Define operator & Generate test problem ---
K = model_setup.get_operator(config, img_shape_hw, num_channels, device, main_logger)

main_logger.info("Generating test problem (corrupted data y_delta)...")
y = K(x_true)
y_delta = y + IPutils.gaussian_noise(y, config["noise_level"])
main_logger.info(f"Test problem generated. y_delta shape: {y_delta.shape}")

# --- Run inversion process ---
z = torch.randn(x_true.shape, requires_grad=True, device=device)
main_logger.info(f"Initialized latent variable z with shape: {z.shape}")

# --- Optimization using DGPSolver ---
solver = solvers.DGPSolver(
    model, scheduler, K, x_true, y_delta, z, config, device, main_logger
)
z, psnr_vec, ssim_vec, loss_vec = solver.run_optimization()

# --- Create final generation from optimized z ---
main_logger.info("Generating final image from optimized z...")
scheduler.set_timesteps(config["generation_timesteps"])

x_final_gen = z
for t in scheduler.timesteps:
    with torch.no_grad():
        noise_pred = model(x_final_gen, t).sample
    x_final_gen = scheduler.step(noise_pred, t, x_final_gen).prev_sample

x_gen_final_normalized = (x_final_gen + 1.0) / 2.0
x_gen_final_normalized = torch.clamp(x_gen_final_normalized, 0.0, 1.0)
main_logger.info("Final image generated.")

# --- Saving ---
main_logger.info(f"Saving final results and artifacts to: {saving_path}")
torch.save(torch.tensor(psnr_vec), os.path.join(saving_path, "psnr_history.pth"))
torch.save(torch.tensor(ssim_vec), os.path.join(saving_path, "ssim_history.pth"))
main_logger.info("Metrics history saved.")

with open(os.path.join(saving_path, "config_used.yaml"), "w") as f:
    yaml.dump(config, f)
main_logger.info("Configuration used for this run saved.")

final_ssim = solver.ssim_metric(x_gen_final_normalized, x_true).item()
utilities.plot_and_save_images(
    x_true,
    y_delta,
    x_gen_final_normalized,
    final_ssim,
    saving_path,
    config,
    main_logger,
)

main_logger.info("Experiment finished successfully.")
