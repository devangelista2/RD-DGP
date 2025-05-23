import argparse
import logging
import os

import numpy as np
import torch

import algorithms
from miscellaneous import model_setup, utilities

# --- Initialization ---
CONFIG_NAME = "configs/mayo_ct.yaml"

parser = argparse.ArgumentParser(
    description="Setup for running Diffusion Models for Image Reconstruction"
)
parser.add_argument(
    "--config",
    type=str,
    default=CONFIG_NAME,
    help="Path to the YAML configuration file (default: config.yaml)",
)
parser.add_argument(
    "--dgp",
    default=False,
    action="store_true",
    help="Whether to test with DGP.",
)
parser.add_argument(
    "--dps",
    default=False,
    action="store_true",
    help="Whether to test with DPS.",
)
parser.add_argument(
    "--diffpir",
    default=False,
    action="store_true",
    help="Whether to test with DiffPIR.",
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

# --- Run experiments
if args.dgp:
    x_dgp, psnr_dgp, ssim_dgp = algorithms.run_dgp(
        model,
        scheduler,
        config,
        main_logger,
        device,
    )

if args.dps:
    x_dps, psnr_dps, ssim_dps = algorithms.run_dps(
        model,
        scheduler,
        config,
        main_logger,
        device,
    )

if args.diffpir:
    x_diffpir, psnr_diffpir, ssim_diffpir = algorithms.run_diffpir(
        model,
        scheduler,
        config,
        main_logger,
        device,
    )

main_logger.info("Experiment finished successfully.")
