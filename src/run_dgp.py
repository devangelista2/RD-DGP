import argparse
import json
import logging

import torch
from diffusers import DDIMScheduler

from materials.dgp.models import load_unet_model
from materials.dgp.physics import RadonTransform
from materials.dgp.pipeline import DGPOptimizer
from materials.dgp.utils import (
    load_image,
    load_unet,
    save_dgp_metrics_csv,
    save_error_map,
    save_image,
    save_plots,
    save_sinogram_vis,
    seed_everything,
    setup_experiment,
    setup_radon,
    simulate_measurement,
)

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="DGP for CT reconstruction with MONAI DDIM prior"
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Path to CLEAN phantom/image"
    )
    parser.add_argument("--weights", type=str, required=True, help="Path to UNet")
    parser.add_argument("--config", type=str, default="configs/default_monai.yaml")

    # CT Physics
    parser.add_argument("--num_angles", type=int, default=180)
    parser.add_argument(
        "--noise_sigma",
        type=float,
        default=0.01,
        help="Noise level added to raw sinogram",
    )

    # DGP Params
    parser.add_argument("--output_dir", type=str, default="outputs/dgp_ct")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=20,
        help="Fixed DDIM steps (used if start/end are not set)",
    )
    parser.add_argument(
        "--ddim_start_steps",
        type=int,
        default=None,
        help="DDIM steps at start of optimization (overrides ddim_steps)",
    )
    parser.add_argument(
        "--ddim_end_steps",
        type=int,
        default=None,
        help="DDIM steps at end of optimization (overrides ddim_steps)",
    )
    parser.add_argument("--opt_steps", type=int, default=150)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--lambda_reg", type=float, default=0.01)
    parser.add_argument(
        "--lambda_tv",
        type=float,
        default=0.1,
        help="Lower TV for CT helps preserve texture",
    )
    parser.add_argument("--seed", type=int, default=123)

    # Snapshot Params
    parser.add_argument(
        "--snapshot_interval",
        type=int,
        default=5,
        help="Save reconstruction snapshot every N iterations",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )
    seed_everything(args.seed)

    timestamp, exp_dir, snapshot_dir = setup_experiment(args.output_dir)
    logger.info("Starting experiment: %s", timestamp)

    with (exp_dir / "config_used.json").open("w") as f:
        json.dump(vars(args), f, indent=4)

    # 1. Load model and scheduler
    unet, sample_size = load_unet(
        args.weights, args.config, args.device, load_unet_model
    )
    scheduler = DDIMScheduler(
        num_train_timesteps=1000, clip_sample=True, prediction_type="epsilon"
    )

    # 2. Physics
    logger.info("Setting up Radon Transform: %d angles", args.num_angles)
    radon = setup_radon(sample_size, args.num_angles, args.device, RadonTransform)

    # 3. Data
    clean_image = load_image(
        args.input, size=(sample_size, sample_size), grayscale=True, device=args.device
    )
    save_image(clean_image, exp_dir / "clean_phantom.png")

    # 4. Simulate (no normalization)
    raw_sinogram, noisy_sinogram = simulate_measurement(
        clean_image, radon, args.noise_sigma, args.seed
    )
    logger.info(
        "Raw Sinogram Range: [%.2f, %.2f]",
        raw_sinogram.min().item(),
        raw_sinogram.max().item(),
    )

    save_sinogram_vis(raw_sinogram, exp_dir / "sinogram_raw.png")
    save_sinogram_vis(noisy_sinogram, exp_dir / "sinogram_input.png")

    # Compute FBP Reconstruction to test the operator
    fbp_recon = radon.fbp(noisy_sinogram)
    save_image(fbp_recon, exp_dir / "fbp_reconstruction.png")

    # 5. Run DGP
    dgp = DGPOptimizer(unet, scheduler, device=args.device)

    def forward_op(x):
        return radon(x)

    def fbp_op(y):
        return radon.fbp(y)

    def save_progress_callback(img_tensor, i):
        path = snapshot_dir / f"iter_{i:04d}.png"
        save_image(img_tensor, path, silent=True)

    logger.info("Starting optimization on raw sinogram domain...")
    recon_image, z, history = dgp.reconstruct(
        target_measurement=noisy_sinogram,
        image_size=sample_size,
        forward_operator=forward_op,
        fbp_operator=fbp_op,
        clean_reference=clean_image,
        noise_sigma=args.noise_sigma,
        num_ddim_steps=args.ddim_steps,
        ddim_start_steps=args.ddim_start_steps,
        ddim_end_steps=args.ddim_end_steps,
        num_opt_steps=args.opt_steps,
        lr=args.lr,
        lambda_reg=args.lambda_reg,
        lambda_tv=args.lambda_tv,
        seed=args.seed,
        snapshot_interval=args.snapshot_interval,
        snapshot_callback=save_progress_callback,
        return_latent=True,
    )

    # 6. Save outputs
    save_image(recon_image, exp_dir / "reconstruction.png")
    save_error_map(clean_image, recon_image, exp_dir / "error_map.png")

    if history.get("psnr"):
        save_plots(history, exp_dir)
        save_dgp_metrics_csv(history, exp_dir)
        logger.info(
            "Final PSNR: %.2f | Final SSIM: %.4f",
            history["psnr"][-1],
            history["ssim"][-1],
        )

    logger.info("Experiment saved to %s", exp_dir)


if __name__ == "__main__":
    main()
