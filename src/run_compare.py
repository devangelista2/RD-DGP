import argparse
import json
import logging
import os

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from diffusers import DDIMScheduler

from materials.ddrm.pipeline import DDRMPipeline
from materials.dgp.models import load_unet_model
from materials.dgp.physics import RadonTransform
from materials.dgp.utils import (
    load_image,
    load_unet,
    save_error_map,
    save_image,
    save_method_metrics_csv,
    save_plots,
    seed_everything,
    setup_experiment,
    setup_radon,
    simulate_measurement,
)
from materials.diffpir.pipeline import DiffPIRPipeline

# Pipeline Imports
from materials.dps.pipeline import DPSPipeline

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Compare DPS, DiffPIR, DDRM on CT reconstruction (MONAI DDIM prior)"
    )

    # Inputs
    parser.add_argument("--input", type=str, required=True, help="Clean image path")
    parser.add_argument("--weights", type=str, required=True, help="Path to UNet")
    parser.add_argument("--config", type=str, default="configs/default_monai.yaml")

    # CT Physics
    parser.add_argument("--num_angles", type=int, default=180)
    parser.add_argument("--noise_sigma", type=float, default=0.01)

    # Methods Steps
    parser.add_argument("--dps_steps", type=int, default=200)
    parser.add_argument("--diffpir_steps", type=int, default=50)
    parser.add_argument("--ddrm_steps", type=int, default=30)

    # Hyperparams (Can be tuned)
    parser.add_argument("--dps_zeta", type=float, default=0.5)
    parser.add_argument("--diffpir_lambda", type=float, default=1.0)
    parser.add_argument("--diffpir_rho", type=float, default=1.0)

    # System
    parser.add_argument("--output_dir", type=str, default="outputs/comparison_ct")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seed", type=int, default=42)

    # Snapshot Params
    parser.add_argument(
        "--snapshot_interval",
        type=int,
        default=5,
        help="Save reconstruction snapshot every N iterations",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )
    seed_everything(args.seed)

    # 1. Setup Output
    timestamp, exp_dir, snapshots_root = setup_experiment(args.output_dir)

    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    # 2. Load Model
    unet, sample_size = load_unet(
        args.weights, args.config, args.device, load_unet_model
    )

    scheduler = DDIMScheduler(
        num_train_timesteps=1000, clip_sample=True, prediction_type="epsilon"
    )

    # 3. Physics & Data
    logger.info(
        "Setup: %d angles, size %d, sigma %.4f",
        args.num_angles,
        sample_size,
        args.noise_sigma,
    )
    radon = setup_radon(sample_size, args.num_angles, args.device, RadonTransform)

    clean_image = load_image(
        args.input, size=(sample_size, sample_size), grayscale=True, device=args.device
    )
    save_image(clean_image, os.path.join(exp_dir, "ref_clean.png"))

    # Generate Sinogram + Normalization
    raw_sinogram, noisy_sinogram = simulate_measurement(
        clean_image, radon, args.noise_sigma, args.seed
    )
    logger.info(
        "Raw Sinogram Range: [%.2f, %.2f]",
        raw_sinogram.min().item(),
        raw_sinogram.max().item(),
    )

    # Operators
    def forward_op(x):
        return radon(x)

    def fbp_op(y):
        return radon.fbp(y)

    # Helper function to generate callbacks
    def make_callback(method_name):
        method_dir = snapshots_root / method_name
        method_dir.mkdir(parents=True, exist_ok=True)

        def callback(img, i):
            save_image(img, method_dir / f"iter_{i:04d}.png", silent=True)

        return callback

    # ------------------------------------------------------------------
    # RUN METHOD 1: DPS
    # ------------------------------------------------------------------
    logger.info("[1/3] Running DPS...")
    dps = DPSPipeline(unet, scheduler, args.device)
    recon_dps, hist_dps = dps.reconstruct(
        measurement=noisy_sinogram,
        image_size=sample_size,
        forward_op=forward_op,
        clean_reference=clean_image,
        num_steps=args.dps_steps,
        zeta=args.dps_zeta,
        snapshot_interval=args.snapshot_interval,
        snapshot_callback=make_callback("dps"),
    )
    save_image(recon_dps, exp_dir / "recon_dps.png")
    save_error_map(clean_image, recon_dps, exp_dir / "error_dps.png")
    save_plots(hist_dps, exp_dir, "DPS")
    save_method_metrics_csv(hist_dps, exp_dir, "DPS")

    # ------------------------------------------------------------------
    # RUN METHOD 2: DiffPIR
    # ------------------------------------------------------------------
    logger.info("[2/3] Running DiffPIR...")
    diffpir = DiffPIRPipeline(unet, scheduler, args.device)
    recon_pir, hist_pir = diffpir.reconstruct(
        measurement=noisy_sinogram,
        image_size=sample_size,
        forward_op=forward_op,
        fbp_op=fbp_op,
        clean_reference=clean_image,
        num_steps=args.diffpir_steps,
        lambda_data=args.diffpir_lambda,
        rho=args.diffpir_rho,
        snapshot_interval=args.snapshot_interval,
        snapshot_callback=make_callback("diffpir"),
    )
    save_image(recon_pir, exp_dir / "recon_diffpir.png")
    save_error_map(clean_image, recon_pir, exp_dir / "error_diffpir.png")
    save_plots(hist_pir, exp_dir, "DiffPIR")
    save_method_metrics_csv(hist_pir, exp_dir, "DiffPIR")

    # ------------------------------------------------------------------
    # RUN METHOD 3: DDRM / DDNM
    # ------------------------------------------------------------------
    logger.info("[3/3] Running DDRM...")
    ddrm = DDRMPipeline(unet, scheduler, args.device)
    recon_ddrm, hist_ddrm = ddrm.reconstruct(
        measurement=noisy_sinogram,
        image_size=sample_size,
        forward_op=forward_op,
        fbp_op=fbp_op,
        clean_reference=clean_image,
        num_steps=args.ddrm_steps,
        snapshot_interval=args.snapshot_interval,
        snapshot_callback=make_callback("ddrm"),
    )
    save_image(recon_ddrm, exp_dir / "recon_ddrm.png")
    save_error_map(clean_image, recon_ddrm, exp_dir / "error_ddrm.png")
    save_plots(hist_ddrm, exp_dir, "DDRM")
    save_method_metrics_csv(hist_ddrm, exp_dir, "DDRM")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    logger.info("Saving summary comparison...")

    def prep(img):
        return (img.detach().cpu().squeeze() * 0.5 + 0.5).clamp(0, 1)

    imgs = [prep(clean_image), prep(recon_dps), prep(recon_pir), prep(recon_ddrm)]

    plt.figure(figsize=(20, 6))
    titles = ["Clean", "DPS", "DiffPIR", "DDRM"]
    final_scores = [
        "",
        f"PSNR: {hist_dps['psnr'][-1]:.2f}" if hist_dps["psnr"] else "N/A",
        f"PSNR: {hist_pir['psnr'][-1]:.2f}" if hist_pir["psnr"] else "N/A",
        f"PSNR: {hist_ddrm['psnr'][-1]:.2f}" if hist_ddrm["psnr"] else "N/A",
    ]

    for i, (img, title, score) in enumerate(zip(imgs, titles, final_scores)):
        plt.subplot(1, 4, i + 1)
        plt.imshow(img, cmap="gray")
        plt.title(f"{title}\n{score}")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(exp_dir / "summary_comparison.png")
    plt.close()

    logger.info("Comparison complete. Results in %s", exp_dir)


if __name__ == "__main__":
    main()
