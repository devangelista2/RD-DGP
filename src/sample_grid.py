import argparse
import logging

import torch
import torchvision
from diffusers import DDIMScheduler
from tqdm import tqdm

from materials.dgp.models import load_unet_model
from materials.dgp.utils import load_config, seed_everything, unet_forward

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Generate a grid of random samples from the MONAI DDIM prior"
    )

    # Configuration
    parser.add_argument(
        "--weights", type=str, required=True, help="Path to .pth UNet state_dict"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_monai.yaml",
        help="Path to MONAI model config",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="sample_grid.png",
        help="Path to save the grid image",
    )

    # Sampling Params
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Total number of images to generate"
    )
    parser.add_argument(
        "--nrow", type=int, default=4, help="Number of images per row in the grid"
    )
    parser.add_argument(
        "--num_steps", type=int, default=50, help="Number of DDIM sampling steps"
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.0,
        help="DDIM eta (0.0 = deterministic, 1.0 = standard DDPM)",
    )

    # System
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--strict_load", action="store_true", help="Enforce strict state_dict loading"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )

    # 1. Setup
    logger.info("Generating sample grid on %s", args.device)
    seed_everything(args.seed)
    config = load_config(args.config)

    # 2. Load Model
    unet = load_unet_model(
        args.weights, config, device=args.device, strict_load=args.strict_load
    )

    # 3. Scheduler
    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
        clip_sample=True,
        prediction_type="epsilon",
    )
    scheduler.set_timesteps(args.num_steps)

    # 4. Generate Random Noise
    shape = (
        args.batch_size,
        config["in_channels"],
        config["sample_size"],
        config["sample_size"],
    )
    logger.info("Sampling %d images of shape %s...", args.batch_size, shape)

    # Initial Gaussian Noise
    latents = torch.randn(shape, device=args.device)

    # 5. Sampling Loop
    with torch.no_grad():
        for t in tqdm(scheduler.timesteps):
            t_input = t.view(1).expand(latents.shape[0])
            model_output = unet_forward(unet, latents, t_input)

            # Compute previous noisy sample x_t -> x_t-1
            step_output = scheduler.step(model_output, t, latents, eta=args.eta)
            latents = step_output.prev_sample

    # 6. Create Grid
    # latents are in range [-1, 1]. make_grid expects tensor, handles padding.
    # We first denormalize visually for the grid tool, or let save_image handle it later.
    # make_grid standardizes range if normalize=True.

    logger.info("Creating grid...")
    grid_tensor = torchvision.utils.make_grid(
        latents,
        nrow=args.nrow,
        padding=2,
        normalize=True,
        value_range=(-1, 1),  # Tells make_grid that input is [-1, 1]
    )

    # 7. Save
    torchvision.utils.save_image(grid_tensor, args.output)
    logger.info("Saved sample grid to %s", args.output)


if __name__ == "__main__":
    main()
