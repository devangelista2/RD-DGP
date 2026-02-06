import argparse
import logging
import os

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler
from monai.data import CacheDataset

# MONAI IMPORTS
from monai.utils import set_determinism
from torch.utils.data import DataLoader
from tqdm import tqdm

from materials.dgp.utils import build_ct_transforms, list_png_files, seed_everything

logger = logging.getLogger(__name__)


def main():
    # 0. PARSING
    parser = argparse.ArgumentParser(
        description="Train MONAI Diffusion Model on CT images"
    )
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to CLEAN training folder"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path where to save UNet parameters and sample images",
    )
    parser.add_argument(
        "--image_size", type=int, default=256, help="Training image size"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Training batch size"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument(
        "--num_workers", type=int, default=0, help="Number of training workers"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate for optimizer"
    )
    parser.add_argument(
        "--val_interval",
        type=int,
        default=5,
        help="Generate sample images every N epochs",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )
    seed_everything(args.seed)

    # 1. SETUP
    os.makedirs(args.output_dir, exist_ok=True)
    set_determinism(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # 2. DATA AUGMENTATION PIPELINE (CT-focused)
    train_transforms = build_ct_transforms(args.image_size, padding_mode="zeros")

    # 3. DATASET & LOADER
    train_files = list_png_files(args.data_path)
    logger.info("Found %d images in %s", len(train_files), args.data_path)

    # CacheDataset accelerates training by caching transformed images in RAM
    # (Set cache_rate=0.0 if you run out of RAM)
    dataset = CacheDataset(
        data=train_files,
        transform=train_transforms,
        cache_rate=0.5,
        num_workers=args.num_workers,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )

    # 4. MODEL DEFINITION (2D Medical UNet)
    model = DiffusionModelUNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        num_channels=(64, 128, 256, 512),
        attention_levels=(
            False,
            False,
            True,
            True,
        ),  # Attention only at deep layers saves memory
        num_res_blocks=2,
        num_head_channels=32,
    ).to(device)

    # 5. SCHEDULER & OPTIMIZER
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 6. TRAINING LOOP
    logger.info("Starting training for %d epochs...", args.num_epochs)

    # Create a fixed noise vector to visualize progress on the SAME latent over time
    fixed_noise = torch.randn((4, 1, args.image_size, args.image_size)).to(device)

    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")

        for step, batch in enumerate(progress_bar):
            images = batch["image"].to(device)

            # --- CRITICAL: Scale inputs to [-1, 1] for Diffusion ---
            # Our transform outputs [0, 1], so we map it here.
            images = images * 2.0 - 1.0

            # Generate Noise
            noise = torch.randn_like(images)
            timesteps = torch.randint(0, 1000, (images.shape[0],), device=device).long()

            # Add Noise to Image
            noisy_images = scheduler.add_noise(
                original_samples=images, noise=noise, timesteps=timesteps
            )

            # Train: Predict Noise
            noise_pred = model(x=noisy_images, timesteps=timesteps)

            loss = F.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

        # End of Epoch Stats
        avg_loss = epoch_loss / len(loader)
        logger.info("Epoch %d Average Loss: %.6f", epoch + 1, avg_loss)

        # 7. VALIDATION / VISUALIZATION STEP
        if (epoch + 1) % args.val_interval == 0:
            model.eval()
            logger.info("Generating sample images...")
            with torch.no_grad():
                # Start from pure noise and denoise
                current_img = fixed_noise.clone()
                # Use the scheduler to denoise step-by-step
                for t in tqdm(scheduler.timesteps, desc="Sampling"):
                    # Model prediction
                    model_output = model(
                        current_img, timesteps=torch.Tensor((t,)).to(device)
                    )
                    # Step update
                    current_img, _ = scheduler.step(model_output, t, current_img)

                # Plot and Save
                fig, axs = plt.subplots(1, 4, figsize=(16, 4))
                for i in range(4):
                    # Convert back to [0, 1] for viewing
                    img_show = current_img[i, 0].cpu().clamp(-1, 1)
                    img_show = (img_show + 1) / 2.0
                    axs[i].imshow(img_show, cmap="gray")
                    axs[i].axis("off")
                    axs[i].set_title(f"Sample {i}")

                plt.suptitle(f"Epoch {epoch+1} (Loss: {avg_loss:.4f})")
                save_path = os.path.join(args.output_dir, f"epoch_{epoch+1}_sample.png")
                plt.savefig(save_path)
                plt.close()
                logger.info("Saved visualization to %s", save_path)

            # Save Model Checkpoint
            torch.save(
                model.state_dict(),
                os.path.join(args.output_dir, "latest_model.pth"),
            )
            torch.save(
                model.state_dict(),
                os.path.join(args.output_dir, f"model_epoch_{epoch+1}.pth"),
            )


if __name__ == "__main__":
    main()
