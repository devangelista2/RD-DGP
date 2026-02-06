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
        description="Fine-tune MONAI Diffusion Model on CT images"
    )
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to CLEAN training folder"
    )
    parser.add_argument(
        "--pretrained_weights",
        type=str,
        default="./monai_checkpoints/latest_model.pth",
        help="Path to previously trained UNet weights for fine-tuning",
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
        "--start_epoch", type=int, default=0, help="Starting epoch for training"
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
    logger.info("Starting fine-tuning on %s", device)

    # 2. UPDATED TRANSFORM PIPELINE (Reflection padding for cleaner borders)
    train_transforms = build_ct_transforms(
        args.image_size, padding_mode="reflection"
    )

    # 3. DATASET
    train_files = list_png_files(args.data_path)
    logger.info("Found %d images for fine-tuning.", len(train_files))
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

    # 4. INITIALIZE MODEL
    model = DiffusionModelUNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        num_channels=(64, 128, 256, 512),
        attention_levels=(False, False, True, True),
        num_res_blocks=2,
        num_head_channels=32,
    ).to(device)

    # 5. LOAD PREVIOUS WEIGHTS
    if os.path.exists(args.pretrained_weights):
        logger.info("Loading weights from: %s", args.pretrained_weights)
        model.load_state_dict(torch.load(args.pretrained_weights))
    else:
        raise FileNotFoundError(
            f"Could not find checkpoint at {args.pretrained_weights}. Make sure the previous training finished or path is correct."
        )

    # 6. OPTIMIZER (With Lower Learning Rate)
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  # 1e-5

    # Fixed noise for consistent visualization
    fixed_noise = torch.randn((4, 1, args.image_size, args.image_size)).to(device)

    # 7. FINE-TUNING LOOP
    for epoch in range(args.start_epoch, args.num_epochs):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(loader, desc=f"Fine-tune Epoch {epoch+1}/{args.num_epochs}")

        for step, batch in enumerate(progress_bar):
            images = batch["image"].to(device)
            images = images * 2.0 - 1.0  # Scale [0,1] -> [-1, 1]

            noise = torch.randn_like(images)
            timesteps = torch.randint(0, 1000, (images.shape[0],), device=device).long()
            noisy_images = scheduler.add_noise(
                original_samples=images, noise=noise, timesteps=timesteps
            )

            noise_pred = model(x=noisy_images, timesteps=timesteps)
            loss = F.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

        # End of Epoch
        avg_loss = epoch_loss / len(loader)
        logger.info("Epoch %d Avg Loss: %.6f", epoch + 1, avg_loss)

        # Visualization
        if (epoch + 1) % args.val_interval == 0:
            model.eval()
            logger.info("Generating polished samples...")
            with torch.no_grad():
                current_img = fixed_noise.clone()
                for t in tqdm(scheduler.timesteps, desc="Sampling", leave=False):
                    model_output = model(
                        current_img, timesteps=torch.Tensor((t,)).to(device)
                    )
                    current_img, _ = scheduler.step(model_output, t, current_img)

                # Plot
                fig, axs = plt.subplots(1, 4, figsize=(16, 4))
                for i in range(4):
                    img_show = current_img[i, 0].cpu().clamp(-1, 1)
                    img_show = (img_show + 1) / 2.0
                    axs[i].imshow(img_show, cmap="gray")
                    axs[i].axis("off")
                    axs[i].set_title(f"FT Sample {i}")

                plt.suptitle(f"Fine-Tuning Epoch {epoch+1}")
                save_path = os.path.join(args.output_dir, f"ft_epoch_{epoch+1}.png")
                plt.savefig(save_path)
                plt.close()
                logger.info("Saved visualization to %s", save_path)

            # Save Checkpoint
            torch.save(
                model.state_dict(),
                os.path.join(args.output_dir, "finetuned_best.pth"),
            )


if __name__ == "__main__":
    main()
