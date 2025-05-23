import datetime
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image


def get_device(config_device, logger):
    """Sets the device for computation."""
    if config_device == "auto":
        try:
            if torch.mps.is_available():
                selected_device = torch.device("mps")
            else:
                selected_device = torch.device("cpu")
        except:
            pass

        if torch.cuda.is_available():
            selected_device = torch.device("cuda")
        else:
            selected_device = torch.device("cpu")
        logger.info(f"Device selected by 'utilities.get_device()': {selected_device}")
        return selected_device
    selected_device = torch.device(config_device)
    logger.info(f"Device set to: {selected_device}")
    return selected_device


def plot_and_save_images(x_true, y_delta, x_gen, ssim_val, save_path, config, logger):
    """Plots and saves the true, corrupted, and reconstructed images."""
    logger.info("Generating and saving output images.")

    def prep_img_for_plot(tensor_img):
        img = tensor_img.detach().cpu().squeeze(0)
        if img.shape[0] == 1:
            return img.squeeze(0).numpy(), "gray"
        else:
            return img.permute(1, 2, 0).numpy(), None

    plt.figure(figsize=(12, 4))

    x_true_plot, cmap_true = prep_img_for_plot(x_true)
    plt.subplot(1, 3, 1)
    plt.imshow(x_true_plot, cmap=cmap_true)
    plt.title("True")
    plt.axis("off")

    if config["operator"] == "CT":
        y_plot = y_delta.detach().cpu().squeeze(0).numpy()
        if y_plot.ndim == 3 and y_plot.shape[0] == 1:
            y_plot = y_plot.squeeze(0)
        plt.subplot(1, 3, 2)
        plt.imshow(y_plot, cmap="gray", aspect="auto")
        plt.title("Corrupted (Sinogram)")
    else:
        y_delta_plot, cmap_y = prep_img_for_plot(y_delta)
        plt.subplot(1, 3, 2)
        plt.imshow(y_delta_plot, cmap=cmap_y)
        plt.title("Corrupted")
    plt.axis("off")

    x_gen_plot, cmap_gen = prep_img_for_plot(x_gen)
    plt.subplot(1, 3, 3)
    plt.imshow(x_gen_plot, cmap=cmap_gen)
    plt.title(f"Reconstructed\n(SSIM {ssim_val:.4f})")
    plt.axis("off")

    plt.tight_layout()
    image_save_path = os.path.join(save_path, "reconstruction.png")
    plt.savefig(image_save_path, dpi=400)
    plt.close()
    logger.info(f"Reconstruction image saved to: {image_save_path}")


def save_generated_images(samples, saving_path, gen_config, logger):
    """Saves generated samples to files."""
    save_format = gen_config.get("save_format", "grid")
    prefix = gen_config.get("output_filename_prefix", "generated_sample")

    # Normalize samples from [-1, 1] to [0, 1] for saving
    samples = (samples.clamp(-1, 1) + 1.0) / 2.0

    if save_format == "grid":
        grid_path = os.path.join(saving_path, f"{prefix}_grid.png")
        # Determine nrow for make_grid, e.g., sqrt(batch_size) or a fixed number like 4
        nrow = (
            int(np.sqrt(samples.shape[0]))
            if int(np.sqrt(samples.shape[0])) ** 2 == samples.shape[0]
            else min(samples.shape[0], 4)
        )
        save_image(samples, grid_path, nrow=nrow)
        logger.info(f"Saved image grid to: {grid_path}")
    elif save_format == "individual":
        for i, sample in enumerate(samples):
            img_path = os.path.join(saving_path, f"{prefix}_{i:03d}.png")
            save_image(
                sample, img_path
            )  # save_image handles individual tensors correctly
        logger.info(f"Saved {len(samples)} individual images to: {saving_path}")
    else:
        logger.warning(f"Unknown save_format: {save_format}. Images not saved.")


def load_and_preprocess_image(config, target_size, device, logger):
    """Loads and preprocesses the true image."""
    img_path = config["image_path"]
    channels = config["image_channels"]
    logger.info(f"Loading image from: {img_path} with {channels} channel(s).")

    if channels == 1:
        image = Image.open(img_path).convert("L")
    elif channels == 3:
        image = Image.open(img_path).convert("RGB")
    else:
        logger.error(f"Unsupported number of channels: {channels}. Must be 1 or 3.")
        raise ValueError(f"Unsupported number of channels: {channels}. Must be 1 or 3.")

    preprocess_transforms = [
        transforms.Resize(target_size),
        transforms.ToTensor(),
    ]
    preprocess = transforms.Compose(preprocess_transforms)
    x_true = preprocess(image).unsqueeze(0).to(device)
    logger.info(f"Image preprocessed and loaded to device. Shape: {x_true.shape}")
    return x_true


# --- Logger Setup ---
def setup_logger(name, log_file, level=logging.INFO):
    """Function to setup as many loggers as you want"""
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # Prevent logging from propagating to the root logger if it's already configured
    # or to avoid duplicate messages if this function is called multiple times with the same name.
    logger.propagate = False

    return logger


def load_config(config_path):
    """Loads YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def setup_paths(config):
    """Creates a unique directory for saving experiment results."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(
        config["results_base_dir"],
        f"{config['experiment_name']}_{config['operator']}_{config['noise_level']}_{config['operator']}_{timestamp}",
    )
    os.makedirs(experiment_dir, exist_ok=True)
    return experiment_dir
