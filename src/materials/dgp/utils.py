import csv
import logging
import os
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
import yaml
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    Lambdad,
    LoadImaged,
    Rand2DElasticd,
    RandAffined,
    RandGaussianNoised,
    Resized,
    Rotate90d,
    ScaleIntensityd,
    ToTensord,
)
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr_func
from skimage.metrics import structural_similarity as ssim_func

logger = logging.getLogger(__name__)


def seed_everything(seed):
    if seed is None:
        return
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path):
    """Load YAML config file."""
    with Path(config_path).open("r") as f:
        return yaml.safe_load(f)


def list_png_files(root_dir):
    root = Path(root_dir)
    files = sorted(root.rglob("*.png"))
    if not files:
        raise ValueError(f"No .png files found in {root_dir}")
    return [{"image": str(path)} for path in files]


def extract_first_channel(x):
    """Keep only the first channel (grayscale)."""
    return x[0:1, :, :]


def build_ct_transforms(image_size, padding_mode="zeros"):
    return Compose(
        [
            LoadImaged(keys=["image"], image_only=True),
            EnsureChannelFirstd(keys=["image"]),
            Rotate90d(keys=["image"], k=3),
            Lambdad(keys=["image"], func=extract_first_channel),
            ScaleIntensityd(keys=["image"]),
            Resized(keys=["image"], spatial_size=(image_size, image_size)),
            Rand2DElasticd(
                keys=["image"],
                prob=0.5,
                spacing=(20, 20),
                magnitude_range=(2, 2),
                padding_mode=padding_mode,
            ),
            RandAffined(
                keys=["image"],
                prob=0.5,
                rotate_range=0.26,
                scale_range=0.1,
                padding_mode=padding_mode,
            ),
            RandGaussianNoised(keys=["image"], prob=0.1, std=0.01),
            ToTensord(keys=["image"]),
            EnsureTyped(keys=["image"]),
        ]
    )


def load_image(image_path, size=None, grayscale=True, device="cpu"):
    """Load an image, resize, convert to tensor, and normalize to [-1, 1]."""
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found at {image_path}")

    image = Image.open(image_path).convert("L" if grayscale else "RGB")

    transforms_list = []
    if size is not None:
        transforms_list.append(T.Resize(size))

    transforms_list.append(T.ToTensor())
    transforms_list.append(T.Normalize(mean=[0.5], std=[0.5]))

    transform = T.Compose(transforms_list)
    tensor = transform(image)
    return tensor.unsqueeze(0).to(device)


def save_image(tensor, output_path, silent=False):
    """Denormalize from [-1, 1] to [0, 255] and save."""
    img = tensor.detach().cpu()
    img = (img / 2 + 0.5).clamp(0, 1)
    if img.dim() == 4:
        img = img.squeeze(0)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    T.ToPILImage()(img).save(output_path)
    if not silent:
        logger.info("Saved result to %s", output_path)


def load_tensor(path, device="cpu"):
    """Load a raw .pt tensor."""
    return torch.load(path, map_location=device)


def get_model_attributes(unet, default_size=256):
    """Return (in_channels, image_size) for MONAI models."""
    if hasattr(unet, "in_channels"):
        return unet.in_channels, default_size
    raise ValueError("Could not detect MONAI model channels.")


def unet_forward(unet, x, t):
    """Unified UNet forward for MONAI models."""
    return unet(x, t)


def compute_psnr_ssim(img_tensor, ref_tensor):
    img = img_tensor.detach().cpu().squeeze()
    ref = ref_tensor.detach().cpu().squeeze()
    img = (img * 0.5 + 0.5).clamp(0, 1).numpy()
    ref = (ref * 0.5 + 0.5).clamp(0, 1).numpy()

    psnr_val = psnr_func(ref, img, data_range=1.0)
    win_size = min(7, img.shape[0], img.shape[1])
    if win_size % 2 == 0:
        win_size -= 1
    ssim_val = ssim_func(ref, img, data_range=1.0, win_size=win_size)
    return psnr_val, ssim_val


def setup_experiment(output_dir):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    exp_dir = Path(output_dir) / timestamp
    exp_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir = exp_dir / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    return timestamp, exp_dir, snapshot_dir


def load_unet(weights_path, config_path, device, load_unet_model):
    config = load_config(config_path)
    unet = load_unet_model(weights_path, config=config, device=device)
    return unet, config["sample_size"]


def setup_radon(sample_size, num_angles, device, radon_cls):
    return radon_cls(image_size=sample_size, num_angles=num_angles, device=device)


def simulate_measurement(clean_image, radon, noise_sigma, seed):
    raw_sinogram = radon(clean_image)
    noise = torch.randn_like(raw_sinogram) * noise_sigma
    noisy_sinogram = raw_sinogram + noise
    noisy_sinogram = torch.clamp(noisy_sinogram, min=0.0)
    return raw_sinogram, noisy_sinogram


def save_sinogram_vis(sinogram, path):
    sino = sinogram.detach().cpu().squeeze()
    plt.figure(figsize=(6, 6))
    plt.imshow(sino.numpy(), cmap="gray", aspect="auto")
    plt.title(f"Sinogram Range: [{sino.min():.1f}, {sino.max():.1f}]")
    plt.axis("off")
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def save_error_map(clean, recon, output_path):
    clean = clean.detach().cpu().squeeze()
    recon = recon.detach().cpu().squeeze()
    clean = (clean * 0.5 + 0.5).clamp(0, 1)
    recon = (recon * 0.5 + 0.5).clamp(0, 1)
    if clean.ndim == 3:
        clean = clean.mean(dim=0)
        recon = recon.mean(dim=0)
    diff = torch.abs(clean - recon)
    plt.figure(figsize=(6, 6))
    plt.imshow(diff.numpy(), cmap="inferno")
    plt.colorbar(label="Absolute Error")
    plt.axis("off")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def save_dgp_metrics_csv(history, output_dir):
    csv_path = Path(output_dir) / "metrics.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Iteration", "Loss", "PSNR", "SSIM"])
        losses = history.get("loss", [])
        psnr_list = history.get("psnr", [])
        ssim_list = history.get("ssim", [])
        for i, loss_val in enumerate(losses):
            psnr = psnr_list[i] if i < len(psnr_list) else 0
            ssim = ssim_list[i] if i < len(ssim_list) else 0
            writer.writerow([i, loss_val, psnr, ssim])


def save_method_metrics_csv(history, output_dir, method_name):
    path = Path(output_dir) / f"{method_name}_metrics.csv"
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Step", "PSNR", "SSIM"])
        steps = history.get("step", [])
        num_records = len(history["psnr"])
        use_fallback = (not steps) or (len(steps) != num_records)

        for i in range(num_records):
            step_val = i if use_fallback else steps[i]
            psnr_val = history["psnr"][i]
            ssim_val = history["ssim"][i]
            writer.writerow([step_val, psnr_val, ssim_val])


def save_plots(history, output_dir, method_name=None):
    if not history.get("psnr"):
        return
    epochs = range(len(history["psnr"]))
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["psnr"], label="PSNR", color="blue")
    plt.title("PSNR (Image Space)" if not method_name else f"{method_name} - PSNR")
    plt.xlabel("Iteration" if not method_name else "Step")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["ssim"], label="SSIM", color="red")
    plt.title("SSIM (Image Space)" if not method_name else f"{method_name} - SSIM")
    plt.xlabel("Iteration" if not method_name else "Step")
    plt.grid(True)

    name = "metrics_plot.png" if not method_name else f"{method_name}_metrics.png"
    plt.savefig(Path(output_dir) / name)
    plt.close()
