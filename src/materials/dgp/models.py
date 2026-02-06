import logging
from pathlib import Path

import torch
from generative.networks.nets import DiffusionModelUNet

logger = logging.getLogger(__name__)


def create_monai_unet(config):
    """Instantiate a MONAI DiffusionModelUNet for 2D CT."""
    return DiffusionModelUNet(
        spatial_dims=2,
        in_channels=config.get("in_channels", 1),
        out_channels=config.get("out_channels", 1),
        num_channels=tuple(config.get("block_out_channels", (64, 128, 256, 512))),
        attention_levels=tuple(config.get("attention_levels", (False, False, True, True))),
        num_res_blocks=config.get("layers_per_block", 2),
        num_head_channels=config.get("num_head_channels", 32),
    )


def load_weights(model, weight_path, device="cpu", strict=False):
    """Load a MONAI state_dict, handling DataParallel prefixes."""
    weight_path = Path(weight_path)
    if not weight_path.exists():
        raise FileNotFoundError(f"Weights not found at {weight_path}")

    state_dict = torch.load(weight_path, map_location=device)
    cleaned = {
        (k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()
    }
    missing, unexpected = model.load_state_dict(cleaned, strict=strict)
    if missing:
        logger.warning("Missing keys: %d", len(missing))
    if unexpected:
        logger.warning("Unexpected keys: %d", len(unexpected))

    model.to(device)
    model.eval()
    return model


def load_unet_model(weights_path, config, device="cpu", strict_load=False):
    """Load a MONAI UNet with CT configuration and weights."""
    model = create_monai_unet(config)
    return load_weights(model, weights_path, device=device, strict=strict_load)
