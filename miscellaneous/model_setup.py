import os

import numpy as np
from diffusers import DDIMScheduler, UNet2DModel

from IPPy import operators


def load_model_and_scheduler(config, device, logger):
    model_id_or_path = config["model_path"]
    logger.info(
        f"Attempting to load model from Hugging Face ID or path: {model_id_or_path}"
    )

    try:
        # Standard way to load a UNet2DModel from Hub or local path containing model files directly
        model = UNet2DModel.from_pretrained(model_id_or_path).to(device)
        logger.info(
            f"Model loaded successfully. Architecture: {model.__class__.__name__}"
        )
    except EnvironmentError as e:  # More specific error for missing files/ID
        logger.warning(
            f"Failed to load model directly from '{model_id_or_path}' (may indicate it's a local path needing subfolders or incorrect ID): {e}"
        )
        logger.info(
            f"Attempting to load model from subfolder 'unet' in '{model_id_or_path}' (for local pre-downloaded models)"
        )
        try:
            model = UNet2DModel.from_pretrained(
                os.path.join(model_id_or_path, "unet")
            ).to(device)
            logger.info("Model loaded successfully from 'unet' subfolder.")
        except Exception as e_sub:
            logger.error(f"Failed to load model from 'unet' subfolder as well: {e_sub}")
            raise  # Re-raise the error if both attempts fail
    except Exception as e:  # Catch other potential errors during model loading
        logger.error(f"An unexpected error occurred while loading the model: {e}")
        raise

    logger.info(f"Attempting to load scheduler for '{model_id_or_path}'")
    try:
        # Many Hugging Face models store scheduler config in a 'scheduler' subfolder
        scheduler = DDIMScheduler.from_pretrained(
            model_id_or_path, subfolder="scheduler"
        )
        logger.info("Scheduler loaded successfully from 'scheduler' subfolder.")
    except (
        EnvironmentError
    ):  # If 'scheduler' subfolder doesn't exist or no scheduler_config.json
        logger.warning(
            f"Could not load scheduler from 'scheduler' subfolder in '{model_id_or_path}'."
        )
        try:
            # Try to load from the main model's config (model.config should have scheduler parameters)
            # This is common if the scheduler type is defined in the model's main config.json
            scheduler = DDIMScheduler.from_config(model.config)
            logger.info(
                "Scheduler loaded successfully using 'from_config(model.config)'."
            )
        except Exception as e_config:
            logger.warning(f"Failed to load scheduler from model.config: {e_config}.")
            # Fallback: Initialize a DDIMScheduler with some defaults.
            # This is a last resort and might not match the model's optimal scheduler settings.
            # You might need to check the model card on Hugging Face for recommended scheduler parameters.
            default_scheduler_config = {
                "beta_start": 0.00085,
                "beta_end": 0.012,
                "beta_schedule": "scaled_linear",
                "num_train_timesteps": 1000,
                "clip_sample": False,  # clip_sample might be in model.config
                "set_alpha_to_one": False,  # common for DDPM, check model specifics
            }
            # Attempt to override defaults with anything relevant from model.config if available
            if hasattr(model.config, "clip_sample"):
                default_scheduler_config["clip_sample"] = model.config.clip_sample
            if hasattr(model.config, "set_alpha_to_one"):
                default_scheduler_config["set_alpha_to_one"] = (
                    model.config.set_alpha_to_one
                )

            scheduler = DDIMScheduler(**default_scheduler_config)
            logger.info(
                "Initialized a DDIMScheduler with common defaults as a fallback. This may not be optimal."
            )

    return model, scheduler


def get_operator(config, img_shape_hw, num_channels, device, logger):
    """Initializes the forward operator."""
    op_type = config["operator"]
    logger.info(f"Initializing operator: {op_type}")

    if op_type == "CT":
        K = operators.CTProjector(
            img_shape=img_shape_hw,
            angles=np.linspace(
                np.deg2rad(config["ct"]["start_angle"]),
                np.deg2rad(config["ct"]["end_angle"]),
                config["ct"]["n_angles"],
            ),
            geometry="parallel",
            det_size=config["ct"]["det_size"],
        )
    elif op_type == "Deblur":
        K = operators.Blurring(
            img_shape=img_shape_hw,
            kernel_type=config["deblur"]["kernel_type"],
            kernel_size=config["deblur"]["kernel_size"],
            motion_angle=config["deblur"]["kernel_angle"],
            in_channels=num_channels,
        )
    else:
        logger.error(f"Operator {op_type} not defined.")
        raise NotImplementedError(f"Operator {op_type} not defined.")
    logger.info(f"Operator {op_type} initialized on device {device}.")
    return K
