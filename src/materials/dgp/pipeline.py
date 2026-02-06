import logging

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm

from materials.dgp.utils import compute_psnr_ssim, get_model_attributes, unet_forward

logger = logging.getLogger(__name__)


class DGPOptimizer:
    def __init__(self, unet, scheduler, device="cuda"):
        self.unet = unet
        self.scheduler = scheduler
        self.device = device
        self.unet.requires_grad_(False)
        self.unet.to(device)

    def ddim_step_differentiable(
        self, model_output, timestep, sample, alpha_bar, prev_alpha_bar
    ):
        """Forward DDIM Step: Noise -> Image"""
        beta_prod_t = 1 - alpha_bar
        pred_original_sample = (sample - beta_prod_t**0.5 * model_output) / (
            alpha_bar**0.5 + 1e-6
        )
        pred_sample_direction = (1 - prev_alpha_bar) ** 0.5 * model_output
        prev_sample = (
            prev_alpha_bar**0.5
        ) * pred_original_sample + pred_sample_direction
        return prev_sample

    def ddim_step_inverse(
        self, model_output, timestep, sample, alpha_bar, next_alpha_bar
    ):
        """
        Inverse DDIM Step: Image -> Noise
        """
        beta_prod_t = 1 - alpha_bar
        pred_original_sample = (sample - beta_prod_t**0.5 * model_output) / (
            alpha_bar**0.5 + 1e-6
        )
        pred_sample_direction = (1 - next_alpha_bar) ** 0.5 * model_output
        next_sample = (
            next_alpha_bar**0.5
        ) * pred_original_sample + pred_sample_direction
        return next_sample

    def invert(self, image, num_steps):
        logger.info("Running DDIM inversion (image -> latent)...")
        self.scheduler.set_timesteps(num_steps)
        timesteps = self.scheduler.timesteps.to(self.device).flip(0)

        x = image.clone()

        for i, t in enumerate(tqdm(timesteps)):
            # --- FIX: Ensure 't' is a 1D tensor matching batch size ---
            # t is a scalar here. We need it to be shape [Batch_Size].
            t_input = t.view(1).expand(x.shape[0])

            with torch.no_grad():
                noise_pred = unet_forward(self.unet, x, t_input)

            alpha_cumprod = self.scheduler.alphas_cumprod.to(self.device)
            # t is still used as an index for alphas, so we keep scalar 't' here
            alpha_bar = alpha_cumprod[t]

            train_step = t.item()
            step_size = self.scheduler.config.num_train_timesteps // num_steps
            next_train_step = min(
                train_step + step_size, self.scheduler.config.num_train_timesteps - 1
            )
            next_alpha_bar = alpha_cumprod[next_train_step]

            x = self.ddim_step_inverse(noise_pred, t, x, alpha_bar, next_alpha_bar)

        return x

    def generate_differentiable(self, z, num_steps):
        self.scheduler.set_timesteps(num_steps)
        timesteps = self.scheduler.timesteps.to(self.device)
        x = z
        for i, t in enumerate(timesteps):
            t_input = t.repeat(x.shape[0])
            # Use wrapper here too
            noise_pred = checkpoint(
                unet_forward, self.unet, x, t_input, use_reentrant=False
            )

            alpha_cumprod = self.scheduler.alphas_cumprod.to(self.device)
            alpha_bar = alpha_cumprod[t]

            prev_t = (
                t
                - self.scheduler.config.num_train_timesteps
                // self.scheduler.num_inference_steps
            )
            if prev_t >= 0:
                prev_alpha_bar = alpha_cumprod[prev_t]
            else:
                prev_alpha_bar = torch.tensor(1.0, device=self.device)

            x = self.ddim_step_differentiable(
                noise_pred, t, x, alpha_bar, prev_alpha_bar
            )
        return x

    def reconstruct(
        self,
        target_measurement,
        image_size=256,
        forward_operator=None,
        fbp_operator=None,
        clean_reference=None,
        noise_sigma=None,
        num_ddim_steps=20,
        ddim_start_steps=None,
        ddim_end_steps=None,
        num_opt_steps=100,
        lr=0.05,
        lambda_reg=0.01,
        lambda_tv=0.0,
        seed=42,
        smart_init=True,
        snapshot_interval=None,
        snapshot_callback=None,
        return_latent=False,
    ):
        generator = torch.Generator(device=self.device).manual_seed(seed)

        if ddim_start_steps is None and ddim_end_steps is None:
            ddim_start_steps = num_ddim_steps
            ddim_end_steps = num_ddim_steps
        elif ddim_start_steps is None:
            ddim_start_steps = num_ddim_steps
        elif ddim_end_steps is None:
            ddim_end_steps = num_ddim_steps

        ddim_start_steps = max(1, int(ddim_start_steps))
        ddim_end_steps = max(1, int(ddim_end_steps))

        # Robustly determine channels and size
        in_channels, detected_size = get_model_attributes(
            self.unet, default_size=image_size
        )

        # Use passed image_size if detection fails or is generic
        final_size = image_size if image_size else detected_size

        shape = (1, in_channels, final_size, final_size)

        # (Rest of your reconstruct code stays exactly the same...)
        dims = shape[1] * shape[2] * shape[3]
        expected_norm = np.sqrt(dims)
        target_mse_floor = (noise_sigma**2) if noise_sigma else 0.0

        # --- SMART INITIALIZATION ---
        z = None
        init_steps = ddim_start_steps
        if smart_init:
            if forward_operator is None:
                logger.info("Initializing via inversion of target image.")
                z = self.invert(target_measurement, num_steps=init_steps)
                z = z.detach().clone()
                if clean_reference is not None:
                    psnr, ssim = compute_psnr_ssim(target_measurement, clean_reference)
                    logger.info("Starting G(z) - PSNR: %.2f | SSIM: %.4f", psnr, ssim)

            elif fbp_operator is not None:
                logger.info("Initializing via FBP -> inversion.")
                with torch.no_grad():
                    approx_image = fbp_operator(target_measurement)
                    approx_image = approx_image.clamp(-1.0, 1.0)
                    if clean_reference is not None:
                        psnr, ssim = compute_psnr_ssim(approx_image, clean_reference)
                        logger.info(
                            "Starting G(z) - PSNR: %.2f | SSIM: %.3f", psnr, ssim
                        )

                z = self.invert(approx_image, num_steps=init_steps)
                z = z.detach().clone()
            else:
                logger.info("No FBP operator provided. Falling back to random init.")
                z = torch.randn(shape, device=self.device, generator=generator)
        else:
            z = torch.randn(shape, device=self.device, generator=generator)

        z.requires_grad = True

        optimizer = Adam([z], lr=lr)
        scheduler_lr = CosineAnnealingLR(optimizer, T_max=num_opt_steps, eta_min=1e-4)

        logger.info("Starting DGP. Target MSE floor: %.5f", target_mse_floor)

        history = {"psnr": [], "ssim": [], "loss": []}
        best_img = None
        best_loss_diff = float("inf")

        pbar = tqdm(range(num_opt_steps))
        last_ddim_steps = ddim_start_steps
        for i in pbar:
            optimizer.zero_grad()

            if not smart_init and i < num_opt_steps * 0.7:
                noise_factor = 0.001 * (1 - i / num_opt_steps)
                with torch.no_grad():
                    z.add_(torch.randn_like(z) * noise_factor)

            if num_opt_steps > 1:
                progress = i / (num_opt_steps - 1)
            else:
                progress = 1.0
            current_steps = int(
                round(ddim_start_steps + (ddim_end_steps - ddim_start_steps) * progress)
            )
            current_steps = max(1, current_steps)
            last_ddim_steps = current_steps

            generated_image = self.generate_differentiable(z, current_steps)

            if forward_operator is not None:
                generated_measurement = forward_operator(generated_image)
            else:
                generated_measurement = generated_image

            mse_loss = torch.nn.functional.mse_loss(
                generated_measurement, target_measurement, reduction="mean"
            )

            current_norm = z.norm()
            reg_loss = ((current_norm - expected_norm) ** 2) * lambda_reg

            tv_loss = 0.0
            if lambda_tv > 0:
                b, c, h, w = generated_image.shape
                tv_h = torch.pow(
                    generated_image[:, :, 1:, :] - generated_image[:, :, :-1, :], 2
                ).sum()
                tv_w = torch.pow(
                    generated_image[:, :, :, 1:] - generated_image[:, :, :, :-1], 2
                ).sum()
                tv_loss = (tv_h + tv_w) / (b * c * h * w) * lambda_tv

            total_loss = mse_loss + reg_loss + tv_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_([z], max_norm=0.1)
            optimizer.step()
            scheduler_lr.step()

            if snapshot_callback is not None and snapshot_interval is not None:
                if i % snapshot_interval == 0:
                    snapshot_callback(generated_image, i)

            current_mse = mse_loss.item()
            dist_to_floor = abs(current_mse - target_mse_floor)
            if dist_to_floor < best_loss_diff:
                best_loss_diff = dist_to_floor
                with torch.no_grad():
                    best_img = generated_image.clone()

            history["loss"].append(current_mse)
            if clean_reference is not None:
                psnr, ssim = compute_psnr_ssim(generated_image, clean_reference)
                history["psnr"].append(psnr)
                history["ssim"].append(ssim)

            msg = f"MSE: {current_mse:.4f}"
            if clean_reference is not None:
                msg += f" | PSNR: {history['psnr'][-1]:.2f} | SSIM: {history['ssim'][-1]:.3f}"
            pbar.set_description(msg)

        if best_img is None:
            with torch.no_grad():
                best_img = self.generate_differentiable(z, last_ddim_steps)
        if return_latent:
            return best_img, z.detach(), history
        return best_img, history
