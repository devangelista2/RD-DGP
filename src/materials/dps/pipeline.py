import logging
import lpips
import torch
from tqdm import tqdm

from materials.dgp.utils import compute_psnr_ssim, get_model_attributes, unet_forward

logger = logging.getLogger(__name__)


class DPSPipeline:
    def __init__(self, unet, scheduler, device="cuda"):
        self.unet = unet
        self.scheduler = scheduler
        self.device = device

        self.lpips = lpips.LPIPS(net="alex").to(self.device).eval()

    def calculate_metrics(self, img_tensor, ref_tensor):
        psnr_val, ssim_val = compute_psnr_ssim(img_tensor, ref_tensor)

        # --- LPIPS (torch, in [-1,1]) ---
        # Make (1,1,H,W) then replicate to (1,3,H,W)
        x = (
            img_tensor.detach()
            .to(self.device)
            .view(1, 1, *img_tensor.shape[-2:])
            .float()
        )
        y = (
            ref_tensor.detach()
            .to(self.device)
            .view(1, 1, *ref_tensor.shape[-2:])
            .float()
        )
        x = x.repeat(1, 3, 1, 1)
        y = y.repeat(1, 3, 1, 1)

        with torch.no_grad():
            lpips_val = float(self.lpips(x, y).item())

        return psnr_val, ssim_val, lpips_val

    def reconstruct(
        self,
        measurement,
        image_size,
        forward_op,
        clean_reference=None,
        num_steps=100,
        zeta=1.0,
        snapshot_interval=None,
        snapshot_callback=None,
    ):
        # Robustly determine channels and size using the new helper
        in_channels, detected_size = get_model_attributes(
            self.unet, default_size=image_size
        )

        # Use passed image_size if detection fails or is generic
        final_size = image_size if image_size else detected_size

        self.scheduler.set_timesteps(num_steps)
        timesteps = self.scheduler.timesteps.to(self.device)

        # Start from random noise
        x = torch.randn(
            (
                1,
                in_channels,
                final_size,
                final_size,
            ),
            device=self.device,
        )

        history = {"psnr": [], "ssim": [], "lpips": [], "step": []}

        pbar = tqdm(timesteps, desc="DPS")
        for i, t in enumerate(pbar):
            # --- FIX: Ensure 't' is a 1D tensor matching batch size ---
            t_input = t.view(1).expand(x.shape[0])

            # 1. Enable grad
            x = x.detach().requires_grad_(True)

            # 2. Predict Noise using WRAPPER (Handles MONAI vs Diffusers)
            model_output = unet_forward(self.unet, x, t_input)

            # 3. Estimate x0_hat (Tweedy's Formula)
            # Ensure alpha_bar is on correct device
            alpha_bar = self.scheduler.alphas_cumprod.to(self.device)[t]
            beta_bar = 1 - alpha_bar
            x0_hat = (x - beta_bar ** (0.5) * model_output) / (alpha_bar ** (0.5))

            # --- STABILITY FIX 1: Clamp x0_hat ---
            x0_hat_clamped = x0_hat.clamp(-1.5, 1.5)

            # 4. Compute Loss
            y_pred = forward_op(x0_hat_clamped)
            loss = torch.norm(y_pred - measurement) ** 2

            # 5. Compute Gradient
            norm_grad = torch.autograd.grad(loss, x)[0]

            # --- STABILITY FIX 2: Gradient Clipping ---
            grad_norm = torch.linalg.norm(norm_grad)
            if grad_norm > 1.0:
                norm_grad = norm_grad / grad_norm

            # 6. Step
            with torch.no_grad():
                step_out = self.scheduler.step(model_output, t, x)
                x_prev = step_out.prev_sample

                # Apply guided gradient
                x = x_prev - zeta * norm_grad

                # --- STABILITY FIX 4: Safety Clamp on x ---
                x = x.clamp(-4.0, 4.0)

            # --- SNAPSHOT SAVING ---
            if snapshot_callback is not None and snapshot_interval is not None:
                if i % snapshot_interval == 0:
                    snapshot_callback(x0_hat, i)

            # Metrics
            if clean_reference is not None:
                current_psnr, current_ssim, current_lpips = self.calculate_metrics(
                    x0_hat, clean_reference
                )
                history["psnr"].append(current_psnr)
                history["ssim"].append(current_ssim)
                history["lpips"].append(current_lpips)
                history["step"].append(i)
                pbar.set_description(
                    f"DPS | PSNR: {current_psnr:.2f} | SSIM: {current_ssim:.3f} | LPIPS: {current_lpips:.3f}"
                )

        return x0_hat.detach(), history
