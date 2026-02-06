import logging
import lpips
import torch
from tqdm import tqdm

from materials.dgp.utils import compute_psnr_ssim, get_model_attributes, unet_forward

logger = logging.getLogger(__name__)


class DDRMPipeline:
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
        fbp_op,
        clean_reference=None,
        num_steps=20,
        snapshot_interval=None,
        snapshot_callback=None,
        projection_strength=0.8,  # Replaces 'parameter-free' assumption. 1.0 = Hard DDRM.
        null_noise_scale=0,  # Adds texture back into the null-space
    ):
        in_channels, detected_size = get_model_attributes(
            self.unet, default_size=image_size
        )
        final_size = image_size if image_size else detected_size

        self.scheduler.set_timesteps(num_steps)
        timesteps = self.scheduler.timesteps.to(self.device)

        # 1. The "Data Anchor" (FBP)
        x_range_0 = fbp_op(measurement)

        # Initialize x_T
        x = torch.randn((1, in_channels, final_size, final_size), device=self.device)

        history = {"psnr": [], "ssim": [], "lpips": [], "step": []}

        # Define start and end strengths
        strength_start = 1.0  # Trust FBP fully at the start
        strength_end = 0.0  # Trust Model fully at the end (removes streaks)

        pbar = tqdm(timesteps, desc="DDRM (Annealing)")

        for i, t in enumerate(pbar):
            t_input = t.view(1).expand(x.shape[0])

            # 1. Predict Noise / x0
            with torch.no_grad():
                model_output = unet_forward(self.unet, x, t_input)

            alpha_bar = self.scheduler.alphas_cumprod[t]
            beta_bar = 1 - alpha_bar

            # Tweedie's Formula: Estimate x0 from current noisy x_t
            x0_hat = (x - beta_bar ** (0.5) * model_output) / (alpha_bar ** (0.5))

            # --- 2. RELAXED PROJECTION (The Fix) ---

            # --- CALCULATE DYNAMIC STRENGTH ---
            # progress goes from 0.0 (start) to 1.0 (end)
            progress = i / len(timesteps)
            current_strength = strength_start - progress * (
                strength_start - strength_end
            )

            # A. Calculate what the model thinks the Range Component (Sinogram-consistent part) is:
            # We project the model's guess forward, then back.
            projected_measure = forward_op(x0_hat)
            x0_range_hat = fbp_op(projected_measure)

            # B. Calculate the "Correction Vector"
            # This vector points from the Model's guess -> The Data's truth
            range_correction = x_range_0 - x0_range_hat

            # C. Apply Soft Update
            # Instead of x0_consistent = x0_hat + range_correction (Hard constraint)
            # We use: x0_consistent = x0_hat + strength * range_correction
            x0_consistent = x0_hat + current_strength * range_correction

            # --- 3. NULL-SPACE NOISE (Texture Fix) ---
            # We add noise ONLY to the null-space (features invisible to CT)
            if null_noise_scale > 0:
                z = torch.randn_like(x0_consistent)
                # Project noise onto null space (Identity - K^T K)
                # Ideally: z_null = z - fbp(forward(z))
                # This ensures we don't add noise that corrupts the sinogram
                z_proj = fbp_op(forward_op(z))
                z_null = z - z_proj

                # Add it to our estimate
                x0_consistent = x0_consistent + null_noise_scale * z_null

            # --- 4. DDIM Step to t-1 ---
            # We use x0_consistent as the "clean image" for the next step

            step_index = (
                (self.scheduler.timesteps.to(self.device) == t).nonzero().item()
            )
            if step_index < len(self.scheduler.timesteps) - 1:
                prev_t = self.scheduler.timesteps[step_index + 1]
                alpha_bar_prev = self.scheduler.alphas_cumprod[prev_t]
            else:
                alpha_bar_prev = torch.tensor(1.0).to(self.device)

            # Re-noise to next step
            # Note: We use 0 sigma for the generative part (deterministic),
            # relying on our manual null-space noise above.
            x_prev = (alpha_bar_prev**0.5) * x0_consistent + (
                1 - alpha_bar_prev
            ) ** 0.5 * model_output
            x = x_prev

            # --- SNAPSHOTS & METRICS ---
            if snapshot_callback and snapshot_interval and i % snapshot_interval == 0:
                snapshot_callback(x0_consistent, i)

            if clean_reference is not None:
                current_psnr, current_ssim, current_lpips = self.calculate_metrics(
                    x0_hat, clean_reference
                )
                history["psnr"].append(current_psnr)
                history["ssim"].append(current_ssim)
                history["lpips"].append(current_lpips)
                history["step"].append(i)
                pbar.set_description(
                    f"DDRM | PSNR: {current_psnr:.2f} | SSIM: {current_ssim:.3f} | LPIPS: {current_lpips:.3f}"
                )

        return x0_consistent, history
