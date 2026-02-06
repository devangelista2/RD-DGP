import logging
import torch
from tqdm import tqdm

from materials.dgp.utils import compute_psnr_ssim, get_model_attributes, unet_forward

logger = logging.getLogger(__name__)


class DiffPIRPipeline:
    def __init__(self, unet, scheduler, device="cuda"):
        self.unet = unet
        self.scheduler = scheduler
        self.device = device

    def calculate_metrics(self, img_tensor, ref_tensor):
        return compute_psnr_ssim(img_tensor, ref_tensor)

    def reconstruct(
        self,
        measurement,
        image_size,  # <--- ADDED explicit argument
        forward_op,
        fbp_op=None,
        clean_reference=None,
        num_steps=100,
        lambda_data=5.0,
        rho=1.0,
        snapshot_interval=None,
        snapshot_callback=None,
    ):
        # Robustly determine channels and size
        in_channels, detected_size = get_model_attributes(
            self.unet, default_size=image_size
        )

        # Use passed image_size if detection fails or is generic
        final_size = image_size if image_size else detected_size

        self.scheduler.set_timesteps(num_steps)
        timesteps = self.scheduler.timesteps.to(self.device)

        # --- SMART INITIALIZATION ---
        if fbp_op is not None:
            logger.info("DiffPIR: initializing from FBP (smart init).")
            # 1. Get crude reconstruction
            x_fbp = fbp_op(measurement)

            # 2. Add noise to match the starting timestep T
            # (DiffPIR expects x_T to be noisy)
            noise = torch.randn_like(x_fbp)

            # 3. Add noise equivalent to the first scheduler step
            # Note: We usually start at pure noise, but mixing in signal helps
            x = self.scheduler.add_noise(
                x_fbp, noise, timesteps[0].view(1).expand(1)  # Timestep T
            )
        else:
            logger.info("DiffPIR: initializing from random noise.")
            x = torch.randn(
                (1, in_channels, final_size, final_size),
                device=self.device,
            )

        history = {"psnr": [], "ssim": [], "step": []}

        pbar = tqdm(timesteps, desc="DiffPIR")
        for i, t in enumerate(pbar):
            # --- FIX: Ensure 't' is a 1D tensor matching batch size ---
            t_input = t.view(1).expand(x.shape[0])

            with torch.no_grad():
                # Use wrapper to predict noise (Safe for MONAI)
                model_output = unet_forward(self.unet, x, t_input)

            # Ensure alpha_bar is on correct device
            alpha_bar = self.scheduler.alphas_cumprod.to(self.device)[t]
            beta_bar = 1 - alpha_bar
            x0_hat = (x - beta_bar ** (0.5) * model_output) / (alpha_bar ** (0.5))

            # --- STABILITY FIX 1: Initial Clamp ---
            x0_opt = x0_hat.clone().detach().clamp(-1, 1).requires_grad_(True)

            # Use SGD or Adam
            optimizer = torch.optim.Adam([x0_opt], lr=0.1)

            # Inner Loop
            for _ in range(5):
                optimizer.zero_grad()
                y_pred = forward_op(x0_opt)

                data_loss = torch.norm(y_pred - measurement) ** 2
                prox_loss = rho * torch.norm(x0_opt - x0_hat.detach()) ** 2

                total_loss = (lambda_data * data_loss) + prox_loss
                total_loss.backward()

                # --- STABILITY FIX 2: Gradient Clipping ---
                torch.nn.utils.clip_grad_norm_([x0_opt], max_norm=0.1)

                optimizer.step()

                # --- STABILITY FIX 3: Project back to valid range ---
                with torch.no_grad():
                    x0_opt.clamp_(-1.0, 1.0)

            x0_refined = x0_opt.detach()

            # Re-noise
            eps_refined = (x - (alpha_bar**0.5) * x0_refined) / (beta_bar**0.5)

            with torch.no_grad():
                step_out = self.scheduler.step(eps_refined, t, x)
                x = step_out.prev_sample

            # --- SNAPSHOT SAVING ---
            if snapshot_callback is not None and snapshot_interval is not None:
                if i % snapshot_interval == 0:
                    snapshot_callback(x0_refined, i)

            if clean_reference is not None:
                current_psnr, current_ssim = self.calculate_metrics(
                    x0_refined, clean_reference
                )
                history["psnr"].append(current_psnr)
                history["ssim"].append(current_ssim)
                pbar.set_description(
                    f"DiffPIR | PSNR: {current_psnr:.2f} | SSIM: {current_ssim:.3f}"
                )

        return x0_refined, history
