import time

import torch
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure


# --- DGPSolver Class ---
class DGPSolver:
    def __init__(
        self, model, scheduler, K_op, x_true, y_delta, z_latent, config, device, logger
    ):
        self.model = model
        self.scheduler = scheduler
        self.K = K_op
        self.x_true = x_true
        self.y_delta = y_delta
        self.z = z_latent  # This z will be optimized
        self.config = config
        self.device = device
        self.logger = logger

        self.loss_fn = torch.nn.MSELoss()
        self.psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

        self.psnr_history = []
        self.ssim_history = []
        self.loss_history = []  # Added for completeness
        self.total_steps_elapsed = (
            0  # For continuous step counting if mixing optimizers
        )

    def _generate_and_normalize(self, current_z):
        """Generates image from latent z and normalizes to [0,1]."""
        # Ensure scheduler uses reconstruction_timesteps for this iterative process
        self.scheduler.set_timesteps(self.config["reconstruction_timesteps"])

        x_gen_iter = (
            current_z.clone()
        )  # Use clone if z is modified elsewhere or by scheduler step
        for t_step in self.scheduler.timesteps:
            with torch.no_grad():
                noise_pred = self.model(x_gen_iter, t_step).sample
            x_gen_iter = self.scheduler.step(noise_pred, t_step, x_gen_iter).prev_sample

        x_gen_normalized = (x_gen_iter + 1.0) / 2.0
        return torch.clamp(x_gen_normalized, 0.0, 1.0)

    def _evaluate_metrics_and_loss(self, x_gen_normalized):
        """Computes loss and metrics, and records them."""
        loss = self.loss_fn(self.K(x_gen_normalized), self.y_delta)
        with torch.no_grad():
            psnr_val = self.psnr_metric(x_gen_normalized, self.x_true).item()
            ssim_val = self.ssim_metric(x_gen_normalized, self.x_true).item()

        self.psnr_history.append(psnr_val)
        self.ssim_history.append(ssim_val)
        self.loss_history.append(loss.item())
        return loss, psnr_val, ssim_val

    def _optimize_adam(self, num_iter, lr, weight_decay, start_time):
        self.logger.info(
            f"Starting Adam optimization for {num_iter} iterations (lr={lr}, wd={weight_decay})."
        )
        optimizer = torch.optim.AdamW([self.z], lr=lr, weight_decay=weight_decay)

        for step in range(num_iter):
            optimizer.zero_grad()
            x_gen_normalized = self._generate_and_normalize(self.z)
            loss, psnr_val, ssim_val = self._evaluate_metrics_and_loss(x_gen_normalized)
            loss.backward()
            optimizer.step()

            self.total_steps_elapsed += 1
            self.logger.info(
                f"(Adam - Time {time.time() - start_time:0.2f}s) Step {self.total_steps_elapsed:03d} (Adam step {step+1:03d}) | "
                f"Loss: {loss.item():.4f} | PSNR: {psnr_val:.2f} | SSIM: {ssim_val:.4f}"
            )
        self.logger.info("Adam optimization phase finished.")

    def _optimize_lbfgs(
        self, num_iter, lr, history_size, max_iter_linesearch, start_time
    ):
        self.logger.info(
            f"Starting LBFGS optimization for {num_iter} iterations (lr={lr}, hist_size={history_size}, max_linesearch_iter={max_iter_linesearch})."
        )
        optimizer = torch.optim.LBFGS(
            [self.z], lr=lr, history_size=history_size, max_iter=max_iter_linesearch
        )

        for step in range(num_iter):

            def closure():
                optimizer.zero_grad()
                # LBFGS closure must re-evaluate the loss and gradients
                x_gen_normalized_closure = self._generate_and_normalize(self.z)
                # We don't record metrics inside closure to avoid multiple recordings per LBFGS step
                loss_closure = self.loss_fn(
                    self.K(x_gen_normalized_closure), self.y_delta
                )
                loss_closure.backward()
                return loss_closure

            optimizer.step(closure)  # This performs one LBFGS update step

            # After the LBFGS step, evaluate and record metrics properly
            with torch.no_grad():  # Ensure z is not changing during this evaluation
                x_gen_final_for_step = self._generate_and_normalize(self.z)
            # Re-evaluate metrics and loss for logging and history with the updated self.z
            loss_val, psnr_val, ssim_val = self._evaluate_metrics_and_loss(
                x_gen_final_for_step
            )

            self.total_steps_elapsed += 1
            self.logger.info(
                f"(LBFGS - Time {time.time() - start_time:0.2f}s) Step {self.total_steps_elapsed:03d} (LBFGS step {step+1:03d}) | "
                f"Loss: {loss_val.item():.4f} | PSNR: {psnr_val:.2f} | SSIM: {ssim_val:.4f}"
            )
        self.logger.info("LBFGS optimization phase finished.")

    def run_optimization(self):
        solver_config = self.config["solver"]
        solver_type = solver_config["type"]
        self.logger.info(f"Starting optimization with solver type: {solver_type}")
        start_time = time.time()

        if solver_type == "Adam":
            params = solver_config["adam_params"]
            self._optimize_adam(
                params["num_iter"], params["lr"], params["weight_decay"], start_time
            )
        elif solver_type == "LBFGS":
            params = solver_config["lbfgs_params"]
            self._optimize_lbfgs(
                params["num_iter"],
                params["lr"],
                params["history_size"],
                params["max_iter_linesearch"],
                start_time,
            )
        elif solver_type == "Adam_LBFGS":
            adam_params = solver_config["adam_params"]
            if adam_params.get("num_iter", 0) > 0:
                self._optimize_adam(
                    adam_params["num_iter"],
                    adam_params["lr"],
                    adam_params["weight_decay"],
                    start_time,
                )

            lbfgs_params = solver_config["lbfgs_params"]
            if lbfgs_params.get("num_iter", 0) > 0:
                # Potentially re-initialize z's grad requirement if LBFGS needs it fresh, though usually not an issue.
                self.z.requires_grad_(True)
                self._optimize_lbfgs(
                    lbfgs_params["num_iter"],
                    lbfgs_params["lr"],
                    lbfgs_params["history_size"],
                    lbfgs_params["max_iter_linesearch"],
                    start_time,
                )
        else:
            self.logger.error(f"Unknown solver type: {solver_type}")
            raise ValueError(f"Unknown solver type: {solver_type}")

        self.logger.info(
            f"Total optimization finished in {time.time() - start_time:.2f}s. Total steps: {self.total_steps_elapsed}"
        )
        return self.z, self.psnr_history, self.ssim_history, self.loss_history


# --- DPSSolver Class ---
class DPSSolver:
    def __init__(self, model, scheduler, K_op, y_delta, x_true, config, device, logger):
        self.model = model
        self.scheduler = scheduler
        self.K_op = K_op
        self.y_delta = y_delta
        self.x_true = x_true  # For metrics
        self.config = config
        self.device = device
        self.logger = logger

        self.dps_conf = config["dps_params"]
        self.loop_timesteps = self.dps_conf["loop_timesteps"]
        self.guidance_scale = self.dps_conf["step_size"]

        self.loss_fn = torch.nn.MSELoss()
        self.psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

        self.psnr_history = []
        self.ssim_history = []
        self.loss_history = []  # Data consistency loss

    def _predict_x0(self, x_t, t_val, model_output_noise):
        """
        Predicts x0 from x_t (current noisy image) and model_output_noise (predicted noise), then clamps.
        t_val is a scalar tensor representing the current timestep (e.g., tensor(999)).
        """
        # self.scheduler.alphas_cumprod is typically on CPU.
        # Ensure the index t_val is also on CPU and is of long type.
        alphas_cumprod_device = self.scheduler.alphas_cumprod.device

        # t_val comes from scheduler.timesteps, which are scalar tensors (0-dim)
        current_timestep_idx = t_val.to(device=alphas_cumprod_device, dtype=torch.long)

        # Defensive check for bounds.
        # `t_val` from `scheduler.timesteps` *should* be a valid timestep value
        # that can be used as an index for `alphas_cumprod`.
        max_idx = len(self.scheduler.alphas_cumprod) - 1
        if not (0 <= current_timestep_idx.item() <= max_idx):
            self.logger.warning(
                f"Timestep index {current_timestep_idx.item()} from scheduler.timesteps is out of bounds "
                f"for alphas_cumprod (len: {len(self.scheduler.alphas_cumprod)}). This might indicate "
                "an issue with scheduler setup (e.g., loop_timesteps vs num_train_timesteps). Clamping index."
            )
            current_timestep_idx = torch.clamp(current_timestep_idx, 0, max_idx)

        # Indexing happens on the device of alphas_cumprod (e.g., CPU)
        alpha_prod_t = self.scheduler.alphas_cumprod[current_timestep_idx]

        # Move the retrieved alpha_prod_t to the target computation device (self.device)
        alpha_prod_t = alpha_prod_t.to(self.device)
        alpha_prod_t = alpha_prod_t.view(-1, *(1,) * (x_t.ndim - 1))

        sqrt_one_minus_alpha_prod_t = (1.0 - alpha_prod_t).sqrt()
        sqrt_alpha_prod_t = alpha_prod_t.sqrt()

        # Ensure all tensors in this calculation are on self.device
        # x_t and model_output_noise should already be on self.device
        x0_pred = (
            x_t - sqrt_one_minus_alpha_prod_t * model_output_noise
        ) / sqrt_alpha_prod_t
        return x0_pred.clamp(-1, 1)

    def run_algorithm(self, initial_x_t):
        self.logger.info(
            f"Starting DPS algorithm for {self.loop_timesteps} timesteps with guidance scale: {self.guidance_scale}."
        )

        self.scheduler.set_timesteps(self.loop_timesteps)
        x_t = initial_x_t.clone()
        final_x0_reconstruction = None
        start_time = time.time()

        for i, t_val in enumerate(self.scheduler.timesteps):
            x_t_for_grad = x_t.clone().detach().requires_grad_(True)

            with torch.no_grad():
                model_output_noise = self.model(x_t_for_grad, t_val).sample

            x0_pred = self._predict_x0(x_t_for_grad, t_val, model_output_noise.detach())
            x0_in_01 = (x0_pred + 1.0) / 2.0  # For K and metrics

            with torch.no_grad():
                current_psnr = self.psnr_metric(x0_in_01, self.x_true).item()
                current_ssim = self.ssim_metric(x0_in_01, self.x_true).item()
                self.psnr_history.append(current_psnr)
                self.ssim_history.append(current_ssim)

            data_loss = self.loss_fn(self.K_op(x0_in_01), self.y_delta)
            self.loss_history.append(data_loss.item())

            grad_x_t = torch.autograd.grad(data_loss, x_t_for_grad)[0]

            x_t_guided = x_t_for_grad.detach() - self.guidance_scale * grad_x_t

            # Use original noise prediction (from non-guided x_t) with the guided x_t
            x_t = self.scheduler.step(
                model_output_noise.detach(), t_val, x_t_guided
            ).prev_sample
            x_t = x_t.detach()

            current_time_val = (
                t_val.item() if isinstance(t_val, torch.Tensor) else t_val
            )
            self.logger.info(
                f"(DPS - Time {time.time() - start_time:0.2f}s) Step {i+1:03d}/{len(self.scheduler.timesteps)} (t={current_time_val}) | "
                f"Loss: {data_loss.item():.4f} | PSNR: {current_psnr:.2f} | SSIM: {current_ssim:.4f}"
            )

            if i == len(self.scheduler.timesteps) - 1:
                final_x0_reconstruction = x0_in_01.detach()

        self.logger.info("DPS algorithm finished.")
        if final_x0_reconstruction is None and len(self.scheduler.timesteps) > 0:
            self.logger.warning(
                "Final x0 not set from loop. Predicting from last x_t as fallback."
            )
            with torch.no_grad():
                last_t = self.scheduler.timesteps[-1]
                last_noise = self.model(x_t, last_t).sample
                final_x0_reconstruction = self._predict_x0(x_t, last_t, last_noise)
                final_x0_reconstruction = (
                    (final_x0_reconstruction + 1.0) / 2.0
                ).detach()

        return (
            final_x0_reconstruction,
            self.psnr_history,
            self.ssim_history,
            self.loss_history,
        )


# --- DiffPIRSolver Class ---
class DiffPIRSolver:
    def __init__(
        self,
        model,
        scheduler,
        K_op,
        y_delta,
        x_true_for_metrics,
        config,
        device,
        logger,
    ):
        self.model = model
        self.scheduler = scheduler
        self.K_op = K_op
        self.y_delta = y_delta.to(device)  # Ensure y_delta is on the correct device
        self.x_true_for_metrics = x_true_for_metrics.to(device)  # For metrics
        self.config = config
        self.device = device
        self.logger = logger

        self.diffpir_conf = config["diffpir_params"]
        self.loop_timesteps = self.diffpir_conf["loop_timesteps"]
        self.lambda_reg = self.diffpir_conf["lambda_reg"]
        self.sigma_n_sq = self.diffpir_conf["sigma_n_sq"]  # This is sigma_n^2

        self.loss_fn = torch.nn.MSELoss()
        self.psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

        self.psnr_history = []
        self.ssim_history = []
        self.fidelity_loss_history = []

    def run_algorithm(self, initial_x):
        self.logger.info(
            f"Starting DiffPIR algorithm for {self.loop_timesteps} timesteps."
        )
        self.logger.info(
            f"Params: lambda_reg={self.lambda_reg}, sigma_n_sq={self.sigma_n_sq}"
        )

        x = initial_x.clone().to(self.device)  # Current estimate, typically in [-1, 1]

        # Scheduler timesteps are used for the prior (denoising) step
        self.scheduler.set_timesteps(self.loop_timesteps)

        final_x_reconstruction_01 = None  # To store final image in [0,1]
        start_time = time.time()

        for i in range(self.loop_timesteps):  # Iterate for loop_timesteps
            x.requires_grad_(True)

            # --- Data Fidelity Step ---
            # Scale x from [-1, 1] to [0, 1] for the operator K
            x_01_for_K = (x.clamp(-1, 1) + 1.0) / 2.0

            # Fidelity loss: (1 / (2 * sigma_n^2)) * ||K(x_01) - y_delta||^2
            # Using y_delta (noisy measurement) as per most inverse problem formulations
            fidelity_loss = (1.0 / (2.0 * self.sigma_n_sq)) * self.loss_fn(
                self.K_op(x_01_for_K), self.y_delta
            )

            fidelity_grad = torch.autograd.grad(fidelity_loss, x, retain_graph=False)[
                0
            ]  # retain_graph=False is safer

            x = x - self.lambda_reg * fidelity_grad
            x = x.detach()  # Detach after gradient update

            # --- Prior Step (Denoising / Refinement) ---
            # Select the timestep 't' for the current iteration 'i' from the scheduler's pre-defined timesteps
            # Ensure 'i' does not exceed the length of scheduler.timesteps if they are different from loop_timesteps
            current_scheduler_timestep_idx = min(i, len(self.scheduler.timesteps) - 1)
            t = self.scheduler.timesteps[current_scheduler_timestep_idx]

            with torch.no_grad():
                # Model expects x in its native range (e.g., [-1, 1])
                noise_pred = self.model(x, t).sample

            # Apply scheduler step to refine x (output is typically x_{t-1} in model's range)
            x = self.scheduler.step(noise_pred, t, x).prev_sample
            x = x.detach()

            # --- Monitoring ---
            # For metrics, scale current x (which is in model's range, e.g. [-1,1]) to [0,1]
            x_01_for_metrics = (x.clamp(-1, 1) + 1.0) / 2.0

            with torch.no_grad():
                current_psnr = self.psnr_metric(
                    x_01_for_metrics, self.x_true_for_metrics
                ).item()
                current_ssim = self.ssim_metric(
                    x_01_for_metrics, self.x_true_for_metrics
                ).item()
                self.psnr_history.append(current_psnr)
                self.ssim_history.append(current_ssim)
                self.fidelity_loss_history.append(
                    fidelity_loss.item()
                )  # Store fidelity loss of this step

            current_time_val = t.item() if isinstance(t, torch.Tensor) else t
            self.logger.info(
                f"(DiffPIR - Time {time.time() - start_time:0.2f}s) Iter {i+1:03d}/{self.loop_timesteps} (t={current_time_val}) | "
                f"Fid. Loss: {fidelity_loss.item():.4f} | PSNR: {current_psnr:.2f} | SSIM: {current_ssim:.4f}"
            )

            if i == self.loop_timesteps - 1:
                final_x_reconstruction_01 = x_01_for_metrics.detach()

        self.logger.info("DiffPIR algorithm finished.")
        if final_x_reconstruction_01 is None and self.loop_timesteps > 0:
            self.logger.warning("Final x01 not set. Using last x_01_for_metrics.")
            final_x_reconstruction_01 = x_01_for_metrics.detach()  # Fallback

        return (
            final_x_reconstruction_01,
            self.psnr_history,
            self.ssim_history,
            self.fidelity_loss_history,
        )
