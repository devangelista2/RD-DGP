import torch
from diffusers import DDIMInverseScheduler


class RandomInitializer:
    def __init__(self, model, scheduler):
        self.model = model
        self.scheduler = scheduler

    def __call__(self, seed=None):
        if seed:
            torch.manual_seed(seed)  # Set seed if required
        return torch.randn(
            (
                1,
                1,
                self.model.config.sample_size,
                self.model.config.sample_size,
            ),
            requires_grad=True,
            device=self.model.device,
        )


class InverseInitializer:
    def __init__(self, model, scheduler):
        self.model = model
        self.scheduler = DDIMInverseScheduler(
            num_train_timesteps=1000, beta_schedule="linear"
        )

    def __call__(self, x_tilde, num_timesteps):
        self.scheduler.set_timesteps(num_inference_steps=num_timesteps)

        # Normalize x_tilde in [-1, 1] range
        x_tilde = 2 * x_tilde - 1

        # --- Inversion loop: go from x_0 to x_T using reversed DDIM
        x = x_tilde.clone()
        with torch.no_grad():
            for i, t in enumerate(reversed(self.scheduler.timesteps)):
                # predict noise ε from x0 and x_{t-1}
                model_output = self.model(x, t).sample  # shape: (1, 3, 32, 32)

                # Reverse DDIM step: find x_t that would have led to x at time t-1
                # So simulate backward DDIM step (forward in noise space)
                # Scheduler step: simulate forward process
                x = self.scheduler.step(
                    model_output=model_output, timestep=t, sample=x, return_dict=True
                ).prev_sample
        # Activate gradient tracking for x
        x.requires_grad_(True)
        return x
