import os

import torch
import torchvision.transforms as T
from diffusers import DDIMInverseScheduler, DDIMScheduler, UNet2DModel
from PIL import Image

# --- Dummy UNet (replace with your pretrained model)
model = UNet2DModel.from_pretrained(
    os.path.join("./model_weights/UNet_256_Large_L/", "unet")
)
model.to("cuda")
model.eval()

# --- Load DDIMInverseScheduler
scheduler = DDIMInverseScheduler(num_train_timesteps=1000, beta_schedule="linear")
scheduler.set_timesteps(num_inference_steps=10)

# Use matching DDIMScheduler for generation
gen_scheduler = DDIMScheduler.from_config(scheduler.config)
gen_scheduler.set_timesteps(num_inference_steps=10)

# --- Load image x_0 (normalized to [-1, 1])
image = Image.open("../data/Mayo/test/C081/0.png").convert("L").resize((256, 256))
transform = T.Compose(
    [
        T.ToTensor(),
        T.Normalize([0.5], [0.5]),  # scale to [-1, 1]
    ]
)
x0 = transform(image).unsqueeze(0).to("cuda")

# --- Invert DDIM denoising: go from x_0 to x_T
x = x0.clone()

with torch.no_grad():
    for i, t in enumerate(scheduler.timesteps):
        # Predict noise
        model_output = model(x, t).sample

        # Inversion step
        x = scheduler.step(model_output=model_output, timestep=t, sample=x).prev_sample

x_T = x
print(f"Recovered x_T shape: {x_T.shape}")

x = x_T.clone()
with torch.no_grad():
    for t in gen_scheduler.timesteps:
        model_output = model(x, t).sample
        x = gen_scheduler.step(
            model_output=model_output, timestep=t, sample=x
        ).prev_sample

x0_recon = x

# Images
import matplotlib.pyplot as plt

plt.subplot(1, 2, 1)
plt.imshow(x0.detach().cpu().numpy()[0, 0], cmap="gray")
plt.title("True")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(x0_recon.detach().cpu().numpy()[0, 0], cmap="gray")
plt.title(f"Reconstructed")
plt.axis("off")
plt.tight_layout()
plt.show()
