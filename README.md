# RD-DGP: Regularized Diffusion-based Deep Generative Prior 🚀

## Overview

RD-DGP is a Python framework designed for solving imaging inverse problems by leveraging the power of pre-trained diffusion models as priors. This project implements and allows experimentation with several cutting-edge algorithms, including:

* **Deep Generative Prior (DGP-style)**: Optimizing in the latent space of a diffusion model.
* **Diffusion Posterior Sampling (DPS)**: Iteratively sampling from an approximate posterior distribution.
* **DiffPIR (Diffusion Model for Plug-and-Play Image Restoration)**: An iterative method combining data consistency with diffusion model denoising.
* **Unconditional Sample Generation**: For generating samples directly from the diffusion model.

The framework emphasizes flexibility and ease of experimentation through a YAML-based configuration system, support for various models (local and Hugging Face Hub), and dynamic handling of RGB or grayscale images.


## ✨ Features

* **Multiple Inversion Algorithms**: Implements DGP, DPS, and DiffPIR.
* **Config-Driven**: Easily manage all experimental parameters via YAML files.
* **Flexible Model Loading**: Supports local model paths and direct loading from Hugging Face Hub (`UNet2DModel`).
* **Image Versatility**: Handles both RGB (3-channel) and grayscale (1-channel) images.
* **Modular Code**: Solver classes for each algorithm and shared utility functions for clean and maintainable code.
* **Comprehensive Logging**: Built-in logging to console and file for each experiment run.
* **Customizable Operators**: Integrates with the `IPPy` library for forward operators (e.g., Deblurring, CT).

## 📂 Project Structure (Example)

```
RD-DGP/
├── configs/                  # Directory for YAML configuration files
│   └── mayo_ct.yaml          # Configuration file for CT-based image reconstruction on Mayo's Image
│   └── mayo_deblur.yaml      # Configuration file for Deblur-based image reconstryction on Mayo's Image
│   └── celeba_deblur.yaml    # Configuration file for Deblur-based image reconstruction on CelebA's Image
├── examples/                 # Directory containing a few test images used in the paper
├── results/                  # Default base directory for saving experiment outputs
├── IPPy/                     # User-provided library for operators 
├── miscellaneous/            # User-provided library for utilities 
│   └── utilities.py          # e.g., device getter
├── algorithms.py             # A list of the functions used for the experiments
├── run_generation.py         # Script for unconditional sample generation
├── main.pt                   # Script for running all the experiments
└── README.md                 # This file
```

## 🛠️ Setup

### Prerequisites

* Python 3.8+
* PyTorch (e.g., 1.10+ or 2.x)
* `diffusers` and `transformers` libraries from Hugging Face
* Other common scientific Python packages (`numpy`, `matplotlib`, etc.)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/devangelista2/RD-DGP RD-DGP
    cd RD-DGP
    ```

2.  **Install dependencies:**
    Run:
    ```bash
    pip install -r requirements.txt
    ```
    to install all the required dependencies.

## ⚙️ Configuration

All experiments are controlled via a YAML configuration files (e.g., `configs/mayo_ct.yaml`). This file contains sections for general settings, model paths, image details, operator parameters, and specific parameters for each algorithm.

**Key Configuration Sections Example:**

```yaml
# General Settings
device: "auto"             # "cuda", "cpu", "auto"
seed: 42
results_base_dir: "./results/MyExperiments/"

# Model Configuration
model_path: "google/ddpm-celebahq-256" # Hugging Face ID or local path like "./model_weights/UNet_Custom/"

# Image Settings (for inverse problems)
image_path: "./data/sample_image.png"
image_channels: 3          # 1 for grayscale, 3 for RGB
image_size: 256            # Target size (height, width), or single int for square

# Operator Settings
operator: "Deblur"         # "Deblur", "CT", etc.
noise_level: 0.05          # Noise added to degrade the clean observation y

# CT Parameters (if operator is "CT")
ct:
  start_angle: 0
  end_angle: 180
  n_angles: 90
  det_size: 256

# Deblurring Parameters (if operator is "Deblur")
deblur:
  kernel_type: "gaussian"  # "gaussian", "motion", "box"
  kernel_size: 21
  # motion_angle: 45 # if kernel_type is "motion"

# --- Algorithm-Specific Parameters ---

# For run_generation.py
generation_script_params:
  experiment_name: "MySampleGeneration"
  batch_size: 4
  num_inference_steps: 50
  eta_ddim: 0.0
  save_format: "grid"    # "grid" or "individual"
  output_filename_prefix: "generated_img"

# For run_dps.py
dps_params:
  # experiment_name is set by setup_paths using config['generation_script_params']['experiment_name'] or a default
  loop_timesteps: 100
  step_size: 0.5         # DPS guidance scale (eta)

# For run_diffpir.py
diffpir_params:
  # experiment_name similar to DPS
  loop_timesteps: 50
  lambda_reg: 0.1
  sigma_n_sq: 0.01       # Assumed variance (sigma_n^2) for fidelity term
  initial_x_fill: "randn" # "zeros" or "randn"

# For run_dgp.py
reconstruction_timesteps: 20 # Timesteps for inner diffusion loop during optimization
generation_timesteps: 50     # Timesteps for final image generation post-optimization
solver:
  # experiment_name similar to DPS
  type: "Adam"           # "Adam", "LBFGS", "Adam_LBFGS"
  adam_params:
    num_iter: 200
    lr: 1.0e-3
    weight_decay: 1.0e-4
  lbfgs_params:
    num_iter: 50
    lr: 0.8
    history_size: 100
    max_iter_linesearch: 20
```

## 🚀 Usage

### Running Individual Experiments

Each algorithm can be run using its dedicated Python script. Ensure your configuration file (e.g., `configs/main_config.yaml`) is correctly set up for the desired experiment.

**Example for running DGP:**
```bash
python main.py --config configs/<config_file_name>.yaml --dgp
```

**Example for running DPS:**
```bash
python main.py --config configs/<config_file_name>.yaml --dps
```

**Example for running DiffPIR:**
```bash
python main.py --config configs/<config_file_name>.yaml --diffpir
```

**Example for running sample generation:**
```bash
python run_generation.py --config configs/<config_file_name>.yaml
```

## 🧪 Implemented Algorithms

* **Deep Generative Prior (DGP-style)**: Optimizes a latent code `z` to minimize a data consistency loss, where the image is generated by `x = G(z)` using a diffusion model `G` as the generator.
* **Diffusion Posterior Sampling (DPS)**: An iterative method that combines the score of a diffusion model (prior) with the gradient of the likelihood (data consistency) to sample from an approximate posterior distribution.
* **DiffPIR (Diffusion Model for Plug-and-Play Image Restoration)**: A plug-and-play restoration method that alternates between a data-fidelity update step (e.g., gradient descent) and a denoising step using a pre-trained diffusion model.
* **Unconditional Generation**: Standard reverse diffusion process (e.g., DDIM sampling) to generate samples from random noise, using a pre-trained `UNet2DModel` and `DDIMScheduler`.

## 📜 License
```
MIT License

Copyright (c) [Current Year] [Your Name or Project Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 🙏 Acknowledgements

* This work builds upon the foundations laid by numerous researchers in diffusion models and inverse problems.
* Utilizes the `diffusers` library by Hugging Face.
* Inspired by [mention key papers, e.g., for DGP, DPS, DiffPIR if you followed specific ones closely, or repositories if applicable].