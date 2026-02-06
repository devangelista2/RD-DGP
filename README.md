# RD-DGP: Regularized Diffusion-based Deep Generative Prior

RD-DGP is a research codebase for CT reconstruction and inverse problems using diffusion-model priors. It includes training/finetuning a MONAI DiffusionModelUNet on CT slices and running several reconstruction pipelines (DGP, DPS, DiffPIR, DDRM) with a DDIM scheduler.

**Key scripts**
- `src/train.py`: Train a MONAI diffusion UNet on a folder of clean PNG slices.
- `src/finetune.py`: Same pipeline as training, intended for finetuning from a checkpoint.
- `src/run_dgp.py`: DGP reconstruction on simulated CT measurements.
- `src/run_compare.py`: Compare DPS, DiffPIR, and DDRM on the same CT measurement.
- `src/sample_grid.py`: Generate a grid of samples from a trained UNet.

**Directory layout**
- `configs/default_monai.yaml`: UNet architecture settings. Must match training settings.
- `src/materials/`: Pipelines and utilities.
- `outputs/`: Experiment outputs (created at runtime).

## Requirements
- Python 3.10+
- PyTorch (CUDA optional but recommended for speed)

Install dependencies:
```bash
pip install -r requirements.txt
```

## Data format
Training expects a directory of PNG files. The loader recursively finds `*.png` under `--data_path` and treats each as a grayscale image. Images are scaled to `[0, 1]` by MONAI transforms, then mapped to `[-1, 1]` before diffusion training.

Samples used in the paper associated with this repository are available at the following link: **TODO**.

## Configuration
`configs/default_monai.yaml` defines the UNet architecture (channels, attention levels, layers). Keep this file consistent with `src/train.py` and `src/finetune.py` if you change the model.

## Usage
### Train a diffusion UNet
```bash
python src/train.py --data_path <path_to_png_folder> --output_dir outputs/train
```

### Finetune a diffusion UNet
```bash
python src/finetune.py --data_path <path_to_png_folder> --output_dir outputs/finetune
```

### Run DGP reconstruction
```bash
python src/run_dgp.py --input <clean_png> --weights <path_to_unet.pth> --config configs/default_monai.yaml
```

### Compare DPS, DiffPIR, DDRM
```bash
python src/run_compare.py --input <clean_png> --weights <path_to_unet.pth> --config configs/default_monai.yaml
```

### Sample generation grid
```bash
python src/sample_grid.py --weights <path_to_unet.pth> --config configs/default_monai.yaml
```

## Pre-trained model
The weights of the pre-trained model used to run the experiments on the paper associated with this repository, can be downloaded from the following link: TODO.

By default, the bash files in the `experiments/` folder expects the downloaded folder of the weights to be placed in the path: `./outputs/model_weights/`.