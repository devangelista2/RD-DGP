import os

import torch
import yaml

from IPPy import utilities as IPutils
from miscellaneous import model_setup, solvers, utilities


def initialize(model, config, logger, device):
    # --- Load data ---
    img_size_to_use = config.get("image_size", model.config.sample_size)
    if isinstance(img_size_to_use, int):
        img_size_to_use = (img_size_to_use, img_size_to_use)
    logger.info(f"Target image size set to: {img_size_to_use}")

    x_true = utilities.load_and_preprocess_image(
        config, img_size_to_use, device, logger
    )
    img_shape_hw = x_true.shape[-2:]
    num_channels = config["image_channels"]

    # --- Define operator & Generate test problem ---
    K = model_setup.get_operator(config, img_shape_hw, num_channels, device, logger)

    logger.info("Generating test problem (corrupted data y_delta)...")
    y = K(x_true)
    y_delta = y + IPutils.gaussian_noise(y, config["noise_level"])
    logger.info(f"Test problem generated. y_delta shape: {y_delta.shape}")

    return K, x_true, y_delta


def run_dgp(model, scheduler, config, logger, device):
    # --- Generate data to reconstruct ---
    K, x_true, y_delta = initialize(model, config, logger, device)

    # --- Run inversion process ---
    z = torch.randn(x_true.shape, requires_grad=True, device=device)
    logger.info(f"Initialized latent variable z with shape: {z.shape}")

    # --- Optimization using DGPSolver ---
    solver = solvers.DGPSolver(
        model, scheduler, K, x_true, y_delta, z, config, device, logger
    )
    z, psnr_vec, ssim_vec, loss_vec = solver.run_optimization()

    # --- Create final generation from optimized z ---
    logger.info("Generating final image from optimized z...")
    scheduler.set_timesteps(config["generation_timesteps"])

    x_final_gen = z
    for t in scheduler.timesteps:
        with torch.no_grad():
            noise_pred = model(x_final_gen, t).sample
        x_final_gen = scheduler.step(noise_pred, t, x_final_gen).prev_sample

    x_gen_final_normalized = (x_final_gen + 1.0) / 2.0
    x_gen_final_normalized = torch.clamp(x_gen_final_normalized, 0.0, 1.0)
    logger.info("Final image generated.")

    # --- Saving ---
    saving_path = utilities.setup_paths(config)
    logger.info(f"Saving final results and artifacts to: {saving_path}")
    torch.save(torch.tensor(psnr_vec), os.path.join(saving_path, "psnr_history.pth"))
    torch.save(torch.tensor(ssim_vec), os.path.join(saving_path, "ssim_history.pth"))
    logger.info("Metrics history saved.")

    with open(os.path.join(saving_path, "dgp_config_used.yaml"), "w") as f:
        yaml.dump(config, f)
    logger.info("Configuration used for this run saved.")

    final_ssim = solver.ssim_metric(x_gen_final_normalized, x_true).item()
    utilities.plot_and_save_images(
        x_true,
        y_delta,
        x_gen_final_normalized,
        final_ssim,
        saving_path,
        config,
        logger,
    )
    return x_gen_final_normalized, psnr_vec, ssim_vec


def run_dps(model, scheduler, config, logger, device):
    # --- Generate data to reconstruct ---
    K, x_true, y_delta = initialize(model, config, logger, device)

    initial_x_t = torch.randn(x_true.shape, device=device)
    logger.info(f"Initial x_t (noise) shape: {initial_x_t.shape}")

    # --- Initialize and Run DPSSolver ---
    solver = solvers.DPSSolver(
        model, scheduler, K, y_delta, x_true, config, device, logger
    )
    x_reconstructed, psnr_vec, ssim_vec, loss_vec = solver.run_algorithm(initial_x_t)

    # --- Saving Results ---
    saving_path = utilities.setup_paths(config)
    logger.info(f"Saving final results to: {saving_path}")
    torch.save(
        torch.tensor(psnr_vec), os.path.join(saving_path, "dps_psnr_history.pth")
    )
    torch.save(
        torch.tensor(ssim_vec), os.path.join(saving_path, "dps_ssim_history.pth")
    )
    torch.save(
        torch.tensor(loss_vec), os.path.join(saving_path, "dps_loss_history.pth")
    )
    logger.info("Metrics history saved.")

    with open(os.path.join(saving_path, "dps_config_used.yaml"), "w") as f:
        yaml.dump(config, f)
    logger.info("Configuration saved.")

    final_ssim_val = solver.ssim_metric(x_reconstructed, x_true).item()
    utilities.plot_and_save_images(
        x_true,
        y_delta,
        x_reconstructed,
        final_ssim_val,
        saving_path,
        config,
        logger,
    )

    return x_reconstructed, psnr_vec, ssim_vec


def run_diffpir(model, scheduler, config, logger, device):
    # --- Generate data to reconstruct ---
    K, x_true, y_delta = initialize(model, config, logger, device)

    initial_x_t = torch.randn(x_true.shape, device=device)
    logger.info(f"Initial x_t (noise) shape: {initial_x_t.shape}")

    # --- Initialize and Run DiffPIRSolver ---
    solver = solvers.DiffPIRSolver(
        model, scheduler, K, y_delta, x_true, config, device, logger
    )
    x_reconstructed, psnr_vec, ssim_vec, loss_vec = solver.run_algorithm(initial_x_t)

    # --- Saving Results ---
    saving_path = utilities.setup_paths(config)
    logger.info(f"Saving final results to: {saving_path}")
    torch.save(
        torch.tensor(psnr_vec), os.path.join(saving_path, "dpir_psnr_history.pth")
    )
    torch.save(
        torch.tensor(ssim_vec), os.path.join(saving_path, "dpir_ssim_history.pth")
    )
    torch.save(
        torch.tensor(loss_vec), os.path.join(saving_path, "dpir_loss_history.pth")
    )
    logger.info("Metrics history saved.")

    with open(os.path.join(saving_path, "diffpir_config_used.yaml"), "w") as f:
        yaml.dump(config, f)
    logger.info("Configuration saved.")

    final_ssim_val = solver.ssim_metric(x_reconstructed, x_true).item()
    utilities.plot_and_save_images(
        x_true,
        y_delta,
        x_reconstructed,
        final_ssim_val,
        saving_path,
        config,
        logger,
    )

    return x_reconstructed, psnr_vec, ssim_vec
