#!/bin/bash

set -euo pipefail

# --- Fixed Configuration ---
INPUT_IMAGE="../data/Mayo/test/C081/35.png"
WEIGHTS="./outputs/model_weights/monai_finetune/finetuned_best.pth"
CONFIG="./configs/default_monai.yaml"
DEVICE="cuda"
SEED=42
NOISE_SIGMA=0.005
OPT_STEPS=100
NUM_ANGLES=90

# DGP base params
DDIM_STEPS=30
LR=1e-3

# --- Hyperparameter Ranges to Explore ---
LAMBDA_REGS=(5e-2 1e-1 5e-1 1e0)
LAMBDA_TVS=3e-4 #(0 1e-4 3e-4 1e-3 3e-3)

echo "Starting 90-angle grid search (lambda_reg x lambda_tv)..."
echo "Input: $INPUT_IMAGE"
echo "Angles: $NUM_ANGLES | DDIM: $DDIM_STEPS | Opt: $OPT_STEPS"
echo "------------------------------------------------"

for reg in "${LAMBDA_REGS[@]}"; do
  for tv in "${LAMBDA_TVS[@]}"; do
    EXP_NAME="angles${NUM_ANGLES}_reg${reg}_tv${tv}"
    OUTPUT_BASE="outputs/grid_90/${EXP_NAME}"

    echo "Running Experiment: $EXP_NAME"

    python ./src/run_dgp.py \
      --input "$INPUT_IMAGE" \
      --weights "$WEIGHTS" \
      --config "$CONFIG" \
      --num_angles "$NUM_ANGLES" \
      --device "$DEVICE" \
      --seed "$SEED" \
      --noise_sigma "$NOISE_SIGMA" \
      --ddim_steps "$DDIM_STEPS" \
      --opt_steps "$OPT_STEPS" \
      --lr "$LR" \
      --lambda_reg "$reg" \
      --lambda_tv "$tv" \
      --output_dir "$OUTPUT_BASE"

    echo "Finished $EXP_NAME"
    echo "------------------------------------------------"
  done
done

echo "Grid search complete. Check 'outputs/grid_90/' for results."
