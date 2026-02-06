#!/bin/bash

# --- Fixed Configuration ---
INPUT_IMAGE="../../data/Mayo/test/C081/35.png"
WEIGHTS="./outputs/model_weights/monai_finetune/finetuned_best.pth"
CONFIG="./configs/default_monai.yaml"
DEVICE="cuda"
SEED=42

# Fixed DGP settings (things you likely don't need to grid search)
NOISE_SIGMA=0.005
DDIM_STEPS=30
OPT_STEPS=100

# Operator settings
NUM_ANGLES=60

# --- Hyperparameter Ranges to Explore ---
# Arrays of values to loop over. Adjust these lists based on your intuition.
LEARNING_RATES=(0.0005 0.001)
LAMBDA_REGS=(1e-5 1e-4)
LAMBDA_TVS=(1e-2 1e-1 1e0)

# --- The Loop ---
echo "Starting Grid Search..."
echo "Input: $INPUT_IMAGE"
echo "Weights: $WEIGHTS"
echo "------------------------------------------------"

for lr in "${LEARNING_RATES[@]}"; do
  for reg in "${LAMBDA_REGS[@]}"; do
    for tv in "${LAMBDA_TVS[@]}"; do
      
      # create a descriptive directory name for this specific run
      EXP_NAME="lr${lr}_reg${reg}_tv${tv}"
      OUTPUT_BASE="outputs/grid_search/${EXP_NAME}"
      
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
        --lr "$lr" \
        --lambda_reg "$reg" \
        --lambda_tv "$tv" \
        --output_dir "$OUTPUT_BASE"

      echo "Finished $EXP_NAME"
      echo "------------------------------------------------"
      
    done
  done
done

echo "Grid search complete. Check 'outputs/grid_search/' for results."
