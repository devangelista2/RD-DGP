#!/bin/bash
set -euo pipefail
export KMP_DUPLICATE_LIB_OK=TRUE

# --- Fixed Configuration ---
INPUT_IMAGE="../data/Mayo/test/C081/35.png" # 35 45 79
WEIGHTS="./outputs/model_weights/monai_finetune/finetuned_best.pth"
CONFIG="./configs/default_monai.yaml"
DEVICE="cuda"
SEED=42
NOISE_SIGMA=0.005
OPT_STEPS=100

# DGP baseline params (from existing run_dgp.sh)
BASE_DDIM_STEPS=30
BASE_LR=1e-3
BASE_LAMBDA_REG=1e-7
BASE_LAMBDA_TV=1e-2

echo "=============================================="
echo "CT Experiments - DGP + Comparison"
echo "Input: $INPUT_IMAGE"
echo "Weights: $WEIGHTS"
echo "Config: $CONFIG"
echo "Device: $DEVICE | Seed: $SEED"
echo "Noise Sigma: $NOISE_SIGMA | Opt Steps: $OPT_STEPS"
echo "=============================================="

run_dgp_fixed() {
  local angles=$1
  local output_dir=$2

  echo "--- DGP (fixed DDIM) | angles=$angles | output=$output_dir ---"
  python ./src/run_dgp.py \
    --input "$INPUT_IMAGE" \
    --weights "$WEIGHTS" \
    --config "$CONFIG" \
    --num_angles "$angles" \
    --device "$DEVICE" \
    --seed "$SEED" \
    --noise_sigma "$NOISE_SIGMA" \
    --ddim_steps "$BASE_DDIM_STEPS" \
    --opt_steps "$OPT_STEPS" \
    --lr "$BASE_LR" \
    --lambda_reg "$BASE_LAMBDA_REG" \
    --lambda_tv "$BASE_LAMBDA_TV" \
    --output_dir "$output_dir"
}

run_compare() {
  local angles=$1
  local dps_zeta=$2
  local diffpir_lambda=$3
  local diffpir_rho=$4

  echo "--- Compare (DPS/DiffPIR/DDRM) | angles=$angles ---"
  python ./src/run_compare.py \
    --input "$INPUT_IMAGE" \
    --weights "$WEIGHTS" \
    --config "$CONFIG" \
    --num_angles "$angles" \
    --noise_sigma "$NOISE_SIGMA" \
    --dps_zeta "$dps_zeta" \
    --diffpir_lambda "$diffpir_lambda" \
    --diffpir_rho "$diffpir_rho" \
    --device "$DEVICE" \
    --seed "$SEED"
}

echo ""
echo "=== DGP Experiments ==="
run_dgp_fixed 30 "outputs/dgp_ct_fixed_30"
run_dgp_fixed 45 "outputs/dgp_ct_fixed_45"
run_dgp_fixed 60 "outputs/dgp_ct_fixed_60"

echo ""
echo "=== Comparison Experiments (DPS/DiffPIR/DDRM) ==="
# Params mirrored from experiments/run_compare.sh
# run_compare 30 0.5 5.0 1.0
# run_compare 45 0.5 5.0 1.0
# run_compare 60 0.5 5.0 1.0

echo ""
echo "All experiments complete."
