#!/bin/bash
# nnMIL Survival Analysis Complete Workflow
# Usage: ./run_survival.sh [DATASET_DIR] [MODEL_TYPE] [CUDA_DEVICE]

set -e

DATASET_DIR=${1:-"/mnt/radonc-Li02_vol2/private/luoxd96/MIL/github/nnMIL_raw_data/Task016_PLCO_Lung"}
MODEL_TYPE=${2:-"simple_mil"}
CUDA_DEVICE=${3:-"2"}

export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE

# Get project root (parent of nnMIL directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# Set Python path
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

echo "=========================================="
echo "nnMIL Survival Analysis Workflow"
echo "=========================================="
echo "Dataset: $DATASET_DIR"
echo "Model: $MODEL_TYPE"
echo "CUDA Device: $CUDA_DEVICE"
echo "=========================================="
echo ""

# Step 1: Planning
echo "Step 1/3: Planning..."
python nnMIL/run/nnMIL_plan_experiment.py -d $DATASET_DIR --seed 42
echo "✅ Planning complete"
echo ""

# Step 2: Training
echo "Step 2/3: Training..."
python nnMIL/run/nnMIL_run_training.py $DATASET_DIR $MODEL_TYPE all
echo "✅ Training complete"
echo ""



# # Step 3: Testing
# echo "Step 3/3: Testing..."
CUDA_VISIBLE_DEVICES=3 python run/nnMIL_predict.py \
--plan_path /mnt/radonc-Li02_vol2/private/luoxd96/MIL/github/nnMIL_raw_data/Task016_PLCO_Lung/dataset_plan.json \
--checkpoint_path /mnt/radonc-Li02_vol2/private/luoxd96/MIL/github/nnMIL_results/Task016_PLCO_Lung/simple_mil/official_split/best_simple_mil.pth \
--input_dir /scratch/luoxd96/omnipath/features/virchow2/nlst/h5_files \
--output_dir /mnt/radonc-Li02_vol2/private/luoxd96/MIL/github/nnMIL_results/Task016_PLCO_Lung/simple_mil/official_split/nlst_test_best &


CUDA_VISIBLE_DEVICES=2 python run/nnMIL_predict.py \
--plan_path /mnt/radonc-Li02_vol2/private/luoxd96/MIL/github/nnMIL_raw_data/Task016_PLCO_Lung/dataset_plan.json \
--checkpoint_path /mnt/radonc-Li02_vol2/private/luoxd96/MIL/github/nnMIL_results/Task016_PLCO_Lung/simple_mil/official_split/latest_simple_mil.pth \
--input_dir /scratch/luoxd96/omnipath/features/virchow2/nlst/h5_files \
--output_dir /mnt/radonc-Li02_vol2/private/luoxd96/MIL/github/nnMIL_results/Task016_PLCO_Lung/simple_mil/official_split/nlst_test_latest