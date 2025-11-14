#!/bin/bash
# nnMIL Classification Complete Workflow
# Usage: ./run_classification.sh [DATASET_DIR] [MODEL_TYPE] [CUDA_DEVICE]

set -e

DATASET_DIR=${1:-"/mnt/radonc-Li02_vol2/private/luoxd96/MIL/github/nnMIL_raw_data/Task011_BCCC_5Class"}
MODEL_TYPE=${2:-"simple_mil"}
CUDA_DEVICE=${3:-"3"}

export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE

# Get project root (parent of nnMIL directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# Set Python path
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

echo "=========================================="
echo "nnMIL Classification Workflow"
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

echo ""
echo "=========================================="
echo "✅ All done!"
echo "=========================================="

