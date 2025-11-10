#!/bin/bash
# nnMIL Survival Analysis Complete Workflow
# Usage: ./run_survival_workflow.sh [DATASET_DIR] [MODEL_TYPE] [CUDA_DEVICE]

set -e  # Exit on error

# Default values
DATASET_DIR=${1:-"/mnt/radonc-Li02_vol2/private/luoxd96/MIL/nnMIL_raw_data/Task010_TCGA-BRCA"}
MODEL_TYPE=${2:-"simple_mil"}
CUDA_DEVICE=${3:-"3"}

# Set CUDA device
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE

# Project root
PROJECT_ROOT="/mnt/radonc-Li02_vol2/private/luoxd96/MIL"
cd $PROJECT_ROOT

# Set Python path
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH

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
CUDA_VISIBLE_DEVICES=3 python nnMIL/run/nnMIL_run_training.py $DATASET_DIR $MODEL_TYPE all
echo "✅ Training complete"
echo ""

# Step 3: Testing
echo "Step 3/3: Testing..."
PLAN_FILE="$DATASET_DIR/dataset_plan.json"
for FOLD in 0 1 2 3 4; do
    CHECKPOINT="nnMIL_results/$(basename $DATASET_DIR)/$MODEL_TYPE/5fold/fold_$FOLD/best_${MODEL_TYPE}.pth"
    OUTPUT_DIR="nnMIL_results/$(basename $DATASET_DIR)/$MODEL_TYPE/5fold/fold_$FOLD/predictions"
    
    if [ -f "$CHECKPOINT" ]; then
        echo "  Fold $FOLD: Predicting..."
        python nnMIL/run/nnMIL_predict.py --plan_path $PLAN_FILE --checkpoint_path $CHECKPOINT --output_dir $OUTPUT_DIR --fold $FOLD
        echo "  ✅ Fold $FOLD complete"
    fi
done
echo "✅ Testing complete"
echo ""
echo "=========================================="
echo "✅ All done!"
echo "=========================================="
