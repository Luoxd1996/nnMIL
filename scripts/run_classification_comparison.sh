#!/bin/bash
# nnMIL Classification Model Comparison Script
# Trains multiple models on the same dataset and compares results
# Usage: ./run_classification_comparison.sh [DATASET_DIR] [MODELS] [CUDA_DEVICE]

set -e

DATASET_DIR=${1:-"examples/Dataset001_ebrains_batch1"}
MODELS=${2:-"ds_mil"}  # Comma-separated list of models
CUDA_DEVICE=${3:-"3"}

export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE

# Get project root (parent of nnMIL directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# Set Python path
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

echo "=========================================="
echo "nnMIL Classification Model Comparison"
echo "=========================================="
echo "Dataset: $DATASET_DIR"
echo "Models: $MODELS"
echo "CUDA Device: $CUDA_DEVICE"
echo "=========================================="
echo ""

# Convert comma-separated models to array
IFS=',' read -ra MODEL_ARRAY <<< "$MODELS"

# Step 1: Planning (only once)
echo "Step 1: Planning..."
if [ ! -f "$DATASET_DIR/dataset_plan.json" ]; then
    python nnMIL/run/nnMIL_plan_experiment.py -d $DATASET_DIR --seed 42
    echo "✅ Planning complete"
else
    echo "✅ Plan file already exists, skipping planning"
fi
echo ""

# Step 2: Training each model
echo "Step 2: Training models..."
for MODEL in "${MODEL_ARRAY[@]}"; do
    MODEL=$(echo "$MODEL" | xargs)  # Trim whitespace
    echo "----------------------------------------"
    echo "Training: $MODEL"
    echo "----------------------------------------"
    
    # Determine fold argument based on evaluation setting
    EVAL_SETTING=$(python -c "import json; plan=json.load(open('$DATASET_DIR/dataset_plan.json')); print(plan.get('evaluation_setting', plan.get('dataset_info', {}).get('evaluation_setting', 'official_split')))")
    
    if [ "$EVAL_SETTING" = "5fold" ] || [ "$EVAL_SETTING" = "5_fold" ]; then
        FOLD_ARG="all"
        echo "Running 5-fold cross-validation"
    else
        FOLD_ARG="None"
        echo "Running official split"
    fi
    
    # For ebrains: simple_mil and ab_mil use batch_size=32
    # Note: batch_size is typically set in the plan file, but can be overridden
    python nnMIL/run/nnMIL_run_training.py $DATASET_DIR $MODEL $FOLD_ARG
    
    if [ $? -eq 0 ]; then
        echo "✅ $MODEL training complete"
    else
        echo "❌ $MODEL training failed"
    fi
    echo ""
done

# Step 3: Testing each model
echo "Step 3: Testing models..."
PLAN_FILE="$DATASET_DIR/dataset_plan.json"
EVAL_SETTING=$(python -c "import json; plan=json.load(open('$PLAN_FILE')); print(plan.get('evaluation_setting', plan.get('dataset_info', {}).get('evaluation_setting', 'official_split')))")

if [ "$EVAL_SETTING" = "5fold" ] || [ "$EVAL_SETTING" = "5_fold" ]; then
    # 5-fold CV: test each fold
    for MODEL in "${MODEL_ARRAY[@]}"; do
        MODEL=$(echo "$MODEL" | xargs)
        echo "Testing: $MODEL"
        for FOLD in 0 1 2 3 4; do
            CHECKPOINT="nnMIL_results/$(basename $DATASET_DIR)/$MODEL/fold_${FOLD}/best_${MODEL}.pth"
            OUTPUT_DIR="nnMIL_results/$(basename $DATASET_DIR)/$MODEL/fold_${FOLD}/predictions"
            
            if [ -f "$CHECKPOINT" ]; then
                python nnMIL/run/nnMIL_predict.py \
                    --plan_path $PLAN_FILE \
                    --checkpoint_path $CHECKPOINT \
                    --output_dir $OUTPUT_DIR \
                    --model_type $MODEL \
                    --fold $FOLD
            fi
        done
        echo "✅ $MODEL testing complete"
    done
else
    # Official split: single test
    for MODEL in "${MODEL_ARRAY[@]}"; do
        MODEL=$(echo "$MODEL" | xargs)
        echo "Testing: $MODEL"
        CHECKPOINT="nnMIL_results/$(basename $DATASET_DIR)/$MODEL/official_split/best_${MODEL}.pth"
        OUTPUT_DIR="nnMIL_results/$(basename $DATASET_DIR)/$MODEL/official_split/predictions"
        
        if [ -f "$CHECKPOINT" ]; then
            python nnMIL/run/nnMIL_predict.py \
                --plan_path $PLAN_FILE \
                --checkpoint_path $CHECKPOINT \
                --output_dir $OUTPUT_DIR \
                --model_type $MODEL
            echo "✅ $MODEL testing complete"
        else
            echo "⚠️  Checkpoint not found: $CHECKPOINT"
        fi
    done
fi

echo ""
echo "=========================================="
echo "✅ All models trained and tested!"
echo "=========================================="
echo ""
echo "Results are in: nnMIL_results/$(basename $DATASET_DIR)/"
echo ""
echo "To compare results, run:"
echo "  python compare_classification_results.py nnMIL_results/$(basename $DATASET_DIR) $MODELS"

