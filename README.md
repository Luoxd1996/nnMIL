# nnMIL: Multiple Instance Learning Framework

A modular MIL (Multiple Instance Learning) framework following nnUNet design principles for medical imaging and histopathology analysis.

## Features

- **Unified Interface**: Single command-line interface for all tasks (classification, regression, survival)
- **Configuration-Driven**: All hyperparameters loaded from plan files
- **Automatic Path Resolution**: Automatically finds datasets, plans, and checkpoints
- **Modular Design**: Follows nnUNet architecture principles
- **Support for Multiple Tasks**: Classification, Regression, and Survival Analysis
- **5-Fold Cross-Validation**: Built-in support for k-fold CV with patient-level stratification

## Directory Structure

```
nnMIL/
├── __init__.py
├── network_architecture/   # MIL model implementations
│   ├── model_factory.py
│   └── models/
│       ├── simple_mil.py
│       ├── ab_mil.py
│       └── ...
├── data/                   # Dataset handling
│   ├── dataset.py          # UnifiedMILDataset
│   └── archive/            # Archived old dataset files
├── training/               # Training modules
│   ├── trainers/           # Unified trainers
│   │   ├── base_trainer.py
│   │   ├── classification_trainer.py
│   │   ├── regression_trainer.py
│   │   ├── survival_trainer.py
│   │   └── survival_porpoise_trainer.py
│   ├── losses/             # Loss functions
│   ├── samplers/           # Batch samplers
│   └── callbacks/          # Callbacks (early stopping, etc.)
├── inference/              # Inference engine
│   ├── inference_engine.py
│   └── predictors/         # Task-specific predictors
├── preprocessing/          # Data preprocessing and experiment planning
│   ├── experiment_planner.py
│   └── generate_dataset_json.py
├── run/                    # Command-line entry points
│   ├── nnMIL_plan_experiment.py
│   ├── nnMIL_run_training.py  # Unified training entry ⭐
│   └── nnMIL_predict.py
├── scripts/                # Complete workflow scripts
│   ├── run_classification.sh
│   └── run_survival.sh
└── utilities/               # Utility functions
    ├── plan_loader.py
    └── utils.py
```

## Complete Workflow

The nnMIL workflow consists of three main steps:

1. **Planning**: Analyze dataset and generate training configuration
2. **Training**: Train models with the generated configuration
3. **Testing**: Evaluate trained models on test sets

## Quick Start

### 1. Dataset Preparation

Create a dataset directory with `dataset.json` and `dataset.csv`:

```
examples/Dataset001_classification/
├── dataset.json          # Dataset metadata and configuration
└── dataset.csv          # Patient/slide information with labels
```

**Example `dataset.json` (Classification):**
```json
{
    "dataset_id": "Dataset001_classification",
    "dataset_name": "Example Classification Dataset",
    "task_type": "classification",
    "labels": [0, 1],
    "feature_dir": "/path/to/features",
    "metric": "auc",
    "evaluation_setting": "5fold"
}
```

**Example `dataset.json` (Survival):**
```json
{
    "dataset_id": "Dataset002_survival",
    "dataset_name": "Example Survival Dataset",
    "task_type": "survival",
    "feature_dir": "/path/to/features",
    "metric": "c_index",
    "evaluation_setting": "5fold"
}
```

**Example `dataset.csv` (Classification):**
```csv
slide_id,patient_id,split,label
slide_001,patient_001,train,0
slide_002,patient_002,train,1
...
```

**Example `dataset.csv` (Survival):**
```csv
slide_id,patient_id,split,event,time
slide_001,patient_001,train,1,45.2
slide_002,patient_002,train,0,60.5
...
```

### 2. Planning (Step 1/3)

The planner analyzes your dataset and generates a training configuration:

```bash
python nnMIL/run/nnMIL_plan_experiment.py -d examples/Dataset001_classification
```

This will:
- Analyze feature files to determine patch counts and feature dimensions
- Calculate recommended `max_seq_length` (median * 0.5)
- Create data splits (patient-level, stratified)
- Generate `dataset_plan.json` with all training configurations

**Output:** `examples/Dataset001_classification/dataset_plan.json`

### 3. Training (Step 2/3)

Train models using the unified training interface:

```bash
# For official split (single fold)
python nnMIL/run/nnMIL_run_training.py examples/Dataset001_classification simple_mil None

# For 5-fold cross-validation
python nnMIL/run/nnMIL_run_training.py examples/Dataset001_classification simple_mil all
```

**Arguments:**
- `dataset_dir`: Path to dataset directory (contains `dataset_plan.json`)
- `model_type`: Model architecture (`simple_mil`, `ab_mil`, `ds_mil`, etc.)
- `fold`: Fold number (`0`, `1`, `2`, `3`, `4`), `all` for 5-fold CV, or `None` for official split

**Output:** `nnMIL_results/Dataset001_classification/simple_mil/fold_X/`

### 4. Testing (Step 3/3)

Evaluate trained models:

```bash
# For official split
python nnMIL/run/nnMIL_predict.py \
    --plan_path examples/Dataset001_classification/dataset_plan.json \
    --checkpoint_path nnMIL_results/Dataset001_classification/simple_mil/official_split/best_simple_mil.pth \
    --output_dir nnMIL_results/Dataset001_classification/simple_mil/official_split/predictions

# For 5-fold CV, iterate through folds
for FOLD in 0 1 2 3 4; do
    python nnMIL/run/nnMIL_predict.py \
        --plan_path examples/Dataset001_classification/dataset_plan.json \
        --checkpoint_path nnMIL_results/Dataset001_classification/simple_mil/fold_${FOLD}/best_simple_mil.pth \
        --output_dir nnMIL_results/Dataset001_classification/simple_mil/fold_${FOLD}/predictions \
        --fold ${FOLD}
done
```

## Complete Workflow Scripts

### Classification Workflow

Use the provided script:

```bash
bash nnMIL/scripts/run_classification.sh examples/Dataset001_classification simple_mil 0
```

The script is located at: `nnMIL/scripts/run_classification.sh`

### Survival Analysis Workflow

Use the provided script:

```bash
bash nnMIL/scripts/run_survival.sh examples/Dataset002_survival simple_mil 0
```

The script is located at: `nnMIL/scripts/run_survival.sh`

## Tutorial

This tutorial will walk you through a complete workflow for both classification and survival analysis tasks.

### Tutorial 1: Classification Task

#### Step 1: Prepare Your Dataset

Create a dataset directory structure:

```bash
mkdir -p examples/MyClassificationDataset
```

**Create `examples/MyClassificationDataset/dataset.json`:**
```json
{
    "dataset_id": "MyClassificationDataset",
    "dataset_name": "My Binary Classification Dataset",
    "task_type": "classification",
    "labels": [0, 1],
    "feature_dir": "/path/to/your/features",
    "metric": "auc",
    "evaluation_setting": "5fold"
}
```

**Create `examples/MyClassificationDataset/dataset.csv`:**
```csv
slide_id,patient_id,split,label
slide_001,patient_001,train,0
slide_002,patient_002,train,1
slide_003,patient_003,train,0
slide_004,patient_004,val,1
slide_005,patient_005,val,0
slide_006,patient_006,test,1
```

**Important Notes:**
- `feature_dir`: Path to directory containing H5 feature files (e.g., `slide_001.h5`, `slide_002.h5`)
- For 5-fold CV, you can use any split column or leave it empty; the planner will create folds
- Feature files should be named as `{slide_id}.h5` and contain `features` array with shape `(n_patches, feature_dim)`

#### Step 2: Run Planning

Generate the training plan:

```bash
python nnMIL/run/nnMIL_plan_experiment.py -d examples/MyClassificationDataset --seed 42
```

**What happens:**
1. Scans all H5 files in `feature_dir`
2. Calculates patch count statistics (min, max, mean, median, percentiles)
3. Recommends `max_seq_length` (typically median * 0.5)
4. Creates patient-level, stratified data splits (5 folds or official split)
5. Generates training configuration (batch size, learning rate, etc.)

**Output:** `examples/MyClassificationDataset/dataset_plan.json`

You can inspect the plan file to see:
- Feature statistics
- Data splits (patient IDs and slide IDs for each fold)
- Recommended training hyperparameters

#### Step 3: Train Models

**Option A: Train all folds (5-fold CV)**

```bash
python nnMIL/run/nnMIL_run_training.py examples/MyClassificationDataset simple_mil all
```

This will train 5 models (fold_0 to fold_4) sequentially.

**Option B: Train a single fold**

```bash
python nnMIL/run/nnMIL_run_training.py examples/MyClassificationDataset simple_mil 0
```

**Option C: Use the workflow script**

```bash
bash nnMIL/scripts/run_classification.sh examples/MyClassificationDataset simple_mil 0
```

This runs planning, training, and testing automatically.

**Training Output Structure:**
```
nnMIL_results/MyClassificationDataset/simple_mil/
├── fold_0/
│   ├── best_simple_mil.pth          # Best model checkpoint
│   ├── training_config.json          # Training configuration used
│   ├── simple_mil_training.log      # Training log
│   └── results_simple_mil.csv       # Validation/test results
├── fold_1/
├── ...
└── fold_4/
```

#### Step 4: Evaluate Models

**For a single fold:**

```bash
python nnMIL/run/nnMIL_predict.py \
    --plan_path examples/MyClassificationDataset/dataset_plan.json \
    --checkpoint_path nnMIL_results/MyClassificationDataset/simple_mil/fold_0/best_simple_mil.pth \
    --output_dir nnMIL_results/MyClassificationDataset/simple_mil/fold_0/predictions \
    --fold 0
```

**For all folds:**

```bash
for FOLD in 0 1 2 3 4; do
    python nnMIL/run/nnMIL_predict.py \
        --plan_path examples/MyClassificationDataset/dataset_plan.json \
        --checkpoint_path nnMIL_results/MyClassificationDataset/simple_mil/fold_${FOLD}/best_simple_mil.pth \
        --output_dir nnMIL_results/MyClassificationDataset/simple_mil/fold_${FOLD}/predictions \
        --fold ${FOLD}
done
```

**Output:** CSV files with predictions for each sample:
- `slide_id`, `patient_id`, `prediction`, `probability_class_0`, `probability_class_1`, `label`

### Tutorial 2: Survival Analysis Task

#### Step 1: Prepare Your Dataset

**Create `examples/MySurvivalDataset/dataset.json`:**
```json
{
    "dataset_id": "MySurvivalDataset",
    "dataset_name": "My Survival Analysis Dataset",
    "task_type": "survival",
    "feature_dir": "/path/to/your/features",
    "metric": "c_index",
    "evaluation_setting": "5fold"
}
```

**Create `examples/MySurvivalDataset/dataset.csv`:**
```csv
slide_id,patient_id,split,event,time
slide_001,patient_001,train,1,45.2
slide_002,patient_002,train,0,60.5
slide_003,patient_003,train,1,30.1
slide_004,patient_004,val,1,50.0
slide_005,patient_005,val,0,72.3
slide_006,patient_006,test,1,25.8
```

**Important Notes:**
- `event`: 1 for event occurred, 0 for censored
- `time`: Survival time (months, days, etc.)
- Same patient can have multiple slides (all with same event/time)

#### Step 2: Run Planning

```bash
python nnMIL/run/nnMIL_plan_experiment.py -d examples/MySurvivalDataset --seed 42
```

The planner will:
- Analyze feature files
- Create patient-level splits stratified by event status
- Determine appropriate batch size (may be set to 1 for survival tasks, which uses NLLSurvLoss)

#### Step 3: Train Models

**Using the workflow script (recommended):**

```bash
bash nnMIL/scripts/run_survival.sh examples/MySurvivalDataset simple_mil 0
```

**Or manually:**

```bash
# Training
python nnMIL/run/nnMIL_run_training.py examples/MySurvivalDataset simple_mil all

# Testing
PLAN_FILE="examples/MySurvivalDataset/dataset_plan.json"
for FOLD in 0 1 2 3 4; do
    python nnMIL/run/nnMIL_predict.py \
        --plan_path $PLAN_FILE \
        --checkpoint_path nnMIL_results/MySurvivalDataset/simple_mil/fold_${FOLD}/best_simple_mil.pth \
        --output_dir nnMIL_results/MySurvivalDataset/simple_mil/fold_${FOLD}/predictions \
        --fold ${FOLD}
done
```

#### Step 4: Evaluate Models

The prediction output includes:
- `slide_id`, `patient_id`, `risk_score`, `event`, `time`

Risk scores can be aggregated at patient level (e.g., mean) for C-index calculation.

### Common Customizations

#### Override Plan Hyperparameters

```bash
python nnMIL/run/nnMIL_run_training.py examples/MyClassificationDataset simple_mil 0 \
    --batch_size 64 \
    --learning_rate 1e-3 \
    --num_epochs 200 \
    --weight_decay 1e-4
```

#### Use Different Models

Available models: `simple_mil`, `ab_mil`, `ds_mil`, `trans_mil`, `wikg_mil`, etc.

```bash
python nnMIL/run/nnMIL_run_training.py examples/MyClassificationDataset ab_mil all
```

#### Official Split Instead of 5-Fold CV

Set `"evaluation_setting": "official_split"` in `dataset.json`, then:

```bash
python nnMIL/run/nnMIL_run_training.py examples/MyClassificationDataset simple_mil None
```

#### Select GPU

```bash
CUDA_VISIBLE_DEVICES=0 bash nnMIL/scripts/run_classification.sh examples/MyClassificationDataset simple_mil 0
```

### Expected Output Examples

**Training Log (excerpt):**
```
2025-11-02 10:00:00 - INFO - Epoch 1/100 - Loss: 0.6234
2025-11-02 10:00:30 - INFO - Val AUC: 0.7234
2025-11-02 10:00:30 - INFO - Validation AUC improved (0.7234). Saving model...
...
2025-11-02 10:15:00 - INFO - Early stopping triggered at epoch 45
```

**Results CSV (Classification):**
```csv
slide_id,patient_id,prediction,probability_class_0,probability_class_1,label
slide_001,patient_001,1,0.123,0.877,1
slide_002,patient_002,0,0.891,0.109,0
```

**Results CSV (Survival):**
```csv
slide_id,patient_id,risk_score,status,time
slide_001,patient_001,2.345,1,45.2
slide_002,patient_002,1.234,0,60.5
```

### Next Steps

- Compare multiple models: Use `compare_cv_results.py` to compare 5-fold CV results
- Ensemble predictions: Average predictions across folds
- Patient-level aggregation: For tasks with multiple slides per patient
- Hyperparameter tuning: Modify plan file or override via command line

## Key Concepts

### Plan Files (`dataset_plan.json`)

The plan file contains:
- **Dataset Information**: Task type, labels, feature dimensions
- **Feature Statistics**: Patch count statistics, recommended `max_seq_length`
- **Data Splits**: Patient-level splits for train/val/test
- **Training Configuration**: Batch size, learning rate, epochs, batch sampler, etc.

### Model Types

- `simple_mil`: Simple MIL with attention
- `ab_mil`: Attention-based MIL
- `ds_mil`: Dual-stream MIL
- `trans_mil`: Transformer-based MIL
- And more...

### Batch Samplers

Automatically selected based on task and metric:
- `risk_set`: For survival C-index (ensures batches have events)
- `auc`: For AUC metric (balanced positive/negative pairs)
- `balanced`: For BACC/F1 metrics (balanced classes)
- `random`: Default random sampling

### Patch Selection Strategy

- **Training**: Random selection up to `max_seq_length` for data augmentation
- **Validation/Testing**: Always use all original patches (no truncation)
- **Original Length Training**: 10% random drop for data augmentation

## Configuration Override

You can override plan configurations via command-line arguments:

```bash
python nnMIL/run/nnMIL_run_training.py examples/Dataset001_classification simple_mil 0 \
    --batch_size 64 \
    --learning_rate 1e-3 \
    --num_epochs 200
```

## Output Structure

```
nnMIL_results/
└── Dataset001_classification/
    └── simple_mil/
        ├── fold_0/
        │   ├── best_simple_mil.pth
        │   ├── training_config.json
        │   ├── simple_mil_training.log
        │   ├── results_simple_mil.csv
        │   └── predictions/
        │       └── results_simple_mil.csv
        ├── fold_1/
        ├── ...
        └── fold_4/
```

## Notes

1. **Always run planning first**: The plan file is required for training
2. **Patient-level splitting**: All splits are done at the patient level to avoid data leakage
3. **Stratified splitting**: Splits maintain class/event distribution
4. **Reproducibility**: Random seeds are set for reproducibility (default: 42)
5. **CUDA Device Selection**: Use `CUDA_VISIBLE_DEVICES` environment variable to select GPUs

## Troubleshooting

### Plan file not found
- Ensure you've run `nnMIL_plan_experiment.py` first
- Check that `dataset_plan.json` exists in the dataset directory

### Out of memory errors
- Reduce `batch_size` in the plan file or via command-line
- Reduce `max_seq_length` in the plan file

### No validation improvement
- Check if `patience` is set appropriately
- Verify data splits are correct
- Check early stopping metric matches your task

## Acknowledgement

We would like to thank the [MIL_BASELINE](https://github.com/lingxitong/MIL_BASELINE) project for providing a comprehensive collection of MIL models. We also sincerely appreciate the [nnUNet](https://github.com/MIC-DKFZ/nnUNet) framework, which has greatly inspired and supported our work.
