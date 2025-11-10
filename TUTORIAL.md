# nnMIL (No-New Multiple Instance Learning) Tutorial

This guide walks through the complete nnMIL workflow using the provided task layouts and shows how to adapt each step to your own dataset. It covers:

- Preparing `nnMIL_raw_data` and `nnMIL_results`
- Editing `dataset.json` and `dataset.csv` for classification and survival tasks
- Running the experiment planner
- Training and inference via both Python entry points and shell scripts
- Customising hyper-parameters and troubleshooting common issues

## 1. Environment Setup

- nnMIL repository cloned to `<PROJECT_ROOT>/nnMIL/`
- Two sibling directories:
  - `<PROJECT_ROOT>/nnMIL_raw_data/`
  - `<PROJECT_ROOT>/nnMIL_results/`
- Conda environment (recommended) created via:
  ```bash
  conda env create -f nnMIL/environment.yml
  conda activate pyrad
  ```
- PyTorch 3.10 nightly + matching torchvision build installed (keep versions in sync to avoid CUDA operator errors).

Optional reference data:

- [Example `nnMIL_raw_data` snapshot](https://drive.google.com/drive/folders/1HF7jjH3FiWDIGCvvWBqD-3Z0Sgv8Dh-g?usp=sharing)
- [Example `nnMIL_results` snapshot](https://drive.google.com/drive/folders/1-DPqIUUy0oYFicGGdEuHzehrQQFyDoXI?usp=sharing)
- Patch-level embedding for quick try:[TCGA/UNI](https://drive.google.com/drive/folders/1gjL3Uhumk35YSkbz1TOFMzUBumZfSBGZ?usp=sharing),[TCGA/Virchow2](https://drive.google.com/drive/folders/1PVuRhnc_ObUn1aFRSiyk5_5vtWjoCeNv?usp=sharing),[EBRAINS](https://drive.google.com/file/d/16tpUS-o21WsQH1U3Jyqi4784sb-OceiB/view?usp=sharing).
## 2. Quick Try with Provided Example Tasks

Start here if you just want nnMIL running immediately. We distribute four ready-to-go tasks; after downloading the features and **fixing the absolute paths of `feature_dir`/`clinical_dir`**, you can plan (if you want to reuse my plans, please fix the paths), train, and evaluate without touching any python code.

Each bundle contains:

- `dataset.json` / `dataset.csv` already configured for nnMIL
- Pre-extracted slide features (HDF5) matching the CSV `slide_id` values
- Optional `dataset_plan.json` snapshots (you can regenerate to match your environment)

| Task ID | Cohort / Description | Feature link |
| ------- | -------------------- | ------------ |
| `Task001_CRC_DSS` | CRC DSS survival (External evaluation, Virchow2) | [Google Drive](https://drive.google.com/drive/folders/1SiF379K70hWAPzq6TOKrwX8VX2DCDyJq?usp=sharing)|
| `Task010_TCGA-BRCA` | BRCA DSS survival (5-fold, Virchow2)| [Google Drive](https://drive.google.com/drive/folders/1PVuRhnc_ObUn1aFRSiyk5_5vtWjoCeNv?usp=sharing) |
| `Task002_EBRAINS_Fine` | EBRAINS coarse classification (official split, UNI) | [Google Drive](https://drive.google.com/file/d/16tpUS-o21WsQH1U3Jyqi4784sb-OceiB/view?usp=sharing) |
| `Task002_EBRAINS_Coarse` | EBRAINS coarse classification  (official split, UNI) | [Google Drive](https://drive.google.com/file/d/16tpUS-o21WsQH1U3Jyqi4784sb-OceiB/view?usp=sharing) |

### Steps

1. Download each archive and place the task directory under your `nnMIL_raw_data/`.
2. Edit the `feature_dir` (and, if necessary, `clinical_dir`) paths in `dataset.json` so they point to the absolute locations on your machine.
3. From the repository root (`/path/to/github/nnMIL`), run the workflow scripts located in `scripts/`, which already chain **planning → training → prediction**:
   ```bash
   cd /path/to/github/nnMIL/scripts

   # Example: EBRAINS classification with official split, UNI
   bash run_classification.sh # you can modify the default settings after you read this script

   # Example: 5-fold BRCA DSS survival
   bash run_survival.sh # you can modify the default settings after you read this script

   # Example: survival task on CRC DSS (official split)
   CUDA_VISIBLE_DEVICES=0 python nnMIL/run/nnMIL_predict.py \
     --plan_path nnMIL_raw_data/Task_001_CRC_DSS/dataset_plan.json \
     --checkpoint_path nnMIL_results/Task_001_CRC_DSS/simple_mil/official_split/best_simple_mil.pth \
     --input_dir /XXXX/SR386_WSIs/h5_files \
     --output_dir nnMIL_results/Task_001_CRC_DSS/simple_mil/official_split_random/SR386_WSIs_test_best \
     # Due to the license, we can not release the embedding of PLCO-CRC and MCO, but we have released the model for evaluation.
   
   ```
   These scripts automatically call `nnMIL_plan_experiment.py`, `nnMIL_run_training.py`, and `nnMIL_predict.py` in sequence, writing outputs under `nnMIL_results/<Task_ID>/<model>/<split>/`.
4. Reuse checkpoints immediately:
   ```bash
   python nnMIL/run/nnMIL_predict.py \
     --plan_path /path/to/nnMIL_raw_data/Task010_TCGA-BRCA/dataset_plan.json \
     --checkpoint_path /path/to/nnMIL_results/Task010_TCGA-BRCA/simple_mil/fold_0/best_simple_mil.pth \
     --output_dir /path/to/nnMIL_results/Task010_TCGA-BRCA/simple_mil/fold_0/predictions \
     --fold 0
   ```
   Swap `fold_0` for other folds or `official_split` checkpoints as needed.

> Tip: the `nnMIL_results` Google Drive snapshot already contains trained checkpoints. Drop it next to your repo and run `nnMIL_predict.py` directly to inspect evaluation metrics.

The rest of this tutorial explains how to adapt the configuration files for your own data once you are comfortable with the quick-try workflow.

## 3. Directory Layout

Each task lives in its own folder under `nnMIL_raw_data/`:

```
nnMIL_raw_data/Task001_CRC_DSS/
├── dataset.json          # Task metadata & planner configuration
├── dataset.csv           # Slide / patient splits and targets
├── dataset_plan.json     # Generated by the planner
└── features/             # Slide-level feature files (optional subdir name)
    ├── slide_0001.h5
    ├── slide_0002.h5
    └── ...
```

The planner reads `dataset.json` and `dataset.csv`, inspects feature files, and overwrites/creates `dataset_plan.json`. Training writes into `nnMIL_results/Task001_CRC_DSS/…`.

## 4. Understanding `dataset.json`

`dataset.json` specifies dataset metadata, label schema, evaluation setting, feature location, and optional overrides. Below is a trimmed example for a classification task:

```json
{
  "dataset_id": "Task001_CRC_DSS",
  "dataset_name": "CRC DSS",
  "task_type": "classification",
  "labels": {
    "0": "Low risk",
    "1": "High risk"
  },
  "feature_dir": "/mnt/.../nnMIL_raw_data/Task001_CRC_DSS/features",
  "clinical_dir": "/mnt/.../nnMIL_raw_data/Task001_CRC_DSS/dataset.csv",
  "metric": "auc",
  "evaluation_setting": "5fold",
  "seed": 42
}
```

### Fields to adjust for your data

| Field | Required | Description | Adaptation tips |
| ----- | -------- | ----------- | --------------- |
| `dataset_id` | ✓ | Unique task identifier used in result paths | Keep consistent across raw/results directories |
| `dataset_name` | ✓ | Human-readable name | Any string |
| `task_type` | ✓ | `classification`, `regression`, or `survival` | Ensure it matches your task |
| `labels` | classification only | Dict mapping class IDs → label names | Use string keys (`"0"`, `"1"`, …) and ensure IDs align with CSV |
| `feature_dir` | ✓ | Absolute path to feature HDF5 files | Each file must be named `<slide_id>.h5` |
| `clinical_dir` | ✓ | Absolute path to `dataset.csv` | Planner reads splits/targets from here |
| `metric` | ✓ | Training metric (`auc`, `bacc`, `mse`, `c_index`, …) | Choose the metric used for early stopping |
| `evaluation_setting` | ✓ | `5fold` or `official_split` | Determines training/inference behaviour |
| `event_column`, `time_column` | survival only | Column names in CSV | Defaults are `event` and `time` |

Keep paths absolute to avoid confusion when running scripts from different working directories.

## 5. Authoring `dataset.csv`

`dataset.csv` lists each slide (instance) and the metadata needed for planning and training. Minimum columns depend on the task:

### Classification / Regression

```csv
slide_id,patient_id,split,label
TCGA-01-0001,Patient_0001,train,0
TCGA-01-0002,Patient_0002,train,1
TCGA-01-0003,Patient_0003,val,0
TCGA-01-0004,Patient_0004,test,1
```

- `slide_id`: must match the base name of the feature file (`slide_id.h5`).
- `patient_id`: used for patient-level splitting; multiple slides can share one patient.
- `split`: `train`, `val`, `test`, or left blank if planner should generate CV folds.
- `label`: integer class ID or numeric target.

### Survival

```csv
slide_id,patient_id,split,event,time
Case_0001,Patient_A,train,1,32.5
Case_0002,Patient_B,train,0,45.7
Case_0003,Patient_C,val,1,12.3
```

- `event`: 1 if the event occurred, 0 if censored.
- `time`: survival duration in consistent units.

### Adapting to your dataset

1. Make sure column names match what `dataset.json` expects (`label`, `event`, `time`, etc.).
2. Ensure label values fall within the range defined in `labels`.
3. If performing coarse mapping (e.g., fine → coarse labels), do it in the CSV before training. Do **not** modify core dataset loading code.
4. Verify that every `slide_id` has a corresponding `.h5` feature file.

## 6. Run the Experiment Planner

From the project root (`<PROJECT_ROOT>`):

```bash
python nnMIL/run/nnMIL_plan_experiment.py -d nnMIL_raw_data/Task001_CRC_DSS --seed 42
```

What the planner does:

- Scans the feature directory and gathers patch-count statistics.
- Builds patient-level training/validation/test splits according to `evaluation_setting`.
- Chooses default hyper-parameters (batch size, learning rate, sampler type, etc.).
- Writes `dataset_plan.json` in the same task directory.

`dataset_plan.json` is required by the training entry points and shell scripts.

## 7. Train Models

### Option A: Python entry point

```bash
# All folds (5-fold CV)
python nnMIL/run/nnMIL_run_training.py nnMIL_raw_data/Task001_CRC_DSS simple_mil all

# Single fold
python nnMIL/run/nnMIL_run_training.py nnMIL_raw_data/Task001_CRC_DSS simple_mil 0

# Official split (no CV)
python nnMIL/run/nnMIL_run_training.py nnMIL_raw_data/Task001_CRC_DSS simple_mil None
```

Arguments:

1. Path to the dataset directory (absolute or relative)
2. Model name (e.g., `simple_mil`, `ab_mil`, `trans_mil`)
3. Fold specifier: `all`, fold index, or `None`

Override hyper-parameters using optional flags. Example:

```bash
python nnMIL/run/nnMIL_run_training.py nnMIL_raw_data/Task001_CRC_DSS simple_mil all \
  --batch_size 32 \
  --learning_rate 3e-4 \
  --num_epochs 150 \
  --weight_decay 1e-4
```

### Option B: Shell scripts

Scripts wrap planning, training, and prediction. They assume the project root is the parent directory of `nnMIL/`.

#### Classification

```bash
bash nnMIL/scripts/run_classification.sh nnMIL_raw_data/Task001_CRC_DSS simple_mil 0 auto
```

Parameters:

1. Dataset directory
2. Model name
3. CUDA device index
4. Split selection (`auto` = read from plan, `all`, `None`, or a fold index)

#### Survival

```bash
bash nnMIL/scripts/run_survival.sh nnMIL_raw_data/Task001_CRC_DSS simple_mil 0
```

Scripts will:

1. Run the planner (overwriting `dataset_plan.json`)
2. Train according to the selected split
3. Generate predictions into `nnMIL_results/.../predictions/`

## 8. Inspecting Outputs

Training produces a directory structure under `nnMIL_results/<Task_ID>/<model>/`:

```
nnMIL_results/Task001_CRC_DSS/simple_mil/
├── official_split/
│   ├── best_simple_mil.pth
│   ├── training_config.json
│   ├── simple_mil_training.log
│   ├── results_simple_mil.csv
│   └── predictions/
│       └── results_simple_mil.csv
├── fold_0/
│   ├── best_simple_mil.pth
│   ├── ...
└── fold_4/
```

- `training_config.json` captures the exact parameters used.
- `results_simple_mil.csv` contains validation metrics per epoch.
- `predictions/` stores inference CSVs (`slide_id`, `patient_id`, predictions, labels, etc.).

## 9. Running Inference Manually

Use `nnMIL/run/nnMIL_predict.py` to evaluate a checkpoint:

```bash
python nnMIL/run/nnMIL_predict.py \
  --plan_path nnMIL_raw_data/Task001_CRC_DSS/dataset_plan.json \
  --checkpoint_path nnMIL_results/Task001_CRC_DSS/simple_mil/fold_0/best_simple_mil.pth \
  --output_dir nnMIL_results/Task001_CRC_DSS/simple_mil/fold_0/predictions \
  --fold 0
```

For official split:

```bash
python nnMIL/run/nnMIL_predict.py \
  --plan_path nnMIL_raw_data/Task001_CRC_DSS/dataset_plan.json \
  --checkpoint_path nnMIL_results/Task001_CRC_DSS/simple_mil/official_split/best_simple_mil.pth \
  --output_dir nnMIL_results/Task001_CRC_DSS/simple_mil/official_split/predictions \
  --fold None
```

## 10. Customising for Your Dataset

### Choosing a new task ID

- Copy an existing task directory as a template:  
  `cp -r nnMIL_raw_data/Task001_CRC_DSS nnMIL_raw_data/Task999_MyStudy`
- Update `dataset.json` and `dataset.csv` with your cohort info.
- Place your feature files in the new directory (names must match `slide_id`).

### Common modifications

- **Label schema**: Update the `labels` dict (classification) and ensure `dataset.csv` only contains those IDs.
- **Evaluation setting**: set `"evaluation_setting": "official_split"` if you provide explicit train/val/test splits in the CSV.
- **Survival column names**: if using custom names (e.g., `"event_col": "Death"`, `"time_col": "Days"`), set `event_column` / `time_column` in `dataset.json`.
- **Metrics**: choose `auc`, `bacc`, `f1`, `mse`, `c_index`, etc. to match your task.
- **Batch sampler**: the planner usually chooses automatically. If you need manual control, set `"batch_sampler": null` in `dataset_plan.json` (after planning) or pass `--batch_sampler random` to the trainer.

## 11. Troubleshooting

- **`FileNotFoundError` for features**: Confirm `feature_dir` points to the folder containing `<slide_id>.h5`.
- **`Assertion 't >= 0 && t < n_classes' failed`**: Label IDs in `dataset.csv` exceed the range declared in `labels`. Fix the CSV mapping.
- **`torchvision::nms` missing**: reinstall torchvision that matches the torch build.
- **`numpy.dtype size changed`**: reinstall compatible `numpy` and `pandas` (e.g., `pip install --force-reinstall pandas==2.3.3 numpy==2.2.6`).
- **Fonts or plotting issues in analysis scripts**: ensure `matplotlib` is installed and fonts are available; refer to project plotting conventions (error bars from 1000 bootstrap iterations, etc.).

## 12. Next Steps

- Aggregate predictions across folds to compute patient-level metrics (e.g., average slide risk per patient).
- Use `nnMIL/run/nnMIL_predict_ensemble.py` for ensemble inference across multiple checkpoints.
- Explore alternative model architectures under `network_architecture/models/`.
- Extend scripts or write new ones for your laboratory workflow; reuse environment assumptions described here.

**If you encounter issues not covered in this tutorial, please reach out at luoxd96 AT stanford DOT edu. Contributions and improvements are welcome.**


