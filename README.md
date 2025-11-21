# nnMIL: No-New Multiple Instance Learning

nnMIL: A simple yet generalizable multiple instance learning framework for computational pathology

- **No-new MIL** training strategies yield benefits in the foundation model era.
- **Unified pipeline** for classification, regression, and survival MIL tasks
- **Plan-driven training** that inspects slide features, builds patient-level splits, and recommends hyper-parameters
- **Consistent inference utilities** for official or k-fold evaluation settings

Looking for the full step-by-step walkthrough? Jump to [`TUTORIAL.md`](./TUTORIAL.md).

## Repository Layout

```
nnMIL/
├── data/                    # Dataset abstractions
├── network_architecture/    # Model factory + implementations
├── preprocessing/           # Experiment planner & helpers
├── run/                     # Python entry points (plan/train/predict)
├── scripts/                 # Shell wrappers for complete workflows
├── training/                # Trainers, losses, samplers, callbacks
└── utilities/               # Shared utils (logging, configs, etc.)
```

Two external directories are expected beside the repo:

- `nnMIL_raw_data/TaskXXX_*` holds each task’s `dataset.json`, `dataset.csv`, generated `dataset_plan.json`, and HDF5 feature files.
- `nnMIL_results/TaskXXX_*` receives logs, checkpoints, predictions, and metrics (`official_split/` or `fold_*` subfolders).

Reference bundles:

- [Datasets & plan files (`nnMIL_raw_data` snapshot)](https://drive.google.com/drive/folders/1HF7jjH3FiWDIGCvvWBqD-3Z0Sgv8Dh-g?usp=sharing)
- [Experiment outputs (`nnMIL_results` snapshot)](https://drive.google.com/drive/folders/1-DPqIUUy0oYFicGGdEuHzehrQQFyDoXI?usp=sharing)
- Extracted patch-level embeddings (TCGA/EBRAINS using Virchow2 and UNI):
  - [TCGA/UNI](https://drive.google.com/drive/folders/1gjL3Uhumk35YSkbz1TOFMzUBumZfSBGZ?usp=sharing)
  - [TCGA/Virchow2](https://drive.google.com/drive/folders/1PVuRhnc_ObUn1aFRSiyk5_5vtWjoCeNv?usp=sharing)
  - [EBRAINS](https://drive.google.com/file/d/16tpUS-o21WsQH1U3Jyqi4784sb-OceiB/view?usp=sharing)

## Environment & Dependencies

1. Install system packages for HDF5/BLAS as required by your platform.
2. Create and activate the project environment (example using conda):
```bash
   conda env create -f environment.yml
   conda activate pyrad
   ```
3. Install PyTorch 2.9 nightly and the matching torchvision build (CUDA 12.6 example):
```bash
   pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/nightly/cu126
   pip install --no-cache-dir torchvision --index-url https://download.pytorch.org/whl/nightly/cu126
   ```

## Quick Start (High Level)

1. **Prepare data** under `nnMIL_raw_data/<Task_ID>/` with:
   - `dataset.json` (task metadata: labels, metrics, feature path)
   - `dataset.csv` (slide/patient metadata, labels or survival fields)
   - slide-level feature files (`<slide_id>.h5`)
2. **Plan** the experiment:
```bash
   python nnMIL/run/nnMIL_plan_experiment.py -d nnMIL_raw_data/Task001_CRC_DSS
   ```
   This produces `dataset_plan.json` with recommended hyper-parameters and patient splits.
3. **Train**:
```bash
   python nnMIL/run/nnMIL_run_training.py nnMIL_raw_data/Task001_CRC_DSS simple_mil all
   ```
   or use `bash nnMIL/scripts/run_classification.sh nnMIL_raw_data/Task001_CRC_DSS simple_mil 0 auto`.
4. **Predict** with `nnMIL/run/nnMIL_predict.py`, pointing to each checkpoint directory.

For detailed guidance (including how to adapt `dataset.json`/`dataset.csv` to your own data), consult [`TUTORIAL.md`](./TUTORIAL.md).

## Workflow Scripts

- `scripts/run_classification.sh <DATASET_DIR> <MODEL> <CUDA_DEVICE> [split]`  
  Automates planning → training → prediction. `split` accepts `auto` (default), `all`, `None`, or a fold index.
- `scripts/run_survival.sh <DATASET_DIR> <MODEL> <CUDA_DEVICE>`  
  Equivalent wrapper tailored for survival experiments.
- `scripts/run_plco_crc.sh`  
  End-to-end recipe for the PLCO CRC cohort.

All scripts assume the project root is the parent of `nnMIL/` and write outputs into `nnMIL_results/`.

## Acknowledgements

We gratefully acknowledge prior work that inspired nnMIL:

- [MIL_BASELINE](https://github.com/lingxitong/MIL_BASELINE) for its comprehensive collection of MIL models.
- [nnUNet](https://github.com/MIC-DKFZ/nnUNet) for the self-configuring design principles that guided our training planner and workflow automation.
  
This project focuses mainly on simple yet generalizable MIL training. For feature extraction, we highly recommend using the excellent project [CLAM](https://github.com/mahmoodlab/CLAM).

 If you use this codebase in your research, please cite the following works:

  @misc{luo2025nnmilgeneralizablemultipleinstance,
        title={nnMIL: A generalizable multiple instance learning framework for computational pathology}, 
        author={Xiangde Luo and Jinxi Xiang and Yuanfeng Ji and Ruijiang Li},
        year={2025},
        eprint={2511.14907},
        archivePrefix={arXiv},
        primaryClass={cs.CV},
        url={https://arxiv.org/abs/2511.14907}, 
  }

## Status & Contact

nnMIL is actively evolving—expect iterative updates to the planner, trainers, and evaluation scripts. Feedback and contributions are welcome. Reach out at luoxd96 at stanford dot edu.

👉 A comprehensive tutorial (classification + survival, custom dataset adaptation, shell scripts) is maintained in [`TUTORIAL.md`](./TUTORIAL.md) and updated alongside code changes.

