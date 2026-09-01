# Reproducibility Guide / 复现说明

This document separates three things that are easy to confuse:

- **Code reproducibility:** source, configuration, tests, and scripts are provided.
- **Data reproducibility:** datasets must be downloaded from their official publishers and converted to the documented CSV layout.
- **Result reproduction:** reported metrics require retraining because checkpoints are intentionally not included.

## 1. Environment

The environment recorded with the experiments was:

- Python 3.8.20
- PyTorch 1.7.1
- NumPy 1.23.5
- pandas 1.5.3
- scikit-learn 1.2.2
- SciPy 1.10.1

Install it from the repository root:

```bash
conda env create -f environment.yml
conda activate ce-bid
python -m unittest discover -s tests -v
```

The shell workflows target Linux or WSL. `run.py` itself is cross-platform and defaults to `num_workers=0` for Windows compatibility. CUDA requests fall back to CPU when CUDA is unavailable; AMP is automatically disabled outside CUDA.

## 2. Data

Use the official sources:

- PVOD: <https://github.com/yaotc/PVODataset> and <https://doi.org/10.11922/sciencedb.01094>
- ST-EVCDP: <https://github.com/IntelligentSystemsLab/ST-EVCDP>

Do not commit downloaded data to this repository. Arrange raw files as:

```text
dataset/
├── party_1/PVOD.csv
├── party_2/PVOD.csv
├── ...
├── party_7/PVOD.csv
└── merge/PVOD.csv
```

Run `scripts/PVOD/Init.sh`, or invoke the alignment step directly:

```bash
python data_provider/prepare_pvod.py \
  --input-root ./dataset --output-root ./dataset/processed --parties 7
```

The preparation step averages duplicate timestamps, takes the timestamp
intersection across all party and merge files, and writes aligned copies under
`dataset/processed/`; raw files are unchanged. Observation gaps are preserved
because PVOD is not a continuous all-day grid.

CSV contract:

1. first column is `date`;
2. last column is the prediction target `OT`;
3. all intermediate columns are numeric features;
4. prepared timestamps are parseable, unique, ascending, and aligned across files;
5. no feature contains `NaN` or infinity;
6. every party uses the same column order and row timestamps.

For split distillation, the dimensional invariant is:

```text
cloud_dim = N × (edge_dim - 1) + 1
```

The final `+1` is the shared `OT` column. The PVOD experiment uses `N=7`, `edge_dim=15`, and `cloud_dim=99`.

## 3. Determinism and data splits

`run.py` seeds Python, NumPy, PyTorch, and all CUDA devices with `--seed 2021` by default. Reported PVOD initialization uses:

- samples `[0:18750]`;
- 80% training and 20% validation inside that window;
- no initial test split;
- `seq_len=12`, `label_len=12`, `pred_len=6`;
- batch size 32 and learning rate `1e-4`.

GPU kernels and dependency versions can still introduce small numerical differences. Record the GPU model, driver, CUDA runtime, full command, seed, and Git commit when reporting a rerun.

The public cleanup corrected eight issues found in the historical experiment code: raw party/merge timestamps are explicitly aligned; scaler fitting is restricted to the chronological training portion of a requested window; local personalization exports the best validation checkpoint rather than the last epoch; each party maintains its own refreshable error reference; event triggers consume only the shared target column rather than every modelled variable; the old reverse feature loss (which could not update a frozen cloud backbone) is replaced by quality-weighted edge-output soft targets; forward stage projectors learn from the teacher's actual output instead of detached untrained teacher projections; and a CNN stage projector maps the full history axis to the forecast horizon instead of relabelling the last historical positions as future steps. Therefore the committed historical tables document the completed experiments, but they are not a bit-for-bit acceptance target for the hardened code. A new publication-quality run should version its processed data snapshot and regenerate all tables.

## 4. Initial three-stage training

The canonical PVOD workflow is [`scripts/PVOD/Init.sh`](../scripts/PVOD/Init.sh):

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/PVOD/Init.sh
```

It performs:

### Stage A — cloud teacher

Train a TimesNet on `dataset/merge/PVOD.csv` with 99 input/output variables. TimesNet uses FFT-selected periods and 2D convolutions to model within-period and across-period variation.

Expected artifact:

```text
checkpoints/Global_Teacher/checkpoint.pth
```

### Stage B — forward heterogeneous distillation

Freeze the cloud teacher and train a CNN with 15 input/output variables. Each party feature block is paired with the shared `OT`. The loss combines hard supervision, teacher soft targets, stage-level OFA projectors, and output-level OFA weighting.

Expected artifact:

```text
checkpoints/Unified_Student/checkpoint.pth
```

### Stage C — local personalization

Copy the general CNN to seven edge sites and fine-tune each copy on its local CSV.

Expected artifacts:

```text
checkpoints/Local_Edges/party_1/checkpoint.pth
checkpoints/Local_Edges/party_1/scaler.pkl
...
checkpoints/Local_Edges/party_7/checkpoint.pth
checkpoints/Local_Edges/party_7/scaler.pkl
```

Each edge scaler is fitted on the chronological initialization-training rows, saved next to that edge checkpoint, and reused for later inference and incremental fine-tuning. A pure inference window fails early if its fitted scaler is missing.

The initialization script exits if a required stage does not create its checkpoint.

## 5. Online cloud-edge simulation

Run:

```bash
CUDA_VISIBLE_DEVICES=0 \
N_PARTY=7 EDGE_DIM=15 CLOUD_DIM=99 \
bash real_time.sh
```

At each step, every edge CNN forecasts its local horizon. The simulator stores predictions and delayed ground truth, then evaluates three trigger conditions:

- **error:** recent MSE exceeds both a relative threshold and an absolute floor;
- **drift:** histogram-based KL divergence between recent and reference target windows exceeds a threshold;
- **timeout:** no update has occurred for the configured maximum interval.

If any party triggers, the workflow runs reverse distillation, then local fine-tuning. After the configured number of reverse updates, it also performs forward distillation to refresh the general student. Commands are launched without `shell=True`, and cleanup is restricted to descendants of the generated `checkpoints/` and `results/` roots.

Useful environment overrides are defined by `Config` in [`exp/real_time_inference.py`](../exp/real_time_inference.py), including `RT_N_PARTY`, `RT_CLOUD_DIM`, `RT_EDGE_DIM`, `RT_INFER_STEP`, `RT_ERR_RATIO`, `RT_KL_THRESH`, epoch counts, and OFA weights.

## 6. Ablation and sensitivity workflows

- Epoch ablation: [`Epoch_ablation.sh`](../Epoch_ablation.sh)
- Trigger sensitivity: [`Sensitivity_experiment.sh`](../Sensitivity_experiment.sh)

These scripts restore the initialization copies under `checkpoints/GT_back/` and
`checkpoints/LE_Backpack/` before each run; `scripts/PVOD/Init.sh` creates those
copies. Raw outputs are generated under ignored directories such as
`ablation_results_v5/` and `sensitivity_results/`. Historical baseline models and
plot-only utilities were removed from this focused public repository; the small,
reviewable result tables used in the README remain under [`docs/results/`](results).

## 7. Validation checklist

Before accepting a reproduced run, verify:

```bash
python -m compileall -q run.py exp models layers data_provider utils
python -m unittest discover -s tests -v
```

Then check:

- the exact data row count, columns, time range, and sampling interval;
- `edge_dim`, `cloud_dim`, and party count satisfy the invariant;
- the teacher/student architecture parameters match the loaded checkpoints;
- validation is chronological rather than randomly split;
- the final report contains all parties and non-empty prediction counts;
- every metric is calculated in the same normalized or inverse-transformed space;
- the Git commit and complete command are saved with the output.

## 8. What is not included

- dataset files and their redistribution rights;
- trained `.pth` checkpoints;
- full per-step predictions and `.npy` tensors;
- generated plots and raw multi-run logs.

These omissions are deliberate. They prevent a multi-gigabyte Git history and avoid presenting third-party data as part of the source distribution.
