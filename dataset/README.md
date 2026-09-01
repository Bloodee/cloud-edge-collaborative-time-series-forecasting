# Dataset layout

Datasets are intentionally excluded from Git because they are large and have
their own distribution terms. Download them from the official sources and
place processed CSV files under this directory.

## PVOD

- Official repository: https://github.com/yaotc/PVODataset
- Dataset DOI: https://doi.org/10.11922/sciencedb.01094
- Sampling interval used by this project: 15 minutes

Raw seven-site layout:

```text
dataset/
├── party_1/PVOD.csv
├── party_2/PVOD.csv
├── ...
├── party_7/PVOD.csv
└── merge/PVOD.csv
```

The retained PVOD files contain occasional gaps, one duplicated timestamp, and
one timestamp missing from the merged cloud file. Do not edit the raw files in
place. The canonical initializer runs:

```bash
python data_provider/prepare_pvod.py \
  --input-root ./dataset --output-root ./dataset/processed --parties 7
```

This averages rows sharing a timestamp, intersects timestamps across all party
and merged files, and writes aligned CSVs under `dataset/processed/`. Training
and online scripts consume that generated directory.

Each local CSV must contain a `date` column first and an `OT` target column
last. All local files must have the same feature order. The merged cloud CSV
contains the per-site feature blocks followed by one shared `OT` column.

## ST-EVCDP

- Official repository: https://github.com/IntelligentSystemsLab/ST-EVCDP

Apply the same party/merge convention after converting the source data into a
regular time grid.

## Data checks

Before training, verify that:

1. timestamps are parseable and non-decreasing; observation gaps are allowed;
2. numeric columns do not contain `NaN` or infinite values;
3. `enc_in`, `dec_in`, and `c_out` match the local CSV dimension;
4. `cloud_dim = (edge_dim - 1) * number_of_parties + 1` in split mode.

Training fits scalers only on chronological training rows. Online inference must
reuse the saved cloud or per-party scaler via `--scaler_path`; it must not fit a
new scaler on the inference window.
