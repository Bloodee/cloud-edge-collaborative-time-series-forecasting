"""Create timestamp-aligned PVOD CSVs without modifying the raw files."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _load_and_collapse(path):
    if not path.is_file():
        raise FileNotFoundError(f'PVOD input not found: {path}')
    frame = pd.read_csv(path)
    if 'date' not in frame.columns or 'OT' not in frame.columns:
        raise ValueError(f'{path} must contain date and OT columns.')
    if frame.columns[0] != 'date' or frame.columns[-1] != 'OT':
        raise ValueError(f'{path} must place date first and OT last.')

    dates = pd.to_datetime(frame['date'], errors='coerce')
    if dates.isna().any():
        raise ValueError(f'{path} contains invalid timestamps.')
    numeric = frame.drop(columns='date').apply(pd.to_numeric, errors='coerce')
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError(f'{path} contains non-numeric, NaN, or infinite values.')

    clean = pd.concat([dates.rename('date'), numeric], axis=1)
    duplicate_rows = int(clean['date'].duplicated(keep=False).sum())
    # Multiple observations labelled with the same instant cannot be aligned
    # across sites. Averaging preserves their scale without choosing one row.
    clean = clean.groupby('date', as_index=False, sort=True).mean(numeric_only=True)
    return clean, duplicate_rows


def prepare_dataset(input_root, output_root, parties=7, data_name='PVOD.csv'):
    """Align party and merged CSVs on their timestamp intersection."""
    input_root = Path(input_root).resolve()
    output_root = Path(output_root).resolve()
    if input_root == output_root:
        raise ValueError('input_root and output_root must be different.')
    if parties <= 0:
        raise ValueError('parties must be positive.')

    inputs = {
        f'party_{index}': input_root / f'party_{index}' / data_name
        for index in range(1, parties + 1)
    }
    inputs['merge'] = input_root / 'merge' / data_name

    frames = {}
    duplicates = {}
    for name, path in inputs.items():
        frames[name], duplicates[name] = _load_and_collapse(path)

    edge_widths = {len(frames[f'party_{i}'].columns) - 1
                   for i in range(1, parties + 1)}
    if len(edge_widths) != 1:
        raise ValueError(f'Party feature widths differ: {sorted(edge_widths)}')
    edge_dim = edge_widths.pop()
    cloud_dim = len(frames['merge'].columns) - 1
    expected_cloud_dim = parties * (edge_dim - 1) + 1
    if cloud_dim != expected_cloud_dim:
        raise ValueError(
            f'Merged width is {cloud_dim}; expected {expected_cloud_dim} from '
            f'{parties} parties and edge width {edge_dim}.')

    common = pd.Index(frames['party_1']['date'])
    for frame in frames.values():
        common = common.intersection(pd.Index(frame['date']), sort=False)
    common = common.sort_values()
    if len(common) == 0:
        raise ValueError('Party and merged files have no common timestamps.')

    summary = {}
    for name, frame in frames.items():
        aligned = frame.set_index('date').loc[common].reset_index()
        destination = output_root / name / data_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        aligned.to_csv(destination, index=False, date_format='%Y-%m-%d %H:%M:%S')
        summary[name] = {
            'input_rows': len(pd.read_csv(inputs[name], usecols=['date'])),
            'duplicate_rows': duplicates[name],
            'output_rows': len(aligned),
        }

    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', default='./dataset')
    parser.add_argument('--output-root', default='./dataset/processed')
    parser.add_argument('--parties', type=int, default=7)
    parser.add_argument('--data-name', default='PVOD.csv')
    args = parser.parse_args()
    summary = prepare_dataset(
        args.input_root, args.output_root, args.parties, args.data_name)
    print('PVOD alignment summary:')
    for name, values in summary.items():
        print(
            f"  {name}: input={values['input_rows']}, "
            f"duplicate_rows={values['duplicate_rows']}, "
            f"output={values['output_rows']}"
        )


if __name__ == '__main__':
    main()
