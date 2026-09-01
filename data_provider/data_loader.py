"""Validated chronological CSV loader for CE-BiD experiments."""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from utils.timefeatures import time_features


def load_scaler(path, expected_features=None):
    """Load a fitted scaler if it exists and matches the feature width."""
    if not path or not os.path.exists(path):
        return None
    try:
        scaler = joblib.load(path)
    except Exception as exc:
        raise RuntimeError(f'Failed to load scaler {path}: {exc}') from exc
    if not hasattr(scaler, 'mean_'):
        raise ValueError(f'Scaler is not fitted: {path}')
    if expected_features is not None and len(scaler.mean_) != expected_features:
        raise ValueError(
            f'Scaler feature width mismatch: expected {expected_features}, '
            f'got {len(scaler.mean_)} from {path}.')
    return scaler


class Dataset_Custom(Dataset):
    """Sliding-window dataset with strict time-series and scaling checks."""

    def __init__(self, args, root_path, flag='train', size=None,
                 features='M', data_path='PVOD.csv', target='OT', scale=True,
                 timeenc=0, freq='h', step=0, total_steps=1,
                 specific_start=None, specific_end=None,
                 train_ratio=None, val_ratio=None, test_ratio=None):
        if flag not in {'train', 'val', 'test'}:
            raise ValueError(f"Unsupported split '{flag}'.")
        if size is None or len(size) != 3 or any(int(value) <= 0 for value in size):
            raise ValueError('size must contain positive seq_len, label_len and pred_len.')
        if int(size[1]) > int(size[0]):
            raise ValueError('label_len cannot exceed seq_len.')
        if features not in {'M', 'S', 'MS'}:
            raise ValueError("features must be one of 'M', 'S' or 'MS'.")
        if (specific_start is None) != (specific_end is None):
            raise ValueError('specific_start and specific_end must be provided together.')
        if total_steps <= 0 or step < 0 or step >= total_steps:
            raise ValueError('step must satisfy 0 <= step < total_steps.')

        ratios_supplied = any(
            value is not None for value in (train_ratio, val_ratio, test_ratio))
        if ratios_supplied:
            if train_ratio is None or val_ratio is None:
                raise ValueError('train_ratio and val_ratio must be provided together.')
            ratios = [train_ratio, val_ratio,
                      0.0 if test_ratio is None else test_ratio]
            if any(value < 0 or value > 1 for value in ratios):
                raise ValueError('Split ratios must be within [0, 1].')
            if abs(sum(ratios) - 1.0) > 1e-6:
                raise ValueError('train_ratio + val_ratio + test_ratio must equal 1.')

        self.args = args
        self.seq_len, self.label_len, self.pred_len = map(int, size)
        self.set_type = {'train': 0, 'val': 1, 'test': 2}[flag]
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.step = step
        self.total_steps = total_steps
        self.specific_start = specific_start
        self.specific_end = specific_end
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.root_path = root_path
        self.data_path = data_path
        self._read_data()

    def _read_data(self):
        data_file = os.path.join(self.root_path, self.data_path)
        if not os.path.isfile(data_file):
            raise FileNotFoundError(f'Data file not found: {data_file}')
        frame = pd.read_csv(data_file)
        if frame.empty:
            raise ValueError(f'Data file is empty: {data_file}')

        missing = [name for name in ('date', self.target) if name not in frame.columns]
        if missing:
            raise ValueError(f'{data_file} is missing required columns: {missing}')
        dates = pd.to_datetime(frame['date'], errors='coerce')
        if dates.isna().any():
            raise ValueError(f'{data_file} contains invalid timestamps.')
        if dates.duplicated().any():
            raise ValueError(f'{data_file} contains duplicate timestamps.')
        if not dates.is_monotonic_increasing:
            raise ValueError(f'{data_file} timestamps must be sorted ascending.')
        frame['date'] = dates

        feature_names = [
            name for name in frame.columns if name not in {'date', self.target}]
        frame = frame[['date'] + feature_names + [self.target]]
        numeric_frame = (frame.iloc[:, 1:] if self.features in {'M', 'MS'}
                         else frame[[self.target]])
        non_numeric = [
            name for name in numeric_frame.columns
            if not pd.api.types.is_numeric_dtype(numeric_frame[name])]
        if non_numeric:
            raise ValueError(f'Non-numeric feature columns: {non_numeric}')
        values = numeric_frame.to_numpy(dtype=np.float64, copy=False)
        if not np.isfinite(values).all():
            invalid = int((~np.isfinite(values)).sum())
            raise ValueError(f'{data_file} contains {invalid} NaN or infinite values.')

        window_start, window_end = None, None
        if self.specific_start is not None:
            window_start = max(0, int(self.specific_start))
            window_end = min(len(values), int(self.specific_end))
            if window_start >= window_end:
                raise ValueError(
                    f'Invalid window [{self.specific_start}:{self.specific_end}] '
                    f'for {len(values)} rows.')

        if self.scale:
            scaler_path = getattr(self.args, 'scaler_path', None)
            self.scaler = (
                load_scaler(scaler_path, values.shape[1]) if scaler_path else None)
            if self.scaler is None:
                if window_start is not None:
                    if self.train_ratio is None:
                        raise ValueError(
                            'Pure inference requires a fitted scaler. Run initial '
                            'training first and pass --scaler_path.')
                    train_end = window_start + int(
                        (window_end - window_start) * self.train_ratio)
                    if train_end <= window_start:
                        raise ValueError('The requested window has no scaler-training rows.')
                    fit_values = values[window_start:train_end]
                else:
                    fit_values = values[:max(1, int(len(values) * 0.7))]
                self.scaler = StandardScaler().fit(fit_values)
            data = self.scaler.transform(values)
        else:
            self.scaler = None
            data = values

        stamps = self._time_features(frame[['date']])
        self._select_split(data, stamps, window_start, window_end)

    def _time_features(self, stamp_frame):
        stamp_frame = stamp_frame.copy()
        if self.timeenc == 0:
            stamp_frame['month'] = stamp_frame.date.dt.month
            stamp_frame['day'] = stamp_frame.date.dt.day
            stamp_frame['weekday'] = stamp_frame.date.dt.weekday
            stamp_frame['hour'] = stamp_frame.date.dt.hour
            return stamp_frame.drop(columns=['date']).to_numpy()
        encoded = time_features(stamp_frame['date'].to_numpy(), freq=self.freq)
        return encoded.transpose(1, 0)

    def _select_split(self, data, stamps, window_start, window_end):
        if window_start is not None:
            chunk = data[window_start:window_end]
            chunk_stamps = stamps[window_start:window_end]
            if self.train_ratio is None:
                begin, end = 0, len(chunk)
            else:
                n_train = int(len(chunk) * self.train_ratio)
                n_val = int(len(chunk) * self.val_ratio)
                n_test = len(chunk) - n_train - n_val
                begins = [
                    0,
                    max(0, n_train - self.seq_len),
                    max(0, n_train + n_val - self.seq_len)
                    if n_test > 0 else len(chunk),
                ]
                ends = [n_train, n_train + n_val,
                        len(chunk) if n_test > 0 else len(chunk)]
                begin, end = begins[self.set_type], ends[self.set_type]
            self.data_x = chunk[begin:end]
            self.data_y = chunk[begin:end]
            self.data_stamp = chunk_stamps[begin:end]
            return

        if self.total_steps > 1:
            chunk_size = len(data) // self.total_steps
            begin = self.step * chunk_size
            end = len(data) if self.step == self.total_steps - 1 else begin + chunk_size
        else:
            n_train = int(len(data) * 0.7)
            n_test = int(len(data) * 0.2)
            n_val = len(data) - n_train - n_test
            begins = [0, n_train - self.seq_len,
                      n_train + n_val - self.seq_len]
            ends = [n_train, n_train + n_val, len(data)]
            begin, end = max(0, begins[self.set_type]), ends[self.set_type]
        self.data_x = data[begin:end]
        self.data_y = data[begin:end]
        self.data_stamp = stamps[begin:end]

    def __getitem__(self, index):
        input_begin = index
        input_end = input_begin + self.seq_len
        target_begin = input_end - self.label_len
        target_end = target_begin + self.label_len + self.pred_len
        return (
            self.data_x[input_begin:input_end],
            self.data_y[target_begin:target_end],
            self.data_stamp[input_begin:input_end],
            self.data_stamp[target_begin:target_end],
        )

    def __len__(self):
        return max(0, len(self.data_x) - self.seq_len - self.pred_len + 1)

    def inverse_transform(self, data):
        if self.scaler is None:
            return data
        return self.scaler.inverse_transform(data)
