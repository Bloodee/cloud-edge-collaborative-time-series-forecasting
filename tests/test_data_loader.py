import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

try:
    import numpy as np
    import pandas as pd
    from data_provider.data_loader import Dataset_Custom
except ImportError as exc:
    Dataset_Custom = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


@unittest.skipIf(Dataset_Custom is None, f'Data stack unavailable: {IMPORT_ERROR}')
class DatasetCustomTests(unittest.TestCase):
    def _write_csv(self, directory, dates):
        rows = len(dates)
        frame = pd.DataFrame({
            'date': dates,
            'feature': np.arange(rows, dtype=float),
            'OT': np.arange(rows, dtype=float) * 2,
        })
        frame.to_csv(Path(directory) / 'sample.csv', index=False)

    def _dataset(self, directory, **overrides):
        args = SimpleNamespace(scaler_path=str(Path(directory) / 'scaler.pkl'))
        options = dict(
            args=args, root_path=directory, flag='train', size=[4, 4, 2],
            features='M', data_path='sample.csv', target='OT', timeenc=0,
            freq='h', specific_start=0, specific_end=40,
            train_ratio=0.5, val_ratio=0.25, test_ratio=0.25,
        )
        options.update(overrides)
        return Dataset_Custom(**options)

    def test_scaler_fits_only_requested_training_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            dates = pd.date_range('2024-01-01', periods=40, freq='h')
            self._write_csv(directory, dates)
            dataset = self._dataset(directory)
            np.testing.assert_allclose(dataset.scaler.mean_, [9.5, 19.0])
            self.assertGreater(len(dataset), 0)

    def test_accepts_forward_gaps_in_observation_times(self):
        with tempfile.TemporaryDirectory() as directory:
            dates = list(pd.date_range('2024-01-01', periods=40, freq='h'))
            dates[20] = dates[19] + pd.Timedelta(minutes=90)
            self._write_csv(directory, dates)
            dataset = self._dataset(directory)
            self.assertGreater(len(dataset), 0)


if __name__ == '__main__':
    unittest.main()
