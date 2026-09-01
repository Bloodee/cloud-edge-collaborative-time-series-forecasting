import tempfile
import unittest
from pathlib import Path

import pandas as pd

from data_provider.prepare_pvod import prepare_dataset


class PreparePvodTests(unittest.TestCase):
    def test_collapses_duplicates_and_intersects_timestamps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw, output = root / 'raw', root / 'processed'
            dates = pd.date_range('2024-01-01', periods=4, freq='15min')
            party_1 = pd.DataFrame({
                'date': [dates[0], dates[1], dates[1], dates[2], dates[3]],
                'feature': [1.0, 2.0, 4.0, 5.0, 6.0],
                'OT': [1.0, 2.0, 4.0, 5.0, 6.0],
            })
            party_2 = pd.DataFrame({
                'date': dates, 'feature': [7.0, 8.0, 9.0, 10.0],
                'OT': [7.0, 8.0, 9.0, 10.0],
            })
            merge = pd.DataFrame({
                'date': [dates[0], dates[1], dates[3]],
                'feature_p1': [1.0, 2.0, 3.0],
                'feature_p2': [4.0, 5.0, 6.0],
                'OT': [1.0, 2.0, 3.0],
            })
            for name, frame in {
                    'party_1': party_1, 'party_2': party_2, 'merge': merge}.items():
                destination = raw / name / 'PVOD.csv'
                destination.parent.mkdir(parents=True)
                frame.to_csv(destination, index=False)

            summary = prepare_dataset(raw, output, parties=2)
            aligned = pd.read_csv(output / 'party_1/PVOD.csv')
            self.assertEqual(len(aligned), 3)
            self.assertEqual(aligned.loc[1, 'feature'], 3.0)
            self.assertEqual(summary['party_1']['duplicate_rows'], 2)
            self.assertEqual(
                pd.read_csv(output / 'merge/PVOD.csv')['date'].tolist(),
                aligned['date'].tolist())


if __name__ == '__main__':
    unittest.main()
