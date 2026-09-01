import csv
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RepositoryContractTests(unittest.TestCase):
    def test_reported_mse_reductions(self):
        with (ROOT / 'docs/results/benchmark.csv').open(encoding='utf-8') as handle:
            rows = list(csv.DictReader(handle))
        by_dataset = {}
        for row in rows:
            by_dataset.setdefault(row['dataset'], {})[row['method']] = float(row['mse'])
        expected = {'PVOD': 3.05, 'ST-EVCDP': 16.08}
        for dataset, expected_percent in expected.items():
            methods = by_dataset[dataset]
            baseline = next(value for name, value in methods.items() if 'baseline' in name)
            proposed = methods['CE-BiD']
            reduction = (baseline - proposed) / baseline * 100
            self.assertAlmostEqual(reduction, expected_percent, places=2)

    def test_main_source_has_no_private_absolute_paths(self):
        source_dirs = ['data_provider', 'exp', 'layers', 'models', 'utils']
        files = [ROOT / 'run.py']
        for name in source_dirs:
            files.extend((ROOT / name).glob('*.py'))
        for path in files:
            content = path.read_text(encoding='utf-8')
            self.assertNotIn('C:/Users/', content, path.as_posix())
            self.assertNotIn('C:\\Users\\', content, path.as_posix())

    def test_only_main_models_are_public(self):
        model_files = {path.name for path in (ROOT / 'models').glob('*.py')}
        self.assertEqual(model_files, {'__init__.py', 'CNN.py', 'TimesNet.py'})


if __name__ == '__main__':
    unittest.main()
