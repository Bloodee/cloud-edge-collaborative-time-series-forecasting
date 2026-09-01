import unittest
from types import SimpleNamespace

try:
    import torch
    from models import CNN, TimesNet
except ImportError as exc:  # Allows repository checks without the ML environment.
    torch = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


@unittest.skipIf(torch is None, f'PyTorch stack unavailable: {IMPORT_ERROR}')
class ForecastModelTests(unittest.TestCase):
    @staticmethod
    def config(model):
        return SimpleNamespace(
            task_name='long_term_forecast', seq_len=12, label_len=12,
            pred_len=6, enc_in=3, dec_in=3, c_out=3, d_model=8,
            d_ff=16, e_layers=2, top_k=20, num_kernels=2,
            embed='timeF', freq='h', dropout=0.0, model=model,
        )

    def test_cnn_output_and_input_guard(self):
        model = CNN.Model(self.config('CNN'))
        output = model(torch.randn(2, 12, 3), None, None, None)
        self.assertEqual(tuple(output.shape), (2, 6, 3))
        with self.assertRaises(ValueError):
            model(torch.randn(2, 11, 3), None, None, None)

    def test_timesnet_caps_top_k_and_handles_constant_input(self):
        model = TimesNet.Model(self.config('TimesNet'))
        output = model(torch.ones(2, 12, 3), None, None, None)
        self.assertEqual(tuple(output.shape), (2, 6, 3))
        self.assertTrue(torch.isfinite(output).all())


if __name__ == '__main__':
    unittest.main()
