import unittest

try:
    import torch
    from layers.OFA_KD import OFAProjector, ofa_regression_loss
except ImportError as exc:
    torch = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


@unittest.skipIf(torch is None, f'PyTorch unavailable: {IMPORT_ERROR}')
class OfaRegressionTests(unittest.TestCase):
    def test_projector_shape(self):
        projector = OFAProjector(
            d_model=8, target_c=3, pred_len=6, input_len=12, dropout=0)
        output = projector(torch.randn(2, 12, 8))
        self.assertEqual(tuple(output.shape), (2, 6, 3))

    def test_identical_predictions_have_zero_loss(self):
        values = torch.randn(2, 6, 3)
        loss = ofa_regression_loss(values, values, torch.zeros_like(values))
        self.assertEqual(loss.item(), 0.0)

    def test_shape_mismatch_fails(self):
        with self.assertRaises(ValueError):
            ofa_regression_loss(
                torch.zeros(1, 6, 3), torch.zeros(1, 5, 3),
                torch.zeros(1, 6, 3))


if __name__ == '__main__':
    unittest.main()
