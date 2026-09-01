"""Shared experiment setup and device/model selection."""

import os

import torch

from models import CNN, TimesNet


class Exp_Basic(object):
    """Base class used by all forecasting and distillation experiments."""

    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'TimesNet': TimesNet,
            'CNN': CNN,
        }
        if args.model not in self.model_dict:
            available = ', '.join(sorted(self.model_dict))
            raise ValueError(f"Unknown model '{args.model}'. Available models: {available}")
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError

    def _acquire_device(self):
        """Select an available accelerator and fall back to CPU safely."""
        wants_gpu = bool(self.args.use_gpu)

        if wants_gpu and self.args.gpu_type == 'cuda' and torch.cuda.is_available():
            if self.args.use_multi_gpu:
                os.environ["CUDA_VISIBLE_DEVICES"] = self.args.devices
                self.device_ids = [int(item) for item in self.args.devices.split(',')]
                if not self.device_ids:
                    raise ValueError('At least one CUDA device id is required for multi-GPU mode.')
                main_gpu = self.device_ids[0]
                device = torch.device(f'cuda:{main_gpu}')
                print(f'Use Multi GPU: {self.args.devices}, main: cuda:{main_gpu}')
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
                device = torch.device(f'cuda:{self.args.gpu}')
                print(f'Use GPU: cuda:{self.args.gpu}')
        elif (wants_gpu and self.args.gpu_type == 'mps'
              and hasattr(torch.backends, 'mps')
              and torch.backends.mps.is_available()):
            device = torch.device('mps')
            print('Use GPU: mps')
        else:
            if wants_gpu:
                print(f"Requested {self.args.gpu_type} accelerator is unavailable; falling back to CPU.")
            self.args.use_gpu = 0
            self.args.use_multi_gpu = False
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        raise NotImplementedError

    def vali(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError
