"""Compact 1D-CNN edge model used in the cloud-edge experiments.

Intermediate blocks are exposed through ``self.model`` so the external
``AuxHeadHelper`` and OFA-KD hooks can reuse them without changing inference.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNBlock(nn.Module):
    """最朴素的 1D conv block: Conv1d -> ReLU. 输入输出形状均为 [B, T, C]."""
    def __init__(self, channels, kernel_size=3):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv1d(channels, channels,
                              kernel_size=kernel_size,
                              padding=kernel_size // 2)

    def forward(self, x):
        x = x.transpose(1, 2)              # [B, C, T] for Conv1d
        x = F.relu(self.conv(x))
        return x.transpose(1, 2)           # back to [B, T, C]


class Model(nn.Module):
    """
    Naive 1D-CNN baseline.

        x_enc: [B, seq_len, enc_in]
            -> 1x1 Conv (enc_in -> d_model)
            -> e_layers 层 [Conv1d(k=3) + ReLU]
            -> Flatten + Linear -> [B, pred_len, c_out]
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        if self.task_name != 'long_term_forecast':
            raise ValueError(
                f"CNN supports long_term_forecast only, got '{self.task_name}'.")
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out
        self.d_model = configs.d_model

        # 1x1 conv: enc_in -> d_model 通道适配
        self.input_proj = nn.Conv1d(configs.enc_in, configs.d_model, kernel_size=1)

        # 主干: 一连串朴素 Conv+ReLU. self.model 名为兼容蒸馏 hook.
        self.model = nn.ModuleList([
            CNNBlock(configs.d_model, kernel_size=3)
            for _ in range(configs.e_layers)
        ])

        self.head = nn.Linear(configs.d_model * configs.seq_len,
                              configs.pred_len * configs.c_out)

    def _encode(self, x_enc):
        """[B, seq_len, enc_in] -> [B, seq_len, d_model]"""
        if x_enc.ndim != 3:
            raise ValueError(f'CNN expects [B, T, C], got {tuple(x_enc.shape)}')
        if x_enc.size(1) != self.seq_len:
            raise ValueError(
                f'Expected seq_len={self.seq_len}, got {x_enc.size(1)}.')
        if x_enc.size(2) != self.enc_in:
            raise ValueError(
                f'Expected enc_in={self.enc_in}, got {x_enc.size(2)}.')
        h = x_enc.transpose(1, 2)
        h = self.input_proj(h)
        h = h.transpose(1, 2)
        for block in self.model:
            h = block(h)
        return h

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        h = self._encode(x_enc)
        h = h.flatten(1)
        out = self.head(h)
        return out.view(-1, self.pred_len, self.c_out)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
