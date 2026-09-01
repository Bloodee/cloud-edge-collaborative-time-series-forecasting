import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1


def FFT_for_Period(x, k=2):
    """Return dominant non-DC periods and their per-sample amplitudes."""
    if x.ndim != 3 or x.size(1) < 2:
        raise ValueError(f'FFT_for_Period expects [B, T>=2, C], got {tuple(x.shape)}')
    if k <= 0:
        raise ValueError('top_k must be positive.')
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    frequency_list = abs(xf).mean(0).mean(-1)
    non_dc = frequency_list[1:]
    effective_k = min(k, non_dc.numel())
    _, top_list = torch.topk(non_dc, effective_k)
    top_list = top_list + 1
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels)
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(len(period_list)):
            period = period_list[i]
            if (self.seq_len + self.pred_len) % period != 0:
                length = (((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
            out = self.conv(out)
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        res = res + x
        return res


class Model(nn.Module):
    """
    TimesNet - 支持可选的中间特征输出（用于特征级蒸馏）
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out
        if self.c_out != self.enc_in:
            raise ValueError(
                'This TimesNet normalization path requires c_out == enc_in; '
                f'got c_out={self.c_out}, enc_in={self.enc_in}.')

        # 编码器嵌入
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)

        # TimesNet 堆叠层
        self.model = nn.ModuleList([TimesBlock(configs) for _ in range(configs.e_layers)])
        self.layer_norm = nn.LayerNorm(configs.d_model)

        # 预测头
        self.predict_linear = nn.Linear(self.seq_len, self.pred_len + self.seq_len)
        self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, return_features=False):
        """
        Args:
            return_features: 如果为True，额外返回encoder最后一层的特征表示
                             用于特征级反向蒸馏
        Returns:
            如果 return_features=False: [B, pred_len, c_out] (默认行为，完全兼容)
            如果 return_features=True:  ([B, pred_len, c_out], [B, pred_len, d_model])
        """
        if x_enc.ndim != 3:
            raise ValueError(f'TimesNet expects [B, T, C], got {tuple(x_enc.shape)}')
        if x_enc.size(1) != self.seq_len or x_enc.size(2) != self.enc_in:
            raise ValueError(
                f'Expected [B, {self.seq_len}, {self.enc_in}], got {tuple(x_enc.shape)}.')

        # Normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc.sub(means)
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc.div(stdev)

        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B, T, d_model]

        # 时间维度扩展
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(0, 2, 1)

        # TimesNet 处理
        for i in range(len(self.model)):
            enc_out = self.layer_norm(self.model[i](enc_out))

        # ✅ 此处 enc_out 是 encoder 的最终特征: [B, seq_len+pred_len, d_model]
        # 取预测部分的特征用于蒸馏
        feature_repr = enc_out[:, -self.pred_len:, :]  # [B, pred_len, d_model]

        # 投影回原始维度
        dec_out = self.projection(enc_out)

        # De-Normalization
        dec_out = dec_out.mul(
            stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out.add(
            means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))

        pred_out = dec_out[:, -self.pred_len:, :]  # [B, pred_len, c_out]

        if return_features:
            return pred_out, feature_repr
        return pred_out
