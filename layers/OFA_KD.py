"""
OFA-KD core module for time-series forecasting.

Reference: Hao et al., "One-for-All: Bridge the Gap Between Heterogeneous
Architectures in Knowledge Distillation", NeurIPS 2023.
https://github.com/Hao840/OFAKD

适配点 (相对原版):
  - 原版分类任务用 softmax + target one-hot mask, 这里改成回归
  - "Adaptive Target Enhancement" 用 teacher 在每个 (timestep, var) 上的归一化误差
    倒数作为权重: teacher 越靠谱的位置 student 学习信号越强
  - "Generic Projector" 先做通道 MLP, 再把历史时间轴映射到预测时间轴,
    将 [B, T, d_model] 投到 [B, pred_len, c_out] (prediction space)
    丢掉架构特异性, 让 TimesNet 2D 周期表征 vs CNN 1D 卷积表征可以对齐
"""
import torch
import torch.nn as nn


# ---------------------------------------------------------------- #
# 1. Generic Projector
# ---------------------------------------------------------------- #
class OFAProjector(nn.Module):
    """
    把单个 stage 的中间特征投影到预测空间.
    输入 : [B, T_total, d_model]   (T_total = seq_len + pred_len)
    输出 : [B, pred_len, target_c]
    """
    def __init__(self, d_model, target_c, pred_len, input_len=None,
                 hidden=None, dropout=0.1):
        super().__init__()
        hidden = hidden or d_model
        self.pred_len = pred_len
        self.input_len = input_len
        self.channel_proj = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, target_c),
        )
        self.temporal_proj = (
            nn.Linear(input_len, pred_len) if input_len is not None else None)

    def forward(self, feat):
        if feat.ndim != 3:
            raise ValueError(f'OFAProjector expects [B, T, D], got {tuple(feat.shape)}')
        if self.temporal_proj is not None:
            if feat.size(1) != self.input_len:
                raise ValueError(
                    f'Expected feature length {self.input_len}, got {feat.size(1)}.')
            projected = self.channel_proj(feat).transpose(1, 2)
            return self.temporal_proj(projected).transpose(1, 2)
        if feat.size(1) < self.pred_len:
            raise ValueError(
                f'Feature length {feat.size(1)} is shorter than pred_len {self.pred_len}.')
        # Backward-compatible path for stages that already expose future slots.
        feat = feat[:, -self.pred_len:, :]
        return self.channel_proj(feat)


# ---------------------------------------------------------------- #
# 2. OFA loss (regression version)
# ---------------------------------------------------------------- #
def ofa_regression_loss(stu_pred, tea_pred, gt, eps=1.5):
    """
    Adaptive target enhancement for regression.

    Args:
        stu_pred : [B, pred_len, C]   学生预测 (或 stage 投影输出)
        tea_pred : [B, pred_len, C]   教师预测 (或 stage 投影输出)
        gt       : [B, pred_len, C]   ground truth
        eps      : 0  -> 等权 MSE 蒸馏
                   >0 -> teacher 越准的位置权重越高 (聚焦)
                   论文常用 1.0 ~ 2.0
    Returns: 标量 loss
    """
    _validate_regression_tensors(stu_pred, tea_pred, gt, eps)
    with torch.no_grad():
        tea_err = (tea_pred - gt).abs()
        max_err = tea_err.amax(dim=1, keepdim=True) + 1e-6   # [B, 1, C]
        tea_conf = 1.0 - tea_err / max_err                    # ∈ [0, 1]
        weight = tea_conf.pow(eps)
    return ((stu_pred - tea_pred).pow(2) * weight).mean()


def ofa_reverse_regression_loss(stage_pred, teacher_pred, gt, eps=1.5):
    """Pull the cloud prediction toward reliable edge-stage predictions."""
    _validate_regression_tensors(stage_pred, teacher_pred, gt, eps)
    with torch.no_grad():
        edge_err = (stage_pred.detach() - gt).abs()
        max_err = edge_err.amax(dim=1, keepdim=True) + 1e-6
        edge_confidence = (1.0 - edge_err / max_err).pow(eps)
    return ((teacher_pred - stage_pred.detach()).pow(2) * edge_confidence).mean()


def _validate_regression_tensors(first, second, gt, eps):
    if first.shape != second.shape or first.shape != gt.shape:
        raise ValueError(
            'OFA regression tensors must have identical shapes; got '
            f'{tuple(first.shape)}, {tuple(second.shape)}, {tuple(gt.shape)}.')
    if first.ndim != 3:
        raise ValueError(f'OFA regression tensors must be [B, T, C], got {first.ndim}D.')
    if eps < 0:
        raise ValueError('eps must be non-negative.')


# ---------------------------------------------------------------- #
# 3. Hook-based feature collector
# ---------------------------------------------------------------- #
class StageFeatureBuffer:
    """
    给 model.model (ModuleList) 注册 forward hook, 用 "覆盖式" slot 存放每个 block 的输出.
    每次 forward 自动覆盖, 不需要手动 clear.

    用法:
        buf = StageFeatureBuffer(model)        # 注册
        out = model(x, ...)                    # forward, hook 自动写入
        feat_i = buf.features[i]               # 拿第 i 个 block 的输出
        buf.remove()                           # 训练结束后释放
    """
    def __init__(self, model, required=True):
        # 兼容 DataParallel
        m = model.module if isinstance(model, nn.DataParallel) else model
        if not hasattr(m, 'model') or not isinstance(m.model, nn.ModuleList):
            if required:
                raise ValueError('model.model must be nn.ModuleList for stage-level OFA-KD.')
            self.n = 0
            self.features = []
            self._handles = []
            return
        self.n = len(m.model)
        self.features = [None] * self.n
        self._handles = []
        for i, blk in enumerate(m.model):
            self._handles.append(blk.register_forward_hook(self._make_hook(i)))

    def _make_hook(self, i):
        def _hook(module, inputs, output):
            # 学生支持 use_aux_kd 时, blk 输出可能是 tuple, 取主输出
            if isinstance(output, (tuple, list)):
                self.features[i] = output[0]
            else:
                self.features[i] = output
        return _hook

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []
