"""
外挂 aux head: 给"没有内置 aux head 的 student"(如朴素 CNN) 提供
multi-head supervision, 但 aux head 不入
student checkpoint, 只在蒸馏阶段存在.

工作流程:
    1. 训练 setup:
        helper = AuxHeadHelper(student, pred_len, c_out, device)
        # 自动在 student.model[i] (除最后一层) 上注册 forward hook
        # 自动创建对应数量的临时 Linear (作为 aux head)
        optimizer.add_param_group({'params': helper.parameters()})

    2. 训练 forward:
        s_out = student(...)                    # 主预测 (走 student.head)
        s_aux_preds = helper.get_aux_predictions()   # 用 hook 抓的中间特征 + aux head

    3. 训练结束:
        helper.remove()
        torch.save(student.state_dict(), ...)   # 不包含 aux head 参数

要求:
    student.model 是 nn.ModuleList (CNN/TimesNet 等都满足);
    若不满足, helper.active = False, get_aux_predictions() 返回 []
"""

import torch
import torch.nn as nn


class AuxHeadHelper:
    """
    Args:
        student: 学生模型. 要求 student.model 是 nn.ModuleList,
                 且 student.seq_len, student.d_model 可读.
        pred_len: 预测长度
        c_out: 输出维度
        device: torch.device
        n_aux_max: aux head 数量上限 (默认 None = e_layers - 1)
    """
    def __init__(self, student, pred_len, c_out, device, n_aux_max=None):
        # 处理 DataParallel 包装
        m = student.module if isinstance(student, nn.DataParallel) else student

        self.active = False
        self._handles = []

        if not (hasattr(m, 'model') and isinstance(m.model, nn.ModuleList)):
            print("  [AuxHeadHelper] student.model 不是 ModuleList; 禁用 aux head")
            return

        L = len(m.model)
        if L < 2:
            print(f"  [AuxHeadHelper] student 只有 {L} 个 block; 没有中间层, 禁用")
            return

        # 抓输入维度: aux head 接 [B, seq_len, d_model] -> Linear -> [B, pred_len*c_out]
        seq_len = getattr(m, 'seq_len', None)
        d_model = getattr(m, 'd_model', None)
        if seq_len is None or d_model is None:
            print("  [AuxHeadHelper] 找不到 student.seq_len 或 student.d_model; 禁用")
            return

        # 决定多少个 aux head: 默认是 e_layers - 1 (最后一层 block 由 main head 出预测)
        n_aux = L - 1 if n_aux_max is None else min(L - 1, n_aux_max)

        self.aux_heads = nn.ModuleList([
            nn.Linear(seq_len * d_model, pred_len * c_out).to(device)
            for _ in range(n_aux)
        ])
        self.pred_len = pred_len
        self.c_out = c_out

        # 给前 n_aux 个 block 注册 hook (这些 block 输出会被 aux head 取走)
        self._features = [None] * n_aux
        for i in range(n_aux):
            h = m.model[i].register_forward_hook(self._make_hook(i))
            self._handles.append(h)

        self.active = True
        print(f"  [AuxHeadHelper] 启用 {n_aux} 个外挂 aux head "
              f"(input_dim = seq_len*d_model = {seq_len * d_model}, "
              f"output_dim = pred_len*c_out = {pred_len * c_out})")

    def _make_hook(self, i):
        def _hook(module, inputs, output):
            if isinstance(output, (tuple, list)):
                self._features[i] = output[0]
            else:
                self._features[i] = output
        return _hook

    def get_aux_predictions(self):
        """
        在 student forward 之后调用. 返回 list of [B, pred_len, c_out].
        每个元素是该层 block 输出经 aux head 出的预测.
        """
        if not self.active:
            return []
        preds = []
        for i, feat in enumerate(self._features):
            if feat is None:
                continue
            B = feat.shape[0]
            flat = feat.flatten(1)                              # [B, T*D]
            out = self.aux_heads[i](flat)                       # [B, P*C]
            preds.append(out.view(B, self.pred_len, self.c_out))
        return preds

    def parameters(self):
        if not self.active:
            return iter([])
        return self.aux_heads.parameters()

    def train(self):
        if self.active:
            self.aux_heads.train()

    def eval(self):
        if self.active:
            self.aux_heads.eval()

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []
        self.active = False
