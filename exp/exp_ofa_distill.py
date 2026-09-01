"""
OFA-KD on top of Exp_Distill.

设计原则:
    - 子类继承 Exp_Distill, 不改父类一行代码
    - --use_ofa_kd 开关: 关闭时退化成原始 Exp_Distill 行为
    - 完整支持 Compression / Split 模式和 per-party 切片
    - OFA loss 作为额外项加在原 loss 上, 不替换原有 hard/soft 蒸馏

最终训练 loss:
    L = (lambda_kd * MSE(s, gt) + (1-lambda_kd) * SmoothL1(s, t))    # 原 Exp_Distill loss
      + ofa_loss_weight * mean_i OFA_i(stu_proj_i, tea_proj_i, gt)   # 新增: stage 级
      + ofa_final_weight * OFA(s, t, gt)                              # 新增: 输出级 (eps 增强)
"""
import torch
import torch.nn as nn

from exp.exp_distill import Exp_Distill
from layers.OFA_KD import OFAProjector, ofa_regression_loss, StageFeatureBuffer


class Exp_OFA_Distill(Exp_Distill):
    """
    OFA-KD 蒸馏实验类.
    通过 args.use_ofa_kd 开关; 关闭时与 Exp_Distill 完全等价.
    """

    def __init__(self, args):
        super().__init__(args)
        # ---- OFA-KD 开关与超参 ----
        self.use_ofa_kd       = getattr(args, 'use_ofa_kd', False)
        self.ofa_eps          = getattr(args, 'ofa_eps', 1.5)
        self.ofa_loss_weight  = getattr(args, 'ofa_loss_weight', 1.0)
        self.ofa_final_weight = getattr(args, 'ofa_final_weight', 1.0)

        self.stu_buf   = None     # StageFeatureBuffer for student
        self.stu_projs = None     # ModuleList of OFAProjector (student side)
        self.stu_stages = []      # 学生各 stage 索引

        # 学生侧的 hook + projector 现在就可以装 (self.model 由父类已构建)
        if self.use_ofa_kd:
            self._setup_student_side()

    def distill(self, setting):
        """Run distillation and always release stage hooks afterwards."""
        try:
            return super().distill(setting)
        finally:
            if self.stu_buf is not None:
                self.stu_buf.remove()

    # ================================================================ #
    # Student-side: hooks + projectors
    # ================================================================ #
    def _setup_student_side(self):
        """给 student 注册 hook, 初始化 student projectors."""
        try:
            self.stu_buf = StageFeatureBuffer(self.model)
        except ValueError as e:
            print(f'   ⚠️  [OFA-KD] Student does not expose model.ModuleList ({e}); '
                  f'OFA-KD disabled for student-side stage features.')
            self.use_ofa_kd = False
            return

        n_stu = self.stu_buf.n
        self.stu_stages = list(range(n_stu))
        # 学生每个 stage 一个 projector: d_model -> enc_in (= 学生最终输出维度)
        self.stu_projs = nn.ModuleList([
            OFAProjector(
                self.args.d_model, self.args.enc_in, self.args.pred_len,
                input_len=self.args.seq_len)
            for _ in self.stu_stages
        ]).to(self.device)
        print(f'   [OFA-KD] Student stages = {n_stu}, '
              f'projector: d_model={self.args.d_model} -> c_out={self.args.enc_in}')

    # ================================================================ #
    # Optimizer: 加上 projector 参数
    # ================================================================ #
    def _select_optimizer(self):
        params = list(self.model.parameters())
        if self.use_ofa_kd:
            if self.stu_projs is not None:
                params += list(self.stu_projs.parameters())
        return torch.optim.Adam(params, lr=self.args.learning_rate)

    # ================================================================ #
    # Per-party 切片 (复用父类 _get_slice 的语义)
    # ================================================================ #
    def _slice_per_party(self, full_tensor, party_idx, feat_per_party, num_ot, full_dim):
        """
        从 full output [B, pred_len, full_dim] 中取出当前 party 对应的 [feat | ot] 切片.
        返回: [B, pred_len, feat_per_party + num_ot]
        """
        if self.distill_mode == 'Compression':
            # full_dim == enc_in, 直接取 [:-num_ot] 作为 feat, [-num_ot:] 作为 ot
            feats = full_tensor[:, :, :full_dim - num_ot]
            ots   = full_tensor[:, :, full_dim - num_ot:]
        else:
            t_start = party_idx * feat_per_party
            t_end   = (party_idx + 1) * feat_per_party
            feats = full_tensor[:, :, t_start:t_end]
            ots   = full_tensor[:, :, full_dim - num_ot:]   # OT 永远在最后
        return feats, ots

    # ================================================================ #
    # 重写蒸馏 loss: 在父类基础上叠加 OFA-KD loss
    # ================================================================ #
    def _compute_distill_loss(self, s_out, gt_y, t_out, party_idx,
                               feat_per_party, num_ot, criterion, distill_loss_fn):
        # 1) 父类原有蒸馏 loss (hard MSE + soft SmoothL1)
        base_loss = super()._compute_distill_loss(
            s_out, gt_y, t_out, party_idx,
            feat_per_party, num_ot, criterion, distill_loss_fn)

        if not self.use_ofa_kd:
            return base_loss

        # 2) 准备 ground-truth slice (per-party, 仅 pred_len 部分)
        gt_feats = self._get_slice(gt_y, party_idx, feat_per_party, num_ot, 'feature')
        gt_feats = gt_feats[:, -self.args.pred_len:, :]                 # [B, P, feat_pp]
        gt_ot    = self._get_slice(gt_y, party_idx, feat_per_party, num_ot, 'ot')
        gt_ot    = gt_ot[:, -self.args.pred_len:, :]                    # [B, P, num_ot]

        # 3) 输出层 OFA loss: 只对 feature 部分做 (OT 单独评估)
        s_out_feats = s_out[:, :, :-num_ot]                              # 学生本来就是 enc_in 维
        if self.distill_mode == 'Compression':
            t_out_feats = t_out[:, :, :-num_ot]
        else:
            t_out_feats = t_out[:, :, party_idx * feat_per_party:(party_idx + 1) * feat_per_party]
        final_ofa = ofa_regression_loss(
            s_out_feats, t_out_feats.detach(), gt_feats, eps=self.ofa_eps)

        # 4) Stage 级 OFA loss: 利用 hook 收集到的中间特征 + projector 投影
        stage_loss = s_out_feats.new_zeros(())
        if self.stu_projs is not None:
            valid_stages = 0
            for i, s_idx in enumerate(self.stu_stages):
                s_feat = self.stu_buf.features[s_idx]   # [B, T_total, d_stu]
                if s_feat is None:
                    continue

                s_logit = self.stu_projs[i](s_feat)                       # [B, P, enc_in]
                s_logit_feats = s_logit[:, :, :-num_ot]                   # [B, P, feat_pp]

                stage_loss = stage_loss + ofa_regression_loss(
                    s_logit_feats, t_out_feats.detach(), gt_feats, eps=self.ofa_eps)
                valid_stages += 1

            if valid_stages > 0:
                stage_loss = stage_loss / valid_stages

        # 5) 合并: 原 loss + stage OFA + final OFA
        return base_loss \
             + self.ofa_loss_weight * stage_loss \
             + self.ofa_final_weight * final_ofa
