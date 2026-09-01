"""
反向蒸馏 + OFA-KD (异构多边端 -> 云端教师感知)

OFA-KD 在反向蒸馏里的三个增强项 (use_ofa_kd 开关控制):
  1. 多 stage hooks: 给 teacher 和每个 student 注册 forward hook,
     抓取 self.model(ModuleList) 中每个 block 的输出
  2. Per-point 自适应增强权重: 反向蒸馏的方向 = 学生越靠谱的点 teacher
     越听学生的, weight = (1 - stu_err / max_err) ^ eps  (per-party gt 计算)
  3. gt anchor: stage projector 同时被 gt 监督, 防止"teacher 投影互相塌陷"

最终 loss = α * edge_soft_kd + (1-α) * ground_truth_loss
         + ofa_loss_weight * stage_ofa_loss
         + weight_reg
"""

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
import torch
import torch.nn as nn
from torch import optim
import os
import numpy as np
import copy
from sklearn.metrics import mean_absolute_error
from layers.OFA_KD import (
    OFAProjector,
    StageFeatureBuffer,
    ofa_reverse_regression_loss,
)

class Exp_ReverseDistill(Exp_Long_Term_Forecast):
    def __init__(self, args):
        self.teacher_configs = copy.deepcopy(args)
        self.student_configs = copy.deepcopy(args)

        # --- Teacher Configs (Cloud, Large TimesNet) ---
        self.teacher_configs.model = 'TimesNet'
        self.teacher_configs.d_model = getattr(args, 'teacher_d_model', 128)
        self.teacher_configs.d_ff = getattr(args, 'teacher_d_ff', 256)
        self.teacher_configs.e_layers = getattr(args, 'teacher_e_layers', 2)
        self.teacher_configs.top_k = getattr(args, 'top_k', 5)
        self.teacher_configs.num_kernels = getattr(args, 'num_kernels', 6)

        cloud_dim = getattr(args, 'cloud_dim', args.enc_in)
        self.teacher_configs.enc_in = cloud_dim
        self.teacher_configs.dec_in = cloud_dim
        self.teacher_configs.c_out = cloud_dim

        # --- Student Configs ---
        self.student_configs.model = getattr(args, 'student_model_name', 'TimesNet')
        self.student_configs.d_layers = getattr(args, 'student_d_layers', 1)
        self.student_configs.d_model = getattr(args, 'student_d_model', 16)
        self.student_configs.d_ff = getattr(args, 'student_d_ff', 32)
        self.student_configs.e_layers = getattr(args, 'student_e_layers', 2)
        self.student_configs.top_k = getattr(args, 'student_top_k', args.top_k)
        self.student_configs.num_kernels = getattr(args, 'num_kernels', 6)
        self.student_configs.enc_in = args.enc_in
        self.student_configs.dec_in = args.dec_in
        self.student_configs.c_out = args.c_out

        # --- 超参数 ---
        self.gamma = getattr(args, 'rev_gamma', 1.5)
        self.reverse_kd_weight = getattr(args, 'rev_kd_weight', 0.3)
        self.weight_reg_lambda = getattr(args, 'rev_weight_reg', 0.01)
        self.rollback_threshold = getattr(args, 'rev_rollback_thresh', 1.1)
        self.teacher_initial_weights = None

        # --- OFA-KD 相关 (新增) ---
        self.use_ofa_kd       = getattr(args, 'use_ofa_kd', False)
        self.ofa_eps          = getattr(args, 'ofa_eps', 1.5)
        self.ofa_loss_weight  = getattr(args, 'ofa_loss_weight', 0.5)
        self.ofa_anchor_weight= getattr(args, 'ofa_anchor_weight', 0.3)

        super(Exp_ReverseDistill, self).__init__(args)

    def _build_model(self):
        model = self.model_dict['TimesNet'].Model(self.teacher_configs).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _load_single_student(self, path):
        student_model_name = self.student_configs.model
        student = self.model_dict[student_model_name].Model(self.student_configs).float()
        if path and os.path.exists(path):
            state_dict = torch.load(path, map_location=self.device)
            student.load_state_dict(state_dict, strict=False)
        else:
            raise ValueError(f"Student path not found: {path}")

        student.to(self.device)
        student.eval()
        for param in student.parameters():
            param.requires_grad = False
        return student

    def evaluate_student_performance(self, student_model, data_loader, party_idx, feat_per_party, num_ot):
        """评估学生质量, 用 MAE 倒数作为分数"""
        preds, trues = [], []
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in data_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                start_col = party_idx * feat_per_party
                end_col = start_col + feat_per_party
                s_batch_x = torch.cat([batch_x[:, :, start_col:end_col], batch_x[:, :, -num_ot:]], dim=2)
                s_dec_inp = torch.cat([dec_inp[:, :, start_col:end_col], dec_inp[:, :, -num_ot:]], dim=2)

                outputs = student_model(s_batch_x, batch_x_mark, s_dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]

                s_batch_y = torch.cat([batch_y[:, :, start_col:end_col], batch_y[:, :, -num_ot:]], dim=2)
                s_batch_y = s_batch_y[:, -self.args.pred_len:, f_dim:]

                preds.append(outputs.cpu().numpy())
                trues.append(s_batch_y.cpu().numpy())

        if not preds:
            return 0.0
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        mae = mean_absolute_error(trues.reshape(-1), preds.reshape(-1))
        return 1.0 / (mae + 1e-3)

    def _evaluate_teacher_mse(self, data_loader):
        """评估教师当前 MSE"""
        losses = []
        self.model.eval()
        criterion = nn.MSELoss()
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in data_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                losses.append(criterion(outputs, batch_y).item())
        return np.mean(losses) if losses else float('inf')

    # ============================================================
    # 主训练函数
    # ============================================================
    def update_cloud_teacher(self, setting):
        """
        ========================================================
        特征级反向蒸馏: 边端特征 → 投影对齐 → 云端感知分布变化
        + 可选 OFA-KD: 多 stage hooks + 自适应增强权重
        ========================================================
        """
        print(f"\n>>> [Feature-Level Reverse Distill]")
        print(f"    Data: {self.args.specific_start} -> {self.args.specific_end}")

        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')

        # ============================================
        # 1. 加载教师模型
        # ============================================
        if not (self.args.pretrained_teacher_path and os.path.exists(self.args.pretrained_teacher_path)):
            raise ValueError("教师模型路径不存在")

        state_dict = torch.load(self.args.pretrained_teacher_path, map_location=self.device)
        self.model.load_state_dict(state_dict)

        self.teacher_initial_weights = {
            name: param.clone().detach()
            for name, param in self.model.named_parameters()
        }

        pre_update_mse = self._evaluate_teacher_mse(vali_loader)
        print(f"    [Baseline] Pre-update MSE: {pre_update_mse:.6f}")

        # ============================================
        # 2. 冻结策略: 只训练 projection
        # ============================================
        for name, param in self.model.named_parameters():
            if "projection" in name:
                param.requires_grad = True
                print(f"    -> Trainable: {name}")
            else:
                param.requires_grad = False

        # ============================================
        # 3. 加载学生 & 计算 beta
        # ============================================
        student_paths = self.args.student_model_path
        if not isinstance(student_paths, list):
            student_paths = [student_paths]

        num_ot = 1
        feat_per_party = self.args.enc_in - num_ot
        party_dim = feat_per_party + num_ot     # 学生输出的总维度
        expected_parties = (self.args.cloud_dim - num_ot) // feat_per_party
        if len(student_paths) != expected_parties:
            raise ValueError(
                f'Expected {expected_parties} student checkpoints from cloud_dim='
                f'{self.args.cloud_dim} and enc_in={self.args.enc_in}, got '
                f'{len(student_paths)}.')

        student_models, student_scores = [], []
        for idx, spath in enumerate(student_paths):
            s_model = self._load_single_student(spath)
            score = self.evaluate_student_performance(s_model, vali_loader, idx, feat_per_party, num_ot)
            print(f"    [Student {idx+1}] Quality: {score:.4f}")
            student_models.append(s_model)
            student_scores.append(score)

        scores = np.array(student_scores)
        min_s, max_s = scores.min(), scores.max()
        norm_scores = (scores - min_s) / (max_s - min_s) if max_s > min_s else np.ones_like(scores) / len(scores)
        betas = np.exp(self.gamma * norm_scores)
        betas = np.maximum(betas / betas.sum(), 0.05)
        betas = betas / betas.sum()

        for idx, beta in enumerate(betas):
            print(f"    [Student {idx+1}] Beta: {beta:.4f}")

        # ============================================
        # 4. OFA-KD 设置 (由 use_ofa_kd 开关)
        # ============================================
        ofa_active = False
        stu_bufs = []
        ofa_projs = nn.ModuleList()
        n_stu_stages = 0

        if self.use_ofa_kd:
            for s_model in student_models:
                stu_bufs.append(StageFeatureBuffer(s_model, required=False))

            stage_counts = [buffer.n for buffer in stu_bufs]
            n_stu_stages = min(stage_counts, default=0)

            if n_stu_stages > 0:
                ofa_active = True
                # 给每个 (student, stage) 分配一个 projector
                # 学生 stage 特征 [B, T, student_d_model] -> [B, pred_len, party_dim]
                for _ in student_models:
                    for _ in range(n_stu_stages):
                        ofa_projs.append(
                            OFAProjector(
                                d_model=self.student_configs.d_model,
                                target_c=party_dim,
                                pred_len=self.args.pred_len,
                                input_len=self.args.seq_len,
                            ).to(self.device)
                        )
                print(f"\n    [OFA-KD] enabled: {n_stu_stages} stages, "
                      f"eps={self.ofa_eps}, weight={self.ofa_loss_weight}, "
                      f"anchor={self.ofa_anchor_weight}")
                print(f"    [OFA-KD] {len(ofa_projs)} stage projectors built")
            else:
                print(f"\n    [OFA-KD] not enabled: student stages unavailable")
                # 清理无用 hook
                for buf in stu_bufs:
                    buf.remove()
                stu_bufs = []

        # ============================================
        # 5. 优化器 (云端输出层 + 训练期 OFA projector)
        # ============================================
        trainable_params = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        if ofa_active:
            trainable_params += list(ofa_projs.parameters())

        model_optim = optim.Adam(trainable_params, lr=self.args.learning_rate)
        criterion_mse = nn.MSELoss()

        print(f"\n    [Config] α(edge_soft_kd) = {self.reverse_kd_weight}")
        print(f"    [Config] Trainable params = {sum(p.numel() for p in trainable_params)}")

        # ============================================
        # 6. 训练循环
        # ============================================
        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        print(f"\n>>> Training...")
        f_dim = -1 if self.args.features == 'MS' else 0

        for epoch in range(self.args.train_epochs):
            loss_total_list, loss_kd_list, loss_hard_list, loss_ofa_list = [], [], [], []
            self.model.train()
            for proj in ofa_projs:
                proj.train()

            for batch_x, batch_y, batch_x_mark, batch_y_mark in train_loader:
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                teacher_pred = self.model(
                    batch_x, batch_x_mark, dec_inp, batch_y_mark)
                teacher_pred = teacher_pred[:, :, f_dim:]
                batch_y_true = batch_y[:, -self.args.pred_len:, f_dim:]

                agg_kd_loss = torch.tensor(0.0, device=self.device)
                agg_hard_loss = torch.tensor(0.0, device=self.device)
                agg_ofa_loss  = torch.tensor(0.0, device=self.device)

                for idx, (s_model, s_beta) in enumerate(zip(student_models, betas)):

                    start_col = idx * feat_per_party
                    end_col = start_col + feat_per_party

                    s_batch_x = torch.cat([batch_x[:, :, start_col:end_col],
                                           batch_x[:, :, -num_ot:]], dim=2)
                    s_dec_inp = torch.cat([dec_inp[:, :, start_col:end_col],
                                           dec_inp[:, :, -num_ot:]], dim=2)

                    # Student output is the base reverse-KD soft target. OFA
                    # hooks capture intermediate student stages at the same time.
                    with torch.no_grad():
                        student_pred = s_model(
                            s_batch_x, batch_x_mark, s_dec_inp, batch_y_mark)
                        student_pred = student_pred[:, -self.args.pred_len:, f_dim:]

                    t_slice = teacher_pred[:, :, start_col:end_col]
                    gt_slice = batch_y_true[:, :, start_col:end_col]
                    t_ot = teacher_pred[:, :, -num_ot:]
                    gt_ot = batch_y_true[:, :, -num_ot:]
                    t_party = torch.cat([t_slice, t_ot], dim=2)
                    gt_party = torch.cat([gt_slice, gt_ot], dim=2)

                    if student_pred.shape != t_party.shape:
                        raise ValueError(
                            f'Student/teacher party outputs differ: '
                            f'{tuple(student_pred.shape)} vs {tuple(t_party.shape)}.')

                    # Quality-weighted edge-to-cloud soft-target distillation.
                    agg_kd_loss = agg_kd_loss + s_beta * criterion_mse(
                        t_party, student_pred.detach())

                    loss_hard = criterion_mse(t_slice, gt_slice)
                    ot_weight = getattr(self.args, 'ot_weight', 1.0)
                    if ot_weight > 0:
                        loss_hard = loss_hard + ot_weight * criterion_mse(t_ot, gt_ot)
                    agg_hard_loss = agg_hard_loss + loss_hard

                    # OFA-KD adds stage-derived soft targets in prediction space.
                    if ofa_active and stu_bufs[idx].n > 0:
                        stage_loss = torch.tensor(0.0, device=self.device)
                        anchor_loss = torch.tensor(0.0, device=self.device)
                        n_valid = 0
                        for stage_i in range(n_stu_stages):
                            stu_stage_feat = stu_bufs[idx].features[stage_i]
                            if stu_stage_feat is None:
                                continue
                            proj_idx = idx * n_stu_stages + stage_i
                            stage_pred = ofa_projs[proj_idx](stu_stage_feat)     # [B, P, party_dim]

                            # OFA reverse loss: teacher 朝 stage_pred 靠拢, 按学生可靠度加权
                            stage_loss = stage_loss + ofa_reverse_regression_loss(
                                stage_pred, t_party, gt_party, eps=self.ofa_eps
                            )
                            # gt anchor: 让 stage projector 也朝 gt 学习, 防止塌陷
                            anchor_loss = anchor_loss + criterion_mse(stage_pred, gt_party)
                            n_valid += 1

                        if n_valid > 0:
                            stage_loss = stage_loss / n_valid
                            anchor_loss = anchor_loss / n_valid
                            agg_ofa_loss = agg_ofa_loss + s_beta * (
                                stage_loss + self.ofa_anchor_weight * anchor_loss
                            )

                n = len(student_models)
                agg_hard_loss = agg_hard_loss / n

                # 总损失
                alpha = self.reverse_kd_weight
                total_loss = alpha * agg_kd_loss + (1 - alpha) * agg_hard_loss
                if ofa_active:
                    total_loss = total_loss + self.ofa_loss_weight * agg_ofa_loss

                # 权重漂移正则化
                if self.teacher_initial_weights is not None:
                    drift = torch.tensor(0.0, device=self.device)
                    for name, param in self.model.named_parameters():
                        if param.requires_grad and name in self.teacher_initial_weights:
                            drift = drift + torch.sum((param - self.teacher_initial_weights[name]) ** 2)
                    total_loss = total_loss + self.weight_reg_lambda * drift

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                model_optim.step()

                loss_total_list.append(total_loss.item())
                loss_kd_list.append(agg_kd_loss.item())
                loss_hard_list.append(agg_hard_loss.item())
                loss_ofa_list.append(agg_ofa_loss.item() if ofa_active else 0.0)

            log_msg = (f"    Epoch {epoch+1} | Total: {np.mean(loss_total_list):.6f} "
                       f"| EdgeKD: {np.mean(loss_kd_list):.6f} "
                       f"| Hard: {np.mean(loss_hard_list):.6f}")
            if ofa_active:
                log_msg += f" | OFA: {np.mean(loss_ofa_list):.6f}"
            print(log_msg)

        # ============================================
        # 7. 更新后性能检查 & 自动回滚
        # ============================================
        post_update_mse = self._evaluate_teacher_mse(vali_loader)
        change_pct = (post_update_mse / pre_update_mse - 1) * 100
        print(f"\n    [Result] Post-update MSE: {post_update_mse:.6f} ({change_pct:+.1f}%)")

        if post_update_mse > pre_update_mse * self.rollback_threshold:
            allowed_pct = (self.rollback_threshold - 1.0) * 100
            print(f"    ⚠️ Validation MSE degraded by more than {allowed_pct:.1f}%; rolling back")
            self.model.load_state_dict(
                {k: v.clone() for k, v in self.teacher_initial_weights.items()})
        else:
            print(f"    ✅ 更新有效")

        save_path = os.path.join(path, 'checkpoint_teacher_updated.pth')
        torch.save(self.model.state_dict(), save_path)
        print(f">>> Saved: {save_path}")

        # ============================================
        # 8. 清理 hook
        # ============================================
        for buf in stu_bufs:
            buf.remove()

        return self.model
