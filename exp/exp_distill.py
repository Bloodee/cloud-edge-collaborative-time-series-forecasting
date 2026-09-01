"""
正向蒸馏 (支持增量式)

核心改进:
  ✅ 支持 pretrained_model_path: 加载上一轮统一学生权重作为起点
     - 首次蒸馏: 随机初始化 → 从零学习
     - 后续蒸馏: 加载上一轮权重 → 增量更新, 不破坏已学知识
  ✅ 保存全局 Scaler
  ✅ 梯度裁剪
  ✅ CNN 辅助蒸馏: 通过 hook 捕获中间特征，并用临时预测头提供多阶段监督；
     辅助头不写入学生模型 checkpoint。
"""

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.aux_head_helper import AuxHeadHelper
from utils.tools import EarlyStopping, adjust_learning_rate
import torch
import torch.nn as nn
import os
import time
import numpy as np
import copy
import joblib

class Exp_Distill(Exp_Long_Term_Forecast):
    def __init__(self, args):
        super(Exp_Distill, self).__init__(args)
        self.lambda_kd = args.lambda_kd

        # Teacher config
        self.teacher_args = copy.deepcopy(args)
        self.teacher_model_name = getattr(args, 'teacher_model_name', 'TimesNet')
        self.teacher_args.model = self.teacher_model_name

        if self.teacher_model_name == 'TimesNet':
            self.teacher_args.d_model = getattr(args, 'teacher_d_model', 256)
            self.teacher_args.d_ff = getattr(args, 'teacher_d_ff', 512)
            self.teacher_args.e_layers = getattr(args, 'teacher_e_layers', args.e_layers)
            self.teacher_args.top_k = getattr(args, 'teacher_top_k', args.top_k)
            self.teacher_args.num_kernels = getattr(args, 'teacher_num_kernels', args.num_kernels)

        self.cloud_dim = getattr(self.args, 'cloud_dim', self.args.enc_in)
        self.distill_mode = 'Compression' if self.cloud_dim == self.args.enc_in else 'Split'
        self.individual_ot = False

        # Optional auxiliary heads are attached only while distilling a CNN.
        self._aux_helper = None

    def _build_teacher_model(self):
        print(f"   [Distill] Teacher: {self.teacher_model_name} "
              f"(d={self.teacher_args.d_model}, ff={self.teacher_args.d_ff}, L={self.teacher_args.e_layers})")

        teacher = self.model_dict[self.teacher_model_name].Model(self.teacher_args).float()

        if self.args.teacher_model_path and os.path.exists(self.args.teacher_model_path):
            teacher.load_state_dict(torch.load(self.args.teacher_model_path, map_location=self.device))
        else:
            raise ValueError(f"Teacher not found: {self.args.teacher_model_path}")

        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad = False

        if self.args.use_multi_gpu and self.args.use_gpu:
            teacher = nn.DataParallel(teacher, device_ids=self.args.device_ids)
        return teacher

    def _get_slice(self, data, idx, feat_len, num_ot, mode='feature'):
        if self.distill_mode == 'Compression':
            return data[:, :, -num_ot:] if mode == 'ot' else data[:, :, :-num_ot]
        if mode == 'ot':
            return data[:, :, -num_ot:]
        start = idx * feat_len
        end = (idx + 1) * feat_len
        return data[:, :, start:end]

    def _compute_distill_loss(self, s_out, gt_y, t_out, party_idx,
                               feat_per_party, num_ot, criterion, distill_loss_fn):
        s_out_feats = s_out[:, :, :-num_ot]
        s_out_ot = s_out[:, :, -num_ot:]

        gt_feats = self._get_slice(gt_y, party_idx, feat_per_party, num_ot, 'feature')
        gt_feats = gt_feats[:, -self.args.pred_len:, :]
        gt_ot = self._get_slice(gt_y, party_idx, feat_per_party, num_ot, 'ot')
        gt_ot = gt_ot[:, -self.args.pred_len:, :]

        if self.distill_mode == 'Compression':
            t_target = t_out[:, :, :-num_ot]
        else:
            t_start = party_idx * feat_per_party
            t_end = (party_idx + 1) * feat_per_party
            t_target = t_out[:, :, t_start:t_end]

        # 把 feat 和 ot 合起来求 mean
        loss_feat_hard = criterion(s_out_feats, gt_feats)
        loss_feat_soft = distill_loss_fn(s_out_feats, t_target)
        loss_ot_hard   = criterion(s_out_ot, gt_ot)

        n_feat, n_ot = feat_per_party, num_ot
        ot_weight = self.args.ot_weight
        n_total = n_feat + ot_weight * n_ot
        hard_combined = (
            n_feat * loss_feat_hard + ot_weight * n_ot * loss_ot_hard
        ) / n_total

        loss = self.lambda_kd * hard_combined + (1 - self.lambda_kd) * loss_feat_soft
        return loss

    def _forward_student(self, s_batch_x, batch_x_mark, s_dec_inp, batch_y_mark, use_aux):
        """Return the student forecast and optional CNN auxiliary forecasts."""
        if use_aux and self.args.model == 'CNN' \
                and self._aux_helper is not None and self._aux_helper.active:
            s_out = self.model(s_batch_x, batch_x_mark, s_dec_inp, batch_y_mark)
            s_aux_preds = self._aux_helper.get_aux_predictions()
            return s_out, s_aux_preds
        else:
            s_out = self.model(s_batch_x, batch_x_mark, s_dec_inp, batch_y_mark)
            return s_out, []

    def distill(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        has_test = len(test_data) > 0

        # 保存全局 Scaler
        if hasattr(train_data, 'scaler') and train_data.scaler is not None:
            scaler_path = (
                getattr(self.args, 'scaler_path', None)
                or os.path.join(self.args.checkpoints, 'global_scaler', 'scaler.pkl')
            )
            os.makedirs(os.path.dirname(scaler_path) or '.', exist_ok=True)
            joblib.dump(train_data.scaler, scaler_path)
            print(f"   ✅ Global Scaler saved: {scaler_path}")

        # Teacher
        self.teacher_args.enc_in = self.cloud_dim
        self.teacher_args.dec_in = self.cloud_dim
        self.teacher_args.c_out = self.cloud_dim
        teacher_model = self._build_teacher_model().to(self.device)

        # Dimensions
        num_ot = 1
        student_dim = self.args.enc_in
        if self.distill_mode == 'Compression':
            num_parties = 1
            feat_per_party = student_dim - num_ot
        else:
            feat_per_party = student_dim - num_ot
            num_parties = (self.cloud_dim - num_ot) // feat_per_party

        print(f"   Mode={self.distill_mode}, Parties={num_parties}, Feat/Party={feat_per_party}")

        # ✅ 增量蒸馏: 加载上一轮统一学生权重作为起点
        if self.args.pretrained_model_path:
            if not os.path.exists(self.args.pretrained_model_path):
                raise FileNotFoundError(
                    f'Incremental student checkpoint not found: '
                    f'{self.args.pretrained_model_path}')
            print(f"   [Incremental] Loading previous student: {self.args.pretrained_model_path}")
            try:
                state_dict = torch.load(self.args.pretrained_model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                print(f"   ✅ Previous student loaded, incremental distillation")
            except Exception as e:
                raise RuntimeError(
                    f'Failed to load incremental student checkpoint: {e}') from e
        else:
            print(f"   [Fresh] No previous student, starting from scratch")

        use_aux = getattr(self.args, 'use_aux_kd', False) and self.args.model == 'CNN'

        if use_aux and self.args.model == 'CNN':
            self._aux_helper = AuxHeadHelper(
                student=self.model,
                pred_len=self.args.pred_len,
                c_out=self.args.c_out,
                device=self.device,
            )
        else:
            self._aux_helper = None

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        distill_loss_fn = nn.SmoothL1Loss(beta=1.0)

        # ★ 新增: 把外挂 aux head 参数加入 optimizer
        if self._aux_helper is not None and self._aux_helper.active:
            aux_params = list(self._aux_helper.parameters())
            model_optim.add_param_group({
                'params': aux_params,
                'lr': self.args.learning_rate,
            })
            print(f"   [Optim] +{sum(p.numel() for p in aux_params)} external aux head params")

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            train_loss = []
            self.model.train()
            if self._aux_helper is not None:
                self._aux_helper.train()
            epoch_time = time.time()
            f_dim = -1 if self.args.features == 'MS' else 0

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                with torch.no_grad():
                    t_out = teacher_model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    t_out = t_out[:, -self.args.pred_len:, f_dim:]

                total_loss = torch.tensor(0.0, device=self.device)
                for p in range(num_parties):
                    gt_feats_x = self._get_slice(batch_x, p, feat_per_party, num_ot, 'feature')
                    gt_ot_x = self._get_slice(batch_x, p, feat_per_party, num_ot, 'ot')
                    s_batch_x = torch.cat([gt_feats_x, gt_ot_x], dim=2)

                    gt_feats_dec = self._get_slice(dec_inp, p, feat_per_party, num_ot, 'feature')
                    gt_ot_dec = self._get_slice(dec_inp, p, feat_per_party, num_ot, 'ot')
                    s_dec_inp = torch.cat([gt_feats_dec, gt_ot_dec], dim=2)

                    # ===== 学生前向 (统一走 _forward_student, 内部分发 use_aux 路径) =====
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            s_out, s_aux_preds = self._forward_student(
                                s_batch_x, batch_x_mark, s_dec_inp, batch_y_mark, use_aux)
                    else:
                        s_out, s_aux_preds = self._forward_student(
                            s_batch_x, batch_x_mark, s_dec_inp, batch_y_mark, use_aux)

                    s_out = s_out[:, -self.args.pred_len:, f_dim:]
                    loss_p = self._compute_distill_loss(
                        s_out, batch_y, t_out, p, feat_per_party, num_ot, criterion, distill_loss_fn)

                    # ===== 多层级 aux 损失 =====
                    # CNN auxiliary outputs are [B, pred_len, c_out].
                    if s_aux_preds:
                        if self.distill_mode == 'Compression':
                            t_target = t_out[:, :, :-num_ot]
                        else:
                            t_start = p * feat_per_party
                            t_end = (p + 1) * feat_per_party
                            t_target = t_out[:, :, t_start:t_end]

                        aux_kd_loss = 0.0
                        for aux_p in s_aux_preds:
                            aux_p_feats = aux_p[:, -self.args.pred_len:, :-num_ot]
                            aux_kd_loss = aux_kd_loss + distill_loss_fn(aux_p_feats, t_target)
                        aux_kd_loss = aux_kd_loss / len(s_aux_preds)
                        loss_p = loss_p + self.args.aux_kd_weight * aux_kd_loss

                    total_loss = total_loss + loss_p

                total_loss = total_loss / num_parties
                train_loss.append(total_loss.item())

                if self.args.use_amp:
                    scaler.scale(total_loss).backward()
                    scaler.unscale_(model_optim)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    model_optim.step()

                if (i + 1) % 100 == 0:
                    print(f"\titers: {i+1}, epoch: {epoch+1} | loss: {total_loss.item():.7f}")

            train_loss_avg = np.mean(train_loss)
            vali_loss = self._vali_distill(vali_loader, criterion, distill_loss_fn, teacher_model,
                                           num_parties, feat_per_party, num_ot)
            test_loss = self._vali_distill(test_loader, criterion, distill_loss_fn, teacher_model,
                                           num_parties, feat_per_party, num_ot) if has_test else 0.0

            print(f"Epoch {epoch+1} ({time.time()-epoch_time:.1f}s) | "
                  f"Train: {train_loss_avg:.7f} Vali: {vali_loss:.7f} Test: {test_loss:.7f}")

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        # ★ 新增: 训练结束, 移除 hook + 释放 aux head (这样保存的 ckpt 是干净的)
        if self._aux_helper is not None:
            self._aux_helper.remove()
            self._aux_helper = None
            print("   [AuxHelper] removed; saved checkpoint contains no aux params")

        self.model.load_state_dict(
            torch.load(os.path.join(path, 'checkpoint.pth'), map_location=self.device))

        # ===== 蒸馏完成后自动测试 =====
        do_test = getattr(self.args, 'do_distill_test', False) or \
                  (getattr(self.args, 'test_ratio', 0.0) and self.args.test_ratio > 0)
        if do_test:
            print(f"\n>>> [Distill Test] 评估蒸馏后的学生模型...")
            self.test(setting, test=0)
            print(f">>> [Distill Test] 完成, 结果在 ./results/{setting}/")

        return self.model

    def _vali_distill(self, loader, criterion, distill_loss_fn, teacher_model,
                      num_parties, feat_per_party, num_ot):
        total_loss = []
        self.model.eval()
        if self._aux_helper is not None:
            self._aux_helper.eval()
        f_dim = -1 if self.args.features == 'MS' else 0

        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                t_out = teacher_model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                t_out = t_out[:, -self.args.pred_len:, f_dim:]

                batch_loss = 0.0
                for p in range(num_parties):
                    gt_feats_x = self._get_slice(batch_x, p, feat_per_party, num_ot, 'feature')
                    gt_ot_x = self._get_slice(batch_x, p, feat_per_party, num_ot, 'ot')
                    s_batch_x = torch.cat([gt_feats_x, gt_ot_x], dim=2)

                    gt_feats_dec = self._get_slice(dec_inp, p, feat_per_party, num_ot, 'feature')
                    gt_ot_dec = self._get_slice(dec_inp, p, feat_per_party, num_ot, 'ot')
                    s_dec_inp = torch.cat([gt_feats_dec, gt_ot_dec], dim=2)

                    # vali 不用 aux KD, 直接走主预测路径
                    outputs = self.model(s_batch_x, batch_x_mark, s_dec_inp, batch_y_mark)
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]

                    batch_loss += self._compute_distill_loss(
                        outputs, batch_y, t_out, p, feat_per_party, num_ot, criterion, distill_loss_fn).item()

                total_loss.append(batch_loss / num_parties)

        self.model.train()
        if self._aux_helper is not None:
            self._aux_helper.train()
        return np.mean(total_loss)
