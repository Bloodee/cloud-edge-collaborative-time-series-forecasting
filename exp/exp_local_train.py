"""Personalize a distilled student on one edge site's recent data."""

import os
import torch
from torch import optim
import numpy as np
import joblib
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from utils.tools import EarlyStopping, adjust_learning_rate


class Exp_LocalTrain(Exp_Long_Term_Forecast):
    def __init__(self, args):
        super(Exp_LocalTrain, self).__init__(args)

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def local_train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')

        # 加载预训练权重
        if self.args.pretrained_model_path and os.path.exists(self.args.pretrained_model_path):
            print(f"[Local Train] Loading from {self.args.pretrained_model_path}")
            try:
                state_dict = torch.load(self.args.pretrained_model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                print("   -> Weights loaded.")
            except Exception as e:
                raise RuntimeError(f'Failed to load local checkpoint: {e}') from e
        else:
            raise FileNotFoundError(
                f'Local checkpoint not found: {self.args.pretrained_model_path}')

        scaler_path = getattr(self.args, 'scaler_path', None)
        if scaler_path and getattr(train_data, 'scaler', None) is not None:
            os.makedirs(os.path.dirname(scaler_path) or '.', exist_ok=True)
            joblib.dump(train_data.scaler, scaler_path)
            print(f'[Local Train] Scaler saved: {scaler_path}')

        # 全量微调
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"   -> Trainable params: {trainable}")

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            self.model.train()
            train_loss = []

            for batch_x, batch_y, batch_x_mark, batch_y_mark in train_loader:
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float().to(self.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            train_loss_avg = np.mean(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)

            print(f"Epoch: {epoch+1} | Train: {train_loss_avg:.7f} Vali: {vali_loss:.7f}")

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        # Restore the best validation checkpoint before exporting the
        # personalized artifact; the last epoch can be worse than the best one.
        best_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_path, map_location=self.device))

        # 保存个性化模型
        final_path = os.path.join(path, 'checkpoint_personalized.pth')
        torch.save(self.model.state_dict(), final_path)
        print(f"✅ Personalized model saved: {final_path}")

        # 测试
        print(f'>>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        self.test(setting)
