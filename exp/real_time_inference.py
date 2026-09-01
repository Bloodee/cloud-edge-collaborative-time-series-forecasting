"""
实时推理模拟器 v5  (per-step CSV + per-party trigger + CDF durations)

每个 party 单独维护一张 csv, 加上 edge_avg.csv:
    step, start_idx, end_idx,
    mse, mae, rmse, r2, corr,
    error_trigger, drift_trigger, update_trigger,
    infer_time

另外有 3 张 CDF 表 (每个事件一行):
    durations_rev_distill.csv : 每次反向蒸馏总耗时
    durations_fwd_distill.csv : 每次正向蒸馏总耗时
    durations_local_ft.csv    : 每次本地微调 (已除以 N_PARTY, 是单边端平均)
"""

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.stats import entropy
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


class Config:
    DATA_ROOT = os.environ.get("RT_DATA_ROOT", "./dataset/processed")
    CKPT_DIR = os.environ.get("RT_CKPT_DIR", "./checkpoints")
    RES_DIR = os.environ.get("RT_RES_DIR", "./realtime_results")
    MERGE_PATH = os.environ.get("RT_MERGE_PATH", "./dataset/processed/merge/")
    DATA_NAME = os.environ.get("RT_DATA_NAME", "PVOD.csv")

    # 边端数量 & 维度
    N_PARTY = int(os.environ.get("RT_N_PARTY", 7))
    CLOUD_DIM = int(os.environ.get("RT_CLOUD_DIM", 99))
    EDGE_DIM = int(os.environ.get("RT_EDGE_DIM", 15))

    # 时序窗口
    SEQ_LEN = int(os.environ.get("RT_SEQ_LEN", 12))
    PRED_LEN = int(os.environ.get("RT_PRED_LEN", 6))
    LABEL_LEN = int(os.environ.get("RT_LABEL_LEN", 12))
    FEATURES = os.environ.get("RT_FEATURES", "M")
    FACTOR = int(os.environ.get("RT_FACTOR", 3))
    EV_LEN = int(os.environ.get("RT_EV_LEN", 24))

    # 教师模型 (TimesNet)
    T_D_MODEL = int(os.environ.get("RT_T_D_MODEL", 128))
    T_D_FF = int(os.environ.get("RT_T_D_FF", 256))
    T_LAYERS = int(os.environ.get("RT_T_LAYERS", 2))
    T_TOP_K = int(os.environ.get("RT_T_TOP_K", 5))

    # 学生模型 (CNN)
    S_MODEL = os.environ.get("RT_S_MODEL", "CNN")
    S_D_MODEL = int(os.environ.get("RT_S_D_MODEL", 16))
    S_D_FF = int(os.environ.get("RT_S_D_FF", 32))
    S_E_LAYERS = int(os.environ.get("RT_S_E_LAYERS", 2))
    S_D_LAYERS = int(os.environ.get("RT_S_D_LAYERS", 1))
    S_TOP_K = int(os.environ.get("RT_S_TOP_K", 5))

    # 推理与触发参数
    INFER_STEP = int(os.environ.get("RT_INFER_STEP", EV_LEN * 4))   # ★ 可被 env 覆盖
    START_IDX = int(os.environ.get("RT_START_IDX", 18750))
    INFER_BATCH_SIZE = int(os.environ.get("RT_INFER_BATCH_SIZE", 32))

    # 反向蒸馏
    REV_EPOCHS = int(os.environ.get("RT_REV_EPOCHS", 1))
    REV_LR = float(os.environ.get("RT_REV_LR", 0.0001))
    REV_BATCH_SIZE = int(os.environ.get("RT_REV_BATCH_SIZE", 32))
    REV_KD_WEIGHT = float(os.environ.get("RT_REV_KD_WEIGHT") or 0.3)
    REV_DATA_WINDOW = EV_LEN * 14 * 4
    print(f"RT_REV_KD_WEIGHT:{REV_KD_WEIGHT}")
    # 正向蒸馏
    FWD_EPOCHS = int(os.environ.get("RT_FWD_EPOCHS", 10))
    FWD_LR = float(os.environ.get("RT_FWD_LR", 0.0001))
    FWD_BATCH_SIZE = int(os.environ.get("RT_FWD_BATCH_SIZE", 32))
    FWD_DATA_START = 0
    FWD_SLIDING_WINDOW = EV_LEN * 365 * 2.5
    FWD_WINDOW_MODE = 'sliding'
    FORWARD_DISTILL_INTERVAL = int(os.environ.get("RT_FORWARD_INTERVAL", 4))

    # 本地微调
    FT_EPOCHS = int(os.environ.get("RT_FT_EPOCHS", 3))
    FT_LR = float(os.environ.get("RT_FT_LR", 0.0001))
    FT_BATCH_SIZE = int(os.environ.get("RT_FT_BATCH_SIZE", 32))
    FT_DATA_WINDOW = EV_LEN * 5

    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.2
    TEST_RATIO = 0.0

    # 触发阈值
    ERR_THRESH_RATIO = float(os.environ.get("RT_ERR_RATIO", 1.5))
    ERR_MIN_ABSOLUTE = float(os.environ.get("RT_ERR_MIN", 0.35))
    KL_THRESH = float(os.environ.get("RT_KL_THRESH", 0.5))
    MAX_NO_UPDATE = int(os.environ.get("RT_MAX_NO_UPDATE", 6 * INFER_STEP - INFER_STEP))

    TRIGGER_MIN_POINTS = INFER_STEP * 2
    KL_DETECT_MIN_POINTS = INFER_STEP * 2
    KL_WINDOW = INFER_STEP * 4

    # 蒸馏控制
    OT_WEIGHT = 1.0
    USE_AUX_KD = os.environ.get("RT_USE_AUX_KD", "0") == "1"
    USE_OFA_KD = os.environ.get("RT_USE_OFA_KD", "1") == "1"

    OFA_EPS = float(os.environ.get("RT_OFA_EPS", 1.5))
    OFA_LOSS_WEIGHT = float(os.environ.get("RT_OFA_LOSS_WEIGHT", 0.5))
    OFA_FINAL_WEIGHT = float(os.environ.get("RT_OFA_FINAL_WEIGHT", 0.5))
    OFA_ANCHOR_WEIGHT = float(os.environ.get("RT_OFA_ANCHOR_WEIGHT", 0.3))

    FWD_PATIENCE = 3

    # 对比实验开关
    ENABLE_REV_DISTILL = True
    ENABLE_FWD_DISTILL = True
    ENABLE_LOCAL_FT = True
    ENABLE_CLOUD_INFER = False     # ★ 默认关闭, 当前实验不再需要 cloud 行


Config.PARTIES = [f'party_{i}' for i in range(1, Config.N_PARTY + 1)]

print(f">>> [Config] N_PARTY={Config.N_PARTY}, CLOUD_DIM={Config.CLOUD_DIM}, EDGE_DIM={Config.EDGE_DIM}")
print(f">>> [Config] PARTIES={Config.PARTIES}")
print(f">>> [Config] Window: seq={Config.SEQ_LEN}, pred={Config.PRED_LEN}, infer_step={Config.INFER_STEP}")
print(f">>> [Config] Student: {Config.S_MODEL} (d={Config.S_D_MODEL}, ff={Config.S_D_FF}, e_layers={Config.S_E_LAYERS})")
print(f">>> [Config] Teacher: TimesNet (d={Config.T_D_MODEL}, ff={Config.T_D_FF}, e_layers={Config.T_LAYERS})")
print(f">>> [Config] USE_AUX_KD={Config.USE_AUX_KD}, USE_OFA_KD={Config.USE_OFA_KD}")
print(f">>> [Config] Triggers: max_no_update={Config.MAX_NO_UPDATE}, "
      f"eta={Config.ERR_THRESH_RATIO}, delta={Config.KL_THRESH}, R={Config.FORWARD_DISTILL_INTERVAL}")
print(f">>> [Config] ENABLE: rev={Config.ENABLE_REV_DISTILL}, fwd={Config.ENABLE_FWD_DISTILL}, "
      f"local_ft={Config.ENABLE_LOCAL_FT}, cloud_infer={Config.ENABLE_CLOUD_INFER}")


def run_cmd(cmd):
    """Run a generated Python command without invoking a command shell."""
    print(f"  CMD: {cmd[:120]}...")
    argv = shlex.split(cmd, posix=os.name != 'nt')
    if not argv:
        raise ValueError('Refusing to run an empty command.')
    argv = [part.strip('"') for part in argv]
    if Path(argv[0]).name.lower() in {'python', 'python3', 'python.exe'}:
        argv[0] = sys.executable
    subprocess.run(argv, shell=False, check=True)


def find_latest_dir(base_path, pattern):
    if not os.path.exists(base_path):
        return None
    candidates = [d for d in os.listdir(base_path) if pattern in d]
    if not candidates:
        return None
    latest = sorted(candidates, key=lambda x: os.path.getmtime(os.path.join(base_path, x)))[-1]
    return os.path.join(base_path, latest)


def cleanup(path):
    """Delete only artifacts below known generated-output roots."""
    if not path:
        return
    target = Path(path).resolve()
    allowed_roots = [Path(Config.CKPT_DIR).resolve(), Path('./results').resolve()]
    if not any(target != root and root in target.parents for root in allowed_roots):
        raise ValueError(f'Refusing to clean path outside generated roots: {target}')
    if target.is_file():
        target.unlink()
    elif target.is_dir():
        shutil.rmtree(target)


# ============================================================
# 文件 I/O 工具
# ============================================================
PER_STEP_HEADER = (
    "step,start_idx,end_idx,"
    "mse,mae,rmse,r2,corr,"
    "error_trigger,drift_trigger,timeout_trigger,update_trigger,"
    "infer_time\n"
)
CDF_HEADER = "update_id,duration_sec\n"


def _init_csv(path, header):
    """如不存在, 写入 header"""
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(header)


def _append_row(path, row):
    """row 应是 string, 末尾自动加 \n"""
    with open(path, "a") as f:
        f.write(row + "\n")


# ============================================================
class RealTimeSimulator:
    def __init__(self):
        self._validate_config()
        os.makedirs(Config.RES_DIR, exist_ok=True)

        df = pd.read_csv(os.path.join(Config.DATA_ROOT, 'party_1', Config.DATA_NAME))
        raw_length = len(df)
        safety_margin = Config.SEQ_LEN + Config.PRED_LEN * 2
        self.total_length = raw_length - safety_margin
        print(f">>> Dataset: raw={raw_length}, usable={self.total_length}")

        self.current_idx = Config.START_IDX
        self.last_update_idx = self.current_idx
        self.base_metric = {party: None for party in Config.PARTIES}
        self.update_counter = 0
        self.step_counter = 0

        # 用于 trigger 检测的滚动 buffer (累积, 跨 step)
        self.results_buffer = {p: {'OT': [], 'Pred': []} for p in Config.PARTIES}

        # 用于 final report 的累积 (整段 sim)
        self.global_eval = {p: {'OT': [], 'Pred': []} for p in Config.PARTIES}

        # 计时
        import time as _time
        self._time = _time

        self.trigger_counts = {'error': 0, 'drift': 0, 'timeout': 0, 'total': 0}

        # ===== 初始化 per-step CSV 表 =====
        for p in Config.PARTIES:
            _init_csv(self._step_csv_path(p), PER_STEP_HEADER)
        _init_csv(self._step_csv_path('edge_avg'), PER_STEP_HEADER)

        # ===== 初始化 CDF 表 =====
        _init_csv(os.path.join(Config.RES_DIR, "durations_rev_distill.csv"), CDF_HEADER)
        _init_csv(os.path.join(Config.RES_DIR, "durations_fwd_distill.csv"), CDF_HEADER)
        _init_csv(os.path.join(Config.RES_DIR, "durations_local_ft.csv"), CDF_HEADER)

        print(f">>> [Setup] 初始化 {len(Config.PARTIES)+1} 张 per-step CSV "
              f"+ 3 张 CDF CSV -> {Config.RES_DIR}")

    @staticmethod
    def _validate_config():
        positive = {
            'N_PARTY': Config.N_PARTY, 'CLOUD_DIM': Config.CLOUD_DIM,
            'EDGE_DIM': Config.EDGE_DIM, 'SEQ_LEN': Config.SEQ_LEN,
            'LABEL_LEN': Config.LABEL_LEN, 'PRED_LEN': Config.PRED_LEN,
            'INFER_STEP': Config.INFER_STEP, 'INFER_BATCH_SIZE': Config.INFER_BATCH_SIZE,
            'REV_EPOCHS': Config.REV_EPOCHS, 'FWD_EPOCHS': Config.FWD_EPOCHS,
            'FT_EPOCHS': Config.FT_EPOCHS,
            'FORWARD_DISTILL_INTERVAL': Config.FORWARD_DISTILL_INTERVAL,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f'Configuration values must be positive: {invalid}')
        if Config.LABEL_LEN > Config.SEQ_LEN:
            raise ValueError('LABEL_LEN cannot exceed SEQ_LEN.')
        expected_cloud_dim = Config.N_PARTY * (Config.EDGE_DIM - 1) + 1
        if Config.CLOUD_DIM != expected_cloud_dim:
            raise ValueError(
                f'CLOUD_DIM must equal N_PARTY*(EDGE_DIM-1)+1; expected '
                f'{expected_cloud_dim}, got {Config.CLOUD_DIM}.')
        if Config.START_IDX < Config.SEQ_LEN:
            raise ValueError('START_IDX must leave at least one full input window.')
        if Config.FEATURES != 'M' or Config.S_MODEL != 'CNN':
            raise ValueError('The public CE-BiD simulator requires FEATURES=M and S_MODEL=CNN.')
        if (Config.ERR_THRESH_RATIO <= 0 or Config.ERR_MIN_ABSOLUTE < 0
                or Config.KL_THRESH < 0 or Config.MAX_NO_UPDATE <= 0):
            raise ValueError('Trigger ratios/windows must be positive and thresholds non-negative.')

        reference_dates = None
        reference_rows = None
        for party in Config.PARTIES:
            party_file = Path(Config.DATA_ROOT, party, Config.DATA_NAME)
            if not party_file.is_file():
                raise FileNotFoundError(f'Party data not found: {party_file}')
            frame = pd.read_csv(party_file)
            if 'date' not in frame or 'OT' not in frame:
                raise ValueError(f'{party_file} must contain date and OT columns.')
            if len(frame.columns) - 1 != Config.EDGE_DIM:
                raise ValueError(
                    f'{party_file} has {len(frame.columns) - 1} numeric columns; '
                    f'EDGE_DIM={Config.EDGE_DIM}.')
            dates = frame['date'].astype(str).to_numpy()
            if reference_dates is None:
                reference_dates, reference_rows = dates, len(frame)
            elif len(frame) != reference_rows or not np.array_equal(dates, reference_dates):
                raise ValueError(f'{party_file} is not time-aligned with party_1.')
            edge_root = Path(Config.CKPT_DIR, 'Local_Edges', party)
            for artifact in ('checkpoint.pth', 'scaler.pkl'):
                if not (edge_root / artifact).is_file():
                    raise FileNotFoundError(
                        f'Edge artifact not found: {edge_root / artifact}')

        minimum_rows = Config.START_IDX + Config.SEQ_LEN + Config.PRED_LEN + 1
        if reference_rows < minimum_rows:
            raise ValueError(
                f'Party data has {reference_rows} rows; at least {minimum_rows} '
                f'are required for START_IDX={Config.START_IDX}.')

        merge_file = Path(Config.MERGE_PATH, Config.DATA_NAME)
        if not merge_file.is_file():
            raise FileNotFoundError(f'Merged cloud data not found: {merge_file}')
        merge_frame = pd.read_csv(merge_file)
        if 'date' not in merge_frame or 'OT' not in merge_frame:
            raise ValueError(f'{merge_file} must contain date and OT columns.')
        if len(merge_frame.columns) - 1 != Config.CLOUD_DIM:
            raise ValueError(
                f'{merge_file} has {len(merge_frame.columns) - 1} numeric columns; '
                f'CLOUD_DIM={Config.CLOUD_DIM}.')
        if (len(merge_frame) != reference_rows
                or not np.array_equal(
                    merge_frame['date'].astype(str).to_numpy(), reference_dates)):
            raise ValueError(f'{merge_file} is not time-aligned with party data.')

        teacher = Path(Config.CKPT_DIR, 'Global_Teacher', 'checkpoint.pth')
        if (Config.ENABLE_REV_DISTILL or Config.ENABLE_FWD_DISTILL) and not teacher.is_file():
            raise FileNotFoundError(f'Cloud teacher checkpoint not found: {teacher}')
        cloud_scaler = Path(Config.CKPT_DIR, 'global_scaler', 'scaler.pkl')
        if (Config.ENABLE_REV_DISTILL or Config.ENABLE_FWD_DISTILL) and not cloud_scaler.is_file():
            raise FileNotFoundError(f'Cloud scaler not found: {cloud_scaler}')
        unified = Path(Config.CKPT_DIR, 'Unified_Student', 'checkpoint.pth')
        if Config.ENABLE_FWD_DISTILL and not unified.is_file():
            raise FileNotFoundError(f'Unified student checkpoint not found: {unified}')

    def _step_csv_path(self, name):
        return os.path.join(Config.RES_DIR, f"{name}.csv")

    # ============================================================
    # Trigger 检测
    # ============================================================
    def calculate_kl(self, curr, ref):
        curr, ref = np.array(curr).flatten(), np.array(ref).flatten()
        lo, hi = min(np.min(curr), np.min(ref)), max(np.max(curr), np.max(ref))
        if hi == lo:
            return 0.0
        h1, edges = np.histogram(curr, bins=20, range=(lo, hi), density=True)
        h2, _ = np.histogram(ref, bins=edges, density=True)
        return entropy(h1 + 1e-9, h2 + 1e-9)

    def check_trigger_party(self, party):
        """
        返回 (error_trig: 0/1, drift_trig: 0/1, msg: str|None)
        msg 用于打印 / 统计 trigger 类型, 优先 Error.
        """
        hp = np.array(self.results_buffer[party]['Pred'])
        hg = np.array(self.results_buffer[party]['OT'])
        if len(hp) < int(Config.TRIGGER_MIN_POINTS):
            return 0, 0, None

        win = int(Config.TRIGGER_MIN_POINTS)
        mse = np.mean((hg[-win:] - hp[-win:]) ** 2)
        err_trig = 0
        if self.base_metric[party] is None:
            self.base_metric[party] = mse
        else:
            threshold = max(
                self.base_metric[party] * Config.ERR_THRESH_RATIO,
                Config.ERR_MIN_ABSOLUTE,
            )
            if mse > threshold:
                err_trig = 1

        drift_trig = 0
        kl_min = int(Config.KL_DETECT_MIN_POINTS * 2)
        kl_win = int(Config.KL_WINDOW)
        if len(hg) > kl_min:
            kl = self.calculate_kl(hg[-kl_win:], hg[:kl_win])
            if kl > Config.KL_THRESH:
                drift_trig = 1

        msg = None
        if err_trig:
            msg = f"{party}:Error(MSE:{mse:.4f})"
        elif drift_trig:
            msg = f"{party}:Drift"
        return err_trig, drift_trig, msg

    def _refresh_error_references(self):
        """Use the most recent observed window as each party's new baseline."""
        window = int(Config.TRIGGER_MIN_POINTS)
        for party in Config.PARTIES:
            pred = np.asarray(self.results_buffer[party]['Pred']).reshape(-1)
            true = np.asarray(self.results_buffer[party]['OT']).reshape(-1)
            if len(pred) >= window and len(true) >= window:
                self.base_metric[party] = float(
                    np.mean((true[-window:] - pred[-window:]) ** 2))

    # ============================================================
    # 单步指标计算
    # ============================================================
    @staticmethod
    def _metrics_from_arrays(pred, true):
        """从 (pred, true) flatten 数组算 5 项指标. 失败返回 NaN."""
        p = np.array(pred).flatten()
        t = np.array(true).flatten()
        valid = np.isfinite(p) & np.isfinite(t)
        if valid.sum() < 2:
            return (float('nan'),) * 5
        p, t = p[valid], t[valid]
        mse = mean_squared_error(t, p)
        mae = mean_absolute_error(t, p)
        rmse = float(np.sqrt(mse))
        r2 = r2_score(t, p)
        # 处理零方差导致 corrcoef nan
        if np.std(p) < 1e-10 or np.std(t) < 1e-10:
            corr = 0.0
        else:
            corr = float(np.corrcoef(t, p)[0, 1])
        return float(mse), float(mae), rmse, float(r2), corr

    def _format_row(self, step, sidx, eidx, metrics, err, drift, timeout, upd, infer_time):
        ms, ma, rm, r2, co = metrics
        def _fmt(v):
            return f"{v:.6f}" if np.isfinite(v) else "NaN"
        return (f"{step},{sidx},{eidx},"
                f"{_fmt(ms)},{_fmt(ma)},{_fmt(rm)},{_fmt(r2)},{_fmt(co)},"
                f"{err},{drift},{timeout},{upd},"
                f"{infer_time:.3f}")

    # ============================================================
    # Phase 1: Reverse Distill
    # ============================================================
    def _do_reverse_distill(self, s_end):
        rs = int(max(0, s_end - Config.REV_DATA_WINDOW - Config.SEQ_LEN))
        tp = f"{Config.CKPT_DIR}/Global_Teacher/checkpoint.pth"
        sps = [f"{Config.CKPT_DIR}/Local_Edges/{p}/checkpoint.pth" for p in Config.PARTIES]
        print(f"   [RevDistill] ep={Config.REV_EPOCHS}, data=[{rs}:{s_end}]")
        cmd = (
            f"python -u run.py --task_name long_term_forecast --is_training 1 --do_reverse_distill "
            f"--checkpoints {Config.CKPT_DIR} "
            f"--root_path {Config.MERGE_PATH} --data_path {Config.DATA_NAME} "
            f"--specific_start {rs} --specific_end {s_end} "
            f"--scaler_path {Config.CKPT_DIR}/global_scaler/scaler.pkl "
            f"--model_id RevDistill_{self.update_counter} --model TimesNet --data custom "
            f"--features {Config.FEATURES} --seq_len {Config.SEQ_LEN} --label_len {Config.LABEL_LEN} --pred_len {Config.PRED_LEN} "
            f"--factor {Config.FACTOR} "
            f"--enc_in {Config.EDGE_DIM} --dec_in {Config.EDGE_DIM} --c_out {Config.EDGE_DIM} "
            f"--cloud_dim {Config.CLOUD_DIM} "
            f"--d_model {Config.T_D_MODEL} --d_ff {Config.T_D_FF} --e_layers {Config.T_LAYERS} --top_k {Config.T_TOP_K} "
            f"--teacher_d_model {Config.T_D_MODEL} --teacher_d_ff {Config.T_D_FF} "
            f"--student_model_path {' '.join(sps)} "
            f"--student_model_name {Config.S_MODEL} "
            f"--student_d_model {Config.S_D_MODEL} --student_d_ff {Config.S_D_FF} "
            f"--student_e_layers {Config.S_E_LAYERS} --student_d_layers {Config.S_D_LAYERS} "
            f"--pretrained_teacher_path {tp} "
            f"--ot_weight {Config.OT_WEIGHT} "
            f"{'--use_ofa_kd --ofa_eps ' + str(Config.OFA_EPS) + ' --ofa_loss_weight ' + str(Config.OFA_LOSS_WEIGHT) + ' --ofa_anchor_weight ' + str(Config.OFA_ANCHOR_WEIGHT) + ' ' if Config.USE_OFA_KD else ''}"
            f"--train_epochs {Config.REV_EPOCHS} --itr 1 --batch_size {Config.REV_BATCH_SIZE} "
            f"--learning_rate {Config.REV_LR} --rev_kd_weight {Config.REV_KD_WEIGHT}"
        )
        run_cmd(cmd)
        directory = find_latest_dir(
            Config.CKPT_DIR, f"RevDistill_{self.update_counter}")
        if not directory:
            raise FileNotFoundError('Reverse distillation output directory was not created.')
        src = f"{directory}/checkpoint_teacher_updated.pth"
        if not os.path.isfile(src):
            raise FileNotFoundError(f'Reverse distillation checkpoint missing: {src}')
        shutil.move(src, tp)
        cleanup(directory)
        print("   ✅ Teacher updated")

    # ============================================================
    # Phase 2: Forward Distill
    # ============================================================
    def _do_forward_distill(self, s_end):
        if Config.FWD_WINDOW_MODE == 'sliding':
            fs = int(max(0, s_end - Config.FWD_SLIDING_WINDOW))
        else:
            fs = int(Config.FWD_DATA_START)

        tp = f"{Config.CKPT_DIR}/Global_Teacher/checkpoint.pth"
        usp = f"{Config.CKPT_DIR}/Unified_Student/checkpoint.pth"

        print(f"   [FwdDistill] ep={Config.FWD_EPOCHS}, data=[{fs}:{s_end}] ({s_end-fs} rows)")
        cmd = (
            f"python -u run.py --task_name long_term_forecast --is_training 1 --do_distill "
            f"--checkpoints {Config.CKPT_DIR} "
            f"--root_path {Config.MERGE_PATH} --data_path {Config.DATA_NAME} "
            f"--specific_start {fs} --specific_end {s_end} "
            f"--scaler_path {Config.CKPT_DIR}/global_scaler/scaler.pkl "
            f"--train_ratio {Config.TRAIN_RATIO} --val_ratio {Config.VAL_RATIO} --test_ratio {Config.TEST_RATIO} "
            f"--model_id FwdDistill_{self.update_counter} --model {Config.S_MODEL} --data custom "
            f"--features {Config.FEATURES} --seq_len {Config.SEQ_LEN} --label_len {Config.LABEL_LEN} --pred_len {Config.PRED_LEN} "
            f"--factor {Config.FACTOR} "
            f"--enc_in {Config.EDGE_DIM} --dec_in {Config.EDGE_DIM} --c_out {Config.EDGE_DIM} "
            f"--cloud_dim {Config.CLOUD_DIM} "
            f"--d_model {Config.S_D_MODEL} --d_ff {Config.S_D_FF} "
            f"--e_layers {Config.S_E_LAYERS} --d_layers {Config.S_D_LAYERS} --top_k {Config.S_TOP_K} "
            f"--teacher_model_path {tp} --teacher_model_name TimesNet "
            f"--teacher_d_model {Config.T_D_MODEL} --teacher_d_ff {Config.T_D_FF} --teacher_e_layers {Config.T_LAYERS} "
            f"--lambda_kd 0.5 --ot_weight {Config.OT_WEIGHT} "
            f"{'--use_aux_kd ' if Config.USE_AUX_KD else ''}"
            f"{'--use_ofa_kd --ofa_eps ' + str(Config.OFA_EPS) + ' --ofa_loss_weight ' + str(Config.OFA_LOSS_WEIGHT) + ' --ofa_final_weight ' + str(Config.OFA_FINAL_WEIGHT) + ' ' if Config.USE_OFA_KD else ''}"
            f"--pretrained_model_path {usp} "
            f"--train_epochs {Config.FWD_EPOCHS} --itr 1 --batch_size {Config.FWD_BATCH_SIZE} "
            f"--patience {Config.FWD_PATIENCE} "
            f"--learning_rate {Config.FWD_LR}"
        )
        run_cmd(cmd)
        directory = find_latest_dir(
            Config.CKPT_DIR, f"FwdDistill_{self.update_counter}")
        if not directory:
            raise FileNotFoundError('Forward distillation output directory was not created.')
        src = f"{directory}/checkpoint.pth"
        if not os.path.isfile(src):
            raise FileNotFoundError(f'Forward distillation checkpoint missing: {src}')
        os.makedirs(os.path.dirname(usp), exist_ok=True)
        shutil.move(src, usp)
        cleanup(directory)
        print("   ✅ Unified Student rebuilt")

    # ============================================================
    # Phase 3: Local Fine-tune
    # ============================================================
    def _do_local_finetune(self, s_end, from_unified=False):
        fs = int(max(0, s_end - Config.FT_DATA_WINDOW - Config.SEQ_LEN))
        usp = f"{Config.CKPT_DIR}/Unified_Student/checkpoint.pth"
        mode = "rebuild" if from_unified else "incremental"
        print(f"   [LocalFT] {mode}, ep={Config.FT_EPOCHS}, data=[{fs}:{s_end}]")

        for p in Config.PARTIES:
            ec = f"{Config.CKPT_DIR}/Local_Edges/{p}/checkpoint.pth"
            if not os.path.isfile(ec):
                raise FileNotFoundError(f'Local edge checkpoint missing: {ec}')
            pre = usp if (from_unified and os.path.exists(usp)) else ec

            cmd = (
                f"python -u run.py --task_name long_term_forecast --is_training 1 --do_local_train "
                f"--checkpoints {Config.CKPT_DIR} "
                f"--root_path {Config.DATA_ROOT}/{p}/ --data_path {Config.DATA_NAME} "
                f"--specific_start {fs} --specific_end {s_end} "
                f"--pretrained_model_path {pre} "
                f"--scaler_path {Config.CKPT_DIR}/Local_Edges/{p}/scaler.pkl "
                f"--model_id LocalFT_{p}_{self.update_counter} --model {Config.S_MODEL} --data custom "
                f"--features {Config.FEATURES} --seq_len {Config.SEQ_LEN} --label_len {Config.LABEL_LEN} --pred_len {Config.PRED_LEN} "
                f"--factor {Config.FACTOR} "
                f"--enc_in {Config.EDGE_DIM} --dec_in {Config.EDGE_DIM} --c_out {Config.EDGE_DIM} "
                f"--d_model {Config.S_D_MODEL} --d_ff {Config.S_D_FF} "
                f"--e_layers {Config.S_E_LAYERS} --d_layers {Config.S_D_LAYERS} --top_k {Config.S_TOP_K} "
                f"--train_epochs {Config.FT_EPOCHS} --itr 1 --batch_size {Config.FT_BATCH_SIZE} "
                f"--learning_rate {Config.FT_LR}"
            )
            run_cmd(cmd)
            directory = find_latest_dir(
                Config.CKPT_DIR, f"LocalFT_{p}_{self.update_counter}")
            if not directory:
                raise FileNotFoundError(
                    f'Local fine-tuning output directory missing for {p}.')
            src = f"{directory}/checkpoint_personalized.pth"
            if not os.path.isfile(src):
                src = f"{directory}/checkpoint.pth"
            if not os.path.isfile(src):
                raise FileNotFoundError(f'Local fine-tuning checkpoint missing: {src}')
            shutil.copy2(src, ec)
            cleanup(directory)
            print(f"      ✅ {p} ({mode})")

    # ============================================================
    # Update Orchestration
    # ============================================================
    def cloud_edge_update(self):
        """
        执行 reverse / forward / local_ft 三个 phase.
        每个 phase 的耗时分别写入对应 CDF csv.
        local_ft 的耗时除以 N_PARTY (得到单边端平均时间).
        """
        s_end = self.current_idx
        if s_end - self.last_update_idx < Config.PRED_LEN:
            return

        self.update_counter += 1
        do_fwd = (self.update_counter % Config.FORWARD_DISTILL_INTERVAL == 0) and Config.ENABLE_FWD_DISTILL

        print(f"\n{'='*60}")
        print(f"⚡ Update #{self.update_counter} | idx={self.last_update_idx}->{s_end}")
        if do_fwd:
            print(f"   🌟 Full cycle: Rev → Fwd → LFT(rebuild)")
        else:
            print(f"   ⚡ Light cycle (Rev + LFT-incremental)")
        print(f"{'='*60}")

        # --- Phase 1: Reverse Distill ---
        if Config.ENABLE_REV_DISTILL:
            print("\n   Phase 1: Reverse Distill...")
            t0 = self._time.time()
            self._do_reverse_distill(s_end)
            dt = self._time.time() - t0
            _append_row(os.path.join(Config.RES_DIR, "durations_rev_distill.csv"),
                        f"{self.update_counter},{dt:.3f}")
            print(f"   [⏱ RevDistill] {dt:.2f}s")

        # --- Phase 2 + 3 ---
        if do_fwd:
            print(f"\n   Phase 2: Forward Distill...")
            t0 = self._time.time()
            self._do_forward_distill(s_end)
            dt = self._time.time() - t0
            _append_row(os.path.join(Config.RES_DIR, "durations_fwd_distill.csv"),
                        f"{self.update_counter},{dt:.3f}")
            print(f"   [⏱ FwdDistill] {dt:.2f}s")

            if Config.ENABLE_LOCAL_FT:
                print(f"\n   Phase 3: Local Fine-tune (rebuild)...")
                t0 = self._time.time()
                self._do_local_finetune(s_end, from_unified=True)
                dt_total = self._time.time() - t0
                dt_per_edge = dt_total / max(Config.N_PARTY, 1)
                _append_row(os.path.join(Config.RES_DIR, "durations_local_ft.csv"),
                            f"{self.update_counter},{dt_per_edge:.3f}")
                print(f"   [⏱ LFT-rebuild] total={dt_total:.2f}s, per-edge={dt_per_edge:.2f}s")
        else:
            if Config.ENABLE_LOCAL_FT:
                print(f"\n   Phase 2: Local Fine-tune (incremental)...")
                t0 = self._time.time()
                self._do_local_finetune(s_end, from_unified=False)
                dt_total = self._time.time() - t0
                dt_per_edge = dt_total / max(Config.N_PARTY, 1)
                _append_row(os.path.join(Config.RES_DIR, "durations_local_ft.csv"),
                            f"{self.update_counter},{dt_per_edge:.3f}")
                print(f"   [⏱ LFT-incremental] total={dt_total:.2f}s, per-edge={dt_per_edge:.2f}s")

        self.last_update_idx = self.current_idx
        # 清理 trigger buffer 防止旧数据干扰下次触发判定
        keep = Config.PRED_LEN * 7
        for p in Config.PARTIES:
            self.results_buffer[p]['OT'] = self.results_buffer[p]['OT'][-keep:]
            self.results_buffer[p]['Pred'] = self.results_buffer[p]['Pred'][-keep:]

        print(f"\n✅ Update #{self.update_counter} done.\n")

    # ============================================================
    # Inference (主循环)
    # ============================================================
    def _infer_one_party_step(self, party, sidx, eidx):
        """
        对单个 party 跑一次 inference, 返回 (pred_2d, true_2d):
            shape = [N, EDGE_DIM] (每个时间点的 step-0 预测)
        """
        rid = f"Infer_{party}_{sidx}"
        cmd = (
            f"python -u run.py --task_name long_term_forecast --is_training 0 "
            f"--checkpoints {Config.CKPT_DIR} "
            f"--root_path {Config.DATA_ROOT}/{party}/ --data_path {Config.DATA_NAME} "
            f"--specific_start {int(sidx)} --specific_end {int(eidx)} "
            f"--model_id {rid} --model {Config.S_MODEL} --data custom "
            f"--features {Config.FEATURES} --seq_len {Config.SEQ_LEN} --label_len {Config.LABEL_LEN} --pred_len {Config.PRED_LEN} "
            f"--factor {Config.FACTOR} "
            f"--enc_in {Config.EDGE_DIM} --dec_in {Config.EDGE_DIM} --c_out {Config.EDGE_DIM} "
            f"--d_model {Config.S_D_MODEL} --d_ff {Config.S_D_FF} "
            f"--e_layers {Config.S_E_LAYERS} --d_layers {Config.S_D_LAYERS} --top_k {Config.S_TOP_K} "
            f"--pretrained_model_path {Config.CKPT_DIR}/Local_Edges/{party}/checkpoint.pth "
            f"--scaler_path {Config.CKPT_DIR}/Local_Edges/{party}/scaler.pkl "
            f"--batch_size {Config.INFER_BATCH_SIZE} --gpu 0"
        )
        try:
            run_cmd(cmd)
            fld = find_latest_dir("./results", rid)
            if not fld:
                return None, None
            pf, tf = f"{fld}/pred.npy", f"{fld}/true.npy"
            if not (os.path.exists(pf) and os.path.exists(tf)):
                cleanup(fld); return None, None
            preds = np.load(pf)
            trues = np.load(tf)
            if preds.ndim == 2:
                n_feat = preds.shape[-1]
                preds = preds.reshape(-1, Config.PRED_LEN, n_feat)
                trues = trues.reshape(-1, Config.PRED_LEN, n_feat)
            sp, sg = preds[:, 0, :], trues[:, 0, :]
            n = min(len(sp), len(sg))
            cleanup(fld)
            return sp[:n], sg[:n]
        except Exception as e:
            print(f"   {party}: {e}")
            return None, None

    def run_simulation(self):
        print(f"\nStart idx={self.current_idx}, total={self.total_length}\n")
        sim_start = self._time.time()

        while self.current_idx < self.total_length - Config.SEQ_LEN - Config.PRED_LEN:
            ie = min(self.current_idx + Config.SEQ_LEN + Config.INFER_STEP + Config.PRED_LEN - 1,
                     self.total_length)
            sidx, eidx = self.current_idx, ie

            print(f"\n[Step {self.step_counter} | idx={sidx}:{eidx}] Inferring...")
            infer_start = self._time.time()

            # 1) 跑每个 party 的 inference, 收集这一步的 (pred, true)
            step_data = {}      # {party: (pred_2d, true_2d)}
            for p in Config.PARTIES:
                sp, sg = self._infer_one_party_step(p, sidx, eidx)
                if sp is None:
                    raise RuntimeError(
                        f'Inference failed for {p} at window [{sidx}:{eidx}].')
                step_data[p] = (sp, sg)
                # 累积到 trigger buffer (用于跨 step 滚动检测)
                # Only the shared target drives error/drift triggers.
                self.results_buffer[p]['Pred'].extend(sp[:, -1].tolist())
                self.results_buffer[p]['OT'].extend(sg[:, -1].tolist())
                # 累积到 global eval (用于 final report)
                self.global_eval[p]['Pred'].extend(sp.tolist())
                self.global_eval[p]['OT'].extend(sg.tolist())
                print(f"   -> {p}: {len(sp)} samples")

            infer_time = self._time.time() - infer_start

            # 2) 每个 party 的 trigger 检测 (注意每个 party 都判)
            party_triggers = {}   # party -> (err, drift, msg)
            any_err = any_drift = 0
            first_msg = None
            for p in Config.PARTIES:
                err, drift, msg = self.check_trigger_party(p)
                party_triggers[p] = (err, drift, msg)
                if err: any_err = 1
                if drift: any_drift = 1
                if msg and first_msg is None:
                    first_msg = msg

            # 3) Timeout 检测
            ni = self.current_idx + Config.INFER_STEP
            timeout = 0
            if ni - self.last_update_idx > Config.MAX_NO_UPDATE:
                timeout = 1
                if first_msg is None:
                    first_msg = "Timeout"

            will_update = (any_err or any_drift or timeout)

            # 4) 写入每个 party 的 per-step CSV
            for p in Config.PARTIES:
                if p in step_data:
                    sp, sg = step_data[p]
                    metrics = self._metrics_from_arrays(sp, sg)
                else:
                    metrics = (float('nan'),) * 5

                err, drift, _ = party_triggers[p]
                row = self._format_row(
                    self.step_counter, sidx, eidx,
                    metrics, err, drift, timeout,
                    1 if will_update else 0,
                    infer_time
                )
                _append_row(self._step_csv_path(p), row)

            # 5) 计算 edge_avg 并写入
            self._write_edge_avg_row(
                step_data, party_triggers,
                sidx, eidx, any_err, any_drift,
                timeout, 1 if will_update else 0, infer_time
            )

            # 6) 决定是否触发更新
            if will_update:
                # 统计 trigger 类型
                if any_err:
                    self.trigger_counts['error'] += 1
                elif any_drift:
                    self.trigger_counts['drift'] += 1
                elif timeout:
                    self.trigger_counts['timeout'] += 1
                self.trigger_counts['total'] += 1

                print(f"\n🔔 Trigger: {first_msg} (err={any_err}, drift={any_drift}, "
                      f"timeout={timeout}) -> update")
                self.current_idx = ni
                self._refresh_error_references()
                self.cloud_edge_update()
            else:
                self.current_idx = ni

            self.step_counter += 1
            if (self.current_idx % (Config.INFER_STEP * 5) == 0
                    and torch.cuda.is_available()):
                torch.cuda.empty_cache()

        sim_total = self._time.time() - sim_start
        print(f"\nDone in {sim_total:.1f}s. Steps={self.step_counter}, "
              f"Updates={self.update_counter}")
        self.generate_final_report(sim_total)

    # ============================================================
    # Edge AVG 行
    # ============================================================
    def _write_edge_avg_row(self, step_data, party_triggers,
                            sidx, eidx, any_err, any_drift,
                            timeout_trig, update_trig, infer_time):
        """对所有有效 party 的本 step 指标求平均, 写一行到 edge_avg.csv"""
        all_pred, all_true = [], []
        for p, (sp, sg) in step_data.items():
            all_pred.append(sp.flatten())
            all_true.append(sg.flatten())
        if all_pred:
            cat_p = np.concatenate(all_pred)
            cat_t = np.concatenate(all_true)
            metrics = self._metrics_from_arrays(cat_p, cat_t)
        else:
            metrics = (float('nan'),) * 5

        row = self._format_row(
            self.step_counter, sidx, eidx, metrics,
            any_err, any_drift, timeout_trig, update_trig, infer_time
        )
        _append_row(self._step_csv_path('edge_avg'), row)

    # ============================================================
    # Final Report
    # ============================================================
    def generate_final_report(self, sim_total):
        rp = os.path.join(Config.RES_DIR, "final_report.txt")
        with open(rp, "w") as f:
            def log(m):
                print(m); f.write(m + "\n")

            log("\n" + "=" * 60)
            log("FINAL REPORT (per-step CSVs already written)")
            log("=" * 60)

            # 边端整体指标 (从累积 global_eval 算, 与 per-step CSV 互补)
            agg = {'mse': [], 'mae': [], 'rmse': [], 'r2': [], 'corr': []}
            for p in Config.PARTIES:
                pr = np.array(self.global_eval[p]['Pred']).flatten()
                tr = np.array(self.global_eval[p]['OT']).flatten()
                if len(pr) == 0:
                    continue
                ms, ma, rm, r2, co = self._metrics_from_arrays(pr, tr)
                log(f"{p:>12}: MSE={ms:.4f} MAE={ma:.4f} RMSE={rm:.4f} R2={r2:.4f} CORR={co:.4f}  "
                    f"(n={len(pr)})")
                agg['mse'].append(ms); agg['mae'].append(ma)
                agg['rmse'].append(rm); agg['r2'].append(r2); agg['corr'].append(co)
            log("-" * 60)
            log("Edge AVG (across-party mean):")
            for k, v in agg.items():
                if v: log(f"  {k.upper()}: {np.mean(v):.4f}")

            # ===== Trigger 统计 =====
            log("\n" + "=" * 60)
            log("TRIGGER STATISTICS")
            log("=" * 60)
            total = self.trigger_counts['total']
            log(f"  Total updates triggered: {total}")
            log(f"  Total steps:             {self.step_counter}")
            if self.step_counter > 0:
                log(f"  Update rate:             {total/self.step_counter*100:.1f}%")
            for ttype in ['error', 'drift', 'timeout']:
                cnt = self.trigger_counts[ttype]
                pct = cnt / max(total, 1) * 100
                log(f"  {ttype:>8}: {cnt:>4} ({pct:.1f}%)")

            # ===== Duration CDF 文件说明 =====
            log("\n" + "=" * 60)
            log("DURATION CDFs (for plotting)")
            log("=" * 60)
            log(f"  - durations_rev_distill.csv  : N={self._count_rows('durations_rev_distill.csv')}")
            log(f"  - durations_fwd_distill.csv  : N={self._count_rows('durations_fwd_distill.csv')}")
            log(f"  - durations_local_ft.csv     : N={self._count_rows('durations_local_ft.csv')}"
                f"  (per-edge time = total / {Config.N_PARTY})")
            log(f"  Total simulation: {sim_total:.1f}s")
            log("=" * 60)

            summary_path = os.path.join(Config.RES_DIR, "sensitivity_summary.csv")
            avg_metrics = {k: (float(np.mean(v)) if v else float('nan')) for k, v in agg.items()}
            duration_means = {
                "avg_rev_distill_time": self._mean_duration("durations_rev_distill.csv"),
                "avg_fwd_distill_time": self._mean_duration("durations_fwd_distill.csv"),
                "avg_local_ft_time": self._mean_duration("durations_local_ft.csv"),
            }
            with open(summary_path, "w") as sf:
                sf.write(
                    "data_name,max_no_update,eta,delta,R,"
                    "avgMAE,avgMSE,avgRMSE,avgR2,avgCorr,"
                    "reverse_count,forward_count,total_updates,total_steps,"
                    "error_count,drift_count,timeout_count,"
                    "avg_rev_distill_time,avg_fwd_distill_time,avg_local_ft_time,total_sim_time\n"
                )
                sf.write(
                    f"{Config.DATA_NAME},{Config.MAX_NO_UPDATE},{Config.ERR_THRESH_RATIO},"
                    f"{Config.KL_THRESH},{Config.FORWARD_DISTILL_INTERVAL},"
                    f"{avg_metrics['mae']:.6f},{avg_metrics['mse']:.6f},"
                    f"{avg_metrics['rmse']:.6f},{avg_metrics['r2']:.6f},{avg_metrics['corr']:.6f},"
                    f"{self._count_rows('durations_rev_distill.csv')},"
                    f"{self._count_rows('durations_fwd_distill.csv')},"
                    f"{total},{self.step_counter},"
                    f"{self.trigger_counts['error']},{self.trigger_counts['drift']},"
                    f"{self.trigger_counts['timeout']},"
                    f"{duration_means['avg_rev_distill_time']:.6f},"
                    f"{duration_means['avg_fwd_distill_time']:.6f},"
                    f"{duration_means['avg_local_ft_time']:.6f},"
                    f"{sim_total:.3f}\n"
                )
            log(f"Summary CSV: {summary_path}")

        print(f"\nReport: {rp}")

    def _count_rows(self, fname):
        path = os.path.join(Config.RES_DIR, fname)
        if not os.path.exists(path):
            return 0
        with open(path) as f:
            return max(0, sum(1 for _ in f) - 1)   # 减 header

    def _mean_duration(self, fname):
        path = os.path.join(Config.RES_DIR, fname)
        if not os.path.exists(path):
            return float('nan')
        values = []
        with open(path) as f:
            next(f, None)
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 2:
                    try:
                        values.append(float(parts[1]))
                    except ValueError:
                        pass
        return float(np.mean(values)) if values else float('nan')


if __name__ == "__main__":
    # 注意: 不再 cleanup(Config.RES_DIR), 否则会把之前 group 的结果一并删掉.
    # bash 调用方负责给每个 group 用独立的 RT_RES_DIR.
    RealTimeSimulator().run_simulation()
