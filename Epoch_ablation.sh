#!/bin/bash
# =============================================================================
#  4-Group Single-Shot Ablation
#  G1: 无 RT, 用 LE_Backpack 初始化做纯推理
#  G2: 有 RT, response-only KD
#  G3: 有 RT, aux KD (外挂 aux head, 蒸馏完丢弃)
#  G4: 有 RT, OFA-KD (本方法)
#
#  ★ 所有可调参数集中在下面 [CONFIG] 区. 顶部改完, 下面所有代码自动同步.
# =============================================================================

set -eo pipefail

# ============================================================================ #
# [CONFIG]   所有可调项都在这里. 改这里就够了, 不需要在 real_time_inference 里改.
# ============================================================================ #

# ---------- GPU ----------
export CUDA_VISIBLE_DEVICES=5
GPU_ID=0

# ---------- 数据集 (ST) ----------
DATA_ROOT="./dataset/processed"
MERGE_PATH="./dataset/processed/merge/"
DATA_NAME="PVOD.csv"
PARTIES=("party_1" "party_2" "party_3" "party_4" "party_5"
         "party_6" "party_7")
CLOUD_DIM=99
EDGE_DIM=15
INIT_START=0
INIT_END=18750       # ★ 训练/验证集结束位置, 之后是实时推理区

# ---------- 时序窗口 ----------
SEQ_LEN=12
PRED_LEN=6
LABEL_LEN=12
FEATURES="M"
FACTOR=3

# ---------- 数据划分 ----------
TRAIN_RATIO=0.8
VAL_RATIO=0.2
TEST_RATIO=0.0
BATCH_SIZE=32

# ---------- 教师模型 (TimesNet) ----------
T_D_MODEL=128
T_D_FF=256
T_E_LAYERS=2

# ---------- 学生模型 (CNN) ----------
S_MODEL="CNN"
S_D_MODEL=16
S_D_FF=32
S_E_LAYERS=2
S_D_LAYERS=1

# ---------- 蒸馏 (初始化用) ----------
DISTILL_EPOCHS=10
STUDENT_LR=0.0001
LAMBDA_KD=0.3
OT_WEIGHT=1.0
REV_KD_WEIGHT=0.01

# ---------- 本地微调 (初始化用) ----------
LOCAL_FT_EPOCHS=5
LOCAL_LR=0.0001

# ---------- 实时推理: 模拟器主循环参数 ----------
# ★ INFER_STEP: 每次推进多少个时间点 (越小越细粒度, 触发越频繁)
INFER_STEP=48

# ---------- 实时推理: 每次 update 的训练强度 ----------
REV_EPOCHS=1      # 反向蒸馏 epoch
FWD_EPOCHS=5      # 正向蒸馏 epoch
FT_EPOCHS=3       # 本地微调 epoch

# ---------- 触发阈值 ----------
ERR_THRESH_RATIO=1.5     # MSE 超过 base * ratio 时触发 error
ERR_MIN_ABSOLUTE=0.25     # MSE 绝对阈值下限
KL_THRESH=0.4            # KL 散度阈值 (drift)

# ---------- OFA-KD 超参数 ----------
OFA_EPS=1.2
OFA_LOSS_WEIGHT=0.2        # ★ 太大可能让 teacher 偏离; 0.1-0.5 之间调
OFA_FINAL_WEIGHT=0.2
OFA_ANCHOR_WEIGHT=0.1

# ---------- 备份路径 (每组开始前从这里恢复) ----------
GT_BACKUP="./checkpoints/GT_back/checkpoint.pth"
SCALER_BACKUP="./checkpoints/GT_back/scaler.pkl"
LE_BACKUP_DIR="./checkpoints/LE_Backpack"

# ---------- 输出 ----------
EXP_DIR="./ablation_results_v5"

# ---------- 4 组实验定义 ----------
# 格式: tag use_rt use_aux use_ofa skip_init
#   skip_init=true → 直接用 LE_Backpack 起跑 (跳过蒸馏 + 本地微调)
#   skip_init=false → 每次重新蒸馏 + 本地微调 (公平对比起点)
GROUPS_CONFIG=(
    "G4_rt_ofa_kd    true    false   true    false"
    "G3_rt_aux_kd    true    true    false   false"
    "G2_rt_no_kd     true    false   false   false"
    "G1_no_rt_no_kd  false   false   false   true"
)

# ============================================================================ #
# [END OF CONFIG]   下面的代码不需要改, 全部从上面变量取值.
# ============================================================================ #


mkdir -p "$EXP_DIR"

# ============ 辅助 ============
get_latest_dir() { ls -td "$1"/*"$2"* 2>/dev/null | head -1; }
gpu_cleanup() { python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true; }

full_cleanup() {
    find ./checkpoints -maxdepth 1 -type d \( -name "*Distill_G*" -o -name "*LFT_*" \
        -o -name "*RevDistill_*" -o -name "*FwdDistill_*" -o -name "*LocalFT_*" \
        -o -name "*Infer_*" \) -exec rm -rf {} + 2>/dev/null
    find ./results -maxdepth 1 -type d -exec rm -rf {} + 2>/dev/null
    gpu_cleanup
}

# ============ 恢复初始状态 ============
restore_initial_state() {
    if [ ! -f "$GT_BACKUP" ]; then
        echo "  ❌ 找不到 $GT_BACKUP, 请先备份教师"
        exit 1
    fi
    if [ ! -f "$SCALER_BACKUP" ]; then
        echo "  ❌ 找不到 $SCALER_BACKUP, 请先运行初始化脚本"
        exit 1
    fi
    mkdir -p ./checkpoints/Global_Teacher
    cp "$GT_BACKUP" ./checkpoints/Global_Teacher/checkpoint.pth
    mkdir -p ./checkpoints/global_scaler
    cp "$SCALER_BACKUP" ./checkpoints/global_scaler/scaler.pkl
    echo "  [RESTORE] Global_Teacher <- $GT_BACKUP"

    for PARTY in "${PARTIES[@]}"; do
        local SRC_DIR="${LE_BACKUP_DIR}/${PARTY}"
        local DST_DIR="./checkpoints/Local_Edges/${PARTY}"
        for ARTIFACT in checkpoint.pth scaler.pkl; do
            if [ ! -f "${SRC_DIR}/${ARTIFACT}" ]; then
                echo "  ❌ 找不到 ${SRC_DIR}/${ARTIFACT}"
                exit 1
            fi
        done
        mkdir -p "$DST_DIR"
        cp "${SRC_DIR}/checkpoint.pth" "${DST_DIR}/checkpoint.pth"
        cp "${SRC_DIR}/scaler.pkl" "${DST_DIR}/scaler.pkl"
    done
    echo "  [RESTORE] Local_Edges <- $LE_BACKUP_DIR (${#PARTIES[@]} parties)"
    rm -f ./checkpoints/Unified_Student/checkpoint.pth
}

# ============ 蒸馏 (初始化学生) ============
run_distill() {
    local TEACHER_PATH=$1; local SAVE_PATH=$2
    local USE_AUX=$3; local USE_OFA=$4; local TAG=$5
    local DID="Distill_${TAG}"

    local AUX_FLAG=""
    [ "$USE_AUX" = "true" ] && AUX_FLAG="--use_aux_kd"
    local OFA_FLAGS=""
    if [ "$USE_OFA" = "true" ]; then
        OFA_FLAGS="--use_ofa_kd --ofa_eps $OFA_EPS --ofa_loss_weight $OFA_LOSS_WEIGHT --ofa_final_weight $OFA_FINAL_WEIGHT"
    fi

    echo "  [Distill] tag=$TAG use_aux=$USE_AUX use_ofa=$USE_OFA epochs=$DISTILL_EPOCHS"
    python -u run.py \
        --task_name long_term_forecast --is_training 1 --do_distill \
        --root_path $MERGE_PATH --data_path $DATA_NAME \
        --scaler_path ./checkpoints/global_scaler/scaler.pkl \
        --specific_start $INIT_START --specific_end $INIT_END \
        --train_ratio $TRAIN_RATIO --val_ratio $VAL_RATIO --test_ratio $TEST_RATIO \
        --model_id "$DID" --model $S_MODEL --data custom \
        --features $FEATURES --seq_len $SEQ_LEN --label_len $LABEL_LEN --pred_len $PRED_LEN \
        --factor $FACTOR \
        --enc_in $EDGE_DIM --dec_in $EDGE_DIM --c_out $EDGE_DIM \
        --d_model $S_D_MODEL --d_ff $S_D_FF --e_layers $S_E_LAYERS --d_layers $S_D_LAYERS \
        --cloud_dim $CLOUD_DIM \
        --teacher_model_name TimesNet --teacher_model_path "$TEACHER_PATH" \
        --teacher_d_model $T_D_MODEL --teacher_d_ff $T_D_FF --teacher_e_layers $T_E_LAYERS \
        --lambda_kd $LAMBDA_KD --ot_weight $OT_WEIGHT \
        $AUX_FLAG $OFA_FLAGS \
        --train_epochs $DISTILL_EPOCHS --batch_size $BATCH_SIZE \
        --learning_rate $STUDENT_LR --gpu $GPU_ID --patience 3

    local DIR=$(get_latest_dir "./checkpoints" "$DID")
    if [ -d "$DIR" ] && [ -f "${DIR}/checkpoint.pth" ]; then
        cp "${DIR}/checkpoint.pth" "$SAVE_PATH"
        rm -rf "$DIR"
        echo "  [Distill] saved -> $SAVE_PATH"
    else
        echo "  [Distill] FAILED"
        return 1
    fi
    gpu_cleanup
}

# ============ 本地微调 ============
run_local_ft() {
    local STUDENT_PATH=$1; local PARTY=$2; local SAVE_PATH=$3
    local FID="LFT_${PARTY}"
    mkdir -p "$(dirname $SAVE_PATH)"

    python -u run.py \
        --task_name long_term_forecast --is_training 1 --do_local_train \
        --pretrained_model_path "$STUDENT_PATH" \
        --scaler_path "./checkpoints/Local_Edges/${PARTY}/scaler.pkl" \
        --root_path "${DATA_ROOT}/${PARTY}/" --data_path $DATA_NAME \
        --specific_start $INIT_START --specific_end $INIT_END \
        --train_ratio $TRAIN_RATIO --val_ratio $VAL_RATIO --test_ratio $TEST_RATIO \
        --model_id "$FID" --model $S_MODEL --data custom \
        --features $FEATURES --seq_len $SEQ_LEN --label_len $LABEL_LEN --pred_len $PRED_LEN \
        --factor $FACTOR \
        --enc_in $EDGE_DIM --dec_in $EDGE_DIM --c_out $EDGE_DIM \
        --d_model $S_D_MODEL --d_ff $S_D_FF --e_layers $S_E_LAYERS --d_layers $S_D_LAYERS \
        --train_epochs $LOCAL_FT_EPOCHS --batch_size $BATCH_SIZE \
        --learning_rate $LOCAL_LR --gpu $GPU_ID --patience 3

    local DIR=$(get_latest_dir "./checkpoints" "$FID")
    if [ -d "$DIR" ]; then
        if [ -f "${DIR}/checkpoint_personalized.pth" ]; then
            cp "${DIR}/checkpoint_personalized.pth" "$SAVE_PATH"
        elif [ -f "${DIR}/checkpoint.pth" ]; then
            cp "${DIR}/checkpoint.pth" "$SAVE_PATH"
        else
            echo "  [LocalFT] checkpoint missing for $PARTY"
            return 1
        fi
        rm -rf "$DIR"
    else
        echo "  [LocalFT] output directory missing for $PARTY"
        return 1
    fi
    gpu_cleanup
}

# ============ G1 纯推理脚本 ============
G1_INFER_SCRIPT=$(cat <<'PYEOF'
"""G1: 对每个 party 跑 run.py --is_training 0 (在 INIT_END 之后),
    按 INFER_STEP 切片写入与其他组一致的 per-step CSV (trigger 全 0)."""
import os, shlex, sys, subprocess, shutil, numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

RES_DIR = sys.argv[1]
PARTIES = sys.argv[2].split(',')
INIT_END = int(sys.argv[3])
INFER_STEP = int(sys.argv[4])
PRED_LEN = int(sys.argv[5])
SEQ_LEN = int(sys.argv[6])
DATA_ROOT = sys.argv[7]
DATA_NAME = sys.argv[8]
EDGE_DIM = int(sys.argv[9])
S_MODEL = sys.argv[10]; S_D_MODEL = int(sys.argv[11]); S_D_FF = int(sys.argv[12])
S_E_LAYERS = int(sys.argv[13]); S_D_LAYERS = int(sys.argv[14])
CKPT_DIR = sys.argv[15]
BATCH_SIZE = int(sys.argv[16]); GPU_ID = int(sys.argv[17])
LABEL_LEN = int(sys.argv[18]); FEATURES = sys.argv[19]; FACTOR = int(sys.argv[20])

os.makedirs(RES_DIR, exist_ok=True)
HEADER = ("step,start_idx,end_idx,mse,mae,rmse,r2,corr,"
          "error_trigger,drift_trigger,update_trigger,infer_time\n")
CDF_H = "update_id,duration_sec\n"

for fname in PARTIES + ['edge_avg']:
    p = os.path.join(RES_DIR, f"{fname}.csv")
    if not os.path.exists(p):
        open(p, "w").write(HEADER)
for fname in ['durations_rev_distill.csv', 'durations_fwd_distill.csv', 'durations_local_ft.csv']:
    p = os.path.join(RES_DIR, fname)
    if not os.path.exists(p):
        open(p, "w").write(CDF_H)

def metrics(p, t):
    p, t = np.array(p).flatten(), np.array(t).flatten()
    v = np.isfinite(p) & np.isfinite(t)
    if v.sum() < 2:
        return [float('nan')] * 5
    p, t = p[v], t[v]
    mse = mean_squared_error(t, p); mae = mean_absolute_error(t, p)
    rmse = float(np.sqrt(mse)); r2 = r2_score(t, p)
    corr = 0.0 if (np.std(p) < 1e-10 or np.std(t) < 1e-10) else float(np.corrcoef(t, p)[0, 1])
    return [float(mse), float(mae), rmse, float(r2), corr]

def fmt(v): return f"{v:.6f}" if np.isfinite(v) else "NaN"

# 读数据集总长 (用于 while 循环边界)
import pandas as pd
df0 = pd.read_csv(os.path.join(DATA_ROOT, PARTIES[0], DATA_NAME))
total_length = len(df0) - (SEQ_LEN + PRED_LEN * 2)
print(f"[G1] usable length: {total_length}, start: {INIT_END}, INFER_STEP: {INFER_STEP}")


def infer_one_chunk(party, sidx, eidx):
    """单 party + 单 chunk 调 run.py 推理 -> (pred_2d, true_2d) or (None, None)
       chunk 太小, Dataset_Custom 的 70/20/10 切分会自动失效, 整段都进 test."""
    rid = f"G1Infer_{party}_{sidx}"
    cmd = (f"python -u run.py --task_name long_term_forecast --is_training 0 "
           f"--root_path {DATA_ROOT}/{party}/ --data_path {DATA_NAME} "
           f"--specific_start {int(sidx)} --specific_end {int(eidx)} "
           f"--model_id {rid} --model {S_MODEL} --data custom "
           f"--features {FEATURES} --seq_len {SEQ_LEN} --label_len {LABEL_LEN} --pred_len {PRED_LEN} "
           f"--factor {FACTOR} "
           f"--enc_in {EDGE_DIM} --dec_in {EDGE_DIM} --c_out {EDGE_DIM} "
           f"--d_model {S_D_MODEL} --d_ff {S_D_FF} "
           f"--e_layers {S_E_LAYERS} --d_layers {S_D_LAYERS} "
           f"--pretrained_model_path {CKPT_DIR}/Local_Edges/{party}/checkpoint.pth "
           f"--scaler_path {CKPT_DIR}/Local_Edges/{party}/scaler.pkl "
           f"--batch_size {BATCH_SIZE} --gpu {GPU_ID}")
    argv = shlex.split(cmd)
    argv[0] = sys.executable
    subprocess.run(argv, shell=False, check=True)
    cands = ([d for d in os.listdir("./results") if rid in d]
             if os.path.exists("./results") else [])
    if not cands:
        raise FileNotFoundError(f"Inference results missing for {party}: {rid}")
    fld = os.path.join("./results",
        sorted(cands, key=lambda x: os.path.getmtime(os.path.join("./results", x)))[-1])
    pf, tf = f"{fld}/pred.npy", f"{fld}/true.npy"
    if not (os.path.exists(pf) and os.path.exists(tf)):
        shutil.rmtree(fld, ignore_errors=True)
        raise FileNotFoundError(f"Prediction arrays missing in {fld}")
    pr = np.load(pf); tr = np.load(tf)
    if pr.ndim == 2:
        n_feat = pr.shape[-1]
        pr = pr.reshape(-1, PRED_LEN, n_feat)
        tr = tr.reshape(-1, PRED_LEN, n_feat)
    shutil.rmtree(fld, ignore_errors=True)
    return pr[:, 0, :], tr[:, 0, :]


# 主循环 (与 G2/G3/G4 完全相同的 chunk-by-chunk 结构)
current_idx = INIT_END
step = 0
global_preds = {p: [] for p in PARTIES}    # 用于 final_report
global_trues = {p: [] for p in PARTIES}

while current_idx < total_length - SEQ_LEN - PRED_LEN:
    eidx = min(current_idx + SEQ_LEN + INFER_STEP + PRED_LEN - 1, total_length)
    print(f"[G1 Step {step}] idx={current_idx}:{eidx}")

    step_data = {}
    for p in PARTIES:
        sp, sg = infer_one_chunk(p, current_idx, eidx)
        if sp is not None:
            step_data[p] = (sp, sg)
            global_preds[p].append(sp); global_trues[p].append(sg)

    # 每 party 写一行
    all_p_step, all_t_step = [], []
    for p in PARTIES:
        if p in step_data:
            sp, sg = step_data[p]
            m = metrics(sp, sg)
            all_p_step.append(sp.flatten()); all_t_step.append(sg.flatten())
            row = (f"{step},{current_idx},{eidx},"
                   f"{fmt(m[0])},{fmt(m[1])},{fmt(m[2])},{fmt(m[3])},{fmt(m[4])},"
                   f"0,0,0,0.000")
        else:
            row = f"{step},{current_idx},{eidx},NaN,NaN,NaN,NaN,NaN,0,0,0,0.000"
        open(os.path.join(RES_DIR, f"{p}.csv"), "a").write(row + "\n")

    if all_p_step:
        m = metrics(np.concatenate(all_p_step), np.concatenate(all_t_step))
        row = (f"{step},{current_idx},{eidx},"
               f"{fmt(m[0])},{fmt(m[1])},{fmt(m[2])},{fmt(m[3])},{fmt(m[4])},"
               f"0,0,0,0.000")
        open(os.path.join(RES_DIR, "edge_avg.csv"), "a").write(row + "\n")

    current_idx += INFER_STEP
    step += 1

print(f"[G1] done, {step} steps written")


# ============ Final report (累积所有 step) ============
report = ["=" * 60, "G1 FINAL REPORT (无 RT 更新, 累积所有 step)", "=" * 60]
agg = {'mse': [], 'mae': [], 'rmse': [], 'r2': [], 'corr': []}
all_p_global, all_t_global = [], []
for p in PARTIES:
    if not global_preds[p]:
        continue
    pr = np.concatenate(global_preds[p], axis=0)
    tr = np.concatenate(global_trues[p], axis=0)
    m = metrics(pr, tr)
    report.append(f"{p:>12}: MSE={m[0]:.4f} MAE={m[1]:.4f} RMSE={m[2]:.4f} "
                  f"R2={m[3]:.4f} CORR={m[4]:.4f}  (n={pr.size})")
    agg['mse'].append(m[0]); agg['mae'].append(m[1])
    agg['rmse'].append(m[2]); agg['r2'].append(m[3]); agg['corr'].append(m[4])
    all_p_global.append(pr.flatten()); all_t_global.append(tr.flatten())

report.append("-" * 60)
report.append("Edge AVG (per-party metrics mean):")
for k in ['mse', 'mae', 'rmse', 'r2', 'corr']:
    if agg[k]:
        report.append(f"  {k.upper()}: {np.mean(agg[k]):.4f}")

if all_p_global:
    cat_p = np.concatenate(all_p_global); cat_t = np.concatenate(all_t_global)
    m_all = metrics(cat_p, cat_t)
    report.append("-" * 60)
    report.append("Pooled (concat all parties):")
    report.append(f"  MSE={m_all[0]:.4f} MAE={m_all[1]:.4f} RMSE={m_all[2]:.4f} "
                  f"R2={m_all[3]:.4f} CORR={m_all[4]:.4f}")
report.append("=" * 60)
text = "\n".join(report)
print(text)
with open(os.path.join(RES_DIR, "final_report.txt"), "w") as f:
    f.write(text + "\n")
PYEOF
)

# ============ 实时推理 (G2/G3/G4) ============
run_realtime() {
    local TAG=$1; local USE_AUX=$2; local USE_OFA=$3
    local RT_OUT="${EXP_DIR}/rt_${TAG}"
    rm -rf "$RT_OUT"
    mkdir -p "$RT_OUT"
    # case "$TAG" in
    #     *G2*)
    #         REV_KD_WEIGHT=0.1
    # esac
    # ★ 所有可调参数全部通过 env var 传给 real_time_inference.py
    export RT_INFER_STEP="$INFER_STEP"
    export RT_REV_EPOCHS="$REV_EPOCHS"
    export RT_FWD_EPOCHS="$FWD_EPOCHS"
    export RT_FT_EPOCHS="$FT_EPOCHS"
    export RT_USE_AUX_KD=$([ "$USE_AUX" = "true" ] && echo 1 || echo 0)
    export RT_USE_OFA_KD=$([ "$USE_OFA" = "true" ] && echo 1 || echo 0)
    export RT_RES_DIR="$RT_OUT"
    export RT_N_PARTY=${#PARTIES[@]}
    export RT_CLOUD_DIM="$CLOUD_DIM"
    export RT_EDGE_DIM="$EDGE_DIM"
    export RT_ERR_MIN="$ERR_MIN_ABSOLUTE"
    export RT_ERR_RATIO="$ERR_THRESH_RATIO"
    export RT_KL_THRESH="$KL_THRESH"
    export RT_REV_KD_WEIGHT="$REV_KD_WEIGHT"

    # OFA-KD 相关
    export RT_OFA_EPS="$OFA_EPS"
    export RT_OFA_LOSS_WEIGHT="$OFA_LOSS_WEIGHT"
    export RT_OFA_FINAL_WEIGHT="$OFA_FINAL_WEIGHT"
    export RT_OFA_ANCHOR_WEIGHT="$OFA_ANCHOR_WEIGHT"

    local LOG_FILE="${EXP_DIR}/log_${TAG}.txt"
    echo "  [RT] $TAG: use_aux=$USE_AUX use_ofa=$USE_OFA"
    echo "  [RT] INFER_STEP=$INFER_STEP FWD=$RT_FWD_EPOCHS REV=$RT_REV_EPOCHS FT=$RT_FT_EPOCHS"
    echo "  [RT] log -> $LOG_FILE"
    python -u exp/real_time_inference.py 2>&1 | tee "$LOG_FILE"

    unset RT_INFER_STEP RT_REV_EPOCHS RT_FWD_EPOCHS RT_FT_EPOCHS \
          RT_USE_AUX_KD RT_USE_OFA_KD RT_RES_DIR RT_N_PARTY RT_CLOUD_DIM RT_EDGE_DIM \
          RT_ERR_MIN RT_ERR_RATIO RT_KL_THRESH \
          RT_OFA_EPS RT_OFA_LOSS_WEIGHT RT_OFA_FINAL_WEIGHT RT_OFA_ANCHOR_WEIGHT \
          RT_REV_KD_WEIGHT
    full_cleanup
}

# ============ 单组完整流程 ============
run_group() {
    local TAG=$1
    local USE_RT=$2
    local USE_AUX=$3
    local USE_OFA=$4
    local SKIP_INIT=$5

    local DONE_FLAG="${EXP_DIR}/rt_${TAG}/.done"
    if [ -f "$DONE_FLAG" ]; then
        echo "  [SKIP] $TAG 已完成"
        return
    fi

    echo ""
    echo "##############################################################"
    echo "##  $TAG"
    echo "##  use_rt=$USE_RT  use_aux=$USE_AUX  use_ofa=$USE_OFA  skip_init=$SKIP_INIT"
    echo "##############################################################"

    restore_initial_state

    if [ "$SKIP_INIT" = "true" ]; then
        echo "  [SKIP] 跳过初始蒸馏 + 本地微调 (用 LE_Backpack 的状态)"
    else
        local STUDENT_PATH="./checkpoints/Unified_Student/checkpoint.pth"
        run_distill "./checkpoints/Global_Teacher/checkpoint.pth" "$STUDENT_PATH" \
                    "$USE_AUX" "$USE_OFA" "$TAG" || {
            echo "  [FAIL] $TAG 蒸馏失败"; return
        }

        echo "  [LFT] 微调 ${#PARTIES[@]} 个边端..."
        for PARTY in "${PARTIES[@]}"; do
            run_local_ft "$STUDENT_PATH" "$PARTY" \
                         "./checkpoints/Local_Edges/${PARTY}/checkpoint.pth"
        done
    fi

    if [ "$USE_RT" = "false" ]; then
        local RT_OUT="${EXP_DIR}/rt_${TAG}"
        mkdir -p "$RT_OUT"
        echo "  [G1] INFER_STEP=$INFER_STEP"
        local PARTIES_CSV
        PARTIES_CSV=$(printf ",%s" "${PARTIES[@]}")
        PARTIES_CSV="${PARTIES_CSV:1}"
        python -c "$G1_INFER_SCRIPT" \
            "$RT_OUT" "$PARTIES_CSV" "$INIT_END" \
            "$INFER_STEP" "$PRED_LEN" "$SEQ_LEN" "$DATA_ROOT" "$DATA_NAME" "$EDGE_DIM" \
            "$S_MODEL" "$S_D_MODEL" "$S_D_FF" "$S_E_LAYERS" "$S_D_LAYERS" \
            "./checkpoints" "$BATCH_SIZE" "$GPU_ID" "$LABEL_LEN" "$FEATURES" "$FACTOR"
        full_cleanup
    else
        run_realtime "$TAG" "$USE_AUX" "$USE_OFA"
    fi

    touch "$DONE_FLAG"
    echo "  [Done] $TAG -> ${EXP_DIR}/rt_${TAG}/"
}


# ============ MAIN ============
echo "============================================================"
echo "  4-Group Single-Shot Ablation"
echo "  Teacher: TimesNet (d=$T_D_MODEL, ff=$T_D_FF)"
echo "  Student: $S_MODEL (d=$S_D_MODEL, ff=$S_D_FF)"
echo "  INFER_STEP = $INFER_STEP"
echo "  RT epochs: rev=$RT_REV_EPOCHS fwd=$RT_FWD_EPOCHS ft=$RT_FT_EPOCHS"
echo "  Output: $EXP_DIR/rt_<tag>/"
echo "============================================================"

# 备份检查
if [ ! -f "$GT_BACKUP" ]; then
    echo "❌ 找不到 $GT_BACKUP"
    exit 1
fi
if [ ! -f "$SCALER_BACKUP" ]; then
    echo "❌ 找不到 $SCALER_BACKUP"
    exit 1
fi
for P in "${PARTIES[@]}"; do
    if [ ! -f "${LE_BACKUP_DIR}/${P}/checkpoint.pth" ]; then
        echo "❌ 找不到 ${LE_BACKUP_DIR}/${P}/checkpoint.pth"
        exit 1
    fi
    if [ ! -f "${LE_BACKUP_DIR}/${P}/scaler.pkl" ]; then
        echo "❌ 找不到 ${LE_BACKUP_DIR}/${P}/scaler.pkl"
        exit 1
    fi
done
echo "✓ 备份检查通过"

# 跑 4 组
for cfg in "${GROUPS_CONFIG[@]}"; do
    read -r tag use_rt use_aux use_ofa skip_init <<< "$cfg"
    run_group "$tag" "$use_rt" "$use_aux" "$use_ofa" "$skip_init"
done

echo ""
echo "============================================================"
echo "All 4 groups done."
echo "Inspect: ls -la $EXP_DIR/rt_*/"
echo "============================================================"
