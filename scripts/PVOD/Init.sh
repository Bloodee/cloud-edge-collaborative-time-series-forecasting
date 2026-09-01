#!/bin/bash

# =========================================================
# PVOD system initialization (TimesNet → CNN, 7 parties)
# 与 real_time_inference.py 默认架构保持一致:
#   - 教师 TimesNet d_model=128, d_ff=256, e_layers=2
#   - 学生 CNN     d_model=16,  d_ff=32,  e_layers=2
#   - 蒸馏方法 OFA-KD (异构蒸馏)
#   - OT_WEIGHT = 1.0
# 含各阶段计时统计
# =========================================================

set -e

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

RAW_DATA_ROOT=${RAW_DATA_ROOT:-"./dataset"}
DATA_ROOT=${DATA_ROOT:-"./dataset/processed"}
MERGE_PATH="${DATA_ROOT}/merge/"; DATA_NAME="PVOD.csv"
PARTIES=("party_1" "party_2" "party_3" "party_4" "party_5" "party_6" "party_7")

SEQ_LEN=12; PRED_LEN=6; LABEL_LEN=12; FEATURES="M"; FACTOR=3
CLOUD_DIM=99; EDGE_DIM=15

# ========== 教师模型 (TimesNet) ==========
T_D_MODEL=128
T_D_FF=256
T_E_LAYERS=2

# ========== 学生模型 (轻量 CNN) ==========
S_MODEL=CNN
S_D_MODEL=16
S_D_FF=32
S_E_LAYERS=2
S_D_LAYERS=1             # CNN 不使用 decoder；保留用于统一命令接口

TRAIN_EPOCHS=10; BATCH_SIZE=32
TEACHER_LR=0.0001; STUDENT_LR=0.0001; LOCAL_LR=0.0001; GPU_ID=0

# ========== 蒸馏控制 ==========
OT_WEIGHT=1.0            # OT 列参与 hard 监督
LAMBDA_KD=0.5

# ========== OFA-KD 配置 ==========
OFA_EPS=1.2
OFA_LOSS_WEIGHT=0.2
OFA_FINAL_WEIGHT=0.2

INIT_START=0; INIT_END=18750
TRAIN_RATIO=0.8; VAL_RATIO=0.2; TEST_RATIO=0.0

TEACHER_PATH="./checkpoints/Global_Teacher/checkpoint.pth"
STUDENT_PATH="./checkpoints/Unified_Student/checkpoint.pth"
GLOBAL_SCALER_PATH="./checkpoints/global_scaler/scaler.pkl"

TIMING_LOG="./init_timing_report.txt"
> "$TIMING_LOG"

log_time() { echo "$1" | tee -a "$TIMING_LOG"; }

get_latest_dir() { ls -td "$1"/*"$2"* 2>/dev/null | head -1; }

echo "========================================================"
echo "PVOD Initialization (TimesNet -> CNN, 7 Parties, OFA-KD)"
echo "  Teacher: TimesNet (d=$T_D_MODEL, ff=$T_D_FF, L=$T_E_LAYERS)"
echo "  Student: $S_MODEL (d=$S_D_MODEL, ff=$S_D_FF, L=$S_E_LAYERS)"
echo "  Distill: OFA-KD (eps=$OFA_EPS, stage_w=$OFA_LOSS_WEIGHT, final_w=$OFA_FINAL_WEIGHT)"
echo "========================================================"

mkdir -p ./checkpoints/Global_Teacher ./checkpoints/Unified_Student ./checkpoints/Local_Edges

log_time "========== Init Timing Report =========="
log_time "Start: $(date '+%Y-%m-%d %H:%M:%S')"
log_time ""

TOTAL_START=$SECONDS

echo "Preparing timestamp-aligned PVOD files..."
python -u data_provider/prepare_pvod.py \
  --input-root "$RAW_DATA_ROOT" --output-root "$DATA_ROOT" \
  --parties "${#PARTIES[@]}" --data-name "$DATA_NAME"

# ========== STEP 1: Cloud Teacher ==========
echo ""; echo "Step 1: Training Cloud Teacher (TimesNet, d=$T_D_MODEL, ff=$T_D_FF)..."
STEP1_START=$SECONDS

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path $MERGE_PATH --data_path $DATA_NAME \
  --scaler_path $GLOBAL_SCALER_PATH \
  --specific_start $INIT_START --specific_end $INIT_END \
  --train_ratio $TRAIN_RATIO --val_ratio $VAL_RATIO --test_ratio $TEST_RATIO \
  --model_id Cloud_Teacher --model TimesNet --data custom \
  --features $FEATURES --seq_len $SEQ_LEN --label_len $LABEL_LEN --pred_len $PRED_LEN \
  --factor $FACTOR --e_layers $T_E_LAYERS --d_model $T_D_MODEL --d_ff $T_D_FF \
  --enc_in $CLOUD_DIM --dec_in $CLOUD_DIM --c_out $CLOUD_DIM \
  --train_epochs $TRAIN_EPOCHS --batch_size $BATCH_SIZE \
  --learning_rate $TEACHER_LR --gpu $GPU_ID --patience 3 --des 'Teacher'

STEP1_TIME=$((SECONDS - STEP1_START))

TEACHER_DIR=$(get_latest_dir "./checkpoints" "Cloud_Teacher")
if [ -d "$TEACHER_DIR" ] && [ -f "${TEACHER_DIR}/checkpoint.pth" ]; then
    SRC="$(realpath ${TEACHER_DIR}/checkpoint.pth 2>/dev/null)"
    DST="$(realpath $TEACHER_PATH 2>/dev/null)"
    [ "$SRC" != "$DST" ] && cp "${TEACHER_DIR}/checkpoint.pth" "$TEACHER_PATH"
    echo "Teacher saved: $TEACHER_PATH"
else
    echo "Teacher training failed"; exit 1
fi

log_time "[Step 1] Cloud Teacher Training: ${STEP1_TIME}s"

# ========== STEP 2: Distillation (OFA-KD: TimesNet -> CNN) ==========
echo ""; echo "Step 2: Cross-Architecture Distillation (TimesNet -> $S_MODEL, OFA-KD)..."
STEP2_START=$SECONDS

python -u run.py \
  --task_name long_term_forecast --is_training 1 --do_distill \
  --root_path $MERGE_PATH --data_path $DATA_NAME \
  --scaler_path $GLOBAL_SCALER_PATH \
  --specific_start $INIT_START --specific_end $INIT_END \
  --train_ratio $TRAIN_RATIO --val_ratio $VAL_RATIO --test_ratio $TEST_RATIO \
  --model_id Unified_Student --model $S_MODEL --data custom \
  --features $FEATURES --seq_len $SEQ_LEN --label_len $LABEL_LEN --pred_len $PRED_LEN \
  --factor $FACTOR --enc_in $EDGE_DIM --dec_in $EDGE_DIM --c_out $EDGE_DIM \
  --d_model $S_D_MODEL --d_ff $S_D_FF --e_layers $S_E_LAYERS --d_layers $S_D_LAYERS \
  --cloud_dim $CLOUD_DIM \
  --teacher_model_name TimesNet --teacher_model_path $TEACHER_PATH \
  --teacher_d_model $T_D_MODEL --teacher_d_ff $T_D_FF --teacher_e_layers $T_E_LAYERS \
  --lambda_kd $LAMBDA_KD --ot_weight $OT_WEIGHT \
  --use_ofa_kd --ofa_eps $OFA_EPS \
  --ofa_loss_weight $OFA_LOSS_WEIGHT --ofa_final_weight $OFA_FINAL_WEIGHT \
  --train_epochs $TRAIN_EPOCHS --batch_size $BATCH_SIZE \
  --learning_rate $STUDENT_LR --gpu $GPU_ID --patience 3 --des 'Distill'

STEP2_TIME=$((SECONDS - STEP2_START))

STUDENT_DIR=$(get_latest_dir "./checkpoints" "Unified_Student")
if [ -d "$STUDENT_DIR" ] && [ -f "${STUDENT_DIR}/checkpoint.pth" ]; then
    SRC="$(realpath ${STUDENT_DIR}/checkpoint.pth 2>/dev/null)"
    DST="$(realpath $STUDENT_PATH 2>/dev/null)"
    [ "$SRC" != "$DST" ] && cp "${STUDENT_DIR}/checkpoint.pth" "$STUDENT_PATH"
    echo "Student saved: $STUDENT_PATH"
else
    echo "Distillation failed"; exit 1
fi

log_time "[Step 2] Distillation: ${STEP2_TIME}s"

# ========== STEP 3: Local Fine-tuning ==========
echo ""; echo "Step 3: Local Fine-tuning ($S_MODEL on each party)..."
STEP3_START=$SECONDS
declare -a FT_TIMES=()

for PARTY in "${PARTIES[@]}"; do
    echo "   -> $PARTY"
    mkdir -p "./checkpoints/Local_Edges/$PARTY"
    FT_START=$SECONDS

    python -u run.py \
      --task_name long_term_forecast --is_training 1 --do_local_train \
      --pretrained_model_path $STUDENT_PATH \
      --scaler_path "./checkpoints/Local_Edges/${PARTY}/scaler.pkl" \
      --root_path "${DATA_ROOT}/${PARTY}/" --data_path $DATA_NAME \
      --specific_start $INIT_START --specific_end $INIT_END \
      --train_ratio $TRAIN_RATIO --val_ratio $VAL_RATIO --test_ratio $TEST_RATIO \
      --model_id "Init_${PARTY}" --model $S_MODEL --data custom \
      --features $FEATURES --seq_len $SEQ_LEN --label_len $LABEL_LEN --pred_len $PRED_LEN \
      --factor $FACTOR --enc_in $EDGE_DIM --dec_in $EDGE_DIM --c_out $EDGE_DIM \
      --d_model $S_D_MODEL --d_ff $S_D_FF --e_layers $S_E_LAYERS --d_layers $S_D_LAYERS \
      --train_epochs $TRAIN_EPOCHS --batch_size $BATCH_SIZE \
      --learning_rate $LOCAL_LR --gpu $GPU_ID --patience 3 --des 'Local'

    FT_ELAPSED=$((SECONDS - FT_START))
    FT_TIMES+=($FT_ELAPSED)

    LOCAL_DIR=$(get_latest_dir "./checkpoints" "Init_${PARTY}")
    LOCAL_PATH="./checkpoints/Local_Edges/${PARTY}/checkpoint.pth"
    if [ -d "$LOCAL_DIR" ]; then
        if [ -f "${LOCAL_DIR}/checkpoint_personalized.pth" ]; then
            cp "${LOCAL_DIR}/checkpoint_personalized.pth" "$LOCAL_PATH"
        elif [ -f "${LOCAL_DIR}/checkpoint.pth" ]; then
            cp "${LOCAL_DIR}/checkpoint.pth" "$LOCAL_PATH"
        fi
        echo "      Done $PARTY (${FT_ELAPSED}s)"
    else
        echo "      Failed $PARTY"
        exit 1
    fi
done

STEP3_TIME=$((SECONDS - STEP3_START))
FT_TOTAL=0; FT_MIN=999999; FT_MAX=0
for t in "${FT_TIMES[@]}"; do
    FT_TOTAL=$((FT_TOTAL + t))
    [ $t -lt $FT_MIN ] && FT_MIN=$t
    [ $t -gt $FT_MAX ] && FT_MAX=$t
done
FT_AVG=$((FT_TOTAL / ${#FT_TIMES[@]}))

log_time "[Step 3] Local Fine-tuning Total: ${STEP3_TIME}s"
log_time "         Per-party: avg=${FT_AVG}s, min=${FT_MIN}s, max=${FT_MAX}s"
for i in "${!PARTIES[@]}"; do
    log_time "         ${PARTIES[$i]}: ${FT_TIMES[$i]}s"
done

TOTAL_TIME=$((SECONDS - TOTAL_START))

log_time ""
log_time "========== Summary =========="
log_time "[Step 1] Cloud Teacher:    ${STEP1_TIME}s"
log_time "[Step 2] Distillation:     ${STEP2_TIME}s"
log_time "[Step 3] Local Fine-tune:  ${STEP3_TIME}s"
log_time "         FT avg/min/max:   ${FT_AVG}s / ${FT_MIN}s / ${FT_MAX}s"
log_time "---"
log_time "[Total]  ${TOTAL_TIME}s"
log_time "End: $(date '+%Y-%m-%d %H:%M:%S')"

echo ""
echo "========================================================"
echo "Timing report saved: $TIMING_LOG"
echo "========================================================"
cat "$TIMING_LOG"

# Ablation and sensitivity scripts restore these immutable initialization copies
# before each run so that repeated experiments start from the same checkpoints.
mkdir -p ./checkpoints/GT_back ./checkpoints/LE_Backpack
cp "$TEACHER_PATH" ./checkpoints/GT_back/checkpoint.pth
cp "$GLOBAL_SCALER_PATH" ./checkpoints/GT_back/scaler.pkl
cp -a ./checkpoints/Local_Edges/. ./checkpoints/LE_Backpack/
echo "Initialization backups created for ablation/sensitivity workflows."
