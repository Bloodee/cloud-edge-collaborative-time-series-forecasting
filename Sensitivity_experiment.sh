#!/bin/bash
# =============================================================================
# CE-BiD trigger-parameter sensitivity experiments
#
# One-factor-at-a-time sensitivity for:
#   DeltaT -> RT_MAX_NO_UPDATE
#   eta    -> RT_ERR_RATIO
#   delta  -> RT_KL_THRESH
#   R      -> RT_FORWARD_INTERVAL
#
# Outputs:
#   ${EXP_DIR}/rt_<case>/edge_avg.csv
#   ${EXP_DIR}/rt_<case>/final_report.txt
#   ${EXP_DIR}/rt_<case>/sensitivity_summary.csv
#   ${EXP_DIR}/all_sensitivity_summary.csv
# =============================================================================

set -eo pipefail

# ============================================================================ #
# [CONFIG]
# ============================================================================ #

export CUDA_VISIBLE_DEVICES=5
GPU_ID=0

DATA_ROOT="./dataset/processed"
MERGE_PATH="./dataset/processed/merge/"
DATA_NAME="PVOD.csv"
PARTIES=("party_1" "party_2" "party_3" "party_4" "party_5" "party_6" "party_7")
CLOUD_DIM=99
EDGE_DIM=15
INIT_START=0
INIT_END=18750

SEQ_LEN=12
PRED_LEN=6
LABEL_LEN=12
FEATURES="M"
FACTOR=3
EV_LEN=24

TRAIN_RATIO=0.8
VAL_RATIO=0.2
TEST_RATIO=0.0
BATCH_SIZE=32

T_D_MODEL=128
T_D_FF=256
T_E_LAYERS=2
T_TOP_K=5

S_MODEL="CNN"
S_D_MODEL=16
S_D_FF=32
S_E_LAYERS=2
S_D_LAYERS=1
S_TOP_K=5

DISTILL_EPOCHS=10
STUDENT_LR=0.0001
LAMBDA_KD=0.3
OT_WEIGHT=1.0
REV_KD_WEIGHT=0.01

LOCAL_FT_EPOCHS=5
LOCAL_LR=0.0001

INFER_STEP=48

REV_EPOCHS=1
FWD_EPOCHS=5
FT_EPOCHS=3

# Base trigger parameters. Adjust here if your paper baseline differs.
BASE_MAX_NO_UPDATE=$((5 * INFER_STEP))
BASE_ERR_RATIO=1.20
BASE_KL_THRESH=0.40
BASE_FORWARD_INTERVAL=3
ERR_MIN_ABSOLUTE=0.25

OFA_EPS=1.2
OFA_LOSS_WEIGHT=0.2
OFA_FINAL_WEIGHT=0.2
OFA_ANCHOR_WEIGHT=0.1

GT_BACKUP="./checkpoints/GT_back/checkpoint.pth"
SCALER_BACKUP="./checkpoints/GT_back/scaler.pkl"
LE_BACKUP_DIR="./checkpoints/LE_Backpack"
REALTIME_SCRIPT="${REALTIME_SCRIPT:-exp/real_time_inference.py}"

EXP_DIR="./sensitivity_results"

# Format:
#   case_tag changed_param max_no_update eta delta R
SENSITIVITY_CONFIG=(
    "Base none   $BASE_MAX_NO_UPDATE $BASE_ERR_RATIO $BASE_KL_THRESH $BASE_FORWARD_INTERVAL"

    "T1   DeltaT $((BASE_MAX_NO_UPDATE / 2)) $BASE_ERR_RATIO $BASE_KL_THRESH $BASE_FORWARD_INTERVAL"
    "T2   DeltaT $BASE_MAX_NO_UPDATE       $BASE_ERR_RATIO $BASE_KL_THRESH $BASE_FORWARD_INTERVAL"
    "T3   DeltaT $((BASE_MAX_NO_UPDATE * 2)) $BASE_ERR_RATIO $BASE_KL_THRESH $BASE_FORWARD_INTERVAL"
    "T4   DeltaT $((BASE_MAX_NO_UPDATE * 4)) $BASE_ERR_RATIO $BASE_KL_THRESH $BASE_FORWARD_INTERVAL"

    "E1   eta    $BASE_MAX_NO_UPDATE 1.05 $BASE_KL_THRESH $BASE_FORWARD_INTERVAL"
    "E2   eta    $BASE_MAX_NO_UPDATE 1.10 $BASE_KL_THRESH $BASE_FORWARD_INTERVAL"
    "E3   eta    $BASE_MAX_NO_UPDATE 1.20 $BASE_KL_THRESH $BASE_FORWARD_INTERVAL"
    "E4   eta    $BASE_MAX_NO_UPDATE 1.50 $BASE_KL_THRESH $BASE_FORWARD_INTERVAL"

    "D1   delta  $BASE_MAX_NO_UPDATE $BASE_ERR_RATIO 0.20 $BASE_FORWARD_INTERVAL"
    "D2   delta  $BASE_MAX_NO_UPDATE $BASE_ERR_RATIO 0.40 $BASE_FORWARD_INTERVAL"
    "D3   delta  $BASE_MAX_NO_UPDATE $BASE_ERR_RATIO 0.80 $BASE_FORWARD_INTERVAL"
    "D4   delta  $BASE_MAX_NO_UPDATE $BASE_ERR_RATIO 1.60 $BASE_FORWARD_INTERVAL"

    "R1   R      $BASE_MAX_NO_UPDATE $BASE_ERR_RATIO $BASE_KL_THRESH 1"
    "R2   R      $BASE_MAX_NO_UPDATE $BASE_ERR_RATIO $BASE_KL_THRESH 2"
    "R3   R      $BASE_MAX_NO_UPDATE $BASE_ERR_RATIO $BASE_KL_THRESH 3"
    "R4   R      $BASE_MAX_NO_UPDATE $BASE_ERR_RATIO $BASE_KL_THRESH 5"
)

# Optional filter. Examples:
#   RUN_ONLY="R eta" bash Sensitivity_experiment.sh
#   RUN_ONLY="R" bash Sensitivity_experiment.sh
RUN_ONLY="${RUN_ONLY:-all}"

# ============================================================================ #
# [HELPERS]
# ============================================================================ #

mkdir -p "$EXP_DIR"

get_latest_dir() { ls -td "$1"/*"$2"* 2>/dev/null | head -1; }
gpu_cleanup() { python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true; }

full_cleanup() {
    find ./checkpoints -maxdepth 1 -type d \( -name "*SensitivityDistill_*" -o -name "*SensitivityLFT_*" \
        -o -name "*RevDistill_*" -o -name "*FwdDistill_*" -o -name "*LocalFT_*" \
        -o -name "*Infer_*" \) -exec rm -rf {} + 2>/dev/null
    find ./results -maxdepth 1 -type d -exec rm -rf {} + 2>/dev/null
    gpu_cleanup
}

should_run_case() {
    local changed=$1
    if [ "$RUN_ONLY" = "all" ]; then
        return 0
    fi
    if [ "$changed" = "none" ]; then
        return 0
    fi
    for item in $RUN_ONLY; do
        if [ "$item" = "$changed" ]; then
            return 0
        fi
    done
    return 1
}

restore_initial_state() {
    if [ ! -f "$GT_BACKUP" ]; then
        echo "Missing $GT_BACKUP"
        exit 1
    fi
    mkdir -p ./checkpoints/Global_Teacher
    cp "$GT_BACKUP" ./checkpoints/Global_Teacher/checkpoint.pth
    mkdir -p ./checkpoints/global_scaler
    cp "$SCALER_BACKUP" ./checkpoints/global_scaler/scaler.pkl

    for PARTY in "${PARTIES[@]}"; do
        local SRC_DIR="${LE_BACKUP_DIR}/${PARTY}"
        local DST_DIR="./checkpoints/Local_Edges/${PARTY}"
        for ARTIFACT in checkpoint.pth scaler.pkl; do
            if [ ! -f "${SRC_DIR}/${ARTIFACT}" ]; then
                echo "Missing ${SRC_DIR}/${ARTIFACT}"
                exit 1
            fi
        done
        mkdir -p "$DST_DIR"
        cp "${SRC_DIR}/checkpoint.pth" "${DST_DIR}/checkpoint.pth"
        cp "${SRC_DIR}/scaler.pkl" "${DST_DIR}/scaler.pkl"
    done
    rm -f ./checkpoints/Unified_Student/checkpoint.pth
}

run_initial_distill() {
    local TAG=$1
    local TEACHER_PATH="./checkpoints/Global_Teacher/checkpoint.pth"
    local SAVE_PATH="./checkpoints/Unified_Student/checkpoint.pth"
    local DID="SensitivityDistill_${TAG}"

    echo "  [Initial Distill] $TAG"
    python -u run.py \
        --task_name long_term_forecast --is_training 1 --do_distill \
        --root_path "$MERGE_PATH" --data_path "$DATA_NAME" \
        --scaler_path ./checkpoints/global_scaler/scaler.pkl \
        --specific_start "$INIT_START" --specific_end "$INIT_END" \
        --train_ratio "$TRAIN_RATIO" --val_ratio "$VAL_RATIO" --test_ratio "$TEST_RATIO" \
        --model_id "$DID" --model "$S_MODEL" --data custom \
        --features "$FEATURES" --seq_len "$SEQ_LEN" --label_len "$LABEL_LEN" --pred_len "$PRED_LEN" \
        --factor "$FACTOR" \
        --enc_in "$EDGE_DIM" --dec_in "$EDGE_DIM" --c_out "$EDGE_DIM" \
        --d_model "$S_D_MODEL" --d_ff "$S_D_FF" --e_layers "$S_E_LAYERS" --d_layers "$S_D_LAYERS" \
        --cloud_dim "$CLOUD_DIM" \
        --teacher_model_name TimesNet --teacher_model_path "$TEACHER_PATH" \
        --teacher_d_model "$T_D_MODEL" --teacher_d_ff "$T_D_FF" --teacher_e_layers "$T_E_LAYERS" \
        --lambda_kd "$LAMBDA_KD" --ot_weight "$OT_WEIGHT" \
        --use_ofa_kd --ofa_eps "$OFA_EPS" --ofa_loss_weight "$OFA_LOSS_WEIGHT" \
        --ofa_final_weight "$OFA_FINAL_WEIGHT" \
        --train_epochs "$DISTILL_EPOCHS" --batch_size "$BATCH_SIZE" \
        --learning_rate "$STUDENT_LR" --gpu "$GPU_ID" --patience 3

    local DIR
    DIR=$(get_latest_dir "./checkpoints" "$DID")
    if [ ! -d "$DIR" ] || [ ! -f "${DIR}/checkpoint.pth" ]; then
        echo "Initial distill failed for $TAG"
        exit 1
    fi
    mkdir -p "$(dirname "$SAVE_PATH")"
    cp "${DIR}/checkpoint.pth" "$SAVE_PATH"
    rm -rf "$DIR"
    gpu_cleanup
}

run_initial_local_ft() {
    local TAG=$1
    local STUDENT_PATH="./checkpoints/Unified_Student/checkpoint.pth"

    echo "  [Initial Local FT] $TAG"
    for PARTY in "${PARTIES[@]}"; do
        local FID="SensitivityLFT_${TAG}_${PARTY}"
        local SAVE_PATH="./checkpoints/Local_Edges/${PARTY}/checkpoint.pth"
        mkdir -p "$(dirname "$SAVE_PATH")"

        python -u run.py \
            --task_name long_term_forecast --is_training 1 --do_local_train \
            --pretrained_model_path "$STUDENT_PATH" \
            --scaler_path "./checkpoints/Local_Edges/${PARTY}/scaler.pkl" \
            --root_path "${DATA_ROOT}/${PARTY}/" --data_path "$DATA_NAME" \
            --specific_start "$INIT_START" --specific_end "$INIT_END" \
            --train_ratio "$TRAIN_RATIO" --val_ratio "$VAL_RATIO" --test_ratio "$TEST_RATIO" \
            --model_id "$FID" --model "$S_MODEL" --data custom \
            --features "$FEATURES" --seq_len "$SEQ_LEN" --label_len "$LABEL_LEN" --pred_len "$PRED_LEN" \
            --factor "$FACTOR" \
            --enc_in "$EDGE_DIM" --dec_in "$EDGE_DIM" --c_out "$EDGE_DIM" \
            --d_model "$S_D_MODEL" --d_ff "$S_D_FF" --e_layers "$S_E_LAYERS" --d_layers "$S_D_LAYERS" \
            --train_epochs "$LOCAL_FT_EPOCHS" --batch_size "$BATCH_SIZE" \
            --learning_rate "$LOCAL_LR" --gpu "$GPU_ID" --patience 3

        local DIR
        DIR=$(get_latest_dir "./checkpoints" "$FID")
        if [ -d "$DIR" ]; then
            if [ -f "${DIR}/checkpoint_personalized.pth" ]; then
                cp "${DIR}/checkpoint_personalized.pth" "$SAVE_PATH"
            elif [ -f "${DIR}/checkpoint.pth" ]; then
                cp "${DIR}/checkpoint.pth" "$SAVE_PATH"
            else
                echo "Local FT checkpoint missing for $PARTY"
                exit 1
            fi
            rm -rf "$DIR"
        else
            echo "Local FT output missing for $PARTY"
            exit 1
        fi
        gpu_cleanup
    done
}

run_realtime_case() {
    local TAG=$1
    local CHANGED=$2
    local MAX_NO_UPDATE=$3
    local ETA=$4
    local DELTA=$5
    local R_VAL=$6

    local RT_OUT="${EXP_DIR}/rt_${TAG}_${CHANGED}"
    local LOG_FILE="${EXP_DIR}/log_${TAG}_${CHANGED}.txt"
    rm -rf "$RT_OUT"
    mkdir -p "$RT_OUT"

    export RT_DATA_ROOT="$DATA_ROOT"
    export RT_CKPT_DIR="./checkpoints"
    export RT_RES_DIR="$RT_OUT"
    export RT_MERGE_PATH="$MERGE_PATH"
    export RT_DATA_NAME="$DATA_NAME"
    export RT_N_PARTY=${#PARTIES[@]}
    export RT_CLOUD_DIM="$CLOUD_DIM"
    export RT_EDGE_DIM="$EDGE_DIM"
    export RT_SEQ_LEN="$SEQ_LEN"
    export RT_PRED_LEN="$PRED_LEN"
    export RT_LABEL_LEN="$LABEL_LEN"
    export RT_FEATURES="$FEATURES"
    export RT_FACTOR="$FACTOR"
    export RT_EV_LEN="$EV_LEN"
    export RT_START_IDX="$INIT_END"
    export RT_INFER_STEP="$INFER_STEP"
    export RT_INFER_BATCH_SIZE="$BATCH_SIZE"

    export RT_T_D_MODEL="$T_D_MODEL"
    export RT_T_D_FF="$T_D_FF"
    export RT_T_LAYERS="$T_E_LAYERS"
    export RT_T_TOP_K="$T_TOP_K"
    export RT_S_MODEL="$S_MODEL"
    export RT_S_D_MODEL="$S_D_MODEL"
    export RT_S_D_FF="$S_D_FF"
    export RT_S_E_LAYERS="$S_E_LAYERS"
    export RT_S_D_LAYERS="$S_D_LAYERS"
    export RT_S_TOP_K="$S_TOP_K"

    export RT_REV_EPOCHS="$REV_EPOCHS"
    export RT_FWD_EPOCHS="$FWD_EPOCHS"
    export RT_FT_EPOCHS="$FT_EPOCHS"
    export RT_REV_KD_WEIGHT="$REV_KD_WEIGHT"

    export RT_ERR_MIN="$ERR_MIN_ABSOLUTE"
    export RT_ERR_RATIO="$ETA"
    export RT_KL_THRESH="$DELTA"
    export RT_MAX_NO_UPDATE="$MAX_NO_UPDATE"
    export RT_FORWARD_INTERVAL="$R_VAL"

    export RT_USE_AUX_KD=0
    export RT_USE_OFA_KD=1
    export RT_OFA_EPS="$OFA_EPS"
    export RT_OFA_LOSS_WEIGHT="$OFA_LOSS_WEIGHT"
    export RT_OFA_FINAL_WEIGHT="$OFA_FINAL_WEIGHT"
    export RT_OFA_ANCHOR_WEIGHT="$OFA_ANCHOR_WEIGHT"

    echo "  [Realtime] $TAG changed=$CHANGED DeltaT=$MAX_NO_UPDATE eta=$ETA delta=$DELTA R=$R_VAL"
    python -u "$REALTIME_SCRIPT" 2>&1 | tee "$LOG_FILE"

    unset RT_DATA_ROOT RT_CKPT_DIR RT_RES_DIR RT_MERGE_PATH RT_DATA_NAME \
          RT_N_PARTY RT_CLOUD_DIM RT_EDGE_DIM RT_SEQ_LEN RT_PRED_LEN RT_LABEL_LEN \
          RT_FEATURES RT_FACTOR RT_EV_LEN RT_START_IDX RT_INFER_STEP RT_INFER_BATCH_SIZE \
          RT_T_D_MODEL RT_T_D_FF RT_T_LAYERS RT_T_TOP_K RT_S_MODEL RT_S_D_MODEL \
          RT_S_D_FF RT_S_E_LAYERS RT_S_D_LAYERS RT_S_TOP_K \
          RT_REV_EPOCHS RT_FWD_EPOCHS RT_FT_EPOCHS RT_REV_KD_WEIGHT \
          RT_ERR_MIN RT_ERR_RATIO RT_KL_THRESH RT_MAX_NO_UPDATE RT_FORWARD_INTERVAL \
          RT_USE_AUX_KD RT_USE_OFA_KD RT_OFA_EPS RT_OFA_LOSS_WEIGHT \
          RT_OFA_FINAL_WEIGHT RT_OFA_ANCHOR_WEIGHT
}

append_summary() {
    local CASE_DIR=$1
    local CASE_TAG=$2
    local CHANGED=$3
    local SUMMARY="${CASE_DIR}/sensitivity_summary.csv"
    local ALL="${EXP_DIR}/all_sensitivity_summary.csv"

    if [ ! -f "$SUMMARY" ]; then
        echo "Warning: missing $SUMMARY"
        return
    fi
    if [ ! -f "$ALL" ]; then
        head -n 1 "$SUMMARY" | sed 's/^/case_tag,changed_param,/' > "$ALL"
    fi
    tail -n 1 "$SUMMARY" | sed "s/^/${CASE_TAG},${CHANGED},/" >> "$ALL"
}

run_case() {
    local TAG=$1
    local CHANGED=$2
    local MAX_NO_UPDATE=$3
    local ETA=$4
    local DELTA=$5
    local R_VAL=$6

    if ! should_run_case "$CHANGED"; then
        echo "[SKIP] $TAG changed=$CHANGED due to RUN_ONLY=$RUN_ONLY"
        return
    fi

    local DONE_FLAG="${EXP_DIR}/rt_${TAG}_${CHANGED}/.done"
    if [ -f "$DONE_FLAG" ]; then
        echo "[SKIP] $TAG already done"
        return
    fi

    echo ""
    echo "============================================================"
    echo "Case $TAG | changed=$CHANGED | DeltaT=$MAX_NO_UPDATE eta=$ETA delta=$DELTA R=$R_VAL"
    echo "============================================================"

    restore_initial_state
    run_initial_distill "$TAG"
    run_initial_local_ft "$TAG"
    run_realtime_case "$TAG" "$CHANGED" "$MAX_NO_UPDATE" "$ETA" "$DELTA" "$R_VAL"
    full_cleanup

    local CASE_DIR="${EXP_DIR}/rt_${TAG}_${CHANGED}"
    touch "${CASE_DIR}/.done"
    append_summary "$CASE_DIR" "$TAG" "$CHANGED"
    echo "[DONE] $TAG -> $CASE_DIR"
}

# ============================================================================ #
# [MAIN]
# ============================================================================ #

echo "============================================================"
echo "CE-BiD sensitivity experiments"
echo "Dataset: $DATA_NAME"
echo "Output : $EXP_DIR"
echo "Script : $REALTIME_SCRIPT"
echo "RUN_ONLY=$RUN_ONLY"
echo "============================================================"

if [ ! -f "$REALTIME_SCRIPT" ]; then
    echo "Missing realtime script: $REALTIME_SCRIPT"
    exit 1
fi
if [ ! -f "$GT_BACKUP" ]; then
    echo "Missing $GT_BACKUP"
    exit 1
fi
if [ ! -f "$SCALER_BACKUP" ]; then
    echo "Missing $SCALER_BACKUP"
    exit 1
fi
for P in "${PARTIES[@]}"; do
    if [ ! -f "${LE_BACKUP_DIR}/${P}/checkpoint.pth" ]; then
        echo "Missing ${LE_BACKUP_DIR}/${P}/checkpoint.pth"
        exit 1
    fi
    if [ ! -f "${LE_BACKUP_DIR}/${P}/scaler.pkl" ]; then
        echo "Missing ${LE_BACKUP_DIR}/${P}/scaler.pkl"
        exit 1
    fi
done

for cfg in "${SENSITIVITY_CONFIG[@]}"; do
    read -r tag changed max_no_update eta delta r_val <<< "$cfg"
    run_case "$tag" "$changed" "$max_no_update" "$eta" "$delta" "$r_val"
done

echo ""
echo "============================================================"
echo "Sensitivity experiments done."
echo "Summary: ${EXP_DIR}/all_sensitivity_summary.csv"
echo "============================================================"
