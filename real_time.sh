#!/bin/bash

# =========================================================
# 实时推理启动脚本 - 支持外部控制 N_PARTY
#
# 用法:
#   bash real_time.sh                 # 默认使用 PVOD 的 7 个边端
#   N_PARTY=5 bash real_time.sh       # 自定义为 5 个边端
#
# 注意事项:
#   - 运行前需确保 ./dataset/party_1, party_2, ..., party_N 目录已准备好
#   - 需确保 ./dataset/merge/ 已经生成 (CLOUD_DIM 维度的合并数据)
#   - CLOUD_DIM 与 EDGE_DIM、N_PARTY 关系: CLOUD_DIM ≈ N_PARTY × (EDGE_DIM - 1) + 1
# =========================================================

set -e
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# ============ 外部可控参数 ============
# N_PARTY: 边端数量
N_PARTY=${N_PARTY:-7}

# 边端单个数据维度 (含OT列)
EDGE_DIM=${EDGE_DIM:-15}

# 云端全局聚合数据维度
# 默认按公式自动计算: N_PARTY * (EDGE_DIM - 1) + 1
CLOUD_DIM=${CLOUD_DIM:-$((N_PARTY * (EDGE_DIM - 1) + 1))}

# 实时推理的 epoch 控制
RT_REV_EPOCHS=${RT_REV_EPOCHS:-1}
RT_FWD_EPOCHS=${RT_FWD_EPOCHS:-5}
RT_FT_EPOCHS=${RT_FT_EPOCHS:-3}

# 结果输出目录
RT_RES_DIR=${RT_RES_DIR:-"./realtime_results_np${N_PARTY}"}
RT_DATA_ROOT=${RT_DATA_ROOT:-"./dataset/processed"}
RT_MERGE_PATH=${RT_MERGE_PATH:-"${RT_DATA_ROOT}/merge/"}
RT_DATA_NAME=${RT_DATA_NAME:-"PVOD.csv"}

echo "========================================================"
echo "🚀 Real-time Inference"
echo "   N_PARTY   = $N_PARTY"
echo "   EDGE_DIM  = $EDGE_DIM"
echo "   CLOUD_DIM = $CLOUD_DIM (formula: $N_PARTY × $((EDGE_DIM - 1)) + 1 = $((N_PARTY * (EDGE_DIM - 1) + 1)))"
echo "   REV_EP=$RT_REV_EPOCHS FWD_EP=$RT_FWD_EPOCHS FT_EP=$RT_FT_EPOCHS"
echo "   Output: $RT_RES_DIR"
echo "========================================================"

# ============ 数据目录检查 ============
for i in $(seq 1 $N_PARTY); do
    if [ ! -f "${RT_DATA_ROOT}/party_${i}/${RT_DATA_NAME}" ]; then
        echo "❌ 缺少数据文件: ${RT_DATA_ROOT}/party_${i}/${RT_DATA_NAME}"
        echo "   请先运行 scripts/PVOD/Init.sh 准备对齐数据"
        exit 1
    fi
done

if [ ! -f "${RT_MERGE_PATH}/${RT_DATA_NAME}" ]; then
    echo "❌ 缺少合并数据文件: ${RT_MERGE_PATH}/${RT_DATA_NAME}"
    exit 1
fi

# ============ 启动实时推理 ============
export RT_N_PARTY=$N_PARTY
export RT_CLOUD_DIM=$CLOUD_DIM
export RT_EDGE_DIM=$EDGE_DIM
export RT_REV_EPOCHS
export RT_FWD_EPOCHS
export RT_FT_EPOCHS
export RT_RES_DIR
export RT_DATA_ROOT
export RT_MERGE_PATH
export RT_DATA_NAME

python -u exp/real_time_inference.py

echo ""
echo "========================================================"
echo "🎉 实时推理完成"
echo "   结果目录: $RT_RES_DIR"
echo "========================================================"
