#!/bin/bash
# =============================================================================
# scripts/train_lightglue_gf.sh
# glue-factory を使った LightGlue Fine-tuning 起動スクリプト
#
# 事前準備:
#   bash scripts/setup_glue_factory.sh
#
# 使用方法:
#   bash scripts/train_lightglue_gf.sh
#   EXPERIMENT=my_run bash scripts/train_lightglue_gf.sh
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."

EXPERIMENT="${EXPERIMENT:-thermal_xfeat_lightglue}"
CONFIG="configs/lightglue_gf_config.yaml"
GF_DIR="third_party/glue-factory"

if [ ! -d "${GF_DIR}" ]; then
    echo "[ERROR] glue-factory が見つかりません。まず以下を実行してください:"
    echo "  bash scripts/setup_glue_factory.sh"
    exit 1
fi

python -c "import gluefactory" 2>/dev/null || {
    echo "[ERROR] gluefactory が未インストールです:"
    echo "  bash scripts/setup_glue_factory.sh"
    exit 1
}

echo "========================================"
echo "  LightGlue Fine-tuning (glue-factory)"
echo "  Experiment : ${EXPERIMENT}"
echo "  Config     : ${CONFIG}"
echo "========================================"

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

python -m gluefactory.train "${EXPERIMENT}" \
    --conf "${CONFIG}" \
    "$@"