#!/bin/bash
# =============================================================================
# scripts/train_lightglue_gf.sh
# glue-factory を使った LightGlue Fine-tuning 起動スクリプト
#
# 使用方法:
#   bash scripts/train_lightglue_gf.sh
#   bash scripts/train_lightglue_gf.sh --experiment my_experiment
#
# 事前準備:
#   pip install git+https://github.com/cvg/glue-factory.git
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."

EXPERIMENT="${EXPERIMENT:-thermal_xfeat_lightglue}"
CONFIG="configs/lightglue_gf_config.yaml"

# gluefactory のインストール確認
python -c "import gluefactory" 2>/dev/null || {
    echo "[ERROR] gluefactory が見つかりません。"
    echo "  pip install git+https://github.com/cvg/glue-factory.git"
    exit 1
}

echo "========================================"
echo "  LightGlue Fine-tuning (glue-factory)"
echo "  Experiment: ${EXPERIMENT}"
echo "  Config:     ${CONFIG}"
echo "========================================"

# プロジェクトルートを PYTHONPATH に追加して gf_modules を importable にする
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# gluefactory の CLI を直接使用
# glue-factory は 'outputs/{experiment}/' に自動的に保存する
python -m gluefactory.train "${EXPERIMENT}" \
    --conf "${CONFIG}" \
    "$@"