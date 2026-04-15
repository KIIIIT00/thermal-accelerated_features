#!/bin/bash
# =============================================================================
# scripts/setup_glue_factory.sh
# glue-factory を third_party にクローンして
# カスタムファイルを glue-factory 内部に配置するセットアップスクリプト。
#
# 使用方法:
#   bash scripts/setup_glue_factory.sh
#
# 実行後:
#   bash scripts/train_lightglue_gf.sh
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

echo "========================================"
echo "  glue-factory セットアップ"
echo "  Repo root: ${REPO_ROOT}"
echo "========================================"

# ── Step 1: glue-factory をクローン ──────────────────────────────────────────
GF_DIR="${REPO_ROOT}/third_party/glue-factory"

if [ -d "${GF_DIR}" ]; then
    echo "[Step 1] glue-factory already exists at ${GF_DIR}"
    echo "         最新版に更新します..."
    cd "${GF_DIR}" && git pull && cd "${REPO_ROOT}"
else
    echo "[Step 1] glue-factory をクローン中..."
    mkdir -p "${REPO_ROOT}/third_party"
    git clone https://github.com/cvg/glue-factory.git "${GF_DIR}"
    echo "[Step 1] クローン完了"
fi

# ── Step 2: glue-factory を開発モードでインストール ───────────────────────────
echo "[Step 2] glue-factory を開発モードでインストール..."
pip install -e "${GF_DIR}" --quiet
echo "[Step 2] インストール完了"

# ── Step 3: カスタムファイルを glue-factory 内に配置 ──────────────────────────
echo "[Step 3] カスタムファイルを配置..."

# 3-a: ThermalXFeat → gluefactory/models/extractors/
EXTRACTORS_DIR="${GF_DIR}/gluefactory/models/extractors"
if [ ! -d "${EXTRACTORS_DIR}" ]; then
    mkdir -p "${EXTRACTORS_DIR}"
    touch "${EXTRACTORS_DIR}/__init__.py"
fi
cp "${REPO_ROOT}/gf_modules/models/thermal_xfeat.py" \
   "${EXTRACTORS_DIR}/thermal_xfeat.py"
echo "  → ${EXTRACTORS_DIR}/thermal_xfeat.py"

# 3-b: ThermalHomographyDataset → gluefactory/datasets/
DATASETS_DIR="${GF_DIR}/gluefactory/datasets"
cp "${REPO_ROOT}/gf_modules/datasets/thermal_homography.py" \
   "${DATASETS_DIR}/thermal_homography.py"
echo "  → ${DATASETS_DIR}/thermal_homography.py"

echo "[Step 3] 配置完了"

# ── Step 4: 動作確認 ──────────────────────────────────────────────────────────
echo "[Step 4] インポート確認..."

python3 -c "
import sys
sys.path.insert(0, '${REPO_ROOT}')
from gluefactory.models.extractors.thermal_xfeat import ThermalXFeat
from gluefactory.datasets.thermal_homography import ThermalHomographyDataset
print('  ThermalXFeat         : OK')
print('  ThermalHomographyDataset : OK')
"

echo ""
echo "========================================"
echo "  セットアップ完了"
echo ""
echo "  学習を開始するには:"
echo "    bash scripts/train_lightglue_gf.sh"
echo "========================================"