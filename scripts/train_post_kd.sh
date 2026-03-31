#!/usr/bin/env bash
# =============================================================================
# scripts/train_post_kd.sh
# Post-KD 訓練シェルスクリプト
#
# 使い方:
#   bash scripts/train_post_kd.sh                             # 全ステージ実行
#   bash scripts/train_post_kd.sh --stages 1                  # Stage 1 のみ
#   bash scripts/train_post_kd.sh --stages 1 2                # Stage 1→2
#   bash scripts/train_post_kd.sh --kd_weights <path>         # KD重み上書き
#   bash scripts/train_post_kd.sh --no_wandb                  # wandb 無効
# =============================================================================

set -euo pipefail

# ── デフォルト値 ──────────────────────────────────────────────────────────────
CONFIG="configs/post_kd_config.yaml"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG="$2"; shift 2 ;;
        *)        EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# ── リポジトリルートに移動 ────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# ── 仮想環境の有効化 ──────────────────────────────────────────────────────────
if [[ -f ".venv/bin/activate" ]]; then
    source .venv/bin/activate
    echo "[train_post_kd.sh] Activated .venv"
fi

# ── 設定ファイルの確認 ────────────────────────────────────────────────────────
if [[ ! -f "${CONFIG}" ]]; then
    echo "[train_post_kd.sh] ERROR: config not found: ${CONFIG}" >&2
    exit 1
fi

echo "[train_post_kd.sh] Config: ${CONFIG}"
echo "[train_post_kd.sh] Extra args: ${EXTRA_ARGS[*]+"${EXTRA_ARGS[*]}"}"
echo ""

# ── KD フェーズが完了しているかチェック ──────────────────────────────────────
# train_config.yaml から kd_weights を読み取る（簡易チェック）
KD_WEIGHTS=$(python -c "
import yaml, sys
cfg = yaml.safe_load(open('${CONFIG}'))
print(cfg.get('kd_weights', ''))
" 2>/dev/null || echo "")

if [[ -n "${KD_WEIGHTS}" ]] && [[ ! -f "${KD_WEIGHTS}" ]]; then
    echo "[train_post_kd.sh] WARNING: kd_weights not found: ${KD_WEIGHTS}"
    echo "  → Run train_kd.py first, then update configs/post_kd_config.yaml"
    echo "  → Continuing anyway (model starts from random weights)..."
    echo ""
fi

# ── Post-KD 訓練実行 ──────────────────────────────────────────────────────────
echo "[train_post_kd.sh] Starting Post-KD training..."

python train_post_kd.py \
    --config "${CONFIG}" \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"

echo "[train_post_kd.sh] Post-KD training completed."