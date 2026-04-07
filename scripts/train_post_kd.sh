#!/usr/bin/env bash
# =============================================================================
# scripts/train_post_kd.sh
# Post-KD 訓練シェルスクリプト
#
# kd_weights は post_kd_config.yaml の kd_train_config から自動解決される。
# train_config.yaml の ckpt_save_path を参照し、
#   {ckpt_save_path}/thermal_kd_student_final.pth
# を kd_weights として使用する。
#
# 使い方:
#   bash scripts/train_post_kd.sh                        # 全ステージ
#   bash scripts/train_post_kd.sh --stages 1             # Stage 1 のみ
#   bash scripts/train_post_kd.sh --stages 2             # Stage 2 のみ
#   bash scripts/train_post_kd.sh --no_wandb             # wandb 無効
#   bash scripts/train_post_kd.sh \
#       --kd_weights path/to/custom.pth                  # 重みを直接指定
# =============================================================================

set -euo pipefail

# ── デフォルト値 ──────────────────────────────────────────────────────────────
POST_KD_CONFIG="configs/post_kd_config.yaml"
KD_CONFIG="configs/train_config.yaml"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)          POST_KD_CONFIG="$2"; shift 2 ;;
        --kd_train_config) KD_CONFIG="$2";       shift 2 ;;
        *)                 EXTRA_ARGS+=("$1");   shift   ;;
    esac
done

# ── リポジトリルートに移動 ─────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# ── 仮想環境の有効化 ───────────────────────────────────────────────────────────
if [[ -f ".venv/bin/activate" ]]; then
    source .venv/bin/activate
    echo "[train_post_kd.sh] Activated .venv"
fi

# ── 設定ファイルの確認 ─────────────────────────────────────────────────────────
if [[ ! -f "${POST_KD_CONFIG}" ]]; then
    echo "[ERROR] post_kd_config not found: ${POST_KD_CONFIG}" >&2
    exit 1
fi
if [[ ! -f "${KD_CONFIG}" ]]; then
    echo "[ERROR] kd_train_config not found: ${KD_CONFIG}" >&2
    exit 1
fi

# ── KD フェーズの重みパスを train_config.yaml から導出して確認 ──────────────────
KD_WEIGHTS=$(python3 -c "
import yaml, os, sys
try:
    cfg = yaml.safe_load(open('${KD_CONFIG}'))
    ckpt = cfg.get('ckpt_save_path', '')
    if not ckpt:
        print('')
        sys.exit(0)
    print(os.path.join(ckpt, 'thermal_kd_student_final.pth'))
except Exception as e:
    print('', file=sys.stderr)
    sys.exit(0)
" 2>/dev/null || echo "")

echo ""
echo "[train_post_kd.sh] ====================================="
echo "[train_post_kd.sh] Post-KD Config : ${POST_KD_CONFIG}"
echo "[train_post_kd.sh] KD Train Config: ${KD_CONFIG}"
echo "[train_post_kd.sh] KD Weights     : ${KD_WEIGHTS:-'(auto-resolved in Python)'}"
echo "[train_post_kd.sh] Extra args     : ${EXTRA_ARGS[*]+"${EXTRA_ARGS[*]}"}"
echo "[train_post_kd.sh] ====================================="
echo ""

# KD 重みが存在するかチェック（警告のみ・エラーにはしない）
if [[ -n "${KD_WEIGHTS}" ]] && [[ ! -f "${KD_WEIGHTS}" ]]; then
    echo "[train_post_kd.sh] WARNING: KD weights not found: ${KD_WEIGHTS}"
    echo "  → KD フェーズを先に完了させてください:"
    echo "       bash scripts/train.sh"
    echo "  → 存在しない場合はランダム重みで Post-KD を開始します。"
    echo ""
fi

# ── Post-KD 訓練実行 ───────────────────────────────────────────────────────────
echo "[train_post_kd.sh] Starting Post-KD training..."

python train_post_kd.py \
    --config          "${POST_KD_CONFIG}" \
    --kd_train_config "${KD_CONFIG}" \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"

echo ""
echo "[train_post_kd.sh] Post-KD training completed."