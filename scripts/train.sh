#!/usr/bin/env bash
# =============================================================================
# scripts/train.sh
# Thermal XFeat KD 訓練シェルスクリプト
#
# 使い方:
#   bash scripts/train.sh                              # デフォルト config
#   bash scripts/train.sh --config configs/train_config.yaml
#   bash scripts/train.sh --device_num 1               # GPU 指定上書き
#   bash scripts/train.sh --lambda_kd_rel 0.2          # 任意引数を上書き
#   bash scripts/train.sh --no_wandb                   # wandb 無効
#   bash scripts/train.sh --mode grid                  # 手動グリッドサーチ
#   bash scripts/train.sh --mode sweep                 # wandb Sweep 起動
#
# このスクリプトは accelerated_features/ リポジトリルートから実行すること。
# =============================================================================

set -euo pipefail

# ── デフォルト値 ─────────────────────────────────────────────────────────────
CONFIG="configs/train_config.yaml"
MODE="single"          # single | grid | sweep
SWEEP_ID=""            # wandb Sweep ID (--mode sweep 時に必須)
ENTITY=""              # wandb エンティティ名
PROJECT="thermal-xfeat-kd"
EXTRA_ARGS=()          # train_kd.py に追加で渡す任意引数

# ── 引数パース ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            CONFIG="$2"; shift 2 ;;
        --mode)
            MODE="$2"; shift 2 ;;
        --sweep_id)
            SWEEP_ID="$2"; shift 2 ;;
        --entity)
            ENTITY="$2"; shift 2 ;;
        --project)
            PROJECT="$2"; shift 2 ;;
        *)
            # 上記以外はすべて train_kd.py に転送する
            EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# ── 共通チェック ──────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [[ ! -f "${CONFIG}" ]]; then
    echo "[train.sh] ERROR: config file not found: ${CONFIG}" >&2
    exit 1
fi

echo "[train.sh] Repository root : ${REPO_ROOT}"
echo "[train.sh] Config file     : ${CONFIG}"
echo "[train.sh] Mode            : ${MODE}"
echo "[train.sh] Extra args      : ${EXTRA_ARGS[*]+"${EXTRA_ARGS[*]}"}"
echo ""

# ── Python 実行ヘルパー ───────────────────────────────────────────────────────
# 仮想環境 (.venv) があれば自動で有効化する
_activate_venv() {
    if [[ -f ".venv/bin/activate" ]]; then
        # shellcheck disable=SC1091
        source .venv/bin/activate
        echo "[train.sh] Activated .venv"
    fi
}

_run_single() {
    local rel="$1"
    local fpn="$2"
    shift 2
    local extra=("$@")

    python train_kd.py \
        --config "${CONFIG}" \
        --lambda_kd_rel "${rel}" \
        --lambda_fpn    "${fpn}" \
        "${extra[@]+"${extra[@]}"}"
}

# =============================================================================
# モード分岐
# =============================================================================

_activate_venv

case "${MODE}" in

    # ── single: config をそのまま使って 1 回訓練 ─────────────────────────
    single)
        echo "[train.sh] === Single run ==="
        python train_kd.py \
            --config "${CONFIG}" \
            "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
        ;;

    # ── grid: λ_rel × λ_fpn の手動グリッドサーチ（wandb 不要）──────────
    grid)
        echo "[train.sh] === Manual grid search (no wandb required) ==="

        REL_VALUES=(0.05 0.1 0.2)
        FPN_VALUES=(0.01 0.05 0.1)

        for rel in "${REL_VALUES[@]}"; do
            for fpn in "${FPN_VALUES[@]}"; do
                RUN_NAME="rel${rel}_fpn${fpn}"
                CKPT_PATH="checkpoints/grid/${RUN_NAME}"

                echo ""
                echo "[train.sh] ── Grid run: lambda_kd_rel=${rel}  lambda_fpn=${fpn} ──"
                echo "[train.sh]    ckpt_save_path: ${CKPT_PATH}"

                python train_kd.py \
                    --config        "${CONFIG}" \
                    --lambda_kd_rel "${rel}" \
                    --lambda_fpn    "${fpn}" \
                    --ckpt_save_path "${CKPT_PATH}" \
                    --wandb_run_name "${RUN_NAME}" \
                    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"

                echo "[train.sh] ── Finished: ${RUN_NAME} ──"
            done
        done

        echo ""
        echo "[train.sh] === All grid runs completed ==="
        ;;

    # ── sweep: wandb Sweep エージェントを GPU ごとに並列起動 ─────────────
    sweep)
        echo "[train.sh] === wandb Sweep mode ==="

        # 新規 Sweep 作成か既存 Sweep ID 使用かを判断
        if [[ -z "${SWEEP_ID}" ]]; then
            echo "[train.sh] Creating new Sweep from configs/sweep_config.yaml ..."
            SWEEP_OUT=$(wandb sweep configs/sweep_config.yaml 2>&1)
            echo "${SWEEP_OUT}"
            # "Created sweep ID: abc12345" から ID を抽出
            SWEEP_ID=$(echo "${SWEEP_OUT}" | grep -oP '(?<=sweep ID: )\S+' || true)
            if [[ -z "${SWEEP_ID}" ]]; then
                echo "[train.sh] ERROR: Could not extract sweep ID from wandb output." >&2
                echo "[train.sh] Run manually: wandb sweep configs/sweep_config.yaml" >&2
                exit 1
            fi
            echo "[train.sh] New Sweep ID: ${SWEEP_ID}"
        else
            echo "[train.sh] Using existing Sweep ID: ${SWEEP_ID}"
        fi

        # エンティティ/プロジェクトを組み立て
        if [[ -n "${ENTITY}" ]]; then
            SWEEP_PATH="${ENTITY}/${PROJECT}/${SWEEP_ID}"
        else
            SWEEP_PATH="${PROJECT}/${SWEEP_ID}"
        fi
        echo "[train.sh] Sweep path: ${SWEEP_PATH}"

        # GPU ごとにエージェントを並列起動（CUDA_VISIBLE_DEVICES で制御）
        # EXTRA_ARGS に --device_num が含まれている場合はそちらを優先
        GPU_LIST=(0 1)   # 使用する GPU のリスト（必要に応じて変更）

        echo "[train.sh] Launching agents on GPU(s): ${GPU_LIST[*]}"
        for gpu in "${GPU_LIST[@]}"; do
            CUDA_VISIBLE_DEVICES="${gpu}" wandb agent "${SWEEP_PATH}" &
            echo "[train.sh] Agent started on GPU ${gpu} (PID $!)"
        done

        echo ""
        echo "[train.sh] All agents launched. Waiting ..."
        wait
        echo "[train.sh] === Sweep completed ==="
        ;;

    *)
        echo "[train.sh] ERROR: Unknown mode '${MODE}'. Choose: single | grid | sweep" >&2
        exit 1
        ;;
esac