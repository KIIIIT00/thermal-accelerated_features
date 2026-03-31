"""
train_kd.py
Thermal XFeat Knowledge Distillation 訓練エントリーポイント。

引数の優先順位（高い順）:
    1. CLI 引数          python train_kd.py --lambda_kd_rel 0.2
    2. --config YAML     configs/train_config.yaml
    3. argparse デフォルト値

起動例:
    # config ファイルを使った通常実行
    python train_kd.py --config configs/train_config.yaml

    # config を使いつつ特定引数を CLI で上書き
    python train_kd.py --config configs/train_config.yaml --lambda_kd_rel 0.2

    # config なし（すべて CLI 指定・後方互換）
    python train_kd.py \
        --teacher_weights   weights/xfeat.pt \
        --ckpt_save_path    checkpoints/thermal_kd/rel010_fpn005 \
        --dataset           freiburg tartanrgbt \
        --lambda_kd_rel     0.1 \
        --lambda_fpn        0.05 \
        --device_num        0

    # オフライン環境（wandb 無効）
    python train_kd.py --config configs/train_config.yaml --no_wandb

    # wandb Sweep エージェント経由
    wandb sweep configs/sweep_config.yaml
    wandb agent <entity>/thermal-xfeat-kd/<sweep_id>

    # シェルスクリプト経由（推奨）
    bash scripts/train.sh
    bash scripts/train.sh --config configs/train_config.yaml --device_num 1
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any


# ---------------------------------------------------------------------------
# YAML 設定ファイルの読み込みと argparse へのマージ
# ---------------------------------------------------------------------------

def _load_yaml(path: str) -> dict:
    """YAML ファイルを読み込んで dict を返す。PyYAML のみに依存する。"""
    try:
        import yaml
    except ImportError:
        print("[Config] PyYAML not installed. Install with: pip install pyyaml",
              file=sys.stderr)
        sys.exit(1)
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return data


def _yaml_to_defaults(cfg: dict) -> dict:
    """
    YAML dict を argparse.set_defaults() に渡せる形式に変換する。

    - null  → None
    - リスト値はそのまま保持
    - bool  はそのまま保持（store_true の上書きに使う）
    """
    result = {}
    for key, val in cfg.items():
        result[key] = None if val is None else val
    return result


# ---------------------------------------------------------------------------
# 引数パーサー
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Thermal XFeat Knowledge Distillation Training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── 設定ファイル（最初に定義することで --help にも表示される）────────
    parser.add_argument(
        '--config', type=str, default=None, metavar='PATH',
        help='Path to YAML config file (e.g. configs/train_config.yaml). '
             'CLI args override values in the config.')

    # ── モデル ──────────────────────────────────────────────────────────
    parser.add_argument(
        '--teacher_weights', type=str, default='weights/xfeat.pt')
    parser.add_argument(
        '--student_init', type=str, default=None,
        help='Student init weights. None → use teacher_weights.')
    parser.add_argument(
        '--ckpt_save_path', type=str, default=None,
        help='Directory to save checkpoints and logs. '
             'Required either via --config or this flag.')

    # ── データセットルートパス ────────────────────────────────────────────
    # 優先順位: --data_roots / config の data_roots
    #         > --{name}_root 個別引数
    #         > 環境変数 ANYTHERMAL_{NAME}_DATA_ROOT
    parser.add_argument(
        '--freiburg_root',   type=str, default=None,
        help='Root directory of the Freiburg thermal dataset.')
    parser.add_argument(
        '--tartanrgbt_root', type=str, default=None,
        help='Root directory of the TartanRGBT dataset.')
    parser.add_argument(
        '--vivid_root',      type=str, default=None,
        help='Root directory of the VIVID++ dataset.')
    parser.add_argument(
        '--sthereo_root',    type=str, default=None,
        help='Root directory of the STHEREO dataset.')
    parser.add_argument(
        '--freiburg_splits_dir',   type=str, default=None,
        help='Splits dir for Freiburg (overrides auto-detect).')
    parser.add_argument(
        '--tartanrgbt_splits_dir', type=str, default=None,
        help='Splits dir for TartanRGBT (overrides auto-detect).')
    parser.add_argument(
        '--vivid_splits_dir',      type=str, default=None,
        help='Splits dir for VIVID++ (overrides auto-detect).')
    parser.add_argument(
        '--sthereo_splits_dir',    type=str, default=None,
        help='Splits dir for STHEREO (overrides auto-detect).')

    # ── データ ──────────────────────────────────────────────────────────
    parser.add_argument(
        '--dataset', nargs='+', default=['freiburg', 'tartanrgbt'])
    parser.add_argument(
        '--eval_dataset', nargs='+', default=['freiburg'])
    parser.add_argument('--batch_size',        type=int,   default=8)
    parser.add_argument('--eval_batch_size',   type=int,   default=8)
    parser.add_argument('--train_num_workers', type=int,   default=4)
    parser.add_argument('--eval_num_workers',  type=int,   default=2)
    parser.add_argument('--seed',              type=int,   default=42)
    parser.add_argument('--augment',           action='store_true', default=True)
    parser.add_argument(
        '--aug_list', nargs='+',
        default=['affine', 'hflip', 'brightness', 'contrast',
                 'gamma', 'diurnal_inversion'])
    parser.add_argument(
        '--intra_dataset_batch', action='store_true', default=True)

    # ── AnyThermal build_dataset 互換引数（スタンドアロン時は無視）─────
    parser.add_argument('--use_odom',   action='store_true', default=False)
    parser.add_argument('--vpr_test',   action='store_true', default=False)
    parser.add_argument('--debug',      action='store_true', default=False)
    parser.add_argument('--subsample_val',               type=int,   default=1)
    parser.add_argument('--dist_thresh',                 type=float, default=10.0)
    parser.add_argument('--val_positive_dist_threshold', type=float, default=10.0)
    parser.add_argument('--neg_ring_outer_radius',       type=float, default=50.0)
    parser.add_argument('--teacher_modality',            type=str,   default='rgb')
    parser.add_argument('--student_modality',            type=str,   default='thermal')
    parser.add_argument('--student_modality_dual',       type=str,   default='thermal')
    parser.add_argument('--thermal_aug_list', nargs='+', default=None)
    parser.add_argument('--cart_split',      action='store_true', default=False)
    parser.add_argument('--dataset_split_for_eval', type=str, default='val')
    parser.add_argument('--common_database', action='store_true', default=False)
    parser.add_argument('--train',           action='store_true', default=True)
    parser.add_argument('--sampling_weight', type=str,  default='uniform')
    parser.add_argument('--crop_images',     action='store_true', default=False)
    parser.add_argument('--rescale_during_crop', action='store_true', default=False)

    # ── 訓練 ────────────────────────────────────────────────────────────
    parser.add_argument('--n_steps',         type=int,   default=100_000)
    parser.add_argument('--lr',              type=float, default=3e-4)
    parser.add_argument('--lr_step',         type=int,   default=30_000)
    parser.add_argument('--lr_gamma',        type=float, default=0.5)
    parser.add_argument('--grad_clip',       type=float, default=1.0)
    parser.add_argument('--save_ckpt_every', type=int,   default=2_000)
    parser.add_argument('--log_every',       type=int,   default=100)

    # ── KD ハイパーパラメータ ────────────────────────────────────────────
    parser.add_argument('--lambda_kd_rel',       type=float, default=0.1)
    parser.add_argument('--lambda_fpn',          type=float, default=0.05)
    parser.add_argument('--fpn_sigma_min',       type=float, default=2.0)
    parser.add_argument('--fpn_sigma_max',       type=float, default=8.0)
    parser.add_argument('--p_diurnal_inversion', type=float, default=0.3)
    parser.add_argument('--n_kd_samples',        type=int,   default=1024)
    parser.add_argument('--infonce_temp',        type=float, default=0.2)

    # ── デバイス ─────────────────────────────────────────────────────────
    parser.add_argument('--device_num', type=str, default='0')

    # ── wandb ────────────────────────────────────────────────────────────
    parser.add_argument(
        '--no_wandb', action='store_true', default=False,
        help='Disable wandb (for offline environments).')
    parser.add_argument('--wandb_project',   type=str,    default='thermal-xfeat-kd')
    parser.add_argument('--wandb_run_name',  type=str,    default=None)
    parser.add_argument('--wandb_group',     type=str,    default='grid_search')
    parser.add_argument('--wandb_tags',      nargs='+',   default=[])
    parser.add_argument('--image_log_every', type=int,    default=500)

    return parser


def parse_arguments() -> argparse.Namespace:
    """
    引数を 3 段階でマージして返す。

    優先順位（高い順）:
        1. CLI 引数
        2. --config で指定した YAML ファイル
        3. argparse ハードコードデフォルト値
    """
    parser = _build_parser()

    # ── Step 1: --config だけ先に取り出す（残りの CLI 引数はそのまま保持）
    pre, _ = parser.parse_known_args()
    config_path = pre.config

    # ── Step 2: YAML を読んで parser のデフォルト値として注入
    if config_path is not None:
        if not os.path.isfile(config_path):
            parser.error(f"--config file not found: {config_path!r}")
        cfg = _load_yaml(config_path)
        defaults = _yaml_to_defaults(cfg)
        parser.set_defaults(**defaults)
        print(f"[Config] Loaded: {config_path}")

    # ── Step 3: CLI 引数で再パース（YAML デフォルトを CLI が上書き）
    args = parser.parse_args()

    # ── 必須値チェック
    if args.ckpt_save_path is None:
        parser.error(
            "--ckpt_save_path is required. "
            "Set it via --config or --ckpt_save_path.")

    # ── CUDA_VISIBLE_DEVICES
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device_num)

    # ── wandb_run_name の自動生成
    if args.wandb_run_name is None:
        args.wandb_run_name = f'rel{args.lambda_kd_rel}_fpn{args.lambda_fpn}'

    # ── 設定サマリーを表示
    _print_config_summary(args)

    return args


def _print_config_summary(args: Any) -> None:
    """重要なハイパーパラメータを起動時にコンソール出力する。"""
    print("=" * 60)
    print("  Thermal XFeat KD — Configuration Summary")
    print("=" * 60)
    keys = [
        'teacher_weights', 'ckpt_save_path',
        'data_roots',
        'freiburg_root', 'tartanrgbt_root', 'vivid_root', 'sthereo_root',
        'dataset', 'batch_size', 'n_steps',
        'lr', 'lr_step', 'lr_gamma', 'grad_clip',
        'lambda_kd_rel', 'lambda_fpn',
        'fpn_sigma_min', 'fpn_sigma_max',
        'p_diurnal_inversion', 'n_kd_samples', 'infonce_temp',
        'aug_list', 'device_num',
        'no_wandb', 'wandb_project', 'wandb_run_name', 'wandb_group',
    ]
    for k in keys:
        print(f"  {k:<30s} = {getattr(args, k, '(not set)')}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_arguments()

    # ── wandb Sweep エージェントが渡す config で args を上書き ────────────
    if not args.no_wandb:
        try:
            import wandb
            from modules.training.train_kd import _init_wandb
            _init_wandb(args)
            if wandb.run is not None and len(dict(wandb.config)) > 0:
                print("[wandb] Sweep config override detected.")
                for key, val in dict(wandb.config).items():
                    if hasattr(args, key):
                        setattr(args, key, val)
                # Sweep 上書き後に run_name とチェックポイントパスを再生成
                args.wandb_run_name = (
                    f'rel{args.lambda_kd_rel}_fpn{args.lambda_fpn}')
                args.ckpt_save_path = os.path.join(
                    'checkpoints', args.wandb_run_name)
                os.makedirs(args.ckpt_save_path, exist_ok=True)
                print(f"[wandb] ckpt_save_path overridden: {args.ckpt_save_path}")
        except ImportError:
            pass

    # ── DataLoader 構築（3 段階優先度）────────────────────────────────────
    from modules.dataset.thermal.loader import (
        build_thermal_loader,
        build_thermal_eval_loader,
    )
    train_loader = build_thermal_loader(args)
    eval_loader  = build_thermal_eval_loader(args)

    if eval_loader is not None:
        print(f"[main] eval_loader: {len(eval_loader.dataset)} pairs total")
    else:
        print("[main] eval_loader: None (eval_dataset not configured)")

    # ── Trainer 構築・訓練開始 ─────────────────────────────────────────────
    from modules.training.train_kd import ThermalXFeatKDTrainer
    trainer = ThermalXFeatKDTrainer(args)
    trainer.train(train_loader, eval_loader=eval_loader)


if __name__ == '__main__':
    main()