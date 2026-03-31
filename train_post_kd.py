"""
train_post_kd.py
Post-KD 訓練エントリーポイント。

引数の優先順位（高い順）:
    1. CLI 引数
    2. --config YAML (configs/post_kd_config.yaml)
    3. argparse デフォルト値

起動例:
    # 全ステージを順番に実行（推奨）
    python train_post_kd.py --config configs/post_kd_config.yaml

    # Stage 1 のみ実行
    python train_post_kd.py --config configs/post_kd_config.yaml --stages 1

    # Stage 1 → Stage 2 を連続実行
    python train_post_kd.py --config configs/post_kd_config.yaml --stages 1 2

    # シェルスクリプト経由（推奨）
    bash scripts/train_post_kd.sh
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any


# ---------------------------------------------------------------------------
# YAML ロードヘルパー（train_kd.py と同じ実装）
# ---------------------------------------------------------------------------

def _load_yaml(path: str) -> dict:
    try:
        import yaml
    except ImportError:
        print("[Config] PyYAML not installed.", file=sys.stderr)
        sys.exit(1)
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _yaml_to_defaults(cfg: dict) -> dict:
    return {k: (None if v is None else v) for k, v in cfg.items()}


# ---------------------------------------------------------------------------
# 引数パーサー
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Thermal XFeat Post-KD Training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('--config', type=str, default=None, metavar='PATH')

    # ── 実行するステージ ──────────────────────────────────────────────
    parser.add_argument(
        '--stages', nargs='+', type=int, default=[1, 2],
        help='実行するステージ番号。例: --stages 1 2')

    # ── モデル ──────────────────────────────────────────────────────────
    parser.add_argument('--kd_weights', type=str, default=None,
                        help='KD 済み生徒モデルの重みファイルパス（必須）')
    parser.add_argument('--ckpt_save_path', type=str, default=None)

    # ── データセット ─────────────────────────────────────────────────────
    # Stage 1 用（HomographicAdapter）
    parser.add_argument('--s1_dataset', nargs='+',
                        default=['freiburg', 'tartanrgbt'])
    parser.add_argument('--s1_n_per_image', type=int, default=2,
                        help='1枚の熱画像から生成するホモグラフィーペア数')
    parser.add_argument('--s1_perspective_range', type=float, default=0.05)
    parser.add_argument('--s1_rotation_range',    type=float, default=15.0)
    parser.add_argument('--s1_scale_range',       type=float, default=0.15)
    parser.add_argument('--s1_translation_range', type=float, default=0.10)

    # Stage 2 用（SequentialDataset）
    parser.add_argument('--s2_dataset', nargs='+',
                        default=['tartanrgbt', 'freiburg'])
    parser.add_argument('--s2_stride', type=int, default=1,
                        help='連続フレームの間隔')

    # 共通データパス（train_config.yaml の data_roots と同じキー）
    parser.add_argument('--data_roots', type=dict, default=None)
    parser.add_argument('--freiburg_root',   type=str, default=None)
    parser.add_argument('--tartanrgbt_root', type=str, default=None)
    parser.add_argument('--vivid_root',      type=str, default=None)
    parser.add_argument('--sthereo_root',    type=str, default=None)
    parser.add_argument('--freiburg_splits_dir',   type=str, default=None)
    parser.add_argument('--tartanrgbt_splits_dir', type=str, default=None)

    parser.add_argument('--batch_size',        type=int, default=4)
    parser.add_argument('--train_num_workers', type=int, default=4)
    parser.add_argument('--seed',              type=int, default=42)

    # ── Stage 1 ハイパーパラメータ ─────────────────────────────────────
    parser.add_argument('--s1_lr',           type=float, default=1e-4)
    parser.add_argument('--s1_lr_step',      type=int,   default=10_000)
    parser.add_argument('--s1_n_steps',      type=int,   default=30_000)
    parser.add_argument('--s1_lambda_fine',  type=float, default=0.5)
    parser.add_argument('--s1_n_pts',        type=int,   default=256)

    # ── Stage 2 ハイパーパラメータ ─────────────────────────────────────
    parser.add_argument('--s2a_lr',           type=float, default=5e-5)
    parser.add_argument('--s2a_n_steps',      type=int,   default=10_000)
    parser.add_argument('--s2b_lr',           type=float, default=1e-5)
    parser.add_argument('--s2b_n_steps',      type=int,   default=5_000)
    parser.add_argument('--s2_lambda_epi',    type=float, default=0.5)
    parser.add_argument('--s2_epi_threshold', type=float, default=2.0)

    # ── 共通訓練設定 ────────────────────────────────────────────────────
    parser.add_argument('--grad_clip',        type=float, default=1.0)
    parser.add_argument('--lr_gamma',         type=float, default=0.5)
    parser.add_argument('--save_ckpt_every',  type=int,   default=2_000)
    parser.add_argument('--log_every',        type=int,   default=100)
    parser.add_argument('--device_num',       type=str,   default='0')

    # ── wandb ─────────────────────────────────────────────────────────────
    parser.add_argument('--no_wandb',       action='store_true', default=False)
    parser.add_argument('--wandb_project',  type=str, default='thermal-xfeat-post-kd')
    parser.add_argument('--wandb_run_name', type=str, default=None)
    parser.add_argument('--wandb_group',    type=str, default='post_kd')
    parser.add_argument('--wandb_tags',     nargs='+', default=[])

    return parser


def parse_arguments() -> argparse.Namespace:
    parser = _build_parser()
    pre, _ = parser.parse_known_args()

    if pre.config is not None:
        if not os.path.isfile(pre.config):
            parser.error(f"--config not found: {pre.config!r}")
        cfg = _load_yaml(pre.config)
        parser.set_defaults(**_yaml_to_defaults(cfg))
        print(f"[Config] Loaded: {pre.config}")

    args = parser.parse_args()

    if args.ckpt_save_path is None:
        parser.error("--ckpt_save_path is required (via --config or CLI).")
    if args.kd_weights is None:
        print("[WARNING] --kd_weights not specified. Model starts from random weights.")

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device_num)

    if args.wandb_run_name is None:
        args.wandb_run_name = 'post_kd_s' + '_'.join(str(s) for s in args.stages)

    return args


# ---------------------------------------------------------------------------
# DataLoader 構築ヘルパー
# ---------------------------------------------------------------------------

def _get_data_root(name: str, args: Any) -> str:
    """data_roots dict → 個別引数 → 環境変数の順でルートパスを取得。"""
    root = (getattr(args, 'data_roots', None) or {}).get(name, '')
    if not root:
        root = getattr(args, f'{name}_root', '') or ''
    if not root:
        env_key = f'ANYTHERMAL_{name.upper()}_DATA_ROOT'
        root = os.environ.get(env_key, '')
    if not root:
        raise RuntimeError(
            f"Root path for '{name}' not configured.\n"
            f"  Set via YAML data_roots, --{name}_root, or env {env_key}"
        )
    return root


def build_stage1_loader(args: Any):
    """Stage 1 用 HomographicAdapter DataLoader を構築する。"""
    import torch
    from torch.utils.data import ConcatDataset, DataLoader

    from modules.dataset.thermal.freiburg   import FreiburgDataset
    from modules.dataset.thermal.tartanrgbt import TartanRGBTDataset
    from modules.dataset.thermal.homographic_adapter import ThermalHomographicDataset

    _CLS = {
        'freiburg':   FreiburgDataset,
        'tartanrgbt': TartanRGBTDataset,
    }

    datasets = []
    for name in args.s1_dataset:
        name_l = name.lower()
        if name_l not in _CLS:
            raise ValueError(f"[Stage1] Unsupported dataset: {name!r}")
        root = _get_data_root(name_l, args)
        splits_dir = getattr(args, f'{name_l}_splits_dir', None)

        base_ds = _CLS[name_l](
            data_root=root,
            splits_dir=splits_dir,
            split='train',
            augment=False,   # HomographicAdapter が拡張を担う
        )
        wrapped = ThermalHomographicDataset(
            base_dataset=base_ds,
            n_per_image=args.s1_n_per_image,
            perspective_range=args.s1_perspective_range,
            rotation_range=args.s1_rotation_range,
            scale_range=args.s1_scale_range,
            translation_range=args.s1_translation_range,
        )
        datasets.append(wrapped)
        print(f"[Stage1] {name}: {len(wrapped)} pairs")

    combined = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
    g = torch.Generator()
    g.manual_seed(args.seed)
    return DataLoader(
        combined,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.train_num_workers,
        pin_memory=True,
        drop_last=True,
        generator=g,
    )


def build_stage2_loader(args: Any):
    """Stage 2 用 SequentialDataset DataLoader を構築する。"""
    import torch
    from torch.utils.data import ConcatDataset, DataLoader

    from modules.dataset.thermal.sequential import (
        TartanRGBTSequentialDataset,
        FreiburgSequentialDataset,
    )

    _CLS = {
        'tartanrgbt': TartanRGBTSequentialDataset,
        'freiburg':   FreiburgSequentialDataset,
    }

    datasets = []
    for name in args.s2_dataset:
        name_l = name.lower()
        if name_l not in _CLS:
            raise ValueError(f"[Stage2] Unsupported dataset: {name!r}")
        root = _get_data_root(name_l, args)
        splits_dir = getattr(args, f'{name_l}_splits_dir', None)

        if name_l == 'tartanrgbt':
            ds = TartanRGBTSequentialDataset(
                data_root=root,
                splits_dir=splits_dir,
                stride=args.s2_stride,
            )
        else:
            ds = FreiburgSequentialDataset(
                data_root=root,
                splits_dir=splits_dir,
                stride=args.s2_stride,
            )
        datasets.append(ds)
        print(f"[Stage2] {name}: {len(ds)} pairs")

    combined = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
    g = torch.Generator()
    g.manual_seed(args.seed)
    return DataLoader(
        combined,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.train_num_workers,
        pin_memory=True,
        drop_last=True,
        generator=g,
    )


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_arguments()

    print("=" * 60)
    print("  Thermal XFeat Post-KD Training")
    print(f"  Stages: {args.stages}")
    print(f"  KD weights: {args.kd_weights}")
    print(f"  Save path: {args.ckpt_save_path}")
    print("=" * 60)

    from modules.training.train_post_kd import PostKDTrainer
    trainer = PostKDTrainer(args)

    try:
        if 1 in args.stages:
            loader1 = build_stage1_loader(args)
            trainer.run_stage1(loader1)

        if 2 in args.stages:
            loader2 = build_stage2_loader(args)
            trainer.run_stage2(loader2)

        if 3 in args.stages:
            trainer.run_stage3_stub()

    finally:
        trainer.finalize()


if __name__ == '__main__':
    main()