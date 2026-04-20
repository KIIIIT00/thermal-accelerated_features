"""
train_lightglue_ft.py
LightGlue Fine-tuning エントリーポイント。

学習データ : Freiburg (train split) + TartanRGBT (train split)
評価データ : SThErEO, VIVID（fine-tuning に一切使用しない）

使用方法:
    python train_lightglue_ft.py --config configs/lightglue_ft_config.yaml

    # ステップ数を CLI で上書き
    python train_lightglue_ft.py --config configs/lightglue_ft_config.yaml \\
        --n_steps 10000

事前に必要なもの:
    pip install git+https://github.com/cvg/LightGlue.git
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any


# ── CUDA_VISIBLE_DEVICES を import torch より前に設定 ──────────────────────
def _set_cuda_visible_devices_early() -> None:
    device_num = '0'
    for i, arg in enumerate(sys.argv):
        if arg == '--device_num' and i + 1 < len(sys.argv):
            device_num = sys.argv[i + 1]
            break
    if device_num == '0':
        config_path = None
        for i, arg in enumerate(sys.argv):
            if arg == '--config' and i + 1 < len(sys.argv):
                config_path = sys.argv[i + 1]
                break
        if config_path and os.path.isfile(config_path):
            try:
                import yaml
                cfg = yaml.safe_load(open(config_path)) or {}
                device_num = str(cfg.get('device_num', '0'))
            except Exception:
                pass
    os.environ['CUDA_VISIBLE_DEVICES'] = device_num
    print(f"[Device] CUDA_VISIBLE_DEVICES={device_num}")


_set_cuda_visible_devices_early()
# ──────────────────────────────────────────────────────────────────────────


import yaml
import torch
from torch.utils.data import ConcatDataset, DataLoader


# ---------------------------------------------------------------------------
# YAML ロード
# ---------------------------------------------------------------------------

def _load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _yaml_to_defaults(cfg: dict) -> dict:
    return {k: (None if v is None else v) for k, v in cfg.items()}


# ---------------------------------------------------------------------------
# 引数パーサー
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='LightGlue Fine-tuning for ThermalXFeat',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--config',          type=str, default=None)
    p.add_argument('--thermal_weights', type=str, default=None)
    p.add_argument('--ckpt_save_path',  type=str, default=None)
    p.add_argument('--n_steps',         type=int, default=None)
    p.add_argument('--batch_size',      type=int, default=None)
    p.add_argument('--lr',              type=float, default=None)
    p.add_argument('--max_keypoints',   type=int, default=None)
    p.add_argument('--device_num',      type=str, default=None)
    p.add_argument('--no_wandb',        action='store_true', default=False)
    p.add_argument('--wandb_project',   type=str, default=None)
    p.add_argument('--wandb_run_name',  type=str, default=None)
    return p


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
        args.ckpt_save_path = 'checkpoints/lightglue_ft/default'

    if getattr(args, 'wandb_run_name', None) is None:
        import time
        args.wandb_run_name = 'lgft_' + time.strftime('%Y%m%d_%H%M%S')

    return args


# ---------------------------------------------------------------------------
# DataLoader 構築
# ---------------------------------------------------------------------------

def _get_splits_dir(name: str, args: Any):
    """loader.py と同じ優先順位で splits_dir を解決する。"""
    splits_dir = (getattr(args, 'splits_roots', None) or {}).get(name, '')
    if not splits_dir:
        splits_dir = getattr(args, f'{name}_splits_dir', '') or ''
    if splits_dir:
        return splits_dir

    _SPLITS_SUBPATH = {
        'freiburg':   os.path.join('freiburg',   'splits', 'frame_list'),
        'tartanrgbt': os.path.join('tartanRGBT', 'splits'),
    }
    _REPO    = os.path.dirname(os.path.abspath(__file__))
    _CUST_DS = os.path.join(_REPO, 'third_party', 'anythermal',
                            'custom_datasets')
    subpath = _SPLITS_SUBPATH.get(name, '')
    if subpath:
        candidate = os.path.join(_CUST_DS, subpath)
        if os.path.isdir(candidate):
            print(f"[LG-FT] {name}: splits_dir auto-detected → {candidate}")
            return candidate
    return None


def build_loader(args: Any) -> DataLoader:
    from modules.dataset.thermal.sequential import (
        TartanRGBTSequentialDataset,
        FreiburgSequentialDataset,
        SThErEOSequentialDataset
    )

    _CLS = {
        'freiburg':   FreiburgSequentialDataset,
        'tartanrgbt': TartanRGBTSequentialDataset,
        'sthereo': SThErEOSequentialDataset
    }
    ft_datasets = getattr(args, 'ft_datasets', ['freiburg', 'tartanrgbt', 'sthereo'])
    data_roots  = getattr(args, 'data_roots', {}) or {}
    stride      = getattr(args, 'stride', 1)

    datasets = []
    for name in ft_datasets:
        name_l = name.lower()
        if name_l not in _CLS:
            print(f"[LG-FT] WARNING: unknown dataset {name!r}, skipping")
            continue

        root = data_roots.get(name_l, '') or \
               getattr(args, f'{name_l}_root', '') or ''
        if not root:
            print(f"[LG-FT] WARNING: data_roots.{name_l} not set, skipping")
            continue

        splits_dir = _get_splits_dir(name_l, args)
        if name_l == 'sthereo':
            # SThErEO は splits_dir を受け取らないため除外する
            ds = _CLS[name_l](
                data_root = root,
                stride    = stride,
            )
        else:
            # 他のデータセット（Freiburg, TartanRGBT）は従来通り
            ds = _CLS[name_l](
                data_root  = root,
                splits_dir = splits_dir,
                stride     = stride,
            )
        # ds = _CLS[name_l](
        #     data_root  = root,
        #     splits_dir = splits_dir,
        #     stride     = stride,
        # )
        datasets.append(ds)
        print(f"[LG-FT] {name}: {len(ds)} pairs")

    if not datasets:
        raise RuntimeError(
            "有効なデータセットがありません。"
            "data_roots を確認してください。"
        )

    combined = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
    g = torch.Generator()
    g.manual_seed(getattr(args, 'seed', 42))

    num_workers = getattr(args, 'train_num_workers', 4)
    return DataLoader(
        combined,
        batch_size   = getattr(args, 'batch_size', 4),
        shuffle      = True,
        num_workers  = num_workers,
        pin_memory   = True,
        drop_last    = True,
        generator    = g,
        persistent_workers = num_workers > 0,
    )


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_arguments()

    print("=" * 60)
    print("  LightGlue Fine-tuning")
    print(f"  thermal_weights : {args.thermal_weights}")
    print(f"  ckpt_save_path  : {args.ckpt_save_path}")
    print(f"  n_steps         : {args.n_steps}")
    print(f"  ft_datasets     : {getattr(args, 'ft_datasets', [])}")
    print("=" * 60)

    # LightGlue のインストール確認
    try:
        import lightglue  # noqa: F401
    except ImportError:
        print("\n[ERROR] LightGlue がインストールされていません。")
        print("  pip install git+https://github.com/cvg/LightGlue.git\n")
        sys.exit(1)

    from modules.training.train_lightglue_ft import LightGlueFTTrainer

    loader  = build_loader(args)
    trainer = LightGlueFTTrainer(args)

    try:
        trainer.run(loader)
    finally:
        trainer.finalize()


if __name__ == '__main__':
    main()