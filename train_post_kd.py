"""
train_post_kd.py
Post-KD 訓練エントリーポイント。

kd_weights の解決優先順位:
    1. CLI --kd_weights
    2. post_kd_config.yaml の kd_weights（null でなければ）
    3. post_kd_config.yaml の kd_train_config → train_config.yaml の
       ckpt_save_path + "/thermal_kd_student_final.pth"

起動例:
    # 推奨: config だけ指定（kd_weights は自動解決）
    python train_post_kd.py --config configs/post_kd_config.yaml

    # Stage 1 のみ
    python train_post_kd.py --config configs/post_kd_config.yaml --stages 1

    # kd_weights を直接上書き
    python train_post_kd.py --config configs/post_kd_config.yaml \
        --kd_weights checkpoints/thermal_kd/run_A/thermal_kd_student_final.pth

    # シェルスクリプト経由（推奨）
    bash scripts/train_post_kd.sh
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Optional

# ── CUDA_VISIBLE_DEVICES を import torch より前に設定 ──────────────────────
# torch を import した時点で GPU が確定するため、
# argparse で取得した device_num を使って最速で設定する。
def _set_cuda_visible_devices_early() -> None:
    """
    sys.argv から --device_num / --config を解析して
    CUDA_VISIBLE_DEVICES を import torch より前に設定する。
    """
    device_num = '0'  # デフォルト

    # 1. --device_num を直接探す
    for i, arg in enumerate(sys.argv):
        if arg == '--device_num' and i + 1 < len(sys.argv):
            device_num = sys.argv[i + 1]
            break

    # 2. --config から device_num を読む（--device_num が未指定の場合）
    if device_num == '0':
        config_path = None
        for i, arg in enumerate(sys.argv):
            if arg == '--config' and i + 1 < len(sys.argv):
                config_path = sys.argv[i + 1]
                break
        if config_path and os.path.isfile(config_path):
            try:
                import yaml  # yaml は torch に依存しないので safe
                cfg = yaml.safe_load(open(config_path)) or {}
                device_num = str(cfg.get('device_num', '0'))
            except Exception:
                pass

    os.environ['CUDA_VISIBLE_DEVICES'] = device_num
    print(f"[Device] CUDA_VISIBLE_DEVICES={device_num}")


_set_cuda_visible_devices_early()
# ──────────────────────────────────────────────────────────────────────────


# ---------------------------------------------------------------------------
# YAML ロードヘルパー
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
# kd_weights 自動解決
# ---------------------------------------------------------------------------

def _resolve_kd_weights(args: argparse.Namespace) -> Optional[str]:
    """
    kd_weights を以下の優先順位で解決して返す。

    1. args.kd_weights が非 None・非空文字列
    2. args.kd_train_config が指定されている場合:
       → その YAML の ckpt_save_path + "/thermal_kd_student_final.pth"
    3. None（警告を出してランダム重みで続行）
    """
    # 優先度1: 直接指定
    kw = getattr(args, 'kd_weights', None)
    if kw and str(kw).strip():
        return str(kw).strip()

    # 優先度2: kd_train_config から導出
    kd_cfg_path = getattr(args, 'kd_train_config', None)
    if kd_cfg_path and os.path.isfile(str(kd_cfg_path)):
        kd_cfg = _load_yaml(str(kd_cfg_path))
        ckpt_save_path = kd_cfg.get('ckpt_save_path', '')
        if ckpt_save_path:
            derived = os.path.join(
                str(ckpt_save_path), 'thermal_kd_student_final.pth')
            print(f"[Config] kd_weights auto-resolved from kd_train_config:")
            print(f"         {kd_cfg_path} → ckpt_save_path={ckpt_save_path}")
            print(f"         → kd_weights={derived}")
            return derived
        else:
            print(f"[Config] WARNING: kd_train_config={kd_cfg_path} に "
                  f"ckpt_save_path が見つかりません。")

    return None


# ---------------------------------------------------------------------------
# 引数パーサー
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Thermal XFeat Post-KD Training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('--config', type=str, default=None, metavar='PATH',
                        help='post_kd_config.yaml のパス')

    # ── KD 参照 ────────────────────────────────────────────────────────────
    parser.add_argument(
        '--kd_train_config', type=str, default=None,
        help='KD フェーズの設定ファイル（train_config.yaml）。'
             'kd_weights を自動導出するために使う。')
    parser.add_argument(
        '--kd_weights', type=str, default=None,
        help='KD 済み生徒モデルの重みファイルパス。'
             '未指定の場合は kd_train_config から自動導出。')

    # ── 実行するステージ ──────────────────────────────────────────────────
    parser.add_argument(
        '--stages', nargs='+', type=int, default=[1, 2],
        help='実行するステージ番号。例: --stages 1 2')

    # ── 保存先 ────────────────────────────────────────────────────────────
    parser.add_argument('--ckpt_save_path', type=str, default=None)

    # ── データセット ──────────────────────────────────────────────────────
    parser.add_argument('--s1_dataset', nargs='+',
                        default=['freiburg', 'tartanrgbt'])
    parser.add_argument('--s1_n_per_image',       type=int,   default=2)
    parser.add_argument('--s1_perspective_range', type=float, default=0.05)
    parser.add_argument('--s1_rotation_range',    type=float, default=15.0)
    parser.add_argument('--s1_scale_range',        type=float, default=0.15)
    parser.add_argument('--s1_translation_range', type=float, default=0.10)

    parser.add_argument('--s2_dataset', nargs='+',
                        default=['tartanrgbt', 'freiburg'])
    parser.add_argument('--s2_stride', type=int, default=1)

    parser.add_argument('--data_roots',          type=dict, default=None)
    parser.add_argument('--freiburg_root',       type=str,  default=None)
    parser.add_argument('--tartanrgbt_root',     type=str,  default=None)
    parser.add_argument('--vivid_root',          type=str,  default=None)
    parser.add_argument('--sthereo_root',        type=str,  default=None)
    parser.add_argument('--freiburg_splits_dir', type=str,  default=None)
    parser.add_argument('--tartanrgbt_splits_dir', type=str, default=None)

    parser.add_argument('--batch_size',        type=int, default=4)
    parser.add_argument('--train_num_workers', type=int, default=4)
    parser.add_argument('--seed',              type=int, default=42)

    # ── Stage 1 ──────────────────────────────────────────────────────────
    parser.add_argument('--s1_lr',          type=float, default=1e-4)
    parser.add_argument('--s1_lr_step',     type=int,   default=10_000)
    parser.add_argument('--s1_n_steps',     type=int,   default=30_000)
    parser.add_argument('--s1_lambda_fine', type=float, default=0.5)
    parser.add_argument('--s1_n_pts',       type=int,   default=256)

    # ── Stage 2 ──────────────────────────────────────────────────────────
    parser.add_argument('--s2a_lr',           type=float, default=5e-5)
    parser.add_argument('--s2a_n_steps',      type=int,   default=10_000)
    parser.add_argument('--s2b_lr',           type=float, default=1e-5)
    parser.add_argument('--s2b_n_steps',      type=int,   default=5_000)
    parser.add_argument('--s2_lambda_epi',    type=float, default=0.5)
    parser.add_argument('--s2_epi_threshold', type=float, default=2.0)

    # ── 共通 ─────────────────────────────────────────────────────────────
    parser.add_argument('--grad_clip',       type=float, default=1.0)
    parser.add_argument('--lr_gamma',        type=float, default=0.5)
    parser.add_argument('--save_ckpt_every', type=int,   default=2_000)
    parser.add_argument('--log_every',       type=int,   default=100)
    parser.add_argument('--device_num',      type=str,   default='0')

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

    # YAML をデフォルト値として注入
    if pre.config is not None:
        if not os.path.isfile(pre.config):
            parser.error(f"--config not found: {pre.config!r}")
        cfg = _load_yaml(pre.config)
        parser.set_defaults(**_yaml_to_defaults(cfg))
        print(f"[Config] Loaded: {pre.config}")

    args = parser.parse_args()

    # ── kd_weights 自動解決 ───────────────────────────────────────────────
    resolved = _resolve_kd_weights(args)
    args.kd_weights = resolved
    if resolved is None:
        print("[WARNING] kd_weights を解決できませんでした。"
              "ランダム重みで Post-KD を開始します。")
    elif not os.path.isfile(resolved):
        print(f"[WARNING] kd_weights が見つかりません: {resolved}")
        print("  KD フェーズ（bash scripts/train.sh）を先に完了させてください。")

    # ── その他のバリデーション ────────────────────────────────────────────
    if args.ckpt_save_path is None:
        parser.error("--ckpt_save_path is required (via --config or CLI).")

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device_num)

    if args.wandb_run_name is None:
        args.wandb_run_name = 'post_kd_s' + '_'.join(str(s) for s in args.stages)

    return args


# ---------------------------------------------------------------------------
# DataLoader 構築
# ---------------------------------------------------------------------------

def _get_data_root(name: str, args: Any) -> str:
    root = (getattr(args, 'data_roots', None) or {}).get(name, '')
    if not root:
        root = getattr(args, f'{name}_root', '') or ''
    if not root:
        env_key = f'ANYTHERMAL_{name.upper()}_DATA_ROOT'
        root = os.environ.get(env_key, '')
    if not root:
        raise RuntimeError(
            f"Root path for '{name}' not configured.\n"
            f"  post_kd_config.yaml の data_roots.{name} に設定してください。")
    return root


def _get_splits_dir(name: str, args: Any) -> object:
    """
    splits_dir を loader.py の _resolve_splits_dir() と同じ優先順位で解決する。

    優先順位:
        1. args.splits_roots[name] / args.{name}_splits_dir
        2. third_party/anythermal/custom_datasets/{name}/splits/... を自動検出
        3. None → 各データセットクラスが data_root/splits/ にフォールバック
    """
    # 優先度1: 明示的な設定
    splits_dir = (getattr(args, 'splits_roots', None) or {}).get(name, '')
    if not splits_dir:
        splits_dir = getattr(args, f'{name}_splits_dir', '') or ''
    if splits_dir:
        return splits_dir

    # 優先度2: third_party/anythermal サブモジュールから自動検出
    _SPLITS_SUBPATH = {
        'freiburg':   os.path.join('freiburg',   'splits', 'frame_list'),
        'tartanrgbt': os.path.join('tartanRGBT', 'splits'),
        'vivid':      os.path.join('vivid',      'splits', 'frame_lists'),
        'sthereo':    os.path.join('sthereo',    'splits', 'frame_lists'),
    }
    # train_post_kd.py は プロジェクトルートに置かれているので __file__ から計算
    _THIS     = os.path.abspath(__file__)
    _REPO     = os.path.dirname(_THIS)
    _SUBMOD   = os.path.join(_REPO, 'third_party', 'anythermal')
    _CUST_DS  = os.path.join(_SUBMOD, 'custom_datasets')

    subpath   = _SPLITS_SUBPATH.get(name, '')
    if subpath:
        candidate = os.path.join(_CUST_DS, subpath)
        if os.path.isdir(candidate):
            print(f"[PostKD] {name}: splits_dir auto-detected → {candidate}")
            return candidate

    # 優先度3: None（データセットクラス側でフォールバック）
    return None


def build_stage1_loader(args: Any):
    import torch
    from torch.utils.data import ConcatDataset, DataLoader
    from modules.dataset.thermal.freiburg   import FreiburgDataset
    from modules.dataset.thermal.tartanrgbt import TartanRGBTDataset
    from modules.dataset.thermal.homographic_adapter import ThermalHomographicDataset

    _CLS = {'freiburg': FreiburgDataset, 'tartanrgbt': TartanRGBTDataset}
    datasets = []

    for name in args.s1_dataset:
        name_l = name.lower()
        if name_l not in _CLS:
            raise ValueError(f"[Stage1] Unsupported dataset: {name!r}")

        root       = _get_data_root(name_l, args)
        splits_dir = _get_splits_dir(name_l, args)   # 自動検出を使う

        base_ds = _CLS[name_l](
            data_root=root, splits_dir=splits_dir,
            split='train', augment=False,
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

    combined = (ConcatDataset(datasets) if len(datasets) > 1 else datasets[0])
    g = torch.Generator(); g.manual_seed(args.seed)
    return DataLoader(combined, batch_size=args.batch_size, shuffle=True,
                      num_workers=args.train_num_workers, pin_memory=True,
                      drop_last=True, generator=g)


def build_stage2_loader(args: Any):
    import torch
    from torch.utils.data import ConcatDataset, DataLoader
    from modules.dataset.thermal.sequential import (
        TartanRGBTSequentialDataset, FreiburgSequentialDataset)

    _CLS = {'tartanrgbt': TartanRGBTSequentialDataset,
            'freiburg':   FreiburgSequentialDataset}
    datasets = []

    for name in args.s2_dataset:
        name_l = name.lower()
        if name_l not in _CLS:
            raise ValueError(f"[Stage2] Unsupported dataset: {name!r}")

        root       = _get_data_root(name_l, args)
        splits_dir = _get_splits_dir(name_l, args)   # 自動検出を使う
        kwargs    = dict(data_root=root, splits_dir=splits_dir,
                         stride=args.s2_stride)
        ds = _CLS[name_l](**kwargs)
        datasets.append(ds)
        print(f"[Stage2] {name}: {len(ds)} pairs")

    combined = (ConcatDataset(datasets) if len(datasets) > 1 else datasets[0])
    g = torch.Generator(); g.manual_seed(args.seed)
    return DataLoader(combined, batch_size=args.batch_size, shuffle=True,
                      num_workers=args.train_num_workers, pin_memory=True,
                      drop_last=True, generator=g)


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_arguments()

    print("=" * 60)
    print("  Thermal XFeat Post-KD Training")
    print(f"  Stages       : {args.stages}")
    print(f"  kd_weights   : {args.kd_weights}")
    print(f"  ckpt_save_path: {args.ckpt_save_path}")
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