"""
modules/dataset/thermal/loader.py
build_thermal_loader() — 3 段階優先度付きデータローダー構築。

優先度:
    ①  third_party/anythermal/  (git submodule・最優先)
    ②  ANYTHERMAL_PROJECT_ROOT  (外部クローン・後方互換)
    ③  modules/dataset/thermal/ (スタンドアロン実装・fallback)

データセットルートパスの解決優先度（スタンドアロン時）:
    1. args.data_roots[name]   YAML / dict
    2. args.{name}_root        個別 CLI 引数
    3. 環境変数 ANYTHERMAL_{NAME}_DATA_ROOT

スプリットディレクトリの解決優先度（スタンドアロン時）:
    1. args.splits_roots[name]  YAML / dict
    2. args.{name}_splits_dir   個別 CLI 引数
    3. third_party/anythermal/custom_datasets/{name}/splits[/frame_list]
       (submodule が初期化済みなら自動検出)
    4. None → 各データセットクラスが data_root/splits/ で解決
"""

from __future__ import annotations

import os
import sys
from typing import Any, Optional

import torch
from torch.utils.data import ConcatDataset, DataLoader


# ---------------------------------------------------------------------------
# ルートパス解決
# ---------------------------------------------------------------------------

def _resolve_data_root(name: str, args: Any) -> str:
    """データ画像のルートを解決する（優先度: args.data_roots > args.{name}_root > 環境変数）。"""

    root = (getattr(args, 'data_roots', None) or {}).get(name, '')
    if not root:
        root = getattr(args, f'{name}_root', '') or ''
    if not root:
        env_key = f'ANYTHERMAL_{name.upper()}_DATA_ROOT'
        root = os.environ.get(env_key, '')
        if root:
            print(f"[Loader] {name}: using env var {env_key}")

    if not root:
        raise RuntimeError(
            f"[Loader] Root path for '{name}' is not configured.\n"
            f"  Set one of the following (in priority order):\n"
            f"    1. YAML:  data_roots:\n"
            f"                {name}: /path/to/{name}\n"
            f"    2. CLI:   --{name}_root /path/to/{name}\n"
            f"    3. ENV:   export ANYTHERMAL_{name.upper()}_DATA_ROOT=/path/to/{name}"
        )
    if not os.path.isdir(root):
        raise RuntimeError(
            f"[Loader] '{name}' data_root is not a valid directory: {root!r}")
    return root


# スプリットファイルが anythermal 内でどこにあるか（データセット名 → 相対パス）
#
# 注意: AnyThermal リポジトリのディレクトリ名は大文字小文字が混在している
#   freiburg   → custom_datasets/freiburg/splits/frame_list/  (frame_list *.txt)
#   tartanRGBT → custom_datasets/tartanRGBT/splits/           (sequence.yaml のみ)
#   vivid      → custom_datasets/vivid/splits/frame_lists/    (frame_lists *.txt)
#   sthereo    → custom_datasets/sthereo/splits/frame_lists/  (frame_lists *.txt)
#
# TartanRGBT は Freiburg/VIVID/STheReO と異なり frame_list ディレクトリを持たず、
# sequence.yaml のみでシーケンスを管理する。
_ANYTHERMAL_SPLITS_SUBPATH = {
    'freiburg':   os.path.join('freiburg',    'splits', 'frame_list'),
    'tartanrgbt': os.path.join('tartanRGBT',  'splits'),               # 大文字注意
    'vivid':      os.path.join('vivid',       'splits', 'frame_lists'),
    'sthereo':    os.path.join('sthereo',     'splits', 'frame_lists'),
    'ms2':        '',  # MS2 はハードコードされたシーケンスリストで管理（splits 不要）
}


def _resolve_splits_dir(name: str, args: Any) -> Optional[str]:
    """
    スプリットファイルのディレクトリを解決する。

    優先度:
        1. args.splits_roots[name]  / args.{name}_splits_dir
        2. third_party/anythermal/custom_datasets/{name}/splits/... (自動検出)
        3. None → 各データセットクラスが data_root/splits/ にフォールバック
    """
    # ── 1. 明示的な設定 ─────────────────────────────────────────────────
    splits_dir = (getattr(args, 'splits_roots', None) or {}).get(name, '')
    if not splits_dir:
        splits_dir = getattr(args, f'{name}_splits_dir', '') or ''
    if splits_dir:
        if not os.path.isdir(splits_dir):
            raise RuntimeError(
                f"[Loader] '{name}' splits_dir is not a valid directory: {splits_dir!r}")
        return splits_dir

    # ── 2. anythermal submodule から自動検出 ────────────────────────────
    _THIS = os.path.abspath(__file__)
    _REPO_ROOT  = os.path.normpath(os.path.join(os.path.dirname(_THIS), '..', '..', '..'))
    _SUBMODULE  = os.path.join(_REPO_ROOT, 'third_party', 'anythermal')
    _CUSTOM_DS  = os.path.join(_SUBMODULE, 'custom_datasets')

    subpath = _ANYTHERMAL_SPLITS_SUBPATH.get(name, '')
    if subpath:
        candidate = os.path.join(_CUSTOM_DS, subpath)
        if os.path.isdir(candidate):
            print(f"[Loader] {name}: splits_dir auto-detected from submodule → {candidate}")
            return candidate

    # ── 3. フォールバック: None（各クラスが data_root/splits/ で解決）──
    return None


# ---------------------------------------------------------------------------
# スタンドアロン DataLoader 構築
# ---------------------------------------------------------------------------

def _build_standalone_loader(args: Any) -> DataLoader:
    from modules.dataset.thermal.freiburg   import FreiburgDataset
    from modules.dataset.thermal.tartanrgbt import TartanRGBTDataset
    from modules.dataset.thermal.vivid      import VividDataset
    from modules.dataset.thermal.sthereo    import SthEreoDataset
    from modules.dataset.thermal.ms2        import MS2Dataset

    _DATASET_CLS = {
        'freiburg':   FreiburgDataset,
        'tartanrgbt': TartanRGBTDataset,
        'vivid':      VividDataset,
        'sthereo':    SthEreoDataset,
        'ms2':        MS2Dataset,
    }

    aug_list    = getattr(args, 'aug_list',            None)
    p_diurnal   = getattr(args, 'p_diurnal_inversion', 0.3)
    augment     = getattr(args, 'augment',             True)
    batch_size  = getattr(args, 'batch_size',          8)
    num_workers = getattr(args, 'train_num_workers',   4)
    seed        = getattr(args, 'seed',                42)

    dataset_names: list = getattr(args, 'dataset', ['freiburg'])

    train_datasets = []
    for name in dataset_names:
        name_lower = name.lower()
        if name_lower not in _DATASET_CLS:
            raise ValueError(
                f"[Loader] Unknown dataset: {name!r}. "
                f"Valid: {list(_DATASET_CLS.keys())}")

        data_root  = _resolve_data_root(name_lower, args)
        splits_dir = _resolve_splits_dir(name_lower, args)

        cls = _DATASET_CLS[name_lower]
        ds  = cls(
            data_root=data_root,
            splits_dir=splits_dir,
            split='train',
            augment=augment,
            aug_list=aug_list,
            p_diurnal_inversion=p_diurnal,
        )
        train_datasets.append(ds)
        print(f"[Loader] {name}: {len(ds)} pairs (train)"
              f" | data={data_root}"
              f" | splits={splits_dir or '(data_root/splits)'}")

    if not train_datasets:
        raise RuntimeError("[Loader] No datasets loaded.")

    combined = ConcatDataset(train_datasets) if len(train_datasets) > 1 \
        else train_datasets[0]

    g = torch.Generator()
    g.manual_seed(seed)

    return DataLoader(
        combined,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        generator=g,
    )


# ---------------------------------------------------------------------------
# 公開 API
# ---------------------------------------------------------------------------

def build_thermal_loader(args: Any) -> DataLoader:
    """
    3 段階優先度でデータローダーを構築する。

    優先度 ①: third_party/anythermal/ (submodule が初期化済みの場合)
    優先度 ②: ANYTHERMAL_PROJECT_ROOT 環境変数 (外部クローン後方互換)
    優先度 ③: modules/dataset/thermal/ スタンドアロン実装 (fallback)
    """
    _THIS      = os.path.abspath(__file__)
    _REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(_THIS), '..', '..', '..'))
    _SUBMODULE = os.path.join(_REPO_ROOT, 'third_party', 'anythermal')
    _SUBMODULE_LOADER = os.path.join(_SUBMODULE, 'custom_datasets', 'multi_dataset_loader.py')

    # ── ① submodule ───────────────────────────────────────────────────────
    if os.path.isfile(_SUBMODULE_LOADER):
        sys.path.insert(0, _SUBMODULE)
        try:
            from custom_datasets.multi_dataset_loader import build_dataset  # type: ignore
            print("[Loader] Using third_party/anythermal (submodule)")
            return build_dataset(args, return_dataloader=True)
        except Exception as e:
            print(f"[Loader] submodule import failed: {e}")
        finally:
            if _SUBMODULE in sys.path:
                sys.path.remove(_SUBMODULE)

    # ── ② 外部クローン（環境変数）────────────────────────────────────────
    ext_root = os.environ.get('ANYTHERMAL_PROJECT_ROOT', '')
    if ext_root and os.path.isdir(ext_root):
        _ext_loader = os.path.join(ext_root, 'custom_datasets', 'multi_dataset_loader.py')
        if os.path.isfile(_ext_loader):
            sys.path.insert(0, ext_root)
            try:
                from custom_datasets.multi_dataset_loader import build_dataset  # type: ignore  # noqa: F811
                print(f"[Loader] Using ANYTHERMAL_PROJECT_ROOT: {ext_root}")
                return build_dataset(args, return_dataloader=True)
            except Exception as e:
                print(f"[Loader] external AnyThermal failed: {e}")
            finally:
                if ext_root in sys.path:
                    sys.path.remove(ext_root)

    # ── ③ スタンドアロン ──────────────────────────────────────────────────
    print("[Loader] Falling back to standalone dataset implementation")
    return _build_standalone_loader(args)


# ---------------------------------------------------------------------------
# eval DataLoader 構築
# ---------------------------------------------------------------------------

def _build_standalone_eval_loader(args: Any) -> Optional[DataLoader]:
    """
    eval_dataset に指定したデータセットの val split を使って
    評価用 DataLoader を構築する。

    eval_dataset が空の場合は None を返す。
    各データセットの val split 定義:
        freiburg  : train_seq_01_night*, train_seq_02_day* が val
        tartanrgbt: sequence.yaml の val ラベル群
        vivid     : campus* グループが val（frame_lists 内の構造に依存）
        sthereo   : kaist_* シーケンスが val
    """
    from modules.dataset.thermal.freiburg   import FreiburgDataset
    from modules.dataset.thermal.tartanrgbt import TartanRGBTDataset
    from modules.dataset.thermal.vivid      import VividDataset
    from modules.dataset.thermal.sthereo    import SthEreoDataset
    from modules.dataset.thermal.ms2        import MS2Dataset

    _DATASET_CLS = {
        'freiburg':   FreiburgDataset,
        'tartanrgbt': TartanRGBTDataset,
        'vivid':      VividDataset,
        'sthereo':    SthEreoDataset,
        'ms2':        MS2Dataset,
    }

    batch_size  = getattr(args, 'eval_batch_size',   getattr(args, 'batch_size', 8))
    num_workers = getattr(args, 'eval_num_workers',  getattr(args, 'train_num_workers', 4))
    seed        = getattr(args, 'seed', 42)

    eval_names: list = getattr(args, 'eval_dataset', [])
    if not eval_names:
        return None

    eval_datasets = []
    for name in eval_names:
        name_lower = name.lower()
        if name_lower not in _DATASET_CLS:
            print(f"[Loader] eval: Unknown dataset {name!r}, skipping.")
            continue

        data_root  = _resolve_data_root(name_lower, args)
        splits_dir = _resolve_splits_dir(name_lower, args)

        cls = _DATASET_CLS[name_lower]
        try:
            ds = cls(
                data_root=data_root,
                splits_dir=splits_dir,
                split='val',
                augment=False,          # 評価時は拡張なし
                aug_list=None,
                p_diurnal_inversion=0.0,
            )
            eval_datasets.append(ds)
            print(f"[Loader] {name}: {len(ds)} pairs (val)"
                  f" | data={data_root}"
                  f" | splits={splits_dir or '(data_root/splits)'}")
        except RuntimeError as e:
            print(f"[Loader] eval: {name} skipped — {e}")

    if not eval_datasets:
        return None

    combined = ConcatDataset(eval_datasets) if len(eval_datasets) > 1 \
        else eval_datasets[0]

    g = torch.Generator()
    g.manual_seed(seed)

    return DataLoader(
        combined,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        generator=g,
    )


def build_thermal_eval_loader(args: Any) -> Optional[DataLoader]:
    """
    評価用 DataLoader を構築する。

    build_thermal_loader と同じ 3 段階優先度で構築する。
    eval_dataset が空の場合は None を返す。
    """
    _THIS      = os.path.abspath(__file__)
    _REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(_THIS), '..', '..', '..'))
    _SUBMODULE = os.path.join(_REPO_ROOT, 'third_party', 'anythermal')
    _SUBMODULE_LOADER = os.path.join(_SUBMODULE, 'custom_datasets', 'multi_dataset_loader.py')

    # ① submodule
    if os.path.isfile(_SUBMODULE_LOADER):
        sys.path.insert(0, _SUBMODULE)
        try:
            # eval_dataset を dataset にコピーして val 用 args を作成
            import copy
            eval_args = copy.copy(args)
            eval_args.dataset = getattr(args, 'eval_dataset', [])
            from custom_datasets.multi_dataset_loader import build_dataset  # type: ignore
            return build_dataset(eval_args, return_dataloader=True)
        except Exception as e:
            print(f"[Loader] eval submodule failed: {e}")
        finally:
            if _SUBMODULE in sys.path:
                sys.path.remove(_SUBMODULE)

    # ② 環境変数
    ext_root = os.environ.get('ANYTHERMAL_PROJECT_ROOT', '')
    if ext_root and os.path.isdir(ext_root):
        _ext_loader = os.path.join(ext_root, 'custom_datasets', 'multi_dataset_loader.py')
        if os.path.isfile(_ext_loader):
            sys.path.insert(0, ext_root)
            try:
                import copy
                eval_args = copy.copy(args)
                eval_args.dataset = getattr(args, 'eval_dataset', [])
                from custom_datasets.multi_dataset_loader import build_dataset  # type: ignore  # noqa: F811
                return build_dataset(eval_args, return_dataloader=True)
            except Exception as e:
                print(f"[Loader] eval external AnyThermal failed: {e}")
            finally:
                if ext_root in sys.path:
                    sys.path.remove(ext_root)

    # ③ スタンドアロン
    return _build_standalone_eval_loader(args)