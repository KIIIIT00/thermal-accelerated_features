"""
gf_modules/datasets/thermal_homography.py
Freiburg + TartanRGBT の熱画像から合成ホモグラフィーで学習ペアを生成する
gluefactory 準拠のデータセット。

設計方針:
    XFeat が LightGlue を学習したのと同じ方式:
    - 熱画像コーパスから1枚サンプリング
    - ランダムホモグラフィー H を生成
    - H を適用して view1 を生成
    - 正確な GT 対応点が取得可能（深度マップ不要）

出力フォーマット（gluefactory TwoViewPipeline 準拠）:
    {
        'view0': {'image': Tensor(3,H,W), 'image_size': Tensor(2,)},
        'view1': {'image': Tensor(3,H,W), 'image_size': Tensor(2,)},
        'H_0to1': Tensor(3,3),
    }

学習データ: Freiburg (train) + TartanRGBT (train)
評価データ: SThErEO, VIVID（このデータセットには一切含まれない）
"""

from __future__ import annotations

import os
import sys
import random
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


def _ensure_repo_in_path() -> None:
    _THIS = os.path.dirname(os.path.abspath(__file__))
    candidate = _THIS
    for _ in range(8):
        candidate = os.path.dirname(candidate)
        if os.path.isfile(os.path.join(candidate, 'modules', 'model.py')):
            if candidate not in sys.path:
                sys.path.insert(0, candidate)
            return
    fallback = os.path.dirname(os.path.dirname(_THIS))
    if fallback not in sys.path:
        sys.path.insert(0, fallback)


try:
    from gluefactory.datasets.base_dataset import BaseDataset
    _HAS_GLUEFACTORY = True
except ImportError:
    BaseDataset = object
    _HAS_GLUEFACTORY = False


# ---------------------------------------------------------------------------
# ホモグラフィー生成
# ---------------------------------------------------------------------------

def _random_homography(
    H: int,
    W: int,
    perspective_range: float = 0.10,
    rotation_range:    float = 15.0,
    scale_range:       float = 0.20,
    translation_range: float = 0.10,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    ランダムホモグラフィーを生成する（3×3 ndarray）。

    XFeat・gluefactory が行う合成ワープと同じ方式。
    """
    if rng is None:
        rng = np.random.default_rng()

    cx, cy = W / 2.0, H / 2.0

    # パースペクティブ変換
    pts_src = np.float32([
        [0, 0], [W, 0], [W, H], [0, H]
    ])
    offset = perspective_range * min(H, W)
    pts_dst = pts_src + rng.uniform(-offset, offset, pts_src.shape).astype(np.float32)
    H_persp = cv2.getPerspectiveTransform(pts_src, pts_dst)

    # 回転・スケール
    angle   = rng.uniform(-rotation_range, rotation_range)
    scale   = rng.uniform(1.0 - scale_range, 1.0 + scale_range)
    H_rot   = cv2.getRotationMatrix2D((cx, cy), angle, scale)
    H_rot   = np.vstack([H_rot, [0, 0, 1]])

    # 平行移動
    tx = rng.uniform(-translation_range * W, translation_range * W)
    ty = rng.uniform(-translation_range * H, translation_range * H)
    H_trans = np.eye(3, dtype=np.float64)
    H_trans[0, 2] = tx
    H_trans[1, 2] = ty

    H_total = H_trans @ H_rot @ H_persp
    return H_total.astype(np.float32)


def _warp_image(img: np.ndarray, H: np.ndarray) -> np.ndarray:
    """ホモグラフィーを適用して画像をワープする。"""
    h, w = img.shape[:2]
    return cv2.warpPerspective(img, H, (w, h),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT_101)


# ---------------------------------------------------------------------------
# 熱画像パス収集
# ---------------------------------------------------------------------------

def _collect_thermal_paths(
    data_roots:   dict,
    splits_roots: dict,
    datasets:     List[str],
    split:        str = 'train',
) -> List[str]:
    """
    Freiburg と TartanRGBT の熱画像パスを収集する。

    split='train' の画像のみを使用する（val 分割は除外）。
    """
    _ensure_repo_in_path()
    paths = []

    for name in datasets:
        name_l = name.lower()
        root   = data_roots.get(name_l, '')
        if not root:
            print(f'[ThermalHomography] WARNING: {name_l} root not set, skipping')
            continue

        if name_l == 'freiburg':
            paths += _collect_freiburg(root, splits_roots.get('freiburg'), split)
        elif name_l == 'tartanrgbt':
            paths += _collect_tartanrgbt(root, split)
        else:
            print(f'[ThermalHomography] WARNING: unknown dataset {name_l}')

    print(f'[ThermalHomography] Collected {len(paths)} thermal images '
          f'from {datasets} ({split})')
    return paths


def _collect_freiburg(
    data_root:  str,
    splits_dir: Optional[str],
    split:      str,
) -> List[str]:
    """Freiburg の熱画像パスを splits txt から収集する。"""
    # splits_dir の自動検出
    if not splits_dir:
        _THIS = os.path.dirname(os.path.abspath(__file__))
        _ROOT = os.path.dirname(os.path.dirname(_THIS))
        candidate = os.path.join(
            _ROOT, 'third_party', 'anythermal', 'custom_datasets',
            'freiburg', 'splits', 'frame_list')
        if os.path.isdir(candidate):
            splits_dir = candidate

    if not splits_dir or not os.path.isdir(splits_dir):
        print(f'[ThermalHomography] Freiburg splits_dir not found, '
              f'falling back to directory walk')
        return _walk_thermal_images(data_root)

    paths = []
    for txt in sorted(Path(splits_dir).glob(f'{split}_*.txt')):
        # ファイル名から seq_name を取得
        stem = txt.stem  # e.g., 'train_seq_00_day_00'
        parts = stem.split('_')
        # seq_00_day_00 or seq_00_night_00
        seq_name = '_'.join(parts[1:])  # e.g., 'seq_00_day_00'
        seq_dir  = os.path.join(data_root, 'train', seq_name)

        if not os.path.isdir(seq_dir):
            # サブシーケンスディレクトリを探す
            for subseq in Path(os.path.join(data_root, 'train', seq_name)).glob('*'):
                if subseq.is_dir():
                    seq_dir = str(subseq)
                    break

        thr_dir = os.path.join(seq_dir, 'thermal8_clahe')
        if not os.path.isdir(thr_dir):
            continue

        with open(txt) as f:
            for line in f:
                fname = line.strip()
                if not fname:
                    continue
                # Freiburg のファイル名は timestamp.png
                # thermal8_clahe/fl_ir_aligned_{timestamp}.png
                ts = fname.replace('.png', '')
                img_path = os.path.join(
                    thr_dir, f'fl_ir_aligned_{ts}.png')
                if os.path.isfile(img_path):
                    paths.append(img_path)

    return paths


def _collect_tartanrgbt(data_root: str, split: str) -> List[str]:
    """TartanRGBT の熱画像パスを収集する。"""
    import yaml

    _THIS = os.path.dirname(os.path.abspath(__file__))
    _ROOT = os.path.dirname(os.path.dirname(_THIS))
    splits_dir = os.path.join(
        _ROOT, 'third_party', 'anythermal', 'custom_datasets',
        'tartanRGBT', 'splits')

    if not os.path.isdir(splits_dir):
        print('[ThermalHomography] TartanRGBT splits_dir not found, '
              'falling back to directory walk')
        return _walk_thermal_images(data_root)

    yaml_path = os.path.join(splits_dir, 'sequence.yaml')
    if not os.path.isfile(yaml_path):
        return _walk_thermal_images(data_root)

    with open(yaml_path) as f:
        seq_map = yaml.safe_load(f) or {}
    traj_map = seq_map.get('traj_list', seq_map)

    paths = []
    for key, label in traj_map.items():
        if not isinstance(label, str):
            continue
        day_prefix = key.split('/')[0]
        seq_dir    = os.path.join(data_root, day_prefix, label)
        if not os.path.isdir(seq_dir):
            seq_dir = os.path.join(data_root, key)
        thr_dir = os.path.join(seq_dir, 'thermal_left_rect_8')
        if not os.path.isdir(thr_dir):
            continue
        for f in sorted(Path(thr_dir).glob('*.png')):
            paths.append(str(f))

    return paths


def _walk_thermal_images(root: str) -> List[str]:
    """データセットルートを再帰的に走査して .png を収集する（フォールバック）。"""
    paths = []
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if fname.lower().endswith('.png'):
                paths.append(os.path.join(dirpath, fname))
    return paths


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class _ThermalHomographySplit(Dataset):
    """
    1 split（train または val）のデータセット。

    __len__ は epoch_size で決まる（画像の枚数ではなく学習ステップ数）。
    __getitem__ は毎回ランダムに画像を選んで合成ワープを行う。
    """

    def __init__(self, conf, paths: List[str], split: str):
        super().__init__()
        self.conf    = conf
        self.paths   = paths
        self.split   = split
        self.rng     = np.random.default_rng(
            getattr(conf, 'seed', 42) + (0 if split == 'train' else 1))

        self.epoch_size = getattr(conf, 'train_size' if split == 'train'
                                  else 'val_size', 10000)
        H, W = getattr(conf, 'image_size', [480, 640])
        self.H = H
        self.W = W

    def __len__(self) -> int:
        return self.epoch_size

    def __getitem__(self, idx: int) -> dict:
        # ランダムに画像を選択（seed + idx で再現性あり）
        rng = np.random.default_rng(
            getattr(self.conf, 'seed', 42) + idx)
        path = self.paths[int(rng.integers(0, len(self.paths)))]

        # 熱画像のロード（8bit / 16bit 対応）
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            gray16 = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
            if gray16 is not None:
                gray = cv2.normalize(
                    gray16, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            else:
                # 読み込み失敗時はゼロ画像で代替
                gray = np.zeros((self.H, self.W), dtype=np.uint8)

        # リサイズ
        gray = cv2.resize(gray, (self.W, self.H))

        # 3チャンネル化（XFeat は 3ch 入力）
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)  # (H, W, 3)

        # ホモグラフィー生成
        H_mat = _random_homography(
            H=self.H, W=self.W,
            perspective_range  = getattr(self.conf, 'perspective_range', 0.10),
            rotation_range     = getattr(self.conf, 'rotation_range',    15.0),
            scale_range        = getattr(self.conf, 'scale_range',       0.20),
            translation_range  = getattr(self.conf, 'translation_range', 0.10),
            rng=rng,
        )

        # view1 の生成（ホモグラフィーを適用）
        img_warped = _warp_image(img, H_mat)

        # Tensor に変換（[0, 1] 正規化、(C, H, W)）
        def to_tensor(x: np.ndarray) -> Tensor:
            t = torch.from_numpy(x).permute(2, 0, 1).float() / 255.0
            return t

        img_t0 = to_tensor(img)
        img_t1 = to_tensor(img_warped)
        H_t    = torch.from_numpy(H_mat).float()   # (3, 3)

        return {
            'view0': {
                'image':      img_t0,                                    # (3,H,W)
                'image_size': torch.tensor([self.H, self.W], dtype=torch.long),
            },
            'view1': {
                'image':      img_t1,
                'image_size': torch.tensor([self.H, self.W], dtype=torch.long),
            },
            'H_0to1': H_t,   # GT ホモグラフィー (3,3)
        }


# ---------------------------------------------------------------------------
# gluefactory BaseDataset 準拠
# ---------------------------------------------------------------------------

class ThermalHomographyDataset(BaseDataset if _HAS_GLUEFACTORY else object):
    """
    Freiburg + TartanRGBT の熱画像から合成ホモグラフィーでペアを生成する。

    gluefactory の BaseDataset を継承し TwoViewPipeline と連携する。
    """

    default_conf = {
        # データソース
        'train_datasets': ['freiburg', 'tartanrgbt'],
        'val_datasets':   ['freiburg'],

        # data_roots: OmegaConf struct 問題を回避するために
        # 既知のキーをすべて明示定義する（null = 未設定）
        'data_roots': {
            'freiburg':   None,
            'tartanrgbt': None,
            'vivid':      None,
            'sthereo':    None,
            'ms2':        None,
        },

        # splits_roots: 同様に全キーを明示定義
        'splits_roots': {
            'freiburg':   None,
            'tartanrgbt': None,
            'vivid':      None,
            'sthereo':    None,
            'ms2':        None,
        },

        # 画像サイズ
        'image_size': [480, 640],   # [H, W]

        # ホモグラフィー生成パラメータ
        'perspective_range':  0.10,
        'rotation_range':    15.0,
        'scale_range':        0.20,
        'translation_range':  0.10,

        # エポックサイズ（実際の画像枚数ではなく学習ステップ数）
        'train_size': 100000,
        'val_size':    2000,
        'seed': 42,
    }

    if _HAS_GLUEFACTORY:
        def _init(self, conf) -> None:
            self._paths_cache: dict = {}

        def get_dataset(self, split: str) -> Dataset:
            if split not in self._paths_cache:
                datasets = (self.conf.train_datasets if split == 'train'
                            else self.conf.val_datasets)
                data_roots   = dict(self.conf.data_roots)
                splits_roots = dict(getattr(self.conf, 'splits_roots', {}))
                self._paths_cache[split] = _collect_thermal_paths(
                    data_roots, splits_roots, datasets, split)
            paths = self._paths_cache[split]
            return _ThermalHomographySplit(self.conf, paths, split)

    else:
        # gluefactory なしの standalone 使用
        def __init__(self, conf_dict: dict):
            from types import SimpleNamespace
            conf = SimpleNamespace(**{**self.default_conf, **conf_dict})
            data_roots   = conf.data_roots
            splits_roots = getattr(conf, 'splits_roots', {})
            self._train_paths = _collect_thermal_paths(
                data_roots, splits_roots, conf.train_datasets, 'train')
            self._val_paths = _collect_thermal_paths(
                data_roots, splits_roots, conf.val_datasets, 'val')
            self._conf = conf

        def get_dataset(self, split: str) -> Dataset:
            paths = (self._train_paths if split == 'train'
                     else self._val_paths)
            return _ThermalHomographySplit(self._conf, paths, split)