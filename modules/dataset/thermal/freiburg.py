"""
modules/dataset/thermal/freiburg.py
Freiburg Thermal データセット。

ディレクトリ構造（AnyThermal freiburg_dataset.py の seq_path() に準拠）:
    {data_root}/
    ├── train/
    │   └── seq_{seq_num}_{day|night}/   例: seq_00_day, seq_01_night
    │       └── {subseq}/               例: 00, 01, 02
    │           ├── fl_rgb/
    │           │   └── fl_rgb_{timestamp}.png
    │           └── thermal8_clahe/
    │               └── fl_ir_aligned_{timestamp}.png
    └── test/
        └── ...（同構造）

スプリットファイルの場所（splits_dir）:
    AnyThermal 使用時: {anythermal}/custom_datasets/freiburg/splits/frame_list/
    スタンドアロン時:  {data_root}/splits/frame_list/  (splits_dir=None の場合)

スプリットファイルの命名規則:
    train_seq_{seq_num}_{day|night}_{subseq}.txt
    例: train_seq_00_day_00.txt
    → split       = 'train'
    → seq_name    = 'seq_00_day'     （items[1:-1] を '_' で結合）
    → subseq      = '00'             （items[-1]）
    → path        = {data_root}/train/seq_00_day/00/

各行: タイムスタンプ形式のファイル名（拡張子込み）
    例: 1570722156_952177040.png
    → RGB  : fl_rgb/fl_rgb_1570722156_952177040.png
    → Thr  : thermal8_clahe/fl_ir_aligned_1570722156_952177040.png

Split（AnyThermal の return_freiburg_split() に準拠）:
    val_prefixes = ['train_seq_01_night', 'train_seq_02_day']
    それ以外は train
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import cv2
import torch
from torch import Tensor

from modules.dataset.thermal.base import ThermalDatasetBase

_VAL_PREFIXES = ('train_seq_01_night', 'train_seq_02_day')

# AnyThermal crop_box_metadata.txt より
# row_start: 0, row_end: 320, col_start: 148, col_end: 858
_ROW_START = 0
_ROW_END   = 320
_COL_START = 148
_COL_END   = 858


def _seq_path(data_root: str, stem: str) -> str:
    """
    AnyThermal freiburg_dataset.py の seq_path() と同じ処理。

    'train_seq_00_day_00'
    → items = ['train', 'seq', '00', 'day', '00']
    → split    = 'train'
    → seq_name = 'seq_00_day'   （items[1:-1]）
    → subseq   = '00'           （items[-1]）
    → path     = {data_root}/train/seq_00_day/00
    """
    items    = stem.split('_')
    split    = items[0]                        # 'train'
    seq_name = '_'.join(items[1:-1])           # 'seq_00_day'
    subseq   = items[-1]                       # '00'
    return os.path.join(data_root, split, seq_name, subseq)


class FreiburgDataset(ThermalDatasetBase):
    """
    Args:
        data_root:  画像データが置かれたディレクトリ (FREIBURG_ROOT)
        splits_dir: スプリット txt が置かれたディレクトリ。
                    None → data_root/splits/frame_list/ を使う。
        split, augment, aug_list, p_diurnal_inversion: 親クラスと同じ
    """

    def __init__(
        self,
        data_root: str,
        splits_dir: Optional[str] = None,
        split: str = 'train',
        augment: bool = True,
        aug_list: Optional[List[str]] = None,
        p_diurnal_inversion: float = 0.3,
    ):
        self.data_root = data_root
        resolved_splits = splits_dir if splits_dir is not None \
            else os.path.join(data_root, 'splits', 'frame_list')
        super().__init__(
            splits_dir=resolved_splits,
            split=split,
            augment=augment,
            aug_list=aug_list,
            p_diurnal_inversion=p_diurnal_inversion,
        )

    def _is_val_file(self, stem: str) -> bool:
        return any(stem.startswith(p) for p in _VAL_PREFIXES)

    def _build_pairs(self) -> List[Tuple[str, str]]:
        splits_dir = self.splits_dir

        if not os.path.isdir(splits_dir):
            raise RuntimeError(
                f"[Freiburg] splits/frame_list not found: {splits_dir}\n"
                f"  AnyThermal 使用時は splits_dir に\n"
                f"  custom_datasets/freiburg/splits/frame_list を指定してください。")

        pairs: List[Tuple[str, str]] = []
        skipped_dirs: List[str] = []

        for txt_name in sorted(os.listdir(splits_dir)):
            if not txt_name.endswith('.txt'):
                continue
            stem = txt_name[:-4]   # 拡張子を除いたステム名

            # train/val 判定
            is_val = self._is_val_file(stem)
            if (self.split == 'val') != is_val:
                continue

            # AnyThermal seq_path() に準拠したパス解決
            seq_dir  = _seq_path(self.data_root, stem)
            rgb_dir  = os.path.join(seq_dir, 'fl_rgb')
            thr_dir  = os.path.join(seq_dir, 'thermal8_clahe')

            if not os.path.isdir(rgb_dir) or not os.path.isdir(thr_dir):
                skipped_dirs.append(seq_dir)
                continue

            with open(os.path.join(splits_dir, txt_name)) as f:
                for line in f:
                    frame = line.strip()   # 例: '1570722156_952177040.png'
                    if not frame:
                        continue

                    # AnyThermal と同じプレフィックス付与
                    rgb_path = os.path.join(rgb_dir, f'fl_rgb_{frame}')
                    thr_path = os.path.join(thr_dir, f'fl_ir_aligned_{frame}')

                    if os.path.isfile(rgb_path) and os.path.isfile(thr_path):
                        pairs.append((rgb_path, thr_path))

        if skipped_dirs:
            print(f"[Freiburg] WARNING: {len(skipped_dirs)} sequence dir(s) not found, skipped.")
            print(f"  例: {skipped_dirs[0]}")
            print(f"  期待する構造: {{data_root}}/train/seq_00_day/00/fl_rgb/")

        if not pairs:
            raise RuntimeError(
                f"[Freiburg] No pairs found for split='{self.split}' in {splits_dir}\n"
                f"  data_root: {self.data_root}\n"
                f"  確認事項:\n"
                f"    1. {{data_root}}/train/seq_{{seq_num}}_{{day|night}}/{{subseq}}/fl_rgb/ が存在するか\n"
                f"    2. {{data_root}}/train/seq_{{seq_num}}_{{day|night}}/{{subseq}}/thermal8_clahe/ が存在するか\n"
                f"    3. txt の各行が '{{timestamp}}.png' 形式か（例: 1570722156_952177040.png）")
        return pairs

    def _load_rgb(self, path: str) -> Tensor:
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"[Freiburg] RGB not found: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # AnyThermal read_rgb() に準拠: resize → crop
        img = cv2.resize(img, (960, 320), interpolation=cv2.INTER_AREA)
        img = img[_ROW_START:_ROW_END, _COL_START:_COL_END]
        return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

    def _load_thr(self, path: str) -> Tensor:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"[Freiburg] Thermal not found: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # AnyThermal read_thermal() に準拠:
        #   img を半分にリサイズ → 高さを 320 にリサイズ
        h, w = img.shape[:2]
        img = cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        img = cv2.resize(img, (img.shape[1], 320), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0