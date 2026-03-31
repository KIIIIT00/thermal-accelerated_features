"""
modules/dataset/thermal/vivid.py
VIVID++ データセット。

AnyThermal vivid_dataset.py に完全準拠した実装。

ディレクトリ構造（data_root = extracted_data/）:
    {data_root}/
    └── {group}/              例: driving_full, driving_vision
        └── {seq}/            例: campus_day1, city_night
            └── img/
                ├── color/                   ← RGB
                └── thermal_fieldscale_clahe/ ← Thermal

スプリットファイル（splits_dir = frame_lists/）:
    {splits_dir}/
    └── {group}/
        └── {seq}/
            ├── rgb_framelist.txt      ← data_root からの相対パス
            └── thermal_framelist.txt  ← data_root からの相対パス

    中身の例（generate_frame_list.py が生成）:
        driving_full/campus_day1/img/color/1621838918.123473.png
        driving_full/campus_day1/img/thermal_fieldscale_clahe/1621838918.123473.png

AnyThermal との対応:
    datasets_folder  = data_root   (extracted_data/ ディレクトリ)
    root_frame_dir   = splits_dir  (frame_lists/ ディレクトリ)

val split（AnyThermal return_vivid_split 準拠）:
    val_sequences = [s for s in all_sequences if "campus" in s]
    → group/seq のパスに "campus" を含むものが val
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import cv2
import torch
from torch import Tensor

from modules.dataset.thermal.base import ThermalDatasetBase


class VividDataset(ThermalDatasetBase):
    """
    Args:
        data_root:  extracted_data/ ディレクトリ（VIVID_ROOT）
                    例: datasets/vivid/extracted_data
        splits_dir: frame_lists/ ディレクトリ（AnyThermal自動検出時に渡される）
                    None → data_root/splits/frame_lists/ にフォールバック
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
            else os.path.join(data_root, 'splits', 'frame_lists')
        super().__init__(
            splits_dir=resolved_splits,
            split=split,
            augment=augment,
            aug_list=aug_list,
            p_diurnal_inversion=p_diurnal_inversion,
        )

    def _is_val_seq(self, group: str, seq: str) -> bool:
        """
        AnyThermal return_vivid_split() に準拠。
        "campus" を含むシーケンスパス（group/seq）が val。
        例: driving_full/campus_day1 → val
            driving_vision/campus_evening → val
            driving_full/city_night → train
        """
        return 'campus' in f'{group}/{seq}'

    def _build_pairs(self) -> List[Tuple[str, str]]:
        """
        AnyThermal Vivid.generate_image_paths() + read_frame_lists() に準拠。

        フレームリストの中身は data_root からの相対パスであるため、
        画像の絶対パスは os.path.join(data_root, relative_path) で構築する。
        """
        frame_lists_root = self.splits_dir
        pairs: List[Tuple[str, str]] = []

        if not os.path.isdir(frame_lists_root):
            raise RuntimeError(
                f"[VIVID++] frame_lists not found: {frame_lists_root}\n"
                f"  splits_dir={self.splits_dir}\n"
                f"  AnyThermal submodule の custom_datasets/vivid/splits/frame_lists/ を\n"
                f"  splits_dir に指定してください。")

        # group: driving_full, driving_vision など
        for group in sorted(os.listdir(frame_lists_root)):
            group_dir = os.path.join(frame_lists_root, group)
            if not os.path.isdir(group_dir):
                continue

            # seq: campus_day1, city_night など
            for seq in sorted(os.listdir(group_dir)):
                seq_fl_dir = os.path.join(group_dir, seq)
                if not os.path.isdir(seq_fl_dir):
                    continue

                # val/train 判定（AnyThermal 準拠: "campus" を含むものが val）
                is_val = self._is_val_seq(group, seq)
                if (self.split == 'val') != is_val:
                    continue

                rgb_list_path = os.path.join(seq_fl_dir, 'rgb_framelist.txt')
                thr_list_path = os.path.join(seq_fl_dir, 'thermal_framelist.txt')

                if not os.path.isfile(rgb_list_path) or \
                        not os.path.isfile(thr_list_path):
                    continue

                with open(rgb_list_path) as f:
                    rgb_rel_paths = [l.strip() for l in f if l.strip()]
                with open(thr_list_path) as f:
                    thr_rel_paths = [l.strip() for l in f if l.strip()]

                n = min(len(rgb_rel_paths), len(thr_rel_paths))
                for i in range(n):
                    # フレームリストの中身は data_root からの相対パス
                    # AnyThermal: output = [os.path.join(self.datasets_folder, x) for x in output]
                    rp = os.path.join(self.data_root, rgb_rel_paths[i])
                    tp = os.path.join(self.data_root, thr_rel_paths[i])
                    if os.path.isfile(rp) and os.path.isfile(tp):
                        pairs.append((rp, tp))

        if not pairs:
            raise RuntimeError(
                f"[VIVID++] No pairs found for split='{self.split}'\n"
                f"  data_root: {self.data_root}\n"
                f"  splits_dir: {self.splits_dir}\n"
                f"  確認事項:\n"
                f"    1. data_root が extracted_data/ ディレクトリを指しているか\n"
                f"       (driving_full/, driving_vision/ が直下にあるはず)\n"
                f"    2. frame_lists/ 内の rgb_framelist.txt が\n"
                f"       data_root からの相対パスを含んでいるか\n"
                f"       例: driving_full/campus_day1/img/color/xxx.png")
        return pairs

    def _load_rgb(self, path: str) -> Tensor:
        """
        AnyThermal Vivid.read_rgb() に準拠。
        クロップあり: img[crop_top:h-crop_bottom, crop_left:w-crop_right]
        ただし KD では固定サイズ正規化を base.py が担うためクロップは省略。
        """
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"[VIVID++] RGB not found: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

    def _load_thr(self, path: str) -> Tensor:
        """
        AnyThermal Vivid.read_thermal() に準拠。
        thermal_fieldscale_clahe/ の画像は Fieldscale 正規化済み 8bit グレースケール。
        """
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"[VIVID++] Thermal not found: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0