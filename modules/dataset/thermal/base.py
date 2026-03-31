"""
modules/dataset/thermal/base.py
熱画像データセットの抽象基底クラス。

splits_dir（スプリットファイル置き場）とデータroot（画像置き場）を分離している。
AnyThermalではスプリットファイルがリポジトリ側 (custom_datasets/{name}/splits/) に、
画像データが別ディレクトリに置かれているため、両者を独立して指定できる。
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset

from modules.dataset.thermal.transforms import RGBThermalAugment

# ---------------------------------------------------------------------------
# 全データセット共通の出力解像度
# ---------------------------------------------------------------------------
# DataLoader で複数データセットを ConcatDataset にまとめるためには、
# 全テンソルのサイズが揃っている必要がある。
# XFeat の内部処理は任意の解像度を受け付けるが、collate のためにここで揃える。
#
# 選定根拠:
#   - Freiburg thermal: 元サイズ約 640×512 → crop後 約 710×320
#   - TartanRGBT thermal: 640×512 (thermal_left_rect_8)
#   - 640×480 は両者に近い標準的なサイズ
_OUTPUT_H = 480
_OUTPUT_W = 640


def _resize_to_common(img: Tensor) -> Tensor:
    """
    (3, H, W) テンソルを (_OUTPUT_H, _OUTPUT_W) にリサイズして返す。
    すでに目標サイズであればそのまま返す。
    """
    if img.shape[-2] == _OUTPUT_H and img.shape[-1] == _OUTPUT_W:
        return img
    return F.interpolate(
        img.unsqueeze(0),
        size=(_OUTPUT_H, _OUTPUT_W),
        mode='bilinear',
        align_corners=False,
    ).squeeze(0)


class ThermalDatasetBase(Dataset, ABC):
    """
    RGB-熱ペアデータセットの共通インターフェイス。

    Args:
        splits_dir: スプリットファイル (*.txt / *.yaml) が置かれたディレクトリ。
                    AnyThermal 使用時は custom_datasets/{name}/splits/ を渡す。
                    None の場合は data_root/splits/ を使う（後方互換）。
        split:      'train' | 'val'
        augment:    データ拡張の有無
        aug_list:   有効にする拡張名のリスト
        p_diurnal_inversion: 昼夜反転拡張の適用確率

    サブクラスは以下を実装する:
        _build_pairs()  → List[Tuple[str, str]]  (rgb_path, thr_path) のリスト
        _load_rgb(path) → Tensor (3, H, W) float [0,1]
        _load_thr(path) → Tensor (3, H, W) float [0,1]

    NOTE:
        __getitem__ は各サブクラスの _load_rgb / _load_thr が返す任意サイズの
        テンソルを共通解像度 (_OUTPUT_H × _OUTPUT_W) にリサイズして返す。
        これにより ConcatDataset でも collate が正しく動作する。
    """

    def __init__(
        self,
        splits_dir: Optional[str] = None,
        split: str = 'train',
        augment: bool = True,
        aug_list: Optional[List[str]] = None,
        p_diurnal_inversion: float = 0.3,
    ):
        self.splits_dir = splits_dir
        self.split = split
        self.augment = augment
        self.transform = RGBThermalAugment(
            aug_list=aug_list,
            p_diurnal_inversion=p_diurnal_inversion,
        ) if augment else None
        self._pairs: List[Tuple[str, str]] = self._build_pairs()

    @abstractmethod
    def _build_pairs(self) -> List[Tuple[str, str]]:
        """(rgb_path, thr_path) のリストを返す。"""
        ...

    @abstractmethod
    def _load_rgb(self, path: str) -> Tensor:
        """RGB 画像を (3, H, W) float [0,1] で返す。"""
        ...

    @abstractmethod
    def _load_thr(self, path: str) -> Tensor:
        """熱画像を (3, H, W) float [0,1] で返す。"""
        ...

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        rgb_path, thr_path = self._pairs[idx]
        rgb = self._load_rgb(rgb_path)
        thr = self._load_thr(thr_path)

        # 全データセット共通解像度にリサイズ（collate のため必須）
        rgb = _resize_to_common(rgb)
        thr = _resize_to_common(thr)

        # augment 前の熱画像を保存（diurnal_inversion 適用率計測用）
        thr_orig = thr.clone()

        if self.transform is not None:
            rgb, thr = self.transform(rgb, thr)

        return {
            'item': [{'rgb': rgb, 'thr': thr}],
            'rgb_path': rgb_path,
            'thr_path': thr_path,
            'thr_raw': thr_orig,   # augment 前の熱画像（diurnal 適用率計測用）
        }