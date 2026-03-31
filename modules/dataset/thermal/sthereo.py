"""
modules/dataset/thermal/sthereo.py
STheReO データセット。

AnyThermal sthereo_dataset.py + generate_frame_list.py に完全準拠した実装。

実際のディレクトリ構造（確認済み）:
    {data_root}/
    ├── SNU/
    │   ├── Morning/
    │   ├── Afternoon/   例: datasets/sthereo/SNU/Afternoon/
    │   └── Evening/
    │       └── image/
    │           ├── stereo_left/           ← RGB
    │           └── thermal8_left_clahe/   ← Thermal (14bit→8bit+CLAHE済み)
    ├── KAIST/
    │   ├── Morning/
    │   ├── Afternoon/
    │   └── Evening/
    └── Valley/
        ├── Morning/
        ├── Afternoon/
        └── Evening/

スプリットファイル（generate_frame_list.py が生成）:
    {splits_dir}/
    └── {SeqName}_frame_pairs.txt
        例: SNU_Afternoon_frame_pairs.txt
            KAIST_Morning_frame_pairs.txt

    ファイル名の生成規則（generate_frame_list.py）:
        seq.relative_to(root_dir).replace("/", "_")
        root/SNU/Afternoon → SNU/Afternoon → SNU_Afternoon

    中身の各行（generate_frame_list.py 準拠）:
        {rgb_filename} {thr_filename} {x:.6f} {y:.6f} {z:.6f}
        例: 1630733313282105385.png 1630733313282105385.png 0.123 4.567 8.910

    ※ rgb_filename / thr_filename はファイル名のみ（パスなし）

AnyThermal との対応:
    datasets_folder  = data_root   (SNU/, KAIST/, Valley/ が直下にある親ディレクトリ)
    root_frame_dir   = splits_dir  (frame_lists/ が置かれたディレクトリ)

    AnyThermal の seq 名（小文字）: 'snu_afternoon', 'kaist_morning'
    実際のディレクトリ（CamelCase）: SNU/Afternoon, KAIST/Morning
    → この変換を _seq_name_to_dir() で行う

val split（AnyThermal return_sthereo_split 準拠）:
    train: snu_*, valley_* → SNU/*, Valley/*
    val:   kaist_*         → KAIST/*

クロップ（AnyThermal sthereo_dataset.py 準拠）:
    crop_top=121, crop_bottom=107, crop_left=52, crop_right=30
    → img[121:h-107, 52:w-30] = 有効領域のみ
"""

from __future__ import annotations
import os
from typing import List, Optional, Tuple
import cv2
import torch
from torch import Tensor
from modules.dataset.thermal.base import ThermalDatasetBase

# 判定用の接頭辞（小文字で統一）
_TRAIN_SEQ_PREFIXES = ('snu_', 'valley_')
_VAL_SEQ_PREFIXES   = ('kaist_',)

# クロップパラメータ
_CROP_TOP    = 121
_CROP_BOTTOM = 107
_CROP_LEFT   = 52
_CROP_RIGHT  = 30

def _seq_name_to_dir(seq_name: str) -> str:
    """
    ディレクトリ構造がフラットなため、シーケンス名をそのままディレクトリ名として返す。
    snu_morning -> snu_morning
    """
    return seq_name

class SthEreoDataset(ThermalDatasetBase):
    def __init__(
        self,
        data_root: str,
        splits_dir: Optional[str] = None,
        split: str = 'train',
        augment: bool = True,
        aug_list: Optional[List[str]] = None,
        p_diurnal_inversion: float = 0.3,
        # 赤外線フォルダ名を変更可能にするための引数（必要に応じて）
        thermal_dir_name: str = 'thermal8_left_clahe' 
    ):
        self.data_root = data_root
        self.thermal_dir_name = thermal_dir_name
        resolved_splits = splits_dir if splits_dir is not None \
            else os.path.join(data_root, 'splits', 'frame_lists')
        
        super().__init__(
            splits_dir=resolved_splits,
            split=split,
            augment=augment,
            aug_list=aug_list,
            p_diurnal_inversion=p_diurnal_inversion,
        )

    def _is_split_seq(self, seq_name: str) -> bool:
        # ディレクトリ名（小文字）に基づいて学習/検証を振り分け
        seq_name_lower = seq_name.lower()
        if self.split == 'val':
            return any(seq_name_lower.startswith(p) for p in _VAL_SEQ_PREFIXES)
        return any(seq_name_lower.startswith(p) for p in _TRAIN_SEQ_PREFIXES)

    def _build_pairs(self) -> List[Tuple[str, str]]:
        frame_lists_dir = self.splits_dir
        if not os.path.isdir(frame_lists_dir):
            raise RuntimeError(f"[STHEREO] frame_lists not found: {frame_lists_dir}")

        pairs: List[Tuple[str, str]] = []
        
        for fname in sorted(os.listdir(frame_lists_dir)):
            if not fname.endswith('_frame_pairs.txt'):
                continue

            # ファイル名からシーケンス名を抽出
            seq_name = fname[:-len('_frame_pairs.txt')]
            if not self._is_split_seq(seq_name):
                continue

            seq_dir = _seq_name_to_dir(seq_name)
            rgb_img_dir = os.path.join(self.data_root, seq_dir, 'image', 'stereo_left')
            # 修正ポイント：フラット構造内の赤外線フォルダを指定
            thr_img_dir = os.path.join(self.data_root, seq_dir, 'image', self.thermal_dir_name)

            if not os.path.isdir(rgb_img_dir):
                print(f"⚠️ [STHEREO] RGB dir not found: {rgb_img_dir}")
                continue
            if not os.path.isdir(thr_img_dir):
                print(f"⚠️ [STHEREO] Thermal dir not found: {thr_img_dir}")
                continue

            txt_path = os.path.join(frame_lists_dir, fname)
            with open(txt_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 2: continue
                    
                    rgb_fn, thr_fn = parts[0], parts[1]
                    rp = os.path.join(rgb_img_dir, rgb_fn)
                    tp = os.path.join(thr_img_dir, thr_fn)
                    
                    if os.path.isfile(rp) and os.path.isfile(tp):
                        pairs.append((rp, tp))

        return pairs

    def _load_rgb(self, path: str) -> Tensor:
        img = cv2.imread(path)
        if img is None: raise FileNotFoundError(f"RGB not found: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # クロップが必要な場合はここで適用（Thermal側と合わせる）
        h, w = img.shape[:2]
        img = img[_CROP_TOP : h - _CROP_BOTTOM, _CROP_LEFT : w - _CROP_RIGHT]
        return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

    def _load_thr(self, path: str) -> Tensor:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None: raise FileNotFoundError(f"Thermal not found: {path}")
        h, w = img.shape
        img = img[_CROP_TOP : h - _CROP_BOTTOM, _CROP_LEFT : w - _CROP_RIGHT]
        # 3チャンネル化
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0