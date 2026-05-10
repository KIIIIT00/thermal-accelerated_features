"""
modules/dataset/thermal/ms2.py
MS2 (Multi-Spectral Stereo) データセット。

論文: "MS2: Multi-Robot Multi-Spectral Stereo Dataset" (ICRA 2023)
URL: https://sites.google.com/view/multi-spectral-stereo-dataset

ディレクトリ構造:
    {data_root}/
    ├── sync_data/
    │   └── {seq}/               例: _2021-08-06-10-59-33
    │       ├── rgb/
    │       │   └── img_left/    ← RGB 画像
    │       │       └── *.png
    │       └── thr/
    │           └── img_left/    ← Thermal 画像（hist_99 + bilateral 正規化済み）
    │               └── *.png
    └── proj_depth/              （使用しない）

AnyThermal の ms2_dataset.py に準拠した設定:
    subsample: [::10]  （10フレームに1枚サンプリング）
    train split: campus, Road1-4, residential（一部）
    val split:   residential（一部）

NOTE:
    MS2 の熱画像は AnyThermal の process_one_image(type="hist_99") で
    前処理済みであることを前提とする。ダウンロードのみで使用可能。

    他データセットと異なり RGB-T が異なるカメラで撮影されているが、
    ステレオキャリブレーション済みのため空間的に対応している。
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import cv2
import torch
from torch import Tensor
from natsort import natsorted  # type: ignore

from modules.dataset.thermal.base import ThermalDatasetBase

# AnyThermal の return_ms2_split() に完全準拠
_TRAIN_SEQS = [
    '_2021-08-06-10-59-33',
    '_2021-08-06-17-44-55',
    '_2021-08-13-17-06-04',
    '_2021-08-13-21-18-04',   # campus
    '_2021-08-13-16-50-57',   # Road2
    '_2021-08-06-16-59-13',
    '_2021-08-13-16-31-10',
    '_2021-08-13-22-16-02',   # Road1
    '_2021-08-13-16-08-46',
    '_2021-08-13-21-58-13',   # Road3
    '_2021-08-13-22-36-41',   # Road4
]

_VAL_SEQS = [
    '_2021-08-06-11-23-45',
    '_2021-08-06-16-45-28',
    '_2021-08-13-16-14-48',
    '_2021-08-13-22-03-03',   # residential
]

_SUBSAMPLE = 10   # AnyThermal と同じ間引き率


class MS2Dataset(ThermalDatasetBase):
    """
    Args:
        data_root:  MS2 データセットのルートディレクトリ
                    (sync_data/ が置かれた親ディレクトリ)
        splits_dir: 使用しない（MS2 はハードコードされたシーケンスリストで管理）。
                    互換性のために引数は受け付けるが無視する。
        split:      'train' | 'val'
        subsample:  フレームの間引き率（デフォルト 10 = AnyThermal 準拠）
    """

    def __init__(
        self,
        data_root: str,
        splits_dir: Optional[str] = None,   # MS2 では使用しない
        split: str = 'train',
        augment: bool = True,
        aug_list: Optional[List[str]] = None,
        p_diurnal_inversion: float = 0.3,
        subsample: int = _SUBSAMPLE,
    ):
        self.data_root = data_root
        self.subsample = subsample
        self._pairs = []
        # MS2 は splits_dir を使わないが基底クラスに渡す
        super().__init__(
            splits_dir=splits_dir or data_root,
            split=split,
            augment=augment,
            aug_list=aug_list,
            p_diurnal_inversion=p_diurnal_inversion,
        )

    def _build_pairs(self) -> List[Tuple[str, str]]:
        """
        AnyThermal MS2Dataset.generate_image_paths() に準拠。

        sync_data/{seq}/rgb/img_left/*.png  と
        sync_data/{seq}/thr/img_left/*.png  をサブサンプリングしてペアを構築する。
        """
        seq_list = _TRAIN_SEQS if self.split == 'train' else _VAL_SEQS
        sync_root = os.path.join(self.data_root, 'sync_data')

        pairs: List[Tuple[str, str]] = []

        for seq in seq_list:
            rgb_dir = os.path.join(sync_root, seq, 'rgb', 'img_left')
            thr_dir = os.path.join(sync_root, seq, 'thr', 'img_left')

            if not os.path.isdir(rgb_dir) or not os.path.isdir(thr_dir):
                continue

            rgb_files = natsorted([
                f for f in os.listdir(rgb_dir)
                if f.lower().endswith(('.png', '.jpg'))
            ])[::self.subsample]

            thr_files = natsorted([
                f for f in os.listdir(thr_dir)
                if f.lower().endswith(('.png', '.jpg'))
            ])[::self.subsample]

            # ファイル数が一致しない場合は短い方に合わせる
            n = min(len(rgb_files), len(thr_files))
            for i in range(n):
                rp = os.path.join(rgb_dir, rgb_files[i])
                tp = os.path.join(thr_dir, thr_files[i])
                if os.path.isfile(rp) and os.path.isfile(tp):
                    pairs.append((rp, tp))

        if not pairs:
            raise RuntimeError(
                f"[MS2] No pairs found for split='{self.split}'\n"
                f"  data_root: {self.data_root}\n"
                f"  確認事項:\n"
                f"    sync_data/{{seq}}/rgb/img_left/ と\n"
                f"    sync_data/{{seq}}/thr/img_left/ が存在するか")
        return pairs

    def _load_rgb(self, path: str) -> Tensor:
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"[MS2] RGB not found: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

    # def _load_thr(self, path: str) -> Tensor:
    #     """
    #     AnyThermal MS2Dataset.read_thermal() に準拠。

    #     thr/img_left/ の画像は hist_99 + bilateral でスケーリング済みの
    #     8bit グレースケール画像。クロップは行わない（AnyThermal 準拠）。
    #     """
    #     img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    #     if img is None:
    #         raise FileNotFoundError(f"[MS2] Thermal not found: {path}")
    #     img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

    def _load_thr(self, path: str) -> dict:
        """
        16-bit Raw の読み込みと、8-bit 前処理画像の生成を同時に行う
        """
        # 1. 16-bit Raw 読み込み (MS2 は 16-bit)
        img_raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img_raw is None:
            raise FileNotFoundError(f"[MS2] Thermal not found: {path}")
        
        # --- 損失関数用の Raw テンソル ---
        # 物理的な温度差を維持するため、float 化のみ行う
        thr_raw = torch.from_numpy(img_raw.astype(np.float32)).unsqueeze(0)

        # --- ネットワーク入力用の 8-bit 前処理 ---
        # 1. hist_99 スケーリング
        v_min, v_max = np.percentile(img_raw, [1.0, 99.0])
        img_8bit = np.clip((img_raw - v_min) / (v_max - v_min + 1e-6) * 255, 0, 255).astype(np.uint8)

        # 2. CLAHE (コントラスト強調)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_8bit = clahe.apply(img_8bit)

        # 3. Bilateral Filter (エッジ保存平滑化)
        img_8bit = cv2.bilateralFilter(img_8bit, 5, 20, 15)

        # 4. Crop (静的ノイズ領域の除去 - sequential.py の設定に準拠)
        h, w = img_8bit.shape[:2]
        # 上:9, 下:35, 左:28, 右:34 をカット
        img_8bit = img_8bit[9:h-35, 28:w-34]
        # Raw 側も同じ範囲でクロップして座標を同期させる
        thr_raw = thr_raw[:, 9:h-35, 28:w-34]

        # テンソル化と正規化 (XFeat 入力用)
        thr_8bit = torch.from_numpy(img_8bit).unsqueeze(0).float() / 255.0

        return {
            'thr_8bit': thr_8bit, # ネットワーク入力
            'thr_raw':  thr_raw   # 物理勾配損失計算用
        }
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, index: int) -> dict:
        rgb_path, thr_path = self.pairs[index]
        
        # RGB はこれまで通り (教師モデル用)
        img_rgb = self._load_rgb(rgb_path) 
        # Thermal はデュアル・テンソル
        thr_dict = self._load_thr(thr_path)
        
        return {
            'rgb': img_rgb,
            'thr_t_8bit': thr_dict['thr_8bit'],
            'thr_t_raw':  thr_dict['thr_raw'],
            'dataset_name': 'ms2',
            'bit_depth': 16
        }