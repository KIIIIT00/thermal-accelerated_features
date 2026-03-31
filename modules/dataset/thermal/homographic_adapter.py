"""
modules/dataset/thermal/homographic_adapter.py
Stage 1 Post-KD 用: Thermal Homographic Adaptation データセット。

設計思想:
  - 既存の熱画像ローダー（FreiburgDataset 等）の上にラッパーとして動作
  - ランダムホモグラフィーで (I_thr, I_warped, H_mat) の三つ組を生成
  - RGB 不要・ペア対応不要・ラベルなし → 自己教師あり

出力フォーマット:
  {
    'thr'     : (3, H, W)     熱画像（正規化済み [0,1]）
    'thr_w'   : (3, H, W)     ホモグラフィー変換後の熱画像
    'H_mat'   : (3, 3)        適用したホモグラフィー行列
    'H_inv'   : (3, 3)        逆ホモグラフィー行列
    'mask_w'  : (1, H, W)     有効ピクセルマスク（変換後）
  }
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# ランダムホモグラフィー生成
# ---------------------------------------------------------------------------

def _random_homography(
    H: int, W: int,
    perspective_range: float = 0.05,
    rotation_range: float = 15.0,
    scale_range: float = 0.15,
    translation_range: float = 0.10,
) -> np.ndarray:
    """
    画像サイズ (H, W) に対してランダムなホモグラフィー行列を生成する。

    パラメータは Thermal SLAM の実使用範囲に合わせて保守的に設定する:
      - perspective_range: 遠近変形の強さ（小さめ = 構造物の平面近似に適合）
      - rotation_range:    ±deg（SLAM での短期変化に合わせる）
      - scale_range:       スケール変化幅
      - translation_range: 画像幅・高さに対する比率
    """
    cx, cy = W / 2.0, H / 2.0

    # ランダムパラメータ
    angle = np.radians(random.uniform(-rotation_range, rotation_range))
    scale = random.uniform(1.0 - scale_range, 1.0 + scale_range)
    tx = random.uniform(-translation_range, translation_range) * W
    ty = random.uniform(-translation_range, translation_range) * H

    # アフィン行列（回転・スケール・平行移動）
    cos_a, sin_a = np.cos(angle) * scale, np.sin(angle) * scale
    M = np.array([
        [cos_a, -sin_a, (1 - cos_a) * cx + sin_a * cy + tx],
        [sin_a,  cos_a, (1 - cos_a) * cy - sin_a * cx + ty],
        [0,      0,     1],
    ], dtype=np.float64)

    # 遠近変形（ランダム4隅摂動）
    pts_src = np.float32([
        [0,   0],   [W-1, 0],
        [W-1, H-1], [0,   H-1],
    ])
    noise = np.random.uniform(
        -perspective_range,
        +perspective_range,
        size=(4, 2)
    ) * np.array([[W, H]])
    pts_dst = (pts_src + noise).astype(np.float32)

    H_persp, _ = cv2.findHomography(pts_src, pts_dst)

    # 合成: 遠近 × アフィン
    H_mat = H_persp @ M
    return H_mat


def _apply_homography_tensor(
    img: Tensor, H_mat: np.ndarray
) -> tuple[Tensor, Tensor]:
    """
    (C, H, W) float [0,1] Tensor にホモグラフィーを適用する。

    Returns:
        warped : (C, H, W) 変換後画像
        mask   : (1, H, W) 有効ピクセルマスク（bool → float）
    """
    C, H, W = img.shape
    img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    warped_np = cv2.warpPerspective(
        img_np, H_mat, (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )
    # 有効マスク: 黒塗りされていない画素
    ones = np.ones((H, W), dtype=np.uint8) * 255
    mask_np = cv2.warpPerspective(
        ones, H_mat, (W, H),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )

    warped = torch.from_numpy(warped_np).permute(2, 0, 1).float() / 255.0
    mask   = torch.from_numpy(mask_np > 127).unsqueeze(0).float()
    return warped, mask


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ThermalHomographicDataset(Dataset):
    """
    既存の熱画像データセット（ThermalDatasetBase サブクラス）をラップし、
    Stage 1 Post-KD 訓練用の (thr, thr_warped, H_mat) 三つ組を返す。

    Args:
        base_dataset : ThermalDatasetBase のインスタンス（augment=False 推奨）
        n_per_image  : 1枚の熱画像から生成するホモグラフィーペア数
        perspective_range, rotation_range, scale_range, translation_range:
                       ホモグラフィー生成パラメータ
    """

    def __init__(
        self,
        base_dataset: Dataset,
        n_per_image: int = 1,
        perspective_range: float = 0.05,
        rotation_range: float = 15.0,
        scale_range: float = 0.15,
        translation_range: float = 0.10,
    ):
        self.base = base_dataset
        self.n_per_image = n_per_image
        self.persp  = perspective_range
        self.rot    = rotation_range
        self.scale  = scale_range
        self.trans  = translation_range

    def __len__(self) -> int:
        return len(self.base) * self.n_per_image

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        base_idx = idx // self.n_per_image
        item = self.base[base_idx]

        # base_dataset は {'item': [{'rgb': ..., 'thr': ...}], ...} を返す
        thr: Tensor = item['item'][0]['thr']   # (3, H, W) float [0,1]

        C, H, W = thr.shape
        H_mat   = _random_homography(H, W, self.persp, self.rot,
                                     self.scale, self.trans)
        thr_w, mask = _apply_homography_tensor(thr, H_mat)

        # torch Tensor に変換
        H_t   = torch.from_numpy(H_mat).float()
        H_inv = torch.from_numpy(np.linalg.inv(H_mat)).float()

        return {
            'thr'  : thr,
            'thr_w': thr_w,
            'H_mat': H_t,
            'H_inv': H_inv,
            'mask_w': mask,
        }