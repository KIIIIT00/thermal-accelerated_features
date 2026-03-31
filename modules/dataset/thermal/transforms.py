"""
modules/dataset/thermal/transforms.py
RGB-Thermal ペア拡張パイプライン。

主な機能:
  - 幾何変換 (affine / hflip) を RGB-熱ペアに同一パラメータで適用
  - 輝度・コントラスト・ガンマをモダリティ独立で適用
  - color_jitter: RGB のみ
  - CLAHE / blur: 熱画像のみ
  - cutout: 独立
  - diurnal_inversion: 熱画像のみ（昼夜コントラスト反転、最後に適用）
"""

from __future__ import annotations

import random
import math
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# 低レベルヘルパー
# ---------------------------------------------------------------------------

def _to_float(img: Tensor) -> Tensor:
    """uint8 Tensor か float Tensor を [0,1] float に正規化する。"""
    if img.dtype == torch.uint8:
        return img.float() / 255.0
    return img.float()


def _apply_affine(img: Tensor, angle: float, scale: float,
                  tx: float, ty: float) -> Tensor:
    """
    バイリニア補間で Affine 変換を適用する。
    img: (C, H, W) float [0,1]
    """
    C, H, W = img.shape
    # アフィン行列を構築
    cos_a = math.cos(math.radians(angle)) * scale
    sin_a = math.sin(math.radians(angle)) * scale
    # normalize to [-1,1] grid
    # theta shape: (1, 2, 3)
    theta = torch.tensor(
        [[cos_a, -sin_a, tx],
         [sin_a,  cos_a, ty]],
        dtype=torch.float32
    ).unsqueeze(0)
    grid = F.affine_grid(theta, (1, C, H, W), align_corners=False)
    out = F.grid_sample(img.unsqueeze(0), grid, mode='bilinear',
                        padding_mode='reflection', align_corners=False)
    return out.squeeze(0)


def _apply_gamma(img: Tensor, gamma: float) -> Tensor:
    """ガンマ補正: img^(1/gamma)。img は [0,1]。"""
    return img.clamp(1e-8, 1.0).pow(1.0 / gamma)


def _apply_clahe_tensor(thr: Tensor, clip_limit: float = 2.0,
                        tile_grid: Tuple[int, int] = (8, 8)) -> Tensor:
    """
    熱画像 Tensor (C=3, H, W) に CLAHE を適用する。
    内部で numpy / cv2 に変換。
    """
    C, H, W = thr.shape
    # (C,H,W) -> (H,W,C) -> numpy uint8
    arr = (thr.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    gray = arr[:, :, 0]  # 熱画像はグレースケール扱い
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    enhanced = clahe.apply(gray)
    out_arr = np.stack([enhanced, enhanced, enhanced], axis=-1)
    return torch.from_numpy(out_arr).permute(2, 0, 1).float() / 255.0


def _apply_gaussian_blur(img: Tensor, kernel_size: int = 5,
                         sigma: float = 1.5) -> Tensor:
    """Gaussian ブラーを (C, H, W) Tensor に適用。"""
    C, H, W = img.shape
    arr = (img.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(arr, (kernel_size, kernel_size), sigma)
    return torch.from_numpy(blurred).permute(2, 0, 1).float() / 255.0


def _apply_cutout(img: Tensor, n_holes: int = 1,
                  hole_frac: float = 0.15) -> Tensor:
    """ランダムな矩形領域を 0 埋めする。img: (C, H, W)。"""
    C, H, W = img.shape
    out = img.clone()
    for _ in range(n_holes):
        h = int(H * hole_frac)
        w = int(W * hole_frac)
        y = random.randint(0, H - h)
        x = random.randint(0, W - w)
        out[:, y:y + h, x:x + w] = 0.0
    return out


# ---------------------------------------------------------------------------
# 昼夜コントラスト反転（Section 6.2 の実装）
# ---------------------------------------------------------------------------

def apply_diurnal_inversion(thr: Tensor, p: float = 0.3) -> Tensor:
    """
    熱画像のコントラスト反転（昼夜シミュレーション）。
    熱画像のみに適用すること。RGB には絶対に適用しない。

    物理的根拠:
        水・植生・土壌は昼（太陽加熱）と夜（放射冷却）でコントラストが
        2〜3 倍変化し、完全に反転する場合がある。

    Args:
        thr: (C, H, W) または (B, C, H, W) float [0,1]
        p:   適用確率 (default=0.3)
    Returns:
        変換後の Tensor (同 shape)
    """
    if random.random() < p:
        thr = 1.0 - thr
    return thr.clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# メインクラス
# ---------------------------------------------------------------------------

class RGBThermalAugment:
    """
    RGB-熱ペアに適用するデータ拡張パイプライン。

    aug_list で有効な変換を制御する。デフォルト変換:
        affine, hflip, brightness, contrast, gamma, diurnal_inversion

    Args:
        aug_list: 適用する変換名のリスト（順序は内部で固定）
        p_diurnal_inversion: diurnal_inversion の適用確率 (default=0.3)
    """

    VALID_AUGS = [
        'affine', 'hflip', 'brightness', 'contrast', 'gamma',
        'color_jitter', 'clahe', 'blur', 'cutout', 'diurnal_inversion',
    ]

    def __init__(
        self,
        aug_list: List[str] | None = None,
        p_diurnal_inversion: float = 0.3,
    ):
        if aug_list is None:
            aug_list = [
                'affine', 'hflip', 'brightness', 'contrast', 'gamma',
                'diurnal_inversion',
            ]
        for name in aug_list:
            if name not in self.VALID_AUGS:
                raise ValueError(f"Unknown augmentation: {name!r}. "
                                 f"Valid: {self.VALID_AUGS}")
        self._aug_set = set(aug_list)
        self.p_diurnal_inversion = p_diurnal_inversion

    def _enabled(self, name: str) -> bool:
        return name in self._aug_set

    def __call__(
        self, rgb: Tensor, thr: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            rgb: (C, H, W) float [0,1]  教師モダリティ
            thr: (C, H, W) float [0,1]  生徒モダリティ
        Returns:
            (rgb_aug, thr_aug): 同 shape の Tensor
        """
        rgb = _to_float(rgb)
        thr = _to_float(thr)

        # ── 1. affine（ペア共通パラメータ）────────────────────────────────
        if self._enabled('affine'):
            angle = random.uniform(-10.0, 10.0)
            scale = random.uniform(0.85, 1.10)
            tx = random.uniform(-0.05, 0.05)
            ty = random.uniform(-0.05, 0.05)
            rgb = _apply_affine(rgb, angle, scale, tx, ty)
            thr = _apply_affine(thr, angle, scale, tx, ty)

        # ── 2. hflip（ペア共通）───────────────────────────────────────────
        if self._enabled('hflip') and random.random() < 0.5:
            rgb = rgb.flip(-1)
            thr = thr.flip(-1)

        # ── 3. brightness（独立）──────────────────────────────────────────
        if self._enabled('brightness'):
            rgb = (rgb + random.uniform(-0.15, 0.15)).clamp(0, 1)
            thr = (thr + random.uniform(-0.15, 0.15)).clamp(0, 1)

        # ── 4. contrast（独立）────────────────────────────────────────────
        if self._enabled('contrast'):
            rgb_alpha = random.uniform(0.8, 1.2)
            thr_alpha = random.uniform(0.8, 1.2)
            rgb = ((rgb - 0.5) * rgb_alpha + 0.5).clamp(0, 1)
            thr = ((thr - 0.5) * thr_alpha + 0.5).clamp(0, 1)

        # ── 5. gamma（独立）───────────────────────────────────────────────
        if self._enabled('gamma'):
            rgb = _apply_gamma(rgb, random.uniform(0.7, 1.4))
            thr = _apply_gamma(thr, random.uniform(0.7, 1.4))

        # ── 6. color_jitter（RGB のみ）────────────────────────────────────
        if self._enabled('color_jitter') and random.random() < 0.5:
            # 各チャネルにランダムな輝度オフセット
            offsets = torch.tensor(
                [random.uniform(-0.1, 0.1) for _ in range(rgb.shape[0])],
                dtype=rgb.dtype
            ).view(-1, 1, 1)
            rgb = (rgb + offsets).clamp(0, 1)

        # ── 7. CLAHE（熱のみ）────────────────────────────────────────────
        if self._enabled('clahe') and random.random() < 0.5:
            thr = _apply_clahe_tensor(thr)

        # ── 8. blur（熱のみ）─────────────────────────────────────────────
        if self._enabled('blur') and random.random() < 0.3:
            thr = _apply_gaussian_blur(thr)

        # ── 9. cutout（独立）──────────────────────────────────────────────
        if self._enabled('cutout') and random.random() < 0.3:
            rgb = _apply_cutout(rgb)
            thr = _apply_cutout(thr)

        # ── 10. diurnal_inversion（熱のみ・最後）─────────────────────────
        if self._enabled('diurnal_inversion'):
            thr = apply_diurnal_inversion(thr, p=self.p_diurnal_inversion)

        return rgb.clamp(0.0, 1.0), thr.clamp(0.0, 1.0)