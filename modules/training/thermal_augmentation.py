"""
modules/training/thermal_augmentation.py
熱画像専用のデータ拡張モジュール。

【設計根拠】
各データセットの画像特性の差異を拡張でカバーする：

  SThErEO（CLAHE, 640×512）:
    - FLIR ADK 固有の FPN 列ノイズ
    - 周辺減光（vignetting）
    - エッジが明確（CLAHE）

  VIVID（CLAHE, 640×480）:
    - 車載高速移動によるモーションブラー
    - 夜間の高ノイズ
    - インデックスマッチング誤差による GT 不正確さ

  MS2（hist_99+bilateral→CLAHE追加, 640×512）:
    - 雨滴による低温輝点（熱画像では雨滴が透明に見える）
    - hist_99 のコントラスト低下特性
    - CLAHE clipLimit の多様性（apply_clahe でランダム化済み）

【KD 学習における拡張の適用方針】
  学習側（student input）のみに適用し、
  教師モデルへの入力（同じ clean 画像）には適用しない。
  例外: FPN ノイズ不変性損失のみ意図的に非対称（losses_kd.py）。
"""

from __future__ import annotations

import random
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# 基本拡張（全データセット共通）
# ---------------------------------------------------------------------------

def random_horizontal_flip(img: Tensor, p: float = 0.5) -> Tensor:
    """ランダム水平反転。"""
    if random.random() < p:
        return torch.flip(img, dims=[-1])
    return img


def brightness_jitter(
    img: Tensor,
    delta_range: Tuple[float, float] = (-0.15, 0.15),
) -> Tensor:
    """
    輝度シフト（昼夜の温度差シミュレーション）。

    熱画像では夜間に相対的なコントラストが下がる（放射差が縮小）。
    輝度シフトで朝・昼・夜の見え方の差を吸収する。
    """
    delta = random.uniform(*delta_range)
    return (img + delta).clamp(0.0, 1.0)


def contrast_jitter(
    img: Tensor,
    scale_range: Tuple[float, float] = (0.8, 1.25),
) -> Tensor:
    """
    コントラスト変動。

    hist_99 （MS2）と CLAHE（SThErEO/VIVID）のヒストグラム分布差を吸収する。
    hist_99 は高温点に引っ張られてコントラストが低くなりがちなため、
    scale < 1.0 の拡張で MS2 の特性を近似できる。
    """
    scale = random.uniform(*scale_range)
    mean  = img.mean()
    return ((img - mean) * scale + mean).clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# SThErEO 向け拡張
# ---------------------------------------------------------------------------

def fpn_column_noise(
    img: Tensor,
    sigma_dn_range: Tuple[float, float] = (2.0, 8.0),
) -> Tensor:
    """
    FLIR LWIR センサー固有の固定パターンノイズ（FPN）シミュレーション。

    物理的根拠:
        LWIR センサーの列並列 ADC に起因する列方向固定パターンノイズ。
        σ = 2-8 DN（Digital Number）が実測値（256階調換算）。
        CLAHE でも除去されず、実際のデータに常に存在する。

    実装: 列単位で一様なノイズを加算（行方向に共通）。
    """
    B, C, H, W = (1, *img.shape) if img.dim() == 3 else img.shape
    sigma = random.uniform(*sigma_dn_range) / 255.0
    col_noise = torch.randn(1, 1, 1, W, device=img.device) * sigma
    if img.dim() == 3:
        return (img + col_noise.squeeze(0)).clamp(0.0, 1.0)
    return (img + col_noise).clamp(0.0, 1.0)


def vignetting(
    img: Tensor,
    strength_range: Tuple[float, float] = (0.05, 0.25),
) -> Tensor:
    """
    周辺減光（vignetting）シミュレーション。

    FLIR ADK などの熱画像カメラは光学系に起因する周辺減光を示す。
    ガウス分布マスクで中心部を明るく、周辺を暗くする。
    """
    _, H, W = img.shape
    strength = random.uniform(*strength_range)

    cy, cx = H / 2.0, W / 2.0
    ys = torch.arange(H, device=img.device, dtype=torch.float32)
    xs = torch.arange(W, device=img.device, dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')

    # 正規化した距離（対角線の半分を 1.0 とする）
    r = torch.sqrt(((yy - cy) / cy) ** 2 + ((xx - cx) / cx) ** 2)
    mask = 1.0 - strength * r.clamp(0.0, 1.0)   # (H, W)
    return (img * mask.unsqueeze(0)).clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# VIVID 向け拡張
# ---------------------------------------------------------------------------

def motion_blur(
    img: Tensor,
    max_kernel: int = 7,
    direction: str = 'horizontal',
) -> Tensor:
    """
    モーションブラー（車載高速移動シミュレーション）。

    VIVID は車載カメラ（キャンパス・市街地）のため、
    高速走行時に水平方向のブラーが生じる。
    """
    k = random.choice(range(3, max_kernel + 1, 2))   # 奇数カーネル
    kernel = np.zeros((k, k), dtype=np.float32)
    if direction == 'horizontal':
        kernel[k // 2, :] = 1.0 / k
    else:
        kernel[:, k // 2] = 1.0 / k

    # テンソル → numpy → フィルタ → テンソル
    arr = np.ascontiguousarray(
        (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
    arr = cv2.filter2D(arr, -1, kernel)
    t   = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    return t.to(img.device)


def gaussian_noise(
    img: Tensor,
    sigma_range: Tuple[float, float] = (0.01, 0.05),
) -> Tensor:
    """
    ガウスノイズ（夜間の高ノイズシミュレーション）。

    夜間・悪天候時には熱画像のノイズが増加する。
    """
    sigma = random.uniform(*sigma_range)
    noise = torch.randn_like(img) * sigma
    return (img + noise).clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# MS2 向け拡張（雨天シミュレーション）
# ---------------------------------------------------------------------------

def rain_drops(
    img: Tensor,
    n_drops_range: Tuple[int, int] = (20, 80),
    drop_intensity_range: Tuple[float, float] = (0.05, 0.20),
) -> Tensor:
    """
    熱画像における擬似雨滴シミュレーション。

    物理的根拠:
        LWIR 熱画像では雨滴は透明（通過）だが、
        雨滴の表面温度が低く、低輝度の輝点として現れる。
        MS2 の雨天シーケンスで観測された現象。

    実装: ランダムな位置に小さな暗い円を描画。
    """
    _, H, W = img.shape
    # permute 後は non-contiguous になるため np.ascontiguousarray で変換
    arr = np.ascontiguousarray(
        (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))

    n_drops = random.randint(*n_drops_range)
    for _ in range(n_drops):
        x = random.randint(0, W - 1)
        y = random.randint(0, H - 1)
        r = random.randint(1, 3)
        intensity = random.uniform(*drop_intensity_range)
        # 熱画像で雨滴は周囲より低温（暗い）
        color = int(arr[y, x, 0] * (1.0 - intensity))
        cv2.circle(arr, (x, y), r, (color, color, color), -1)

    t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    return t.to(img.device)


def random_clahe_strength(
    img: Tensor,
    clip_range: Tuple[float, float] = (1.0, 4.0),
) -> Tensor:
    """
    CLAHE 強度のランダム化。

    設計根拠:
        SThErEO の CLAHE clipLimit ≈ 2.0（標準）
        MS2 の hist_99 は CLAHE clipLimit ≈ 1.0 に近い（低コントラスト）
        この範囲 [1.0, 4.0] をランダムに適用することで、
        両データセットの前処理差異を吸収できる。
    """
    clip = random.uniform(*clip_range)
    arr  = (img[0].cpu().numpy() * 255).astype(np.uint8)   # グレースケール1ch
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    arr   = clahe.apply(arr)
    # 3ch に戻す
    arr3  = np.stack([arr, arr, arr], axis=0)
    return torch.from_numpy(arr3).float().to(img.device) / 255.0


# ---------------------------------------------------------------------------
# 統合拡張パイプライン
# ---------------------------------------------------------------------------

class ThermalAugmentation:
    """
    KD 学習用の熱画像統合データ拡張クラス。

    各データセットの特性に対応した拡張を組み合わせる。
    すべての拡張は確率的に適用される。

    Args:
        p_flip:         水平反転の確率
        p_brightness:   輝度シフトの確率
        p_contrast:     コントラスト変動の確率
        p_fpn:          FPN 列ノイズの確率
        p_vignetting:   周辺減光の確率
        p_motion_blur:  モーションブラーの確率
        p_gaussian:     ガウスノイズの確率
        p_rain:         擬似雨滴の確率
        p_clahe_rand:   CLAHE 強度ランダム化の確率
    """

    def __init__(
        self,
        p_flip:        float = 0.5,
        p_brightness:  float = 0.5,
        p_contrast:    float = 0.5,
        p_fpn:         float = 0.4,
        p_vignetting:  float = 0.3,
        p_motion_blur: float = 0.2,
        p_gaussian:    float = 0.3,
        p_rain:        float = 0.15,
        p_clahe_rand:  float = 0.4,
    ):
        self.probs = {
            'flip':        p_flip,
            'brightness':  p_brightness,
            'contrast':    p_contrast,
            'fpn':         p_fpn,
            'vignetting':  p_vignetting,
            'motion_blur': p_motion_blur,
            'gaussian':    p_gaussian,
            'rain':        p_rain,
            'clahe_rand':  p_clahe_rand,
        }

    def __call__(self, img: Tensor) -> Tensor:
        """
        img: (3, H, W) float32 [0, 1] の熱画像テンソル
        Returns: 拡張後の (3, H, W) テンソル
        """
        if random.random() < self.probs['flip']:
            img = random_horizontal_flip(img, p=1.0)

        if random.random() < self.probs['clahe_rand']:
            img = random_clahe_strength(img)

        if random.random() < self.probs['contrast']:
            img = contrast_jitter(img)

        if random.random() < self.probs['brightness']:
            img = brightness_jitter(img)

        if random.random() < self.probs['fpn']:
            img = fpn_column_noise(img)

        if random.random() < self.probs['vignetting']:
            img = vignetting(img)

        if random.random() < self.probs['motion_blur']:
            img = motion_blur(img)

        if random.random() < self.probs['gaussian']:
            img = gaussian_noise(img)

        if random.random() < self.probs['rain']:
            img = rain_drops(img)

        return img


# ---------------------------------------------------------------------------
# デフォルトインスタンス
# ---------------------------------------------------------------------------

DEFAULT_AUGMENTATION = ThermalAugmentation()
STRONG_AUGMENTATION  = ThermalAugmentation(
    p_brightness=0.7,
    p_contrast=0.7,
    p_fpn=0.6,
    p_motion_blur=0.4,
    p_rain=0.3,
    p_clahe_rand=0.6,
)