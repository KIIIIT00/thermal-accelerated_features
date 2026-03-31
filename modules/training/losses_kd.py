"""
modules/training/losses_kd.py
Thermal XFeat KD 用損失関数。

ALIKE・alike_wrapper に一切依存しない独立実装。
losses.py を import しないこと（alike_wrapper がトップレベル import されているため）。

損失構成:
    L_total = L_KD
            + λ_rel × L_KD_rel
            + λ_fpn × L_FPN

NOTE: fpn_invariance_loss 内では student が追加 1 回フォワードされる。
      clean 側は torch.no_grad() で包む（stop_gradient）。
      逆方向（clean を noisy に引き寄せる）は特徴が劣化するため禁止。
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# 7.2  dual_softmax_loss (losses.py の完全等価再実装・ALIKE 依存なし)
# ---------------------------------------------------------------------------

def dual_softmax_loss(
    X: Tensor,
    Y: Tensor,
    temp: float = 0.2,
) -> Tuple[Tensor, Tensor]:
    """
    Dual-softmax InfoNCE 損失。

    Args:
        X: (N, C)  L2 正規化済み特徴ベクトル
        Y: (N, C)  L2 正規化済み特徴ベクトル
        temp: 温度パラメータ

    Returns:
        loss: scalar Tensor
        conf: (N,) マッチング確信度
    """
    if X.size() != Y.size() or X.dim() != 2 or Y.dim() != 2:
        raise RuntimeError(
            'dual_softmax_loss: X and Y must be 2D matrices with the same shape')

    dist_mat = (X @ Y.t()) * temp
    log_p12 = F.log_softmax(dist_mat,     dim=1)
    log_p21 = F.log_softmax(dist_mat.t(), dim=1)

    with torch.no_grad():
        conf = log_p12.exp().max(dim=-1)[0] * log_p21.exp().max(dim=-1)[0]

    target = torch.arange(len(X), device=X.device)
    loss = F.nll_loss(log_p12, target) + F.nll_loss(log_p21, target)

    return loss, conf


# ---------------------------------------------------------------------------
# 7.3  L_KD: 特徴マップ KD
# ---------------------------------------------------------------------------

def kd_feature_loss(
    feats_s: Tensor,
    feats_t: Tensor,
    n_samples: int = 1024,
    temp: float = 0.2,
) -> Tensor:
    """
    特徴マップ Knowledge Distillation 損失。
    空間次元からランダムサブサンプリングして dual_softmax_loss を計算する。

    Args:
        feats_s: (B, 64, H/8, W/8)  生徒特徴（L2 正規化済み）
        feats_t: (B, 64, H/8, W/8)  教師特徴（L2 正規化済み・detach 済み）
        n_samples: サンプリング数（上限 H*W）
        temp: 温度パラメータ

    Returns:
        loss: scalar Tensor
    """
    B, C, Hf, Wf = feats_s.shape
    HW = Hf * Wf
    n = min(HW, n_samples)

    # (B, C, H*W) → (B, H*W, C)
    s_flat = feats_s.reshape(B, C, HW).permute(0, 2, 1)  # (B, HW, C)
    t_flat = feats_t.reshape(B, C, HW).permute(0, 2, 1)

    total_loss = feats_s.new_zeros(1)
    for b in range(B):
        idx = torch.randperm(HW, device=feats_s.device)[:n]
        s_b = s_flat[b, idx]   # (n, C)
        t_b = t_flat[b, idx]   # (n, C)
        loss_b, _ = dual_softmax_loss(s_b, t_b, temp=temp)
        total_loss = total_loss + loss_b

    return total_loss / B


# ---------------------------------------------------------------------------
# 7.4  L_KD_rel: 信頼性マップ KD
# ---------------------------------------------------------------------------

def kd_reliability_loss(
    hmap_s: Tensor,
    hmap_t: Tensor,
    img_thr: Tensor,
) -> Tensor:
    """
    信頼性マップ KD 損失（温度勾配確信度加重 MSE）。

    Args:
        hmap_s : (B, 1, H/8, W/8)  生徒信頼性マップ [0,1]
        hmap_t : (B, 1, H/8, W/8)  教師信頼性マップ（detach 済み）
        img_thr: (B, 3, H,   W)    熱画像（確信度マスク計算に使用）

    Returns:
        loss: scalar Tensor
    """
    B, _, Hf, Wf = hmap_s.shape

    # Sobel 勾配強度を計算して確信度マスクを生成
    with torch.no_grad():
        # グレースケール化 (B, 1, H, W)
        gray = img_thr.mean(dim=1, keepdim=True)

        # Sobel カーネル
        sobel_x = torch.tensor(
            [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
            device=img_thr.device
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]],
            device=img_thr.device
        ).view(1, 1, 3, 3)

        gx = F.conv2d(gray, sobel_x, padding=1)
        gy = F.conv2d(gray, sobel_y, padding=1)
        grad_mag = (gx ** 2 + gy ** 2).sqrt()  # (B, 1, H, W)

        # ヒートマップ解像度にダウンサンプル → 確信度マスク
        conf_mask = F.interpolate(
            grad_mag, size=(Hf, Wf), mode='bilinear', align_corners=False
        )  # (B, 1, Hf, Wf)

        # [0,1] 正規化
        conf_max = conf_mask.flatten(2).max(dim=-1)[0].view(B, 1, 1, 1) + 1e-8
        conf_mask = conf_mask / conf_max

    sq_err = (hmap_s - hmap_t) ** 2          # (B, 1, Hf, Wf)
    loss = (conf_mask * sq_err).mean()
    return loss


# ---------------------------------------------------------------------------
# 7.5  L_FPN: FPN 不変性損失（物理考慮損失）
# ---------------------------------------------------------------------------

def fpn_invariance_loss(
    student: nn.Module,
    img_thr: Tensor,
    sigma_min: float = 2.0,
    sigma_max: float = 8.0,
    return_sigma: bool = False,
) -> Tensor:
    """
    固定パターンノイズ (FPN) 不変性損失。

    物理的根拠:
        LWIR センサーの列並列 ADC に起因する列方向固定パターンノイズ。
        CLAHE では除去されず公開データセット前処理後も残存する。
        均一温度黒体撮影で σ_col を実測し、損失設計値と照合可能。

    ノイズ生成:
        sigma     ~ Uniform(sigma_min/255, sigma_max/255)
        col_noise  = randn(W) * sigma      (列方向固定、行方向共通)
        img_fpn    = clamp(img_thr + col_noise, 0, 1)

    損失方向（重要）:
        clean 側 → stop_gradient (no_grad)  = 目標
        noisy 側 → 学習側
        MSE(feats_fpn, feats_clean.detach())
        ← "FPN があっても同じ特徴を出せる" よう student を学習
        ← 逆方向（clean → noisy）は特徴劣化するため禁止

    Args:
        student:      XFeatModel（訓練中）
        img_thr:      (B, 3, H, W) float [0,1]  熱画像
        sigma_min:    列ノイズ標準偏差下限 (DN units / 255)
        sigma_max:    列ノイズ標準偏差上限 (DN units / 255)
        return_sigma: True のとき (loss, sigma_mean) のタプルを返す
                      wandb で実際のノイズ強度を確認するために使用

    Returns:
        loss: scalar Tensor  (return_sigma=False)
        (loss, sigma_mean): Tuple  (return_sigma=True)
    """
    B, C, H, W = img_thr.shape

    # FPN ノイズ生成（バッチ共通でなく各バッチ独立に生成）
    sigma = (
        torch.rand(B, 1, device=img_thr.device)
        * (sigma_max - sigma_min) / 255.0
        + sigma_min / 255.0
    )  # (B, 1)

    # 列方向固定ノイズ: (B, 1, 1, W)
    col_noise = torch.randn(B, 1, 1, W, device=img_thr.device) \
        * sigma.view(B, 1, 1, 1)
    col_noise = col_noise.expand(B, C, H, W)

    img_fpn = (img_thr + col_noise).clamp(0.0, 1.0)

    # clean 側: stop_gradient（目標）
    with torch.no_grad():
        feats_clean, _, _ = student(img_thr)
        feats_clean = F.normalize(feats_clean, dim=1)

    # noisy 側: 学習側
    feats_fpn, _, _ = student(img_fpn)
    feats_fpn = F.normalize(feats_fpn, dim=1)

    loss = F.mse_loss(feats_fpn, feats_clean.detach())

    if return_sigma:
        # DN 単位に戻して返す（sigma_min〜sigma_max の範囲内か確認用）
        sigma_mean = (sigma.mean() * 255.0).item()
        return loss, sigma_mean
    return loss