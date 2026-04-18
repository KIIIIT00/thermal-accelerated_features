"""
modules/thermal_kp_selector.py
温度勾配ベースのキーポイント選択。

【根拠】
XFeat の hmap は RGB 画像で学習された信頼度マップのため、
熱画像の均一領域（低テクスチャ）で KP スコアが不安定になる。

温度勾配（Sobel フィルタ）は熱画像固有の顕著性を表し、
勾配が大きい境界（物体の輪郭・温度変化領域）は
時間的に安定した特徴点が存在しやすい。

combined_score = kp_score * thermal_gradient_normalized
→ 均一領域の不安定な点を除去し、安定した KP のみを選択する。
"""

from __future__ import annotations
from typing import Tuple

import cv2
import numpy as np
import torch


def compute_thermal_saliency(
    gray_img: np.ndarray,
    blur_ksize: int = 5,
    normalize: bool = True,
) -> np.ndarray:
    """
    熱画像から温度勾配顕著性マップを計算する。

    Args:
        gray_img:   グレースケール熱画像 (H, W) uint8 or float
        blur_ksize: Sobel 前のガウスぼかしカーネルサイズ（ノイズ低減）
        normalize:  出力を [0, 1] に正規化するか

    Returns:
        saliency: (H, W) float32  温度勾配の大きさ
    """
    img = gray_img.astype(np.float32)

    # ノイズ低減
    if blur_ksize > 1:
        img = cv2.GaussianBlur(img, (blur_ksize, blur_ksize), 0)

    # Sobel 勾配（熱画像は 8bit なので scale=1）
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    saliency = np.sqrt(gx**2 + gy**2)

    if normalize:
        max_val = saliency.max()
        if max_val > 1e-6:
            saliency = saliency / max_val

    return saliency.astype(np.float32)


def select_keypoints_with_thermal_gradient(
    kpts:       np.ndarray,   # (N, 2) [x, y] pixel coords
    scores:     np.ndarray,   # (N,)   XFeat の検出スコア
    gray_img:   np.ndarray,   # (H, W) グレースケール熱画像
    image_size: Tuple[int, int],   # (W, H)
    top_k:      int = 1024,
    alpha:      float = 0.5,
    grad_threshold: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    温度勾配スコアと XFeat スコアを組み合わせてキーポイントを選択する。

    combined_score = (1 - alpha) * kp_score + alpha * thermal_gradient

    Args:
        kpts:           (N, 2) キーポイント座標
        scores:         (N,)   XFeat 検出スコア
        gray_img:       (H, W) 熱画像（グレースケール）
        image_size:     (W, H) 画像サイズ（kpts のスケール変換に使用）
        top_k:          選択するキーポイント数
        alpha:          温度勾配の重み（0=XFeat スコアのみ、1=勾配のみ）
        grad_threshold: 勾配が閾値以下の点は除外（均一領域の排除）

    Returns:
        selected_kpts:   (M, 2) 選択後のキーポイント
        selected_scores: (M,)   選択後のスコア
    """
    if len(kpts) == 0:
        return kpts, scores

    W, H = image_size

    # 温度勾配顕著性マップを計算
    if gray_img.shape[:2] != (H, W):
        gray_resized = cv2.resize(gray_img, (W, H))
    else:
        gray_resized = gray_img

    saliency = compute_thermal_saliency(gray_resized, normalize=True)

    # キーポイント座標で顕著性をサンプリング
    kx = np.clip(kpts[:, 0].astype(np.int32), 0, W - 1)
    ky = np.clip(kpts[:, 1].astype(np.int32), 0, H - 1)
    grad_scores = saliency[ky, kx]   # (N,)

    # 均一領域の点を除外（勾配が極めて小さい点はノイズ）
    valid_mask = grad_scores >= grad_threshold
    if valid_mask.sum() < 10:
        # 閾値が厳しすぎる場合は全点を許容
        valid_mask = np.ones(len(kpts), dtype=bool)

    # combined score で並べ替え
    kp_scores_norm = scores / (scores.max() + 1e-8)  # [0, 1] に正規化
    combined = ((1 - alpha) * kp_scores_norm + alpha * grad_scores)
    combined[~valid_mask] = -1.0

    # top_k を選択
    k = min(top_k, len(kpts))
    top_idx = np.argsort(-combined)[:k]
    top_idx = top_idx[combined[top_idx] >= 0]   # 無効点（-1）を除外

    return kpts[top_idx], scores[top_idx]


def patch_detect_with_thermal_gradient(
    detect_fn,
    alpha: float = 0.5,
    top_k: int = 1024,
    grad_threshold: float = 0.05,
):
    """
    既存の detect() 関数をラップして温度勾配選択を追加する。

    使用方法:
        from modules.thermal_kp_selector import patch_detect_with_thermal_gradient
        from evaluate.eval_matching import detect as _detect_orig

        detect_thermal = patch_detect_with_thermal_gradient(_detect_orig)

        # 通常の detect と同じように使える
        kpts, descs = detect_thermal(model, img_tensor, max_kp,
                                     gray_img=gray_np)
    """
    import functools

    @functools.wraps(detect_fn)
    def wrapper(model, img_t, max_kp, gray_img=None, **kwargs):
        kpts, descs = detect_fn(model, img_t, max_kp, **kwargs)

        if gray_img is None or len(kpts) == 0:
            return kpts, descs

        # img_t が (1, 3, H, W) テンソルなら gray_img を自動生成
        if gray_img is None and img_t is not None:
            arr = (img_t.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255
                   ).astype(np.uint8)
            gray_img = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

        H, W = img_t.shape[-2], img_t.shape[-1]
        scores = np.ones(len(kpts))   # XFeat の score が取れない場合は均一

        new_kpts, _ = select_keypoints_with_thermal_gradient(
            kpts, scores, gray_img, (W, H),
            top_k=max_kp, alpha=alpha, grad_threshold=grad_threshold)

        # descs を new_kpts のインデックスで選択
        # kpts と new_kpts の対応を取る（brute force、kpts が 1024 以下なので OK）
        if len(new_kpts) < len(kpts):
            dist = np.linalg.norm(
                kpts[:, None] - new_kpts[None], axis=2)  # (N, M)
            idx  = np.argmin(dist, axis=0)
            new_descs = descs[idx]
        else:
            new_descs = descs

        return new_kpts, new_descs

    return wrapper