"""
modules/training/losses_post_kd.py
Post-KD 訓練の損失関数。

Stage 1: Thermal Homographic Adaptation
  - repeatability_loss()        : キーポイント再現性損失
  - fine_matching_loss()        : fine matcher サブピクセル損失

Stage 2: 幾何整合ファインチューニング
  - reprojection_loss()         : 再投影誤差損失
  - epipolar_loss()             : エピポーラ拘束損失
  - sampson_distance()          : Sampson距離（エピポーラの滑らかな近似）

NOTE:
  losses.py / alike_wrapper は import しない（ALIKE 依存禁止）
  losses_kd.py は import してよい（dual_softmax_loss を再利用）
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# ユーティリティ
# ---------------------------------------------------------------------------

def _warp_map(feat_map: Tensor, H_mat: Tensor,
              mode: str = 'bilinear') -> Tensor:
    """
    特徴マップ feat_map: (B, C, H, W) を ホモグラフィー H_mat: (B, 3, 3) で変換する。
    PyTorch の grid_sample を使用。

    Returns:
        warped: (B, C, H, W)
    """
    B, C, H, W = feat_map.shape
    device = feat_map.device
    dtype  = feat_map.dtype

    # メッシュグリッドを生成（画像座標系）
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device, dtype=dtype),
        torch.linspace(-1, 1, W, device=device, dtype=dtype),
        indexing='ij',
    )
    # 正規化座標 → 画素座標に変換
    xs = (x + 1) / 2 * (W - 1)
    ys = (y + 1) / 2 * (H - 1)
    ones = torch.ones_like(xs)
    grid_h = torch.stack([xs, ys, ones], dim=-1)   # (H, W, 3)
    grid_h = grid_h.view(1, H * W, 3).expand(B, -1, -1)  # (B, H*W, 3)

    # H_mat を適用: p' = H @ p
    H_t = H_mat.to(dtype)                              # (B, 3, 3)
    warped_pts = torch.bmm(H_t, grid_h.transpose(1, 2))  # (B, 3, H*W)
    warped_pts = warped_pts.transpose(1, 2)               # (B, H*W, 3)

    # 同次座標から正規化
    w_coord = warped_pts[:, :, 2:3].clamp(min=1e-8)
    warped_pts = warped_pts[:, :, :2] / w_coord          # (B, H*W, 2)

    # 画素座標 → [-1,1] 正規化グリッドへ変換
    warped_pts[:, :, 0] = warped_pts[:, :, 0] / (W - 1) * 2 - 1
    warped_pts[:, :, 1] = warped_pts[:, :, 1] / (H - 1) * 2 - 1
    grid_norm = warped_pts.view(B, H, W, 2)

    warped = F.grid_sample(
        feat_map, grid_norm, mode=mode,
        align_corners=True, padding_mode='zeros'
    )
    return warped


def _get_kpts_heatmap(kp_logits: Tensor, temp: float = 1.0) -> Tensor:
    """
    XFeatModel.keypoint_head の出力 (B, 65, Hf, Wf) から
    キーポイントヒートマップ (B, 1, H, W) を復元する。

    65ch の構成: [0:64]=8×8グリッドオフセット, [64]=dustbin

    Returns:
        heatmap: (B, 1, H=Hf*8, W=Wf*8) キーポイントスコア
    """
    scores = F.softmax(kp_logits * temp, dim=1)[:, :64]  # (B, 64, Hf, Wf)
    B, _, Hf, Wf = scores.shape
    heatmap = scores.permute(0, 2, 3, 1).reshape(B, Hf, Wf, 8, 8)
    heatmap = heatmap.permute(0, 1, 3, 2, 4).reshape(B, 1, Hf * 8, Wf * 8)
    return heatmap


# ---------------------------------------------------------------------------
# Stage 1: Thermal Homographic Adaptation 損失
# ---------------------------------------------------------------------------

def repeatability_loss(
    kp_logits: Tensor,
    kp_logits_w: Tensor,
    H_mat: Tensor,
    hmap_frozen: Tensor,
    hmap_w_frozen: Tensor,
    threshold: float = 0.01,
    eps: float = 1e-6,
) -> Tensor:
    """
    キーポイント再現性損失（Stage 1 主損失）。

    物理的根拠:
      KD で適応済みの信頼性マップ（hmap_frozen）は
      「熱画像のどこが特徴的か」を既に知っている。
      これをキーポイントブランチの教師信号として活用し、
      ホモグラフィー変換下で同じ点を繰り返し検出するよう学習させる。

    Args:
        kp_logits    : (B, 65, Hf, Wf)  元画像のキーポイントロジット（学習中）
        kp_logits_w  : (B, 65, Hf, Wf)  変換後画像のキーポイントロジット（学習中）
        H_mat        : (B, 3, 3)         ホモグラフィー行列（元→変換後）
        hmap_frozen  : (B, 1, H, W)      元画像の信頼性マップ（KD済み・frozen）
        hmap_w_frozen: (B, 1, H, W)      変換後画像の信頼性マップ（KD済み・frozen）
        threshold    : 信頼性マスクの閾値

    Returns:
        loss: scalar
    """
    # キーポイントスコアに変換 (B, 1, H, W)
    kp_score   = _get_kpts_heatmap(kp_logits)
    kp_score_w = _get_kpts_heatmap(kp_logits_w)

    # kp_score を H_mat で変換後画像へワープ
    kp_score_warped = _warp_map(kp_score, H_mat, mode='bilinear')  # (B,1,H,W)

    # 信頼性マスク: 両方で高信頼な領域のみで損失を計算
    # ワープした領域で hmap_frozen も一緒にワープ
    hmap_warped = _warp_map(hmap_frozen, H_mat, mode='bilinear')
    mask = ((hmap_warped > threshold) & (hmap_w_frozen > threshold)).float()

    # 再現性損失: ワープ後スコア ≈ 変換後スコア（マスク領域で）
    diff = (kp_score_warped - kp_score_w) ** 2
    denom = mask.sum().clamp(min=1.0)
    loss_repeat = (diff * mask).sum() / denom

    # 信頼性ガイド損失: 信頼性が高い場所でキーポイントスコアも高くなるよう誘導
    # L = -mean(hmap * kp_score) → 最大化 = hmap が高い場所で kp_score も高くする
    loss_guide = -(hmap_frozen * kp_score).mean() \
               - (hmap_w_frozen * kp_score_w).mean()

    return loss_repeat + 0.1 * loss_guide


def fine_matching_loss(
    feats_frozen: Tensor,
    feats_w_frozen: Tensor,
    kp_logits: Tensor,
    kp_logits_w: Tensor,
    fine_matcher: nn.Module,
    H_mat: Tensor,
    n_pts: int = 256,
    ws: int = 8,
) -> Tensor:
    """
    fine matcher サブピクセルオフセット回帰損失（Stage 1）。

    KD で適応済みの frozen 特徴マップから対応点を生成し、
    fine_matcher が正しいサブピクセルオフセットを予測するよう学習させる。

    Args:
        feats_frozen   : (B, 64, Hf, Wf) frozen 特徴マップ（元画像）
        feats_w_frozen : (B, 64, Hf, Wf) frozen 特徴マップ（変換後画像）
        kp_logits      : (B, 65, Hf, Wf) キーポイントロジット（元画像）
        kp_logits_w    : (B, 65, Hf, Wf) キーポイントロジット（変換後）
        fine_matcher   : XFeatModel.fine_matcher（学習中）
        H_mat          : (B, 3, 3) ホモグラフィー行列
        n_pts          : 対応点サンプリング数
        ws             : サブピクセルグリッドサイズ（XFeat では 8×8）

    Returns:
        loss: scalar
    """
    B, C, Hf, Wf = feats_frozen.shape
    device = feats_frozen.device
    total_loss = feats_frozen.new_zeros(1)

    for b in range(B):
        # ランダムにソース点を選択（Hf×Wf から n_pts 個）
        HW = Hf * Wf
        n = min(n_pts, HW)
        idx = torch.randperm(HW, device=device)[:n]
        iy = idx // Wf
        ix = idx % Wf

        # ソース特徴 (n, 64)
        f1 = feats_frozen[b, :, iy, ix].permute(1, 0)  # (n, 64)

        # ソース点の画像座標 → H_mat で変換後座標へ
        pts_src = torch.stack([
            (ix.float() * 8 + 4),   # セル中心 x
            (iy.float() * 8 + 4),   # セル中心 y
            torch.ones(n, device=device),
        ], dim=1)  # (n, 3)

        H_b = H_mat[b]  # (3, 3)
        pts_dst_h = (H_b @ pts_src.T)  # (3, n)
        w_coord = pts_dst_h[2:3, :].clamp(min=1e-8)
        pts_dst = (pts_dst_h[:2, :] / w_coord).T  # (n, 2) [x, y]

        # 変換後点のセル座標
        ix2 = (pts_dst[:, 0] / 8).long()
        iy2 = (pts_dst[:, 1] / 8).long()

        # 境界チェック
        valid = (ix2 >= 0) & (ix2 < Wf) & (iy2 >= 0) & (iy2 < Hf)
        if valid.sum() < 4:
            continue

        ix2 = ix2[valid]
        iy2 = iy2[valid]
        f1_v = f1[valid]
        pts_dst_v = pts_dst[valid]

        # ターゲット特徴 (n_valid, 64)
        f2 = feats_w_frozen[b, :, iy2, ix2].permute(1, 0)

        # fine_matcher でオフセット予測
        # XFeat の fine_matcher: (n, 128) → (n, 64) → subpix_softmax2d → (n, 2)
        offsets_pred = fine_matcher(
            torch.cat([f1_v, f2], dim=-1))  # (n_valid, 64)

        # GT オフセット: セル内でのサブピクセル位置
        # セル中心からのオフセット（-ws/2 〜 +ws/2）
        cell_cx = (ix2.float() * 8 + 4)  # セル中心 x (変換後)
        cell_cy = (iy2.float() * 8 + 4)  # セル中心 y (変換後)
        offset_x = (pts_dst_v[:, 0] - cell_cx).clamp(-ws/2, ws/2)
        offset_y = (pts_dst_v[:, 1] - cell_cy).clamp(-ws/2, ws/2)
        offset_gt = torch.stack([offset_x, offset_y], dim=-1)  # (n_valid, 2)

        # サブピクセルソフトマックスで予測オフセットに変換
        # (n, 64) → (n, ws, ws) → weighted sum
        n_valid = offsets_pred.shape[0]
        heat = offsets_pred.view(n_valid, ws, ws)
        heat = F.softmax(heat.view(n_valid, -1) * 0.25, dim=-1).view(n_valid, ws, ws)
        gx, gy = torch.meshgrid(
            torch.arange(ws, device=device, dtype=torch.float32) - ws // 2,
            torch.arange(ws, device=device, dtype=torch.float32) - ws // 2,
            indexing='xy',
        )
        pred_x = (gx[None] * heat).sum(dim=(-2, -1))
        pred_y = (gy[None] * heat).sum(dim=(-2, -1))
        pred_offset = torch.stack([pred_x, pred_y], dim=-1)

        loss_b = F.smooth_l1_loss(pred_offset, offset_gt, beta=1.0)
        total_loss = total_loss + loss_b

    return total_loss / B


# ---------------------------------------------------------------------------
# Stage 2: 幾何整合損失
# ---------------------------------------------------------------------------

def sampson_distance(
    pts1: Tensor, pts2: Tensor, F_mat: Tensor, eps: float = 1e-8
) -> Tensor:
    """
    Sampson距離（エピポーラ拘束の滑らかな近似）。
    F行列が未知の場合は8点アルゴリズムで推定して使用する。

    Args:
        pts1  : (N, 2) 画像1上のキーポイント座標（画素）
        pts2  : (N, 2) 画像2上の対応キーポイント座標（画素）
        F_mat : (3, 3) 基本行列（Fundamental matrix）

    Returns:
        dist : (N,) 各点ペアのSampson距離
    """
    N = pts1.shape[0]
    ones = pts1.new_ones(N, 1)
    p1h = torch.cat([pts1, ones], dim=1)  # (N, 3)
    p2h = torch.cat([pts2, ones], dim=1)  # (N, 3)

    Fp1 = (F_mat @ p1h.T).T    # (N, 3)
    Ftp2 = (F_mat.T @ p2h.T).T # (N, 3)

    numer = (p2h * Fp1).sum(dim=1) ** 2   # (N,)
    denom = Fp1[:, 0] ** 2 + Fp1[:, 1] ** 2 \
          + Ftp2[:, 0] ** 2 + Ftp2[:, 1] ** 2
    return numer / (denom.clamp(min=eps))


def epipolar_loss(
    pts1: Tensor,
    pts2: Tensor,
    K: Tensor,
    T_rel: Tensor,
    use_gt_pose: bool = True,
    inlier_threshold: float = 2.0,
) -> Tensor:
    """
    エピポーラ拘束損失（Stage 2）。

    既知の相対姿勢 T_rel から Essential matrix E を計算し、
    さらに Fundamental matrix F = K^{-T} E K^{-1} を求めて
    Sampson 距離で損失を計算する。

    Args:
        pts1   : (N, 2) 画像1のキーポイント（画素座標）
        pts2   : (N, 2) 画像2のキーポイント（画素座標）
        K      : (3, 3) カメラ内部パラメータ
        T_rel  : (4, 4) 相対姿勢 T_{t→t+1}
        use_gt_pose: True → T_rel から F を計算 / False → 8点法で F を推定

    Returns:
        loss: scalar
    """
    if pts1.shape[0] < 8:
        return pts1.new_zeros(1).squeeze()

    if use_gt_pose:
        # E = [t]× R から計算
        R = T_rel[:3, :3]
        t = T_rel[:3, 3]
        t_skew = torch.zeros(3, 3, device=t.device, dtype=t.dtype)
        t_skew[0, 1] = -t[2]; t_skew[0, 2] =  t[1]
        t_skew[1, 0] =  t[2]; t_skew[1, 2] = -t[0]
        t_skew[2, 0] = -t[1]; t_skew[2, 1] =  t[0]
        E = t_skew @ R                          # (3, 3)

        K_inv = torch.inverse(K)
        F_mat = K_inv.T @ E @ K_inv             # (3, 3)
    else:
        # 8点アルゴリズム（OpenCV 経由、no_grad）
        import cv2, numpy as np
        with torch.no_grad():
            p1_np = pts1.cpu().numpy().astype(np.float32)
            p2_np = pts2.cpu().numpy().astype(np.float32)
            F_np, _ = cv2.findFundamentalMat(
                p1_np, p2_np, cv2.FM_8POINT)
            if F_np is None:
                return pts1.new_zeros(1).squeeze()
            F_mat = torch.from_numpy(F_np).float().to(pts1.device)

    dist = sampson_distance(pts1, pts2, F_mat)

    # ソフトインライア重み: 閾値以内の点に高重みを付与
    weights = torch.exp(-dist / (inlier_threshold ** 2))
    loss = (dist * weights).mean()
    return loss


def reprojection_loss(
    pts1: Tensor,
    pts2: Tensor,
    K: Tensor,
    T_rel: Tensor,
    depths: Optional[Tensor] = None,
    inlier_threshold: float = 2.0,
) -> Tensor:
    """
    再投影誤差損失（Stage 2）。

    既知の相対姿勢 T_rel と深度情報 depths を使って
    pts1 を 3D に逆投影し、pts2 に再投影したときの誤差を計算する。
    depths が None の場合は規格化座標での転置で近似する。

    Args:
        pts1   : (N, 2) 画像1のキーポイント（画素座標）
        pts2   : (N, 2) 画像2の対応点（画素座標）
        K      : (3, 3) カメラ内部パラメータ
        T_rel  : (4, 4) 相対姿勢 T_{t→t+1}
        depths : (N,) 各キーポイントの深度（None = 深度未知）
        inlier_threshold: ソフトHuber損失の折れ点（画素単位）

    Returns:
        loss: scalar
    """
    N = pts1.shape[0]
    if N < 4:
        return pts1.new_zeros(1).squeeze()

    K_inv = torch.inverse(K)

    # 正規化座標に変換
    ones = pts1.new_ones(N, 1)
    p1h = torch.cat([pts1, ones], dim=1).T   # (3, N)
    p1n = K_inv @ p1h                         # (3, N) 正規化座標

    # 深度がある場合: 逆投影 → 変換 → 再投影
    if depths is not None:
        d = depths.view(1, N)
        pts3d = p1n * d                        # (3, N)
        ones3d = pts1.new_ones(1, N)
        pts3d_h = torch.cat([pts3d, ones3d], dim=0)  # (4, N)
        pts3d_cam2 = T_rel @ pts3d_h           # (4, N)
        pts2d_h = K @ pts3d_cam2[:3, :]        # (3, N)
        d2 = pts2d_h[2:3, :].clamp(min=1e-8)
        pts2_proj = (pts2d_h[:2, :] / d2).T   # (N, 2)
    else:
        # 深度未知: 規格化座標で回転のみ適用（近似）
        R = T_rel[:3, :3]
        p2n = R @ p1n                          # (3, N)
        pts2_proj = (K @ p2n).T                # (N, 3)
        d2 = pts2_proj[:, 2:3].clamp(min=1e-8)
        pts2_proj = pts2_proj[:, :2] / d2      # (N, 2)

    # Huber 損失
    err = torch.norm(pts2_proj - pts2, dim=1)  # (N,)
    loss = F.huber_loss(err, torch.zeros_like(err),
                        delta=inlier_threshold, reduction='mean')
    return loss

# Optional インポート処理
try:
    from typing import Optional
except ImportError:
    Optional = None