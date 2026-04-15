"""
modules/training/losses_post_kd.py
Post-KD 訓練の損失関数。

Stage 1: Thermal Homographic Adaptation
  - repeatability_loss()                   : キーポイント再現性損失
  - fine_matching_loss()                   : fine matcher サブピクセル損失

Stage 2: 幾何整合ファインチューニング（案C: GT投影特徴損失）
  - geometric_feature_consistency_loss()   : GT投影+特徴空間損失（メイン）
      1. hmap_t 上位N点を pts1 として選択（no_grad）
      2. T_rel と K で pts1 を pts2_gt に投影（no_grad・純幾何計算）
      3. feats_t[pts1] vs feats_t1[pts2_gt] のコサイン類似度損失（勾配あり）
      4. エピポーラ Sampson 距離を重みとして付与（勾配なし）
  - sampson_distance()                     : Sampson距離（エピポーラの近似）

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

    【修正履歴】
      旧実装の問題:
        1. loss_guide = -(hmap_frozen * kp_score).mean()
           → hmap_frozen が上部に集中していると kp_score も上部に引き寄せられる
           → 下部では hmap_frozen ≈ 0 なので勾配がゼロ = 下部が学習されない
        2. mask = (hmap_warped > threshold) & (hmap_w_frozen > threshold)
           → hmap が上部にしか高い値を持たないとき、下部は mask=0 で
              repeatability_loss の勾配もゼロ = 下部が一切学習されない
        3. kp_score を softmax(64ch).max() で計算
           → dustbin チャンネル(ch.64)を無視するため「キーポイントなし」を
             正しく表現できない

      修正内容:
        1. loss_guide を削除（hmap のバイアスを kp_logits に伝播させない）
        2. mask を均一（ホモグラフィーの有効領域のみ）に変更
           → 画像全体でキーポイント検出を学習する
        3. kp_score を 65ch softmax の非dustbin確率の和として計算
           = P(キーポイントあり) = 1 - P(dustbin)
           → dustbin が高い領域では kp_score が低くなり正しく表現できる

    【マスクの意味】
      valid_mask: ホモグラフィー変換後に画像内に収まる領域のみで損失を計算。
      hmap に依存しないため空間バイアスを生じさせない。

    Args:
        kp_logits    : (B, 65, Hf, Wf)  元画像のキーポイントロジット（学習中）
        kp_logits_w  : (B, 65, Hf, Wf)  変換後画像のキーポイントロジット（学習中）
        H_mat        : (B, 3, 3)         ホモグラフィー行列（元→変換後）
        hmap_frozen  : (B, 1,  Hf, Wf)  信頼性マップ（本修正では mask に使用しない）
        hmap_w_frozen: (B, 1,  Hf, Wf)  信頼性マップ（本修正では mask に使用しない）
        threshold    : 未使用（後方互換のために引数は保持）

    Returns:
        loss: scalar
    """
    # ── キーポイントスコアの計算（65ch softmax で dustbin を考慮）────────────
    # P(keypoint) = sum of non-dustbin probs = 1 - P(dustbin)
    # 旧: softmax(64ch).max() → dustbin を無視 → 常に高い値になる
    # 新: softmax(65ch)[:64].sum() → dustbin が高いとスコアが低くなる
    probs     = F.softmax(kp_logits,   dim=1)          # (B, 65, Hf, Wf)
    probs_w   = F.softmax(kp_logits_w, dim=1)
    kp_score   = probs[:, :64].sum(dim=1, keepdim=True)    # (B, 1, Hf, Wf)
    kp_score_w = probs_w[:, :64].sum(dim=1, keepdim=True)  # (B, 1, Hf, Wf)

    # ── ワープ ───────────────────────────────────────────────────────────────
    kp_score_warped = _warp_map(kp_score, H_mat, mode='bilinear')  # (B,1,Hf,Wf)

    # ── 均一マスク: ホモグラフィーの有効領域のみ ────────────────────────────
    # ゼロ埋めで外側にはみ出た画素は grid_sample で 0 になる。
    # warped 後の kp_score が 0 より大きい領域 = 有効領域。
    # hmap に依存しないため空間バイアスを生じさせない。
    with torch.no_grad():
        # 全1テンソルをワープして有効領域マスクを生成
        ones    = torch.ones_like(kp_score)
        valid_mask = (_warp_map(ones, H_mat, mode='nearest') > 0.5).float()
    # (B, 1, Hf, Wf)

    # ── 再現性損失 ────────────────────────────────────────────────────────────
    diff  = (kp_score_warped - kp_score_w) ** 2   # (B, 1, Hf, Wf)
    denom = valid_mask.sum().clamp(min=1.0)
    loss_repeat = (diff * valid_mask).sum() / denom

    # loss_guide は削除（hmap バイアスの伝播を防ぐ）
    # 旧: loss_repeat + 0.1 * loss_guide
    return loss_repeat


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


# ---------------------------------------------------------------------------
# Stage 2: 幾何整合損失（案C: GT投影特徴損失）
# ---------------------------------------------------------------------------

def _project_pts(
    pts1: Tensor,
    K: Tensor,
    T_rel: Tensor,
) -> Tensor:
    """
    pts1 を T_rel と K を使って画像2上に投影する（純粋な幾何計算）。

    Args:
        pts1  : (N, 2) 画像1上の点（画素座標、[x, y]）
        K     : (3, 3) カメラ内部行列
        T_rel : (4, 4) 相対姿勢 T_{1→2}

    Returns:
        pts2_proj : (N, 2) 画像2上の投影点（画素座標）
        valid_mask: (N,)   bool（投影点がdepth > 0 の有効点）
    """
    N = pts1.shape[0]
    ones = pts1.new_ones(N, 1)
    p1h  = torch.cat([pts1, ones], dim=1).T          # (3, N)

    # torch.inverse は GPU 上の小行列で cuSOLVER エラーになる場合がある
    # K は (3,3) の小行列なので CPU で逆行列を計算して GPU に戻す
    K_cpu   = K.cpu().double()
    K_inv   = torch.inverse(K_cpu).float().to(pts1.device)

    p1n   = K_inv @ p1h                              # (3, N) 正規化座標

    R = T_rel[:3, :3]
    t = T_rel[:3, 3:4]                               # (3, 1)
    p2n = R @ p1n + t                                # (3, N)

    pts2_proj_h = K @ p2n                            # (3, N)
    depth       = p2n[2:3, :]                        # (1, N)
    valid_mask  = (depth[0] > 1e-4)                  # (N,)

    d_safe      = depth.clamp(min=1e-8)
    pts2_proj   = (pts2_proj_h[:2, :] / d_safe).T   # (N, 2)

    return pts2_proj, valid_mask


def _sample_feats(
    feats: Tensor,
    pts_px: Tensor,
    H: int,
    W: int,
) -> Tensor:
    """
    特徴マップ feats: (C, Hf, Wf) から pts_px の位置の特徴を双線形補間でサンプリング。

    Args:
        feats  : (C, Hf, Wf)  特徴マップ（1バッチ分）
        pts_px : (N, 2)       画素座標 [x, y]（フル解像度）
        H, W   : int          元画像のフル解像度

    Returns:
        sampled: (N, C)
    """
    C, Hf, Wf = feats.shape
    N = pts_px.shape[0]

    # 画素座標 → 特徴マップ座標 → grid_sample 用正規化座標 [-1, 1]
    # feats は stride=8 → 特徴マップ座標 = pts_px / 8
    fx = pts_px[:, 0] / W * 2.0 - 1.0   # x を [-1, 1] に正規化
    fy = pts_px[:, 1] / H * 2.0 - 1.0   # y を [-1, 1] に正規化
    grid = torch.stack([fx, fy], dim=1)  # (N, 2)
    # grid_sample は (B, C, H, W) x (B, N_out_H, N_out_W, 2) の形式
    grid = grid.view(1, 1, N, 2)         # (1, 1, N, 2)
    feats4d = feats.unsqueeze(0)         # (1, C, Hf, Wf)

    sampled = F.grid_sample(
        feats4d, grid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=False,
    )  # (1, C, 1, N)
    sampled = sampled[0, :, 0, :].T      # (N, C)
    return sampled


def geometric_feature_consistency_loss(
    feats_t:  Tensor,
    feats_t1: Tensor,
    hmap_t:   Tensor,
    K:        Tensor,
    T_rel:    Tensor,
    H: int,
    W: int,
    n_pts: int = 256,
    epi_weight_scale: float = 2.0,
    margin: int = 8,
) -> Tensor:
    """
    GT投影特徴整合損失（Stage 2 メイン損失）。

    【設計方針: 案C】
      対応点を「現在のモデルの予測」ではなく
      「GTポーズ T_rel と K による3D投影」で決定する。
      これにより:
        - 幾何情報（T_rel, K）を正しく使用
        - 対応点がモデル出力に依存しない（鶏と卵の問題なし）
        - feats への勾配が正常に流れる（backward() が成功する）

    【処理フロー】
      1. hmap_t 上位 n_pts 点を pts1 として選択（no_grad）
      2. T_rel, K で pts1 → pts2_gt に投影（no_grad・純幾何）
      3. 画像範囲内の点のみに絞る（valid_mask）
      4. feats_t[pts1] と feats_t1[pts2_gt] を双線形補間でサンプリング
      5. コサイン類似度損失（正の対応 = 一致させる）
      6. エピポーラ Sampson 距離を重みとして付与（detach・勾配なし）

    Args:
        feats_t   : (C, Hf, Wf)  フレーム t の特徴マップ（学習側・勾配あり）
        feats_t1  : (C, Hf, Wf)  フレーム t+1 の特徴マップ（学習側・勾配あり）
        hmap_t    : (1, Hf, Wf)  フレーム t の信頼性マップ（no_grad で取得）
        K         : (3, 3)        カメラ内部行列
        T_rel     : (4, 4)        相対姿勢 T_{t→t+1}
        H, W      : int           元画像のフル解像度
        n_pts     : int           使用するキーポイント数
        epi_weight_scale: float  Sampson 距離のソフト重みのスケール
        margin    : int           画像端のマージン（画素）

    Returns:
        loss: scalar Tensor（勾配あり）
    """
    # ── 1. キーポイント選択（hmap 上位 n_pts 点） ────────────────────────
    # hmap は no_grad で取得済み → ここは純粋なインデックス操作
    Hf, Wf = hmap_t.shape[-2], hmap_t.shape[-1]
    flat    = hmap_t[0].reshape(-1)                          # (Hf*Wf,)
    k       = min(n_pts, flat.numel())
    _, topk = torch.topk(flat, k=k)
    iy1f    = topk // Wf                                     # 特徴マップ y
    ix1f    = topk % Wf                                      # 特徴マップ x

    # 特徴マップ座標 → 画素座標（セル中心: *8 + 4）
    px1 = ix1f.float() * 8.0 + 4.0                          # (k,)
    py1 = iy1f.float() * 8.0 + 4.0
    pts1 = torch.stack([px1, py1], dim=1)                    # (k, 2)

    # ── 2. GT投影: T_rel, K で pts2_gt を計算（no_grad） ────────────────
    with torch.no_grad():
        pts2_gt, valid_depth = _project_pts(pts1, K, T_rel)

        # 画像範囲チェック
        in_bounds = (
            (pts2_gt[:, 0] >= margin) &
            (pts2_gt[:, 0] <  W - margin) &
            (pts2_gt[:, 1] >= margin) &
            (pts2_gt[:, 1] <  H - margin) &
            valid_depth
        )
        # pts1 も範囲チェック（念のため）
        in_bounds_1 = (
            (pts1[:, 0] >= margin) &
            (pts1[:, 0] <  W - margin) &
            (pts1[:, 1] >= margin) &
            (pts1[:, 1] <  H - margin)
        )
        valid = in_bounds & in_bounds_1

    if valid.sum() < 4:
        # 有効点が少なすぎる場合は None を返してループ側でスキップする
        # （feats_t.new_zeros() は requires_grad=False なので backward が壊れる）
        return None

    pts1_v   = pts1[valid]                                   # (N_v, 2)
    pts2_v   = pts2_gt[valid]                                # (N_v, 2)

    # ── 3. 特徴サンプリング（双線形補間・勾配あり） ──────────────────────
    f1 = _sample_feats(feats_t,  pts1_v, H, W)              # (N_v, C)
    f2 = _sample_feats(feats_t1, pts2_v, H, W)              # (N_v, C)

    # ── 4. コサイン類似度損失 ────────────────────────────────────────────
    cos_sim  = F.cosine_similarity(f1, f2, dim=1)            # (N_v,)
    loss_cos = 1.0 - cos_sim                                 # 低いほど良い

    # ── 5. エピポーラ Sampson 重み（detach・勾配なし） ───────────────────
    # エピポーラ拘束に合う点ほど高い重みで学習させる
    # ただし重み自体は座標値から計算するため detach して勾配を切る
    with torch.no_grad():
        R   = T_rel[:3, :3]
        t   = T_rel[:3, 3]
        t_s = torch.zeros(3, 3, device=t.device, dtype=t.dtype)
        t_s[0,1] = -t[2]; t_s[0,2] =  t[1]
        t_s[1,0] =  t[2]; t_s[1,2] = -t[0]
        t_s[2,0] = -t[1]; t_s[2,1] =  t[0]
        E     = t_s @ R
        # torch.inverse は GPU 小行列で cuSOLVER エラーになる場合があるため CPU で計算
        K_inv = torch.inverse(K.cpu().double()).float().to(K.device)
        F_mat = K_inv.T @ E @ K_inv
        epi_dist = sampson_distance(pts1_v, pts2_v, F_mat)  # (N_v,)
        epi_w    = torch.exp(
            -epi_dist / (epi_weight_scale ** 2)
        ).clamp(min=0.0, max=1.0)                           # (N_v,) ∈ [0,1]

    # 重み付き平均損失
    loss = (loss_cos * epi_w).sum() / epi_w.sum().clamp(min=1.0)
    return loss