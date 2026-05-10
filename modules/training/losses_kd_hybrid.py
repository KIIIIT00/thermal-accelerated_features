"""
modules/training/losses_kd.py
Thermal XFeat KD 用損失関数。

ALIKE・alike_wrapper に一切依存しない独立実装。
losses.py を import しないこと（alike_wrapper がトップレベル import されているため）。

損失構成:
    L_total = L_KD                     ← cross-modal alignment
            + λ_rel    × L_KD_rel      ← 信頼性マップ転移
            + λ_fpn    × L_FPN         ← FPN ノイズ不変性（物理考慮）
            + λ_relkd  × L_relational  ← Relational KD（intra-modal 構造転移）

【損失の役割分担】
    L_KD:         feats_s × feats_t^T の対角を最大化
                  → cross-modal alignment（thermal を RGB 特徴空間に引き込む）
                  → thermal↔RGB クロスモーダルマッチングに必要

    L_relational: (feats_s × feats_s^T) ≈ (feats_t × feats_t^T) を最適化
                  → intra-modal 構造の転移（Park et al., CVPR 2019）
                  → 「同じ点を2度見たとき同じ特徴が出る」Repeatability
                  → 「異なる点は異なる特徴が出る」Discriminability
                  → L_KD が直接最適化しない行列を補完する

    L_FPN:        FPN ノイズ付き→なしの特徴一致
                  → 熱画像固有の列ノイズへの不変性

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


# ---------------------------------------------------------------------------
# 7.6  L_relational: Relational KD（intra-modal 構造転移）
# ---------------------------------------------------------------------------

def relational_kd_loss(
    feats_s: Tensor,
    feats_t: Tensor,
    n_samples: int = 512,
) -> Tensor:
    """
    Relational Knowledge Distillation 損失。
    （Park et al., "Relational Knowledge Distillation", CVPR 2019）

    【役割】
        L_KD（cross-modal InfoNCE）は feats_s × feats_t^T の対角を最大化し、
        「熱画像の特徴 i が RGB の特徴 i に近づく」cross-modal alignment を学習する。

        しかしこれだけでは feats_s × feats_s^T の構造（点間の相対関係）が
        feats_t × feats_t^T に等しくなる保証がない。
        不完全な cross-modal 収束のもとでは両行列が大きく乖離し、
        thermal-thermal マッチングの Repeatability・Discriminability が低下する。

        本損失は教師の intra-modal 類似度構造（T×T^T）を
        生徒の intra-modal 構造（S×S^T）に直接転移することで、
        L_KD が補えない行列を補完する。

    【数学的根拠】
        最適化する量:
            MSE( S×S^T, T×T^T )
        S = feats_s のサブサンプリング（L2 正規化済み）
        T = feats_t のサブサンプリング（L2 正規化済み・detach 済み）

        S×S^T: 生徒の点間類似度行列（(n, n)）
        T×T^T: 教師の点間類似度行列（(n, n)）← 目標（stop_gradient）

        これを最小化すると:
            点 i, j が教師で類似 → 生徒でも類似（同一物体・同一テクスチャ）
            点 i, j が教師で非類似 → 生徒でも非類似（異なる位置）
        → Repeatability: 同一点を変換前後で見たときの特徴一致性
        → Discriminability: 異なる点の特徴の弁別性

    【L_KD との関係】
        L_KD:        feats_s × feats_t^T を最適化（cross-modal）
        L_relational: feats_s × feats_s^T を最適化（intra-modal）
        → 独立した情報を補完的に最適化する

    【stop_gradient の方向】
        T×T^T.detach() → 教師の構造を目標として固定
        S×S^T          → 学習側（生徒が近づける）
        逆方向は特徴崩壊のリスクがあるため禁止

    Args:
        feats_s:   (B, 64, H/8, W/8)  生徒特徴（L2 正規化済み）
        feats_t:   (B, 64, H/8, W/8)  教師特徴（L2 正規化済み・detach 済み）
        n_samples: サブサンプリング数（メモリ節約のため n×n 行列を制限）
                   n=512 → 512×512 行列（256K 要素）
                   n=1024 → 1024×1024 行列（1M 要素）: メモリ大
                   推奨: 256〜512

    Returns:
        loss: scalar Tensor
    """
    B, C, Hf, Wf = feats_s.shape
    HW = Hf * Wf
    n  = min(HW, n_samples)

    s_flat = feats_s.reshape(B, C, HW).permute(0, 2, 1)  # (B, HW, C)
    t_flat = feats_t.reshape(B, C, HW).permute(0, 2, 1)

    total_loss = feats_s.new_zeros(1)
    for b in range(B):
        idx = torch.randperm(HW, device=feats_s.device)[:n]

        # L2 正規化（念のため再正規化）
        s_b = F.normalize(s_flat[b, idx], dim=1)   # (n, C)
        t_b = F.normalize(t_flat[b, idx], dim=1)   # (n, C)

        # intra-modal 類似度行列 (n, n)
        # S×S^T: 生徒の点間コサイン類似度
        # T×T^T: 教師の点間コサイン類似度（目標・stop_gradient）
        S_sim = s_b @ s_b.t()   # (n, n)
        T_sim = t_b @ t_b.t()   # (n, n)

        # 教師の intra-modal 構造を生徒に転移
        # T_sim.detach() で stop_gradient を明示（T は既に detach 済みだが二重保護）
        total_loss = total_loss + F.mse_loss(S_sim, T_sim.detach())

    return total_loss / B


# ---------------------------------------------------------------------------
# 7.5b  fpn_noise_forward: FPN ノイズ画像を生成して学習側フォワードのみ行う
#        （train_kd.py のループで feats_clean を再利用するため）
# ---------------------------------------------------------------------------

def make_fpn_noise(
    img_thr: Tensor,
    sigma_min: float = 2.0,
    sigma_max: float = 8.0,
) -> Tuple[Tensor, float]:
    """
    FPN 列ノイズを生成してノイズ付き画像を返す。

    Returns:
        img_fpn  : (B, 3, H, W) ノイズ付き熱画像
        sigma_mean: float 平均ノイズ強度 (DN単位)
    """
    B, C, H, W = img_thr.shape
    sigma = (
        torch.rand(B, 1, device=img_thr.device)
        * (sigma_max - sigma_min) / 255.0
        + sigma_min / 255.0
    )
    col_noise = torch.randn(B, 1, 1, W, device=img_thr.device)         * sigma.view(B, 1, 1, 1)
    col_noise = col_noise.expand(B, C, H, W)
    img_fpn = (img_thr + col_noise).clamp(0.0, 1.0)
    sigma_mean = (sigma.mean() * 255.0).item()
    return img_fpn, sigma_mean


def fpn_invariance_loss_fast(
    feats_fpn: Tensor,
    feats_clean: Tensor,
) -> Tensor:
    """
    FPN 不変性損失（高速版）。

    呼び出し側でノイズ付きフォワードと clean フォワードを行い、
    その結果だけを受け取って損失を計算する。
    fpn_invariance_loss() と異なり、内部でフォワードしない。

    Args:
        feats_fpn   : (B, C, Hf, Wf) ノイズ付き画像の特徴（学習側・L2正規化済み）
        feats_clean : (B, C, Hf, Wf) クリーン画像の特徴（stop_gradient・L2正規化済み）

    Returns:
        loss: scalar Tensor
    """
    return F.mse_loss(feats_fpn, feats_clean.detach())


# ---------------------------------------------------------------------------
# NEW: L_spatial: KP 空間分布エントロピー損失
# ---------------------------------------------------------------------------

def spatial_entropy_loss(kpts_heatmap, scores_heatmap, image_shape=None, grid_size=8):
    """
    バッチ処理・密なヒートマップ(Dense Heatmap)完全対応版：空間分散損失
    CNNが直接出力する 4次元テンソル (B, C, H, W) を受け取り、空間エントロピーを計算します。
    """
    # XFeatの出力ヒートマップ (B, 1, H, W)
    B, C, H, W = kpts_heatmap.shape

    combined_heatmap = torch.relu(kpts_heatmap * scores_heatmap) + 1e-8
    
    # 1. キーポイント活性度と信頼度を掛け合わせて、総合的な活性度マップを作成
    # combined_heatmap = kpts_heatmap * scores_heatmap
    
    # 2. 画像全体を grid_size x grid_size のセルに分割し、各セルの平均活性度を一括取得
    # （インデックス計算や scatter_add を使わない、最も高速でバグの無いアプローチ）
    pooled_density = F.adaptive_avg_pool2d(combined_heatmap, (grid_size, grid_size)) # (B, 1, 8, 8)
    
    # 3. 確率分布を計算するために 2次元テンソル (B, 64) に平坦化
    density = pooled_density.reshape(B, -1)
    
    # 各セルの合計が1になるように正規化 (確率分布 p)
    prob = density / (density.sum(dim=1, keepdim=True) + 1e-8)
    
    # 4. エントロピーの計算: H = -sum(p * log(p))
    entropy = -torch.sum(prob * torch.log(prob + 1e-8), dim=1) # (B,)
    
    # 理論上の最大エントロピー (活性度が全64セルに完全に均等に散らばった状態)
    num_cells = grid_size * grid_size
    max_entropy = torch.log(torch.tensor(num_cells, dtype=torch.float32, device=kpts_heatmap.device))
    
    # Loss: 最大エントロピーに近づく（=画像全体に散らばる）ほど Loss が 0 になる
    loss = (max_entropy - entropy.mean()) / max_entropy
    
    return loss


# ---------------------------------------------------------------------------
# NEW: L_thermal: 温度勾配領域への KP 誘導損失
# ---------------------------------------------------------------------------

def thermal_gradient_loss(
    kpts:    torch.Tensor,
    scores:  torch.Tensor,
    img_thr: torch.Tensor,
) -> torch.Tensor:
    """
    温度勾配が大きい領域（物体輪郭・温度境界）に KP を誘導する損失。

    【根拠】
    熱画像の均一領域（路面・空）の KP は時間的に不安定で Repeatability が低い。
    温度境界は物体の物理的な縁であり時間的に安定した特徴点が存在する。
    現在の XFeat hmap は RGB パターンを学習したもので熱画像に不適切。

    実装:
        Sobel フィルタで温度勾配マップを計算し、
        KP 位置での勾配値が低いほど損失が大きくなる。
        つまり「勾配が大きい領域の KP を高スコアにする」よう学習。

    Args:
        kpts:    (N, 2) キーポイント座標 [x, y] in pixel
        scores:  (N,)   検出スコア（learn_able な量）
        img_thr: (1, 3, H, W) 熱画像テンソル [0, 1]

    Returns:
        loss: scalar Tensor（勾配大の KP が高スコアのとき小さくなる）
    """
    if len(kpts) == 0 or img_thr is None:
        return kpts.new_zeros(1).squeeze()

    B, C, H, W = img_thr.shape

    # グレースケール化 → Sobel 勾配
    with torch.no_grad():
        gray = img_thr.mean(dim=1, keepdim=True)   # (B, 1, H, W)
        sobel_x = torch.tensor(
            [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
            device=img_thr.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]],
            device=img_thr.device).view(1, 1, 3, 3)
        gx = F.conv2d(gray, sobel_x, padding=1)
        gy = F.conv2d(gray, sobel_y, padding=1)
        grad_mag = (gx ** 2 + gy ** 2).sqrt().squeeze()   # (H, W)
        # [0, 1] 正規化
        grad_mag = grad_mag / (grad_mag.max() + 1e-8)

    # KP 位置での勾配値をバイリニア補間でサンプリング
    kx = kpts[:, 0].clamp(0, W - 1).long()
    ky = kpts[:, 1].clamp(0, H - 1).long()
    grad_at_kpts = grad_mag[ky, kx]   # (N,)

    # スコア加重: 勾配が大きい KP に高スコアがつくよう学習
    # 目標: scores が高いほど grad_at_kpts も高いこと
    # = 1 - Σ(scores_normalized × grad_at_kpts) を最小化
    scores_norm = scores / (scores.sum() + 1e-8)
    expected_grad = (scores_norm * grad_at_kpts).sum()
    return 1.0 - expected_grad

# modules/training/losses_kd.py に追加
# def hybrid_thermal_gradient_loss(kpts_heatmap, scores, img_raw, bit_depth=16, tau_fixed=200.0):
#     """
#     熱画像の物理勾配を用いたKP誘導損失 (高解像度対応版)。
#     CNN出力(1/8解像度)とRaw画像(1/1解像度)のスケールを合わせて計算する。
#     """
#     # kpts_heatmap は CNN の出力 (B, 1, H/8, W/8) などの4次元テンソルを想定
#     B, C, H, W = img_raw.shape
    
#     with torch.no_grad():
#         # 1. Sobelフィルタによる物理勾配の計算 (1/1 解像度)
#         sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=img_raw.device).float().view(1,1,3,3)
#         sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=img_raw.device).float().view(1,1,3,3)
#         gx = F.conv2d(img_raw, sobel_x, padding=1)
#         gy = F.conv2d(img_raw, sobel_y, padding=1)
#         grad_mag = (gx**2 + gy**2).sqrt() # (B, 1, H, W)
        
#         # 2. 物理閾値適用: センサノイズ以下の勾配をカット
#         grad_mag = F.softplus(grad_mag - tau_fixed)
#         # バッチ・チャンネルごとに [0, 1] に正規化
#         grad_mag = grad_mag / (grad_mag.reshape(B, -1).max(dim=1)[0].reshape(B, 1, 1, 1) + 1e-8)

#         # =============================================================
#         # 🎯 修正の核心：空間解像度の同期 (1/1 -> 1/8)
#         # =============================================================
#         # CNN出力の空間サイズ (H/8, W/8) を取得
#         target_size = kpts_heatmap.shape[2:] 
        
#         # Adaptive Max Pool を用いて、物理勾配の強いエッジ情報を残しつつ 1/8 にダウンサンプリング
#         grad_mag_down = F.adaptive_max_pool2d(grad_mag, output_size=target_size)
#         # =============================================================

#     # 3. 損失の計算
#     # ネットワーク出力 (ヒートマップ) が確率分布になるように正規化
#     scores_norm = kpts_heatmap / (kpts_heatmap.reshape(B, -1).sum(dim=1).reshape(B, 1, 1, 1) + 1e-8)
    
#     # 確率(予測) × 物理勾配(正解) の期待値を計算
#     # サイズが完全に一致した状態 (B, 1, 60, 80) 同士で要素ごとの掛け算を行う
#     expected_grad = (scores_norm * grad_mag_down).reshape(B, -1).sum(dim=1).mean()
    
#     # 期待値が大きい(正解エッジに確率が集中している)ほど Loss が 0 に近づく
#     return 1.0 - expected_grad

def hybrid_thermal_gradient_loss(kpts_heatmap, scores, img_raw, bit_depth=16, tau_fixed=200.0):
    """
    熱画像の物理勾配を用いたKP誘導損失 (高解像度・完全防壁対応版)。
    CNN出力(1/8解像度)とRaw画像(1/1解像度)のスケールを合わせて計算する。
    """
    B, C, H, W = img_raw.shape
    
    with torch.no_grad():
        # 1. Sobelフィルタによる物理勾配の計算 (1/1 解像度)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=img_raw.device).float().view(1,1,3,3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=img_raw.device).float().view(1,1,3,3)
        gx = F.conv2d(img_raw, sobel_x, padding=1)
        gy = F.conv2d(img_raw, sobel_y, padding=1)
        grad_mag = (gx**2 + gy**2).sqrt() # (B, 1, H, W)
        
        # 2. 物理閾値適用: センサノイズ以下の勾配をカット (tau_fixedをデフォルト500に引き上げ)
        grad_mag = F.softplus(grad_mag - tau_fixed)
        
        # バッチ・チャンネルごとに [0, 1] に正規化
        grad_max = grad_mag.reshape(B, -1).max(dim=1)[0].reshape(B, 1, 1, 1)
        grad_mag = grad_mag / (grad_max + 1e-8)

        # 空間解像度の同期 (1/1 -> 1/8)
        target_size = kpts_heatmap.shape[2:] 
        grad_mag_down = F.adaptive_max_pool2d(grad_mag, output_size=target_size)

    # =============================================================
    # 🎯 修正の核心：数学的爆発を防ぐ安全な確率分布化
    # =============================================================
    # ReLUを適用して負のロジットを完全に排除（0以上にする）
    heatmap_pos = torch.relu(kpts_heatmap)
    
    # 空間全体の和を計算（分母）
    heatmap_sum = heatmap_pos.reshape(B, -1).sum(dim=1).reshape(B, 1, 1, 1)
    
    # 正規化（和が0の場合は均等分布にならないよう、確実に 0 に留める）
    scores_norm = heatmap_pos / (heatmap_sum + 1e-8)

    # 期待値の計算（scores_norm は合計が最大1.0、grad_mag_down は 0~1.0 なので、期待値は必ず 0~1.0 の範囲に収まる）
    expected_grad = (scores_norm * grad_mag_down).reshape(B, -1).sum(dim=1).mean()
    
    # 最終Lossの計算
    raw_loss = 1.0 - expected_grad
    
    # 最後の安全装置: Loss が [0.0, 1.0] の範囲外に飛び出すことを物理的に許さない
    safe_loss = torch.clamp(raw_loss, min=0.0, max=1.0)

    return safe_loss