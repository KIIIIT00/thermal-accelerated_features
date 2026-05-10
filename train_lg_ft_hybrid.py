"""
train_lg_ft_hybrid.py
Stage 2: LightGlue Geometry-Safe Finetuning
"""
import argparse
import cv2
import os
import sys
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
import wandb
import yaml
from tqdm import tqdm

from modules.model import XFeatModel
try:
    from lightglue import LightGlue
    import lightglue.lightglue as lg_module  # 🎯 LightGlueの内部モジュールをインポート

    # =================================================================
    # 🎯 モンキーパッチ: LightGlue内部のバグを回避する防御壁
    # =================================================================
    _orig_filter_matches = lg_module.filter_matches

    def _safe_filter_matches(scores, th):
        if th is None:
            th = -1.0  # None が来た場合、エラーを防ぐため -1.0 (全通過) に変換
        return _orig_filter_matches(scores, th)

    # 内部関数を安全なバージョンにすり替える
    lg_module.filter_matches = _safe_filter_matches

except ImportError:
    raise ImportError("LightGlue is not installed. Run: pip install git+https://github.com/cvg/LightGlue.git")

# from modules.dataset.thermal.stage23_geometry_datasets import (
#     Stage23_VIVIDDataset, Stage23_TartanRGBTDataset, Stage23_SThErEODataset
# )
from modules.dataset.thermal.sequential import (
    VividSequentialDataset,
    MS2SequentialDataset,
    SThErEOSequentialDataset,
    TartanRGBTSequentialDataset
)

from modules.training.visualization import log_stage23_geometry_to_wandb
# 🎯 高度なメトリクスと可視化モジュールのインポート
from modules.training.metrics_vis import compute_pose_metrics, log_matching_and_metrics

def compute_reprojection_gt(kpts0, kpts1, depth0, T_rel, K, th_pos=3.0):
    if kpts0.shape[0] == 0 or kpts1.shape[0] == 0:
        return torch.zeros((0, 2), dtype=torch.long, device=kpts0.device)

    K_np = K.cpu().numpy().astype(np.float64)
    T_np = T_rel.cpu().numpy().astype(np.float64)
    
    # 🌟 【最重要修正】T_rel は 1->0 なので、逆行列をとって 0->1 にする
    T_0_to_1 = np.linalg.inv(T_np)
    R_inv = T_0_to_1[:3, :3]
    t_inv = T_0_to_1[:3, 3].reshape(3, 1)
    
    depth0_np = depth0.cpu().numpy()
    H, W = depth0_np.shape
    
    valid_matches = []
    kpts0_np = kpts0.cpu().numpy()
    kpts1_np = kpts1.cpu().numpy()
    
    for i, pt0 in enumerate(kpts0_np):
        u, v = int(round(pt0[0])), int(round(pt0[1]))
        if u < 0 or u >= W or v < 0 or v >= H:
            continue
            
        window = depth0_np[max(0, v-1):min(H, v+2), max(0, u-1):min(W, u+2)]
        valid_depths = window[window > 0.1]
        if len(valid_depths) == 0: continue
            
        Z0 = np.median(valid_depths)
        
        # 1. 2D -> 3D (逆投影)
        p0_homo = np.array([[pt0[0]], [pt0[1]], [1.0]])
        P0_3d = Z0 * (np.linalg.inv(K_np) @ p0_homo)
        
        # 2. 3D -> 3D (0 -> 1への正しい座標変換)
        P1_3d = R_inv @ P0_3d + t_inv
        
        if P1_3d[2, 0] <= 0: continue # カメラ裏
            
        # 3. 3D -> 2D (再投影)
        p1_proj = K_np @ P1_3d
        u1_proj, v1_proj = p1_proj[0, 0] / p1_proj[2, 0], p1_proj[1, 0] / p1_proj[2, 0]
        
        # 4. 距離判定
        diff = kpts1_np - np.array([u1_proj, v1_proj])
        dists = np.linalg.norm(diff, axis=1)
        
        min_idx = np.argmin(dists)
        if dists[min_idx] < th_pos:
            valid_matches.append([i, min_idx])
            
    return torch.tensor(valid_matches, dtype=torch.long, device=kpts0.device)

def _prepare_image_for_cv2(img_tensor):
    """ PyTorchテンソルをOpenCVで安全に描画できる完璧な画像に変換する共通ヘルパー """
    img = img_tensor.cpu().numpy().transpose(1, 2, 0)
    
    # 1. 完璧なMin-Max正規化 (確実に 0.0 ~ 1.0 にする)
    img_min, img_max = img.min(), img.max()
    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min)
    
    img = (img * 255).astype(np.uint8)
    
    # 2. OpenCVのサイレントエラーを防ぐためのメモリ連続化 (最重要)
    img = np.ascontiguousarray(img)
    
    if img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
    return img

def save_keypoints_visualization(img_tensor, kpts, save_path, color=(0, 255, 0)):
    img = _prepare_image_for_cv2(img_tensor)
    kpts_np = kpts.cpu().numpy()
    
    h, w = img.shape[:2]
    
    # 3. 座標が正規化(0~1)されている場合の自動補正
    if len(kpts_np) > 0 and kpts_np.max() <= 2.0:
        kpts_np = kpts_np.copy()
        kpts_np[:, 0] *= w
        kpts_np[:, 1] *= h

    drawn_count = 0
    for pt in kpts_np:
        x, y = int(pt[0]), int(pt[1])
        # 画面内にある点だけを描画
        if 0 <= x < w and 0 <= y < h:
            # 視認性を爆発的に上げるため、黒いフチドリをつけてから色を塗る
            cv2.circle(img, (x, y), 5, (0, 0, 0), -1)   # 黒フチ (少し大きめ)
            cv2.circle(img, (x, y), 3, color, -1)       # 本体の色
            drawn_count += 1
            
    # ターミナルに描画結果を報告 (デバッグ用)
    print(f"✅ [DEBUG] {os.path.basename(save_path)} に {drawn_count} 個の点を描画しました。")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, img)


def save_gt_matches_visualization(img0_tensor, img1_tensor, kpts0, kpts1, gt_matches, save_path):
    img0 = _prepare_image_for_cv2(img0_tensor)
    img1 = _prepare_image_for_cv2(img1_tensor)

    h0, w0, _ = img0.shape
    h1, w1, _ = img1.shape
    canvas = np.zeros((max(h0, h1), w0 + w1, 3), dtype=np.uint8)
    canvas[:h0, :w0] = img0
    canvas[:h1, w0:w0+w1] = img1

    kpts0_np = kpts0.cpu().numpy()
    kpts1_np = kpts1.cpu().numpy()
    gt_matches_np = gt_matches.cpu().numpy()

    # 座標の自動補正
    if len(kpts0_np) > 0 and kpts0_np.max() <= 2.0:
        kpts0_np = kpts0_np.copy(); kpts0_np[:, 0] *= w0; kpts0_np[:, 1] *= h0
    if len(kpts1_np) > 0 and kpts1_np.max() <= 2.0:
        kpts1_np = kpts1_np.copy(); kpts1_np[:, 0] *= w1; kpts1_np[:, 1] *= h1

    drawn_matches = 0
    for m in gt_matches_np:
        idx0, idx1 = m[0], m[1]
        pt1 = (int(kpts0_np[idx0, 0]), int(kpts0_np[idx0, 1]))
        pt2 = (int(kpts1_np[idx1, 0]) + w0, int(kpts1_np[idx1, 1]))

        color = tuple(np.random.randint(50, 255, 3).tolist())
        # 線と点に黒フチをつけて見やすくする
        cv2.line(canvas, pt1, pt2, (0,0,0), 3)
        cv2.line(canvas, pt1, pt2, color, 1)
        cv2.circle(canvas, pt1, 4, (0,0,0), -1); cv2.circle(canvas, pt1, 2, color, -1)
        cv2.circle(canvas, pt2, 4, (0,0,0), -1); cv2.circle(canvas, pt2, 2, color, -1)
        drawn_matches += 1

    print(f"✅ [DEBUG] {os.path.basename(save_path)} に {drawn_matches} 本のGT線を引きました。")
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, canvas)


def dynamic_pad_collate(batch):
    """
    サイズの異なる画像（VIVIDとSThErEO等）を同じバッチにまとめるための関数。
    幾何学（K行列）を破壊しないよう、バッチ内の最大サイズに合わせて右下にゼロパディングする。
    """
    # 1. バッチ内の最大の高さ(H)と幅(W)を見つける
    raw_max_h = max(max(item['image0'].shape[1], item['image1'].shape[1]) for item in batch)
    raw_max_w = max(max(item['image0'].shape[2], item['image1'].shape[2]) for item in batch)

    # 🌟 【最重要修正】CNNのための 32の倍数アライメント
    align = 32
    max_h = math.ceil(raw_max_h / align) * align
    max_w = math.ceil(raw_max_w / align) * align

    collated = {key: [] for key in batch[0].keys()}

    # 2. 各データに対して右・下にパディングを適用
    for item in batch:
        for key, val in item.items():
            if key in ['image0', 'image1']:
                pad_h = max_h - val.shape[1]
                pad_w = max_w - val.shape[2]
                if pad_h > 0 or pad_w > 0:
                    val = F.pad(val, (0, pad_w, 0, pad_h), mode='constant', value=0.0)
                collated[key].append(val)
            else:
                collated[key].append(val)

    # 3. リストにまとめたテンソルをスタック
    for key in collated.keys():
        if isinstance(collated[key][0], torch.Tensor):
            collated[key] = torch.stack(collated[key], dim=0)

    return collated

def save_geometry_dataset_checks(datasets, output_root, num_samples=5):
    """
    Stage 2/3 用: 幾何学学習データセットの連続フレームペアを取り出し、
    ローカルに保存して目視確認（パディングや時系列のズレがないか）を可能にする
    """
    import torchvision
    debug_dir = os.path.join(output_root, "dataset_checks")
    os.makedirs(debug_dir, exist_ok=True)
    
    print(f"\n🔍 Checking Geometry Datasets... saving {num_samples} sequential pairs per dataset to {debug_dir}")
    
    for idx, ds in enumerate(datasets):
        for i in range(min(num_samples, len(ds))):
            data = ds[i]
            img0 = data['image0'] # 時刻 t の画像 (3, H, W)
            img1 = data['image1'] # 時刻 t+stride の画像 (3, H, W)
            ds_name = data.get('dataset_name', f'dataset_{idx}')
            
            # 時系列フレームを横に結合して比較しやすくする
            comparison = torch.cat([img0, img1], dim=2)
            fname = f"sample_{ds_name}_seq_{i:03d}.png"
            
            # 画像として保存
            torchvision.utils.save_image(comparison, os.path.join(debug_dir, fname))
            
    print(f"✅ Geometry Dataset check completed. Images saved in {debug_dir}\n")

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, default='config_master.yaml', help='Path to config file')

    parser.add_argument('--xfeat_weights', type=str, required=True)
    parser.add_argument('--lg_weights', type=str, default=None, help='Path to pre-trained LightGlue weights for XFeat')
    parser.add_argument('--vivid_root', type=str, default='datasets/vivid')
    parser.add_argument('--tartanrgbt_root', type=str, default='datasets/tartanRGBT')
    parser.add_argument('--sthereo_root', type=str, default='datasets/sthereo', help='Path to SThErEO dataset')
    parser.add_argument('--ms2_root', type=str, default='', help='Path to MS2 dataset')
    parser.add_argument('--freiburg_root', type=str, default='', help='Path to Freiburg dataset')
    parser.add_argument('--output', type=str, default='checkpoints/stage2_lg')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--max_keypoints', type=int, default=None, help='Maximum number of keypoints to extract per image')
    parser.add_argument('--n_steps', type=int, default=10000)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--wandb_run_name', type=str, default='stage2_lg')
    parser.add_argument('--no_wandb', action='store_true')
    parser.add_argument('--save_debug_images', action='store_true', help='学習前に各データセットの時系列サンプル画像を保存して終了する')
    return parser.parse_args()

# ==========================================
# 幾何学 & Loss ユーティリティ
# ==========================================
# def compute_epipolar_gt(kpts0, kpts1, T_rel, K, th_pos=3.0):
#     N, M = kpts0.shape[0], kpts1.shape[0]
#     if N == 0 or M == 0: return torch.zeros((0, 2), dtype=torch.long, device=kpts0.device)

#     K_np = K.cpu().numpy().astype(np.float64)
#     T_np = T_rel.cpu().numpy().astype(np.float64)
#     R, t = T_np[:3, :3], T_np[:3, 3]
#     t_cross = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
#     E = t_cross @ R
#     Ki = np.linalg.inv(K_np)
#     F_mat = (Ki.T @ E @ Ki).astype(np.float32)

#     p0h = torch.cat([kpts0, torch.ones_like(kpts0[:, :1])], dim=-1).cpu().numpy()
#     p1h = torch.cat([kpts1, torch.ones_like(kpts1[:, :1])], dim=-1).cpu().numpy()

#     lines = (F_mat @ p0h.T).T
#     num = np.abs(p1h @ lines.T)
#     denom = np.sqrt(lines[:, 0]**2 + lines[:, 1]**2)[None] + 1e-8
#     dist = (num / denom).T

#     min_idx = dist.argmin(axis=1)
#     min_dist = dist[np.arange(N), min_idx]

#     matches = [[i, min_idx[i]] for i in range(N) if min_dist[i] < th_pos]
#     return torch.tensor(matches, dtype=torch.long, device=kpts0.device)

def compute_epipolar_gt(kpts0, kpts1, T_rel, K, th_pos=3.0):
    """
    対称エピポーラ距離（Symmetric Epipolar Distance）と
    相互近傍チェック（Mutual Nearest Neighbor）を用いた厳密なGT生成。
    """
    if kpts0.dim() != 2 or kpts1.dim() != 2:
        return torch.zeros((0, 2), dtype=torch.long, device=kpts0.device)
        
    N, M = kpts0.shape[0], kpts1.shape[0]
    if N < 1 or M < 1: 
        return torch.zeros((0, 2), dtype=torch.long, device=kpts0.device)
    
    if N > 4096 or M > 4096:
        N, M = min(N, 4096), min(M, 4096)
        kpts0 = kpts0[:N]
        kpts1 = kpts1[:M]

    K_np = K.cpu().numpy().astype(np.float64)
    T_np = T_rel.cpu().numpy().astype(np.float64)
    R, t = T_np[:3, :3], T_np[:3, 3]

    # =====================================================================
    # 🌟 [修正] ベースライン（カメラの移動距離）の厳格なガード
    # 自動運転データセットでは、10cm(0.1m)未満の移動はIMU/GPSノイズの可能性が高く、
    # F行列の計算が破綻するため、強制的にマッチングなし(0個)とする。
    # =====================================================================
    baseline_length = np.linalg.norm(t)
    if baseline_length < 0.1:
        # デバッグ用に、なぜGTが0になったかをターミナルに小さく報告（不要なら消してOKです）
        # print(f"  [Info] Skipped Epipolar GT: Baseline too short ({baseline_length:.4f}m < 0.1m)")
        return torch.zeros((0, 2), dtype=torch.long, device=kpts0.device)

    # t_cross = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    # E = t_cross @ R
    # Ki = np.linalg.inv(K_np)
    # F_mat = (Ki.T @ E @ Ki).astype(np.float32)

    # ones0 = torch.ones((N, 1), dtype=kpts0.dtype, device=kpts0.device)
    # ones1 = torch.ones((M, 1), dtype=kpts1.dtype, device=kpts1.device)
    
    # p0h = torch.cat([kpts0, ones0], dim=-1).cpu().numpy()
    # p1h = torch.cat([kpts1, ones1], dim=-1).cpu().numpy()

    # # 1. 画像0から画像1へのエピポーラ距離 (dist0to1)
    # lines1 = (F_mat @ p0h.T).T # 画像1上に引かれるエピポーラ線 (N, 3)
    # num1 = np.abs(p1h @ lines1.T) # (M, N)
    # denom1 = np.sqrt(lines1[:, 0]**2 + lines1[:, 1]**2)[None] + 1e-8 # (1, N)
    # dist0to1 = (num1 / denom1).T # (N, M) の距離行列

    # # 2. 画像1から画像0へのエピポーラ距離 (dist1to0)
    # lines0 = (F_mat.T @ p1h.T).T # 画像0上に引かれるエピポーラ線 (M, 3)
    # num0 = np.abs(p0h @ lines0.T) # (N, M)
    # denom0 = np.sqrt(lines0[:, 0]**2 + lines0[:, 1]**2)[None] + 1e-8 # (1, M)
    # dist1to0 = (num0 / denom0) # (N, M) の距離行列
    t_cross = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    E = t_cross @ R
    Ki = np.linalg.inv(K_np)
    F_mat = (Ki.T @ E @ Ki).astype(np.float32)

    ones0 = torch.ones((N, 1), dtype=kpts0.dtype, device=kpts0.device)
    ones1 = torch.ones((M, 1), dtype=kpts1.dtype, device=kpts1.device)
    p0h = torch.cat([kpts0, ones0], dim=-1).cpu().numpy()
    p1h = torch.cat([kpts1, ones1], dim=-1).cpu().numpy()

    # 🌟 [修正1] lines1 の計算には F_mat.T を使用する
    lines1 = (F_mat @ p0h.T).T 
    num1 = np.abs(p1h @ lines1.T) 
    denom1 = np.sqrt(lines1[:, 0]**2 + lines1[:, 1]**2)[None] + 1e-8 
    dist0to1 = (num1 / denom1).T 

    # 2. 画像1から画像0へのエピポーラ距離 (画像0の線 = F^T * p1)
    lines0 = (F_mat.T @ p1h.T).T 
    num0 = np.abs(p0h @ lines0.T) 
    denom0 = np.sqrt(lines0[:, 0]**2 + lines0[:, 1]**2)[None] + 1e-8 
    dist1to0 = (num0 / denom0)

    # 3. 対称エピポーラ距離（両方向の平均）
    dist = (dist0to1 + dist1to0) / 2.0 # (N, M)

    # 4. 相互近傍チェック (Mutual Nearest Neighbor)
    min_idx_0to1 = dist.argmin(axis=1) # 画像0から見て一番近い画像1のインデックス (N,)
    min_idx_1to0 = dist.argmin(axis=0) # 画像1から見て一番近い画像0のインデックス (M,)

    matches = []
    for i in range(N):
        j = min_idx_0to1[i]
        # 相互チェック: お互いが一番近い場合のみ
        if min_idx_1to0[j] == i:
            # エピポーラ距離が閾値未満か確認
            if dist[i, j] < th_pos:
                matches.append([i, j])
                
    return torch.tensor(matches, dtype=torch.long, device=kpts0.device)

# def compute_masked_match_loss(log_assignment, kpts0_list, kpts1_list, gt_matches_list, orig_size0, orig_size1):
#     B = len(kpts0_list)
#     total_loss, valid_b = 0.0, 0
    
#     for b in range(B):
#         w0, h0 = orig_size0[b][0].item(), orig_size0[b][1].item()
#         w1, h1 = orig_size1[b][0].item(), orig_size1[b][1].item()
        
#         valid_m0 = (kpts0_list[b][:, 0] < w0) & (kpts0_list[b][:, 1] < h0)
#         valid_m1 = (kpts1_list[b][:, 0] < w1) & (kpts1_list[b][:, 1] < h1)
        
#         matches = gt_matches_list[b]
#         if len(matches) == 0: continue
            
#         m0_idx, m1_idx = matches[:, 0], matches[:, 1]
#         valid_match_mask = valid_m0[m0_idx] & valid_m1[m1_idx]
#         f_m0, f_m1 = m0_idx[valid_match_mask], m1_idx[valid_match_mask]
        
#         if len(f_m0) == 0: continue
            
#         loss = -log_assignment[b, f_m0, f_m1].mean()
#         total_loss += loss
#         valid_b += 1
        
#     return total_loss / valid_b if valid_b > 0 else torch.tensor(0.0, device=log_assignment.device, requires_grad=True)
# def compute_masked_match_loss(log_assignment, kpts0_list, kpts1_list, gt_matches_list, orig_size0, orig_size1):
#     B = len(kpts0_list)
#     total_loss, valid_b = 0.0, 0
    
#     for b in range(B):
#         # log_assignment はダストビンを含む [B, M+1, N+1] を想定
#         M_plus_1, N_plus_1 = log_assignment.shape[1], log_assignment.shape[2]
#         M, N = M_plus_1 - 1, N_plus_1 - 1 # 実際の点の数
        
#         w0, h0 = orig_size0[b][0].item(), orig_size0[b][1].item()
#         w1, h1 = orig_size1[b][0].item(), orig_size1[b][1].item()
        
#         valid_m0 = (kpts0_list[b][:, 0] < w0) & (kpts0_list[b][:, 1] < h0)
#         valid_m1 = (kpts1_list[b][:, 0] < w1) & (kpts1_list[b][:, 1] < h1)
        
#         matches = gt_matches_list[b]
#         if len(matches) == 0: 
#             # 🌟 [重要] GTが0件の場合、すべての点をダストビンに捨てる学習をする
#             loss = (-log_assignment[b, :M, N].mean() - log_assignment[b, M, :N].mean()) / 2.0
#             total_loss += loss
#             valid_b += 1
#             continue
            
#         m0_idx, m1_idx = matches[:, 0], matches[:, 1]
#         valid_match_mask = valid_m0[m0_idx] & valid_m1[m1_idx]
#         f_m0, f_m1 = m0_idx[valid_match_mask], m1_idx[valid_match_mask]
        
#         if len(f_m0) == 0: continue
            
#         # 1. Positive Loss: 正解ペアの確率を最大化
#         pos_loss = -log_assignment[b, f_m0, f_m1].mean()
        
#         # 2. Negative Loss (Dustbin): マッチしなかった点をゴミ箱に捨てる確率を最大化
#         # 画像0の Unmatched を N (画像1のゴミ箱) へ
#         unmatched_m0 = torch.ones(M, dtype=torch.bool, device=log_assignment.device)
#         unmatched_m0[f_m0] = False
#         neg_loss0 = -log_assignment[b, unmatched_m0, N].mean() if unmatched_m0.any() else 0.0
        
#         # 画像1の Unmatched を M (画像0のゴミ箱) へ
#         unmatched_m1 = torch.ones(N, dtype=torch.bool, device=log_assignment.device)
#         unmatched_m1[f_m1] = False
#         neg_loss1 = -log_assignment[b, M, unmatched_m1].mean() if unmatched_m1.any() else 0.0
        
#         # 総合Loss (正解と不正解をバランスよく学習)
#         loss = pos_loss + (neg_loss0 + neg_loss1) / 2.0
        
#         total_loss += loss
#         valid_b += 1
        
#     return total_loss / valid_b if valid_b > 0 else torch.tensor(0.0, device=log_assignment.device, requires_grad=True)

# def compute_masked_match_loss(log_assignment, kpts0_list, kpts1_list, gt_matches_list, orig_size0, orig_size1):
#     B = len(kpts0_list)
#     total_loss, valid_b = 0.0, 0
    
#     for b in range(B):
#         if log_assignment is None or log_assignment.numel() == 0:
#             continue
        
#         if log_assignment.dim() == 2:
#             log_assignment = log_assignment.unsqueeze(0)
#         elif log_assignment.dim() < 3:
#             # 1次元や0次元に崩壊している場合はどうしようもないのでスキップ
#             print(f"\n⚠️ [WARNING] Broken tensor shape detected: {log_assignment.shape}. Skipping batch.")
#             continue

#         # 🌟 真の点の数を取得
#         M_true = kpts0_list[b].shape[0]
#         N_true = kpts1_list[b].shape[0]
        
#         # 🌟 動的判定: 行列のサイズが点の数より大きければ、ダストビンが存在するとみなす
#         has_dustbin = (log_assignment.shape[1] > M_true) and (log_assignment.shape[2] > N_true)
        
#         if has_dustbin:
#             M, N = log_assignment.shape[1] - 1, log_assignment.shape[2] - 1
#         else:
#             M, N = log_assignment.shape[1], log_assignment.shape[2]
            
#         w0, h0 = orig_size0[b][0].item(), orig_size0[b][1].item()
#         w1, h1 = orig_size1[b][0].item(), orig_size1[b][1].item()
        
#         valid_m0 = (kpts0_list[b][:, 0] < w0) & (kpts0_list[b][:, 1] < h0)
#         valid_m1 = (kpts1_list[b][:, 0] < w1) & (kpts1_list[b][:, 1] < h1)
        
#         matches = gt_matches_list[b]
        
#         # 🎯 GTが0件の場合の Negative Learning
#         if len(matches) == 0: 
#             if has_dustbin:
#                 # ダストビンがある場合: ゴミ箱への割り当て確率を最大化
#                 loss = (-log_assignment[b, :M, N].mean() - log_assignment[b, M, :N].mean()) / 2.0
#             else:
#                 # ダストビンが無い場合: 全てのマッチング確率をゼロに押し下げる (expで確率に戻してペナルティ)
#                 loss = log_assignment[b].exp().mean()
                
#             total_loss += loss
#             valid_b += 1
#             continue
            
#         m0_idx, m1_idx = matches[:, 0], matches[:, 1]
#         valid_match_mask = valid_m0[m0_idx] & valid_m1[m1_idx]
#         f_m0, f_m1 = m0_idx[valid_match_mask], m1_idx[valid_match_mask]
        
#         if len(f_m0) == 0: continue
            
#         # 1. Positive Loss (正解ペア)
#         pos_loss = -log_assignment[b, f_m0, f_m1].mean()
        
#         # 2. Negative Loss
#         if has_dustbin:
#             unmatched_m0 = torch.ones(M, dtype=torch.bool, device=log_assignment.device)
#             unmatched_m0[f_m0] = False
#             neg_loss0 = -log_assignment[b, unmatched_m0, N].mean() if unmatched_m0.any() else 0.0
            
#             unmatched_m1 = torch.ones(N, dtype=torch.bool, device=log_assignment.device)
#             unmatched_m1[f_m1] = False
#             neg_loss1 = -log_assignment[b, M, unmatched_m1].mean() if unmatched_m1.any() else 0.0
            
#             loss = pos_loss + (neg_loss0 + neg_loss1) / 2.0
#         else:
#             # ダストビンが無い場合は、Positive Loss だけで学習を進める
#             loss = pos_loss
        
#         total_loss += loss
#         valid_b += 1

#         if valid_b > 0:
#             return total_loss / valid_b
#         else:
#             None
        
    # return total_loss / valid_b if valid_b > 0 else torch.tensor(0.0, device=log_assignment.device, requires_grad=True)

def compute_masked_match_loss(log_assignment, kpts0_list, kpts1_list, gt_matches_list, orig_size0, orig_size1):
    """
    LightGlueのプルーニングに対応し、ダストビン学習を含む NLL Loss を計算する堅牢な関数。
    
    Args:
        log_assignment: [B, M+1, N+1] または [B, M, N] の対数割り当て行列
        kpts0_list: 画像0の特徴点リスト (長さ B)
        kpts1_list: 画像1の特徴点リスト (長さ B)
        gt_matches_list: GTマッチのリスト (長さ B)
        orig_size0: 画像0の元サイズ [B, 2]
        orig_size1: 画像1の元サイズ [B, 2]
        
    Returns:
        loss (torch.Tensor): バッチ内の平均損失。有効なバッチがない場合は None。
    """
    B = len(kpts0_list)
    total_loss = 0.0
    valid_b = 0
    
    # 🌟 [追加] スキップされた理由を数えるカウンター
    skip_reasons = {"empty_tensor": 0, "bad_dim": 0, "all_pruned": 0}
    
    for b in range(B):
        if log_assignment is None or log_assignment.numel() == 0:
            skip_reasons["empty_tensor"] += 1
            continue
            
        if log_assignment.dim() == 2:
            log_assignment_b = log_assignment.unsqueeze(0)
        elif log_assignment.dim() == 3:
            log_assignment_b = log_assignment[[b]]
        else:
            skip_reasons["bad_dim"] += 1
            continue

        dim1, dim2 = log_assignment_b.shape[1], log_assignment_b.shape[2]
        has_dustbin = True 
        M = dim1 - 1 if has_dustbin else dim1
        N = dim2 - 1 if has_dustbin else dim2

        if M == 0 or N == 0:
            skip_reasons["all_pruned"] += 1  # 🌟 カウント追加
            continue
        # =========================================================

        w0, h0 = orig_size0[b][0].item(), orig_size0[b][1].item()
        w1, h1 = orig_size1[b][0].item(), orig_size1[b][1].item()
        
        valid_m0 = (kpts0_list[b][:, 0] < w0) & (kpts0_list[b][:, 1] < h0)
        valid_m1 = (kpts1_list[b][:, 0] < w1) & (kpts1_list[b][:, 1] < h1)
        matches = gt_matches_list[b]
        
        # 🎯 経路A: GT=0 の場合
        if len(matches) == 0: 
            if has_dustbin:
                loss = (-log_assignment_b[0, :M, N].mean() - log_assignment_b[0, M, :N].mean()) / 2.0
            else:
                loss = log_assignment_b[0].exp().mean()
            total_loss += loss
            valid_b += 1
            continue
            
        # 🎯 経路B: GTが存在する場合
        m0_idx, m1_idx = matches[:, 0], matches[:, 1]
        valid_match_mask = valid_m0[m0_idx] & valid_m1[m1_idx]
        f_m0, f_m1 = m0_idx[valid_match_mask], m1_idx[valid_match_mask]
        
        if len(f_m0) == 0: continue
            
        valid_bounds = (f_m0 < M) & (f_m1 < N)
        f_m0 = f_m0[valid_bounds]
        f_m1 = f_m1[valid_bounds]
        
        if len(f_m0) == 0:
            if has_dustbin:
                loss = (-log_assignment_b[0, :M, N].mean() - log_assignment_b[0, M, :N].mean()) / 2.0
                total_loss += loss
                valid_b += 1
            continue

        # 1. Positive Loss
        pos_loss = -log_assignment_b[0, f_m0, f_m1].mean()
        
        # 2. Negative Loss
        if has_dustbin:
            unmatched_m0 = torch.ones(M, dtype=torch.bool, device=log_assignment_b.device)
            unmatched_m0[f_m0] = False
            # 🌟 [修正1] まず :M で1024行をスライスしてから、1024のマスクを適用する
            if unmatched_m0.any():
                neg_loss0 = -log_assignment_b[0, :M, N][unmatched_m0].mean()
            else:
                neg_loss0 = torch.tensor(0.0, device=log_assignment_b.device)
            
            unmatched_m1 = torch.ones(N, dtype=torch.bool, device=log_assignment_b.device)
            unmatched_m1[f_m1] = False
            # 🌟 [修正2] まず :N で1024列をスライスしてから、1024のマスクを適用する
            if unmatched_m1.any():
                neg_loss1 = -log_assignment_b[0, M, :N][unmatched_m1].mean()
            else:
                neg_loss1 = torch.tensor(0.0, device=log_assignment_b.device)
            
            loss = pos_loss + (neg_loss0 + neg_loss1) / 2.0
        else:
            loss = pos_loss
         
        total_loss += loss
        valid_b += 1
        
    if valid_b == 0 and B > 0:
        print(f"\n⚠️ [DEBUG] Batch completely skipped! Reasons: {skip_reasons}")
        
    if valid_b > 0:
        return total_loss / valid_b
    else:
        return None

# ==========================================
# メイン学習ループ
# ==========================================
# def main():
#     args = get_args()
#     os.makedirs(args.output, exist_ok=True)
#     device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
#     wandb.init(project="thermal-xfeat-hybrid", name=args.wandb_run_name, mode="disabled" if args.no_wandb else "online")

#     # モデル
#     xfeat = XFeatModel().to(device).eval()
#     xfeat.load_state_dict(torch.load(args.xfeat_weights, map_location=device))
#     for p in xfeat.parameters(): p.requires_grad = False

#     # lg = LightGlue(features=None, input_dim=64, filter_threshold=-1.0).to(device).train()
#     lg = LightGlue(features=None, input_dim=64, filter_threshold=None, depth_confidence=-1.0).to(device).train()
#     optimizer = torch.optim.Adam(lg.parameters(), lr=args.lr)

#     # データセット
#     datasets = []
#     if args.vivid_root: 
#         datasets.append(Stage23_VIVIDDataset(args.vivid_root, stride=1))
#         print(f"[Dataset] Added VIVID from {args.vivid_root}")
    
#     if args.tartanrgbt_root: datasets.append(Stage23_TartanRGBTDataset(args.tartanrgbt_root, stride=1))
#     if args.sthereo_root:
#         from modules.dataset.thermal.stage23_geometry_datasets import Stage23_SThErEODataset
#         sthereo_ds = Stage23_SThErEODataset(data_root=args.sthereo_root, stride=1, split='train')
#         datasets.append(sthereo_ds)
#         print(f"[Dataset] Added SThErEO from {args.sthereo_root}")

#     # 🌟 [追加] MS2 のロード
#     if args.ms2_root:
#         from modules.dataset.thermal.ms2 import MS2Dataset # クラス名は環境に合わせてください
#         ms2_ds = MS2Dataset(data_root=args.ms2_root, split='train')
#         datasets.append(ms2_ds)
#         print(f"[Dataset] Added MS2 from {args.ms2_root}")
    
#     if len(datasets) == 0:
#         raise RuntimeError("❌ 学習データセットが見つかりません。")
    
#     from torch.utils.data import WeightedRandomSampler
#     valid_datasets = []
#     for ds in datasets:
#         if len(ds) > 0:
#             valid_datasets.append(ds)
#         else:
#             # どのデータセットが 0 なのかを表示して警告する
#             print(f"⚠️ Warning: Dataset {type(ds).__name__} is empty and will be skipped.")

#     if len(valid_datasets) == 0:
#         raise RuntimeError("❌ 全てのデータセットが空です。パスやアノテーションを確認してください。")

#     # バランスを考慮したサンプラーの構築
#     full_dataset = ConcatDataset(valid_datasets)
#     weights = []
#     for ds in valid_datasets:
#         ds_len = len(ds)
#         # ここで ds_len > 0 が保証されるため、ZeroDivisionError は発生しません
#         weight = 1.0 / ds_len
#         weights.extend([weight] * ds_len)
    
#     weights = torch.DoubleTensor(weights)

#     sampler = WeightedRandomSampler(weights, num_samples=len(full_dataset), replacement=True)

#     # 🌟 DataLoader にサンプラーを渡す (shuffle=True は sampler と併用できないため消す)
#     loader = DataLoader(
#         full_dataset, 
#         batch_size=args.batch_size, 
#         sampler=sampler, 
#         drop_last=True,
#         num_workers=4, # 読み込み高速化のため推奨
#         collate_fn=dynamic_pad_collate
#     )
    
    

#     # 🎯 学習前のデータセット・サンプル保存と安全な終了
#     if args.save_debug_images:
#         save_geometry_dataset_checks(datasets, args.output, num_samples=5)
#         print("🛑 連続フレーム画像の保存が完了しました。目視確認のため、ここでプログラムを終了します。")
#         sys.exit(0)

#     # loader = DataLoader(ConcatDataset(datasets), batch_size=args.batch_size, shuffle=True, drop_last=True)
#     loader_iter = iter(loader)

#     best_loss = float('inf')
#     interval_loss_sum = 0.0  
#     eval_interval = 500      

#     # 🎯 メトリクス蓄積用リスト
#     running_pose_errors = []
#     running_precisions = []

#     pbar = tqdm(range(args.n_steps))
#     for step in pbar:
#         try: batch = next(loader_iter)
#         except StopIteration:
#             loader_iter = iter(loader)
#             batch = next(loader_iter)

#         img0, img1 = batch['image0'].to(device), batch['image1'].to(device)
#         T_rel, K = batch['T_rel'].to(device), batch['K'].to(device)
#         orig0, orig1 = batch['orig_size0'], batch['orig_size1']

#         optimizer.zero_grad()
        
#         with torch.no_grad():
#             f0_map, kpts0, sc0 = xfeat(img0)
#             f1_map, kpts1, sc1 = xfeat(img1)
            
#             k0_list, k1_list, gt_list = [], [], []
#             f0_list, f1_list, sc0_list, sc1_list = [], [], [], []

#             max_k = args.max_keypoints
#             print(f"[DEBUG] max_k: {max_k}")
            
#             for b in range(img0.shape[0]):
#                 k0_list.append(kpts0[b].reshape(-1, 2)[:max_k])
#                 k1_list.append(kpts1[b].reshape(-1, 2)[:max_k])
#                 f0_list.append(f0_map[b].reshape(-1, 64)[:max_k])
#                 f1_list.append(f1_map[b].reshape(-1, 64)[:max_k])
#                 sc0_list.append(sc0[b].reshape(-1)[:max_k])
#                 sc1_list.append(sc1[b].reshape(-1)[:max_k])
#                 gt_list.append(compute_epipolar_gt(k0_list[-1], k1_list[-1], T_rel[b], K[b]))
#                 gt_matches = compute_epipolar_gt(k0_list[-1], k1_list[-1], T_rel[b], K[b], th_pos=10.0)

#                 if b == 0:
#                     num_k0 = k0_list[-1].shape[0]
#                     num_k1 = k1_list[-1].shape[0]
#                     t_norm = torch.norm(T_rel[b][:3, 3]).item()
#                     print(f"\n[DEBUG Step {step}] Kpts0: {num_k0}, Kpts1: {num_k1} | T_norm: {t_norm:.3f}m | GT Matches: {len(gt_matches)}")

#         # ====================================================================
#         # 🌟 マイクロバッチング: LightGlueのバグを回避しつつバッチ処理を再現する
#         # ====================================================================
#         total_loss = 0.0
#         valid_b = 0
#         matches_b0 = None
#         pred = {} 
        
#         for b in range(img0.shape[0]):
#             if len(gt_list[b]) == 0: 
#                 continue

#             # 🌟 [修正1] LightGlueへの入力に必ず image_size を含める！
#             # orig0[b], orig1[b] は [width, height] のテンソルである必要があります
#             lg_input_b = {
#                 'image0': {
#                     'keypoints': k0_list[b].unsqueeze(0),
#                     'descriptors': f0_list[b].unsqueeze(0),
#                     'image_size': orig0[b].unsqueeze(0) # 👈 必須！空間認識を正常化
#                 },
#                 'image1': {
#                     'keypoints': k1_list[b].unsqueeze(0),
#                     'descriptors': f1_list[b].unsqueeze(0),
#                     'image_size': orig1[b].unsqueeze(0) # 👈 必須！
#                 }
#             }
            
#             # 🌟 [確認] モデルが確実にTrainモードであることを保証
#             lg.train() 
#             pred_b = lg(lg_input_b)
            
#             # 🌟 [修正2] ダストビンを含む完全な log_assignment を取得
#             # trainモードなら、各レイヤーの log_assignment がリストで返る仕様が一般的
#             log_assignment = pred_b.get('log_assignment', None)
            
#             if log_assignment is None:
#                 pbar.set_postfix({'Status': 'log_assignment missing'})
#                 continue

#             if isinstance(log_assignment, (list, tuple)):
#                 # 最終レイヤーの log_assignment を使用
#                 log_assignment = log_assignment[-1] 
            
#             # log_assignment は通常 [1, M+1, N+1] の形状
#             # これを切り刻まずに、そのまま loss 計算関数に渡す
            
#             loss_b = compute_masked_match_loss(
#                 log_assignment,      # 👈 ダストビンを含む行列
#                 [k0_list[b]], 
#                 [k1_list[b]], 
#                 [gt_list[b]],        # 相互チェック済みの高品質なGT
#                 [orig0[b]], 
#                 [orig1[b]]
#             )
            
#             total_loss += loss_b
#             valid_b += 1
            
#             if b == 0:
#                 pred = pred_b
                
#         # --- バッチ内の全処理が終了 ---
        
#         if valid_b == 0:
#             # 🌟 [修正] なぜスキップされたのか理由を明確にする
#             pbar.set_postfix({'Status': 'Skipped (valid_b=0)'})
#             continue 
            
#         loss = total_loss / valid_b
        
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(lg.parameters(), 1.0)
#         optimizer.step()

#         interval_loss_sum += loss.item()
        
#         # 👇 ついにここに到達し、MatchLoss が表示される！
#         pbar.set_postfix({'MatchLoss': f"{loss.item():.4f}"})
#         # total_loss = 0.0
#         # valid_b = 0
#         # matches_b0 = None
#         # pred = {} 
        
#         # for b in range(img0.shape[0]):
#         #     if len(gt_list[b]) == 0: 
#         #         continue
            
#         #     # 常に「バッチサイズ1」としてLightGlueに入力
#         #     lg_input_b = {
#         #         'image0': {
#         #             'keypoints': k0_list[b].unsqueeze(0),
#         #             'descriptors': f0_list[b].unsqueeze(0),
#         #             'keypoint_scores': sc0_list[b].unsqueeze(0)
#         #         },
#         #         'image1': {
#         #             'keypoints': k1_list[b].unsqueeze(0),
#         #             'descriptors': f1_list[b].unsqueeze(0),
#         #             'keypoint_scores': sc1_list[b].unsqueeze(0)
#         #         }
#         #     }
            
#         #     pred_b = lg(lg_input_b)
            
#         #     # scores_matrix = pred_b['scores']
            
#         #     # if isinstance(scores_matrix, (list, tuple)):
#         #     #     scores_matrix = scores_matrix[-1]
            
#         #     # # 2. テンソルであることを確認
#         #     # if isinstance(scores_matrix, torch.Tensor):
#         #     #     # LightGlueがスコアを1次元(例: [M*N] や空)にしてしまった場合の緊急回避
#         #     #     # 本来 M=512, N=512 なので、再構成を試みる
#         #     #     m_pts = k0_list[b].shape[0]
#         #     #     n_pts = k1_list[b].shape[0]
                
#         #     #     # もし次元数が足りなければ、強制的に [1, M, N] にReshape
#         #     #     if scores_matrix.dim() < 3:
#         #     #         try:
#         #     #             # まず [M, N] の2次元に整形
#         #     #             scores_matrix = scores_matrix.reshape(m_pts, n_pts)
#         #     #             # 次にバッチ次元を足して [1, M, N] にする
#         #     #             scores_matrix = scores_matrix.unsqueeze(0)
#         #     #         except RuntimeError:
#         #     #             # 要素数が合わなくてReshapeに失敗した場合(空テンソルなど)は、このバッチを諦める
#         #     #             continue
#         #     # else:
#         #     #     # テンソルですらない異常データが来たらスキップ
#         #     #     continue

#         #     scores_matrix = pred_b['scores']
#         #     if isinstance(scores_matrix, (list, tuple)):
#         #         scores_matrix = scores_matrix[-1]

#         #     if isinstance(scores_matrix, torch.Tensor):
#         #         # バッチ次元を一時的に消して [H, W] にする
#         #         s = scores_matrix.squeeze() 
#         #         m_pts = k0_list[b].shape[0]
#         #         n_pts = k1_list[b].shape[0]

#         #         # 🌟 [修正] どのような形状でも (m_pts, n_pts) に強制的に切り出す
#         #         # ダストビン(M+1次元目など)が含まれていても、最初の m_pts 行、n_pts 列だけを使う
#         #         try:
#         #             # もし1次元に潰れていたらまず reshape を試みる
#         #             if s.dim() == 1:
#         #                 # 割り切れるか確認せず、必要な分だけスライスして reshape
#         #                 scores_final = s[:m_pts * n_pts].reshape(m_pts, n_pts)
#         #             else:
#         #                 # 2次元以上なら、左上から必要なサイズだけを切り出す
#         #                 scores_final = s[:m_pts, :n_pts]
                    
#         #             # 損失関数用に [1, M, N] の形に戻す
#         #             scores_matrix = scores_final.unsqueeze(0)
#         #         except Exception as e:
#         #             # それでもダメな場合のみスキップ
#         #             pbar.set_postfix({'Status': f'Shape Error: {s.shape}'})
#         #             continue
#         #     else:
#         #         continue

#         #     # 完璧な3次元テンソル [1, M, N] をLoss計算関数に渡す
#         #     loss_b = compute_masked_match_loss(
#         #         scores_matrix, 
#         #         [k0_list[b]], [k1_list[b]], [gt_list[b]], 
#         #         [orig0[b]], [orig1[b]]
#         #     )
#         #     total_loss += loss_b
#         #     valid_b += 1

#         #     gt_matches = compute_epipolar_gt(k0_list[-1], k1_list[-1], T_rel[b], K[b], th_pos=10.0)
#         #     gt_list.append(gt_matches)

#         #     # 🌟 [ハック2] バッチの先頭(b=0)だけ、中身の数値をターミナルに表示させる
#         #     if b == 0:
#         #         num_k0 = k0_list[-1].shape[0]
#         #         num_k1 = k1_list[-1].shape[0]
#         #         t_norm = torch.norm(T_rel[b][:3, 3]).item()
#         #         print(f"\n[DEBUG Step {step}] Kpts0: {num_k0}, Kpts1: {num_k1} | T_norm: {t_norm:.3f}m | GT Matches: {len(gt_matches)}")
#         #         pred = pred_b
            
#         #     # if b == 0:
#         #     #     pred = pred_b
                
#         # # --- バッチ内の全処理が終了 ---
        
#         # if valid_b == 0:
#         #     pbar.set_postfix({'Status': 'Skipped (No Valid GT)'})
#         #     continue # 有効なペアが1つもなければスキップ
            
#         # loss = total_loss / valid_b
        
#         # loss.backward()
#         # torch.nn.utils.clip_grad_norm_(lg.parameters(), 1.0)
#         # optimizer.step()

#         # interval_loss_sum += loss.item()
        
#         # # 🎯 予測マッチの取得 (Metrics計算用)
#         # # ここも 'log_assignment' ではなく 'scores' をチェックするように修正
#         # matches_b0 = pred.get('matches0', pred.get('matches', [None]))[0]
#         # if matches_b0 is None and 'scores' in pred:
#         #     assign = pred['scores'][0].exp() # 対数から確率に変換
#         #     mvals, m1_idx = assign.max(dim=1)
#         #     m0_idx = torch.nonzero(mvals > 0.2).squeeze(-1)
#         #     matches_b0 = torch.stack([m0_idx, m1_idx[m0_idx]], dim=-1) if len(m0_idx) > 0 else torch.zeros((0, 2), dtype=torch.long, device=device)
        
#     #     with torch.no_grad():
#     #         f0_map, kpts0, sc0 = xfeat(img0)
#     #         f1_map, kpts1, sc1 = xfeat(img1)
            
#     #         k0_list, k1_list, gt_list = [], [], []
#     #         f0_list, f1_list, sc0_list, sc1_list = [], [], [], []
            
#     #         for b in range(img0.shape[0]):
#     #             k0, k1 = kpts0[b][:512], kpts1[b][:512]
#     #             k0_list.append(k0); k1_list.append(k1)
#     #             f0_list.append(f0_map[b][:512]); f1_list.append(f1_map[b][:512])
#     #             sc0_list.append(sc0[b][:512]); sc1_list.append(sc1[b][:512])
                
#     #             gt_list.append(compute_epipolar_gt(k0, k1, T_rel[b], K[b]))

#     #     # lg_input = {
#     #     #     'image0': {'keypoints': torch.stack(k0_list), 'descriptors': torch.stack(f0_list), 'keypoint_scores': torch.stack(sc0_list)},
#     #     #     'image1': {'keypoints': torch.stack(k1_list), 'descriptors': torch.stack(f1_list), 'keypoint_scores': torch.stack(sc1_list)}
#     #     # }
#     #     from torch.nn.utils.rnn import pad_sequence
        
#     #     lg_input = {
#     #         'image0': {
#     #             'keypoints': pad_sequence(k0_list, batch_first=True),
#     #             'descriptors': pad_sequence(f0_list, batch_first=True),
#     #             'keypoint_scores': pad_sequence(sc0_list, batch_first=True)
#     #         },
#     #         'image1': {
#     #             'keypoints': pad_sequence(k1_list, batch_first=True),
#     #             'descriptors': pad_sequence(f1_list, batch_first=True),
#     #             'keypoint_scores': pad_sequence(sc1_list, batch_first=True)
#     #         }
#     # }
        
#         # pred = lg(lg_input)
#         # loss = compute_masked_match_loss(pred['log_assignment'], k0_list, k1_list, gt_list, orig0, orig1)

#         # loss.backward()
#         # torch.nn.utils.clip_grad_norm_(lg.parameters(), 1.0)
#         # optimizer.step()

#         # interval_loss_sum += loss.item()

#         # # 🎯 予測マッチの取得 (Metrics計算用)
#         # matches_b0 = pred.get('matches0', pred.get('matches', [None]))[0]
#         # if matches_b0 is None and 'log_assignment' in pred:
#         #     assign = pred['log_assignment'][0].exp()
#         #     mvals, m1_idx = assign.max(dim=1)
#         #     m0_idx = torch.nonzero(mvals > 0.2).squeeze(-1)
#         #     matches_b0 = torch.stack([m0_idx, m1_idx[m0_idx]], dim=-1) if len(m0_idx) > 0 else torch.zeros((0, 2), dtype=torch.long, device=device)

#         # 50ステップごとに誤差を計算・蓄積 (計算コスト削減のため)
#         if step % 50 == 0 and len(matches_b0) > 0:
#             err, prec = compute_pose_metrics(matches_b0, k0_list[0], k1_list[0], T_rel[0], K[0])
#             running_pose_errors.append(err)
#             running_precisions.append(prec)

#         if (step + 1) % eval_interval == 0:
#             avg_interval_loss = interval_loss_sum / eval_interval
#             if avg_interval_loss < best_loss:
#                 best_loss = avg_interval_loss
#                 best_path = os.path.join(args.output, 'lg_stage2_best.pth')
#                 torch.save(lg.state_dict(), best_path)
#                 print(f"\n🌟 Best LightGlue updated at Step {step+1} (Avg Match Loss: {best_loss:.4f})")
                
#             interval_loss_sum = 0.0  

#             # 🎯 PoseAUC と Precision の計算・保存
#             if len(running_pose_errors) > 0 and not args.no_wandb:
#                 err_arr = np.array(running_pose_errors)
#                 auc5  = np.mean(err_arr < 5)  * 100
#                 auc10 = np.mean(err_arr < 10) * 100
#                 auc20 = np.mean(err_arr < 20) * 100
#                 avg_prec = np.mean(running_precisions) * 100

#                 wandb.log({
#                     "metrics/PoseAUC@5": auc5,
#                     "metrics/PoseAUC@10": auc10,
#                     "metrics/PoseAUC@20": auc20,
#                     "metrics/Precision": avg_prec
#                 }, step=step)

#                 running_pose_errors.clear()
#                 running_precisions.clear()
        
#         if step % 10 == 0 and not args.no_wandb:
#             wandb.log({"loss/match": loss.item()}, step=step)
            
#         if step % 500 == 0 and not args.no_wandb:
#             log_stage23_geometry_to_wandb(img0[0], k0_list[0], sc0_list[0], orig0[0], step, "Stage2_LG")
#             # 🎯 マッチング可視化
#             if len(matches_b0) > 0:
#                 log_matching_and_metrics(img0[0], img1[0], k0_list[0], k1_list[0], matches_b0, T_rel[0], K[0], step, "Stage2_Matching")

#         pbar.set_postfix({'MatchLoss': f"{loss.item():.4f}"})

#     torch.save(lg.state_dict(), os.path.join(args.output, 'lg_stage2_final.pth'))
#     print("✅ Stage 2 Training completed.")

# if __name__ == '__main__':
#     main()
# ==========================================
# メイン学習ループ
# ==========================================
# ==========================================
# メイン学習ループ
# ==========================================
def main():
    args = get_args()

    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
            # Stage 2 の設定ブロックを取得 (無ければ空辞書)
            stage2_config = config.get('stage2', {})
    else:
        print(f"⚠️ Warning: Config file {args.config} not found. Using defaults.")
        stage2_config = {}

    # 🌟 1. 優先順位に基づいた max_keypoints の最終決定
    if args.max_keypoints is not None:
        final_max_k = args.max_keypoints
        print(f"🔧 Config Overridden by CLI: max_keypoints = {final_max_k}")
    elif 'max_keypoints' in stage2_config:
        final_max_k = stage2_config['max_keypoints']
        print(f"📄 Config Loaded from YAML: max_keypoints = {final_max_k}")
    else:
        final_max_k = 512
        print(f"⚙️ Using Default Config: max_keypoints = {final_max_k}")

    # 🌟 2. Depth Reprojection GT の使用フラグを取得
    use_depth_gt = stage2_config.get('use_depth_gt', False)
    if use_depth_gt:
        print("🎯 Config: Depth Reprojection GT is ENABLED (Fallback to Epipolar for non-Depth datasets).")
    else:
        print("📏 Config: Epipolar Geometry GT is ENABLED for all datasets.")

    os.makedirs(args.output, exist_ok=True)
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    wandb.init(project="thermal-xfeat-hybrid", name=args.wandb_run_name, mode="disabled" if args.no_wandb else "online")

    # モデルのセットアップ
    xfeat = XFeatModel().to(device).eval()
    xfeat.load_state_dict(torch.load(args.xfeat_weights, map_location=device))
    for p in xfeat.parameters(): p.requires_grad = False

    lg = LightGlue(features=None, input_dim=64, filter_threshold=0.0, depth_confidence=-1.0).to(device).train()
    if args.lg_weights is not None and os.path.exists(args.lg_weights):
        print(f"\n📥 Loading pre-trained LightGlue weights from: {args.lg_weights}")
        lg.load_state_dict(torch.load(args.lg_weights, map_location=device), strict=False)
        print("✅ LightGlue weights loaded successfully.")
    else:
        print("\n⚠️ [WARNING] No pre-trained LightGlue weights provided. Training from RANDOM initialization.")
    
    lg.train()
    optimizer = torch.optim.Adam(lg.parameters(), lr=args.lr)

    # データセットのセットアップ
    datasets = []
    if args.vivid_root: 
        datasets.append(VividSequentialDataset(args.vivid_root, stride=1))
        print(f"[Dataset] Added VIVID from {args.vivid_root}")
    
    if args.tartanrgbt_root: datasets.append(TartanRGBTSequentialDataset(args.tartanrgbt_root, stride=1))
    if args.sthereo_root:
        sthereo_ds = SThErEOSequentialDataset(data_root=args.sthereo_root, stride=1, split='train')
        datasets.append(sthereo_ds)
        print(f"[Dataset] Added SThErEO from {args.sthereo_root}")

    if args.ms2_root:
        ms2_ds = MS2SequentialDataset(data_root=args.ms2_root, split='train')
        datasets.append(ms2_ds)
        print(f"[Dataset] Added MS2 from {args.ms2_root}")
    
    from torch.utils.data import WeightedRandomSampler
    valid_datasets = [ds for ds in datasets if len(ds) > 0]

    if len(valid_datasets) == 0:
        raise RuntimeError("❌ 全てのデータセットが空です。パスやアノテーションを確認してください。")

    full_dataset = ConcatDataset(valid_datasets)
    weights = []
    for ds in valid_datasets:
        weight = 1.0 / len(ds)
        weights.extend([weight] * len(ds))
    
    sampler = WeightedRandomSampler(torch.DoubleTensor(weights), num_samples=len(full_dataset), replacement=True)

    loader = DataLoader(
        full_dataset, 
        batch_size=args.batch_size, 
        sampler=sampler, 
        drop_last=True,
        num_workers=4, 
        collate_fn=dynamic_pad_collate
    )

    if args.save_debug_images:
        save_geometry_dataset_checks(datasets, args.output, num_samples=5)
        print("🛑 連続フレーム画像の保存が完了しました。目視確認のため、ここでプログラムを終了します。")
        sys.exit(0)

    loader_iter = iter(loader)
    best_loss = float('inf')
    interval_loss_sum = 0.0  
    eval_interval = 500      

    running_pose_errors = []
    running_precisions = []

    pbar = tqdm(range(args.n_steps))
    for step in pbar:
        try: batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        img0, img1 = batch['image0'].to(device), batch['image1'].to(device)
        T_rel, K = batch['T_rel'].to(device), batch['K'].to(device)
        orig0, orig1 = batch['orig_size0'], batch['orig_size1']
        
        # 🌟 3. DepthとDataset Nameの安全な取得 (存在しない場合はNone)
        depth0 = batch['depth0'].to(device) if 'depth0' in batch else None
        dataset_names = batch.get('dataset_name', ['unknown'] * img0.shape[0])

        optimizer.zero_grad()
        matches_b0 = torch.zeros((0, 2), dtype=torch.long, device=device)
        
        with torch.no_grad():
            max_k = final_max_k
            
            try:
                f0_map, kpts0, sc0 = xfeat(img0, top_k=max_k)
                f1_map, kpts1, sc1 = xfeat(img1, top_k=max_k)
            except TypeError:
                f0_map, kpts0, sc0 = xfeat(img0)
                f1_map, kpts1, sc1 = xfeat(img1)
            
            k0_list, k1_list, gt_list = [], [], []
            f0_list, f1_list = [], []

            B, C_f, H_f0, W_f0 = f0_map.shape
            _, _, H_f1, W_f1 = f1_map.shape
            stride = 8.0 

            grid_y0, grid_x0 = torch.meshgrid(torch.arange(H_f0, device=device), torch.arange(W_f0, device=device), indexing='ij')
            kpts0_base_flat = (torch.stack([grid_x0, grid_y0], dim=-1).float() * stride + (stride / 2.0)).reshape(-1, 2)

            grid_y1, grid_x1 = torch.meshgrid(torch.arange(H_f1, device=device), torch.arange(W_f1, device=device), indexing='ij')
            kpts1_base_flat = (torch.stack([grid_x1, grid_y1], dim=-1).float() * stride + (stride / 2.0)).reshape(-1, 2)

            for b in range(B):
                scores0_flat = sc0[b].flatten()
                scores1_flat = sc1[b].flatten()

                f0_flat = f0_map[b].reshape(C_f, -1).t()
                f1_flat = f1_map[b].reshape(C_f, -1).t()

                w0, h0 = orig0[b][0].item(), orig0[b][1].item()
                w1, h1 = orig1[b][0].item(), orig1[b][1].item()

                valid_mask0 = (kpts0_base_flat[:, 0] < w0) & (kpts0_base_flat[:, 1] < h0) & (scores0_flat > 0.001)
                valid_mask1 = (kpts1_base_flat[:, 0] < w1) & (kpts1_base_flat[:, 1] < h1) & (scores1_flat > 0.001)

                kpts0_valid = kpts0_base_flat[valid_mask0]
                f0_valid = f0_flat[valid_mask0]
                scores0_valid = scores0_flat[valid_mask0]

                kpts1_valid = kpts1_base_flat[valid_mask1]
                f1_valid = f1_flat[valid_mask1]
                scores1_valid = scores1_flat[valid_mask1]

                k_0 = min(max_k, scores0_valid.shape[0])
                k_1 = min(max_k, scores1_valid.shape[0])

                if k_0 == 0 or k_1 == 0:
                    k0_list.append(torch.zeros((0, 2), device=device))
                    k1_list.append(torch.zeros((0, 2), device=device))
                    f0_list.append(torch.zeros((0, 64), device=device))
                    f1_list.append(torch.zeros((0, 64), device=device))
                    gt_list.append(torch.zeros((0, 2), dtype=torch.long, device=device))
                    continue

                _, topk_idx0 = torch.topk(scores0_valid, k_0)
                _, topk_idx1 = torch.topk(scores1_valid, k_1)

                k0_list.append(kpts0_valid[topk_idx0])
                k1_list.append(kpts1_valid[topk_idx1])
                f0_list.append(f0_valid[topk_idx0])
                f1_list.append(f1_valid[topk_idx1])

                # =================================================================
                # 🌟 4. Configとデータセットに基づくGT計算の動的切り替え
                # =================================================================
                current_dataset = dataset_names[b] if isinstance(dataset_names, (list, tuple)) else dataset_names

                if use_depth_gt and (depth0 is not None) and (current_dataset == 'ms2'):
                    # MS2かつDepthが有効な場合は再投影(Reprojection)を使用
                    gt_matches = compute_reprojection_gt(
                        k0_list[-1], k1_list[-1], depth0[b], T_rel[b], K[b], th_pos=3.0
                    )
                else:
                    # VIVIDなどDepthがない、あるいはConfigがFalseの場合はエピポーラを使用
                    gt_matches = compute_epipolar_gt(
                        k0_list[-1], k1_list[-1], T_rel[b], K[b], th_pos=3.0
                    )
                
                gt_list.append(gt_matches)

                # --- 既存のデバッグ画像保存コード ---
                vis_interval = 100        
                num_vis_per_batch = 10     

                if step % vis_interval == 0 and b < num_vis_per_batch:
                    vis_dir = os.path.join(args.output, "debug_gt_vis")
                    
                    vis_path_gt = os.path.join(vis_dir, f"gt_matches_step{step:06d}_b{b:02d}.jpg")
                    save_gt_matches_visualization(
                        img0[b], img1[b], k0_list[-1], k1_list[-1], gt_matches, vis_path_gt
                    )
                    
                    vis_path_kpts0 = os.path.join(vis_dir, f"kpts_img0_step{step:06d}_b{b:02d}.jpg")
                    save_keypoints_visualization(img0[b], k0_list[-1], vis_path_kpts0, color=(0, 255, 0))

                    vis_path_kpts1 = os.path.join(vis_dir, f"kpts_img1_step{step:06d}_b{b:02d}.jpg")
                    save_keypoints_visualization(img1[b], k1_list[-1], vis_path_kpts1, color=(0, 165, 255))

        # --- LightGlue マイクロバッチ学習 ---
        total_loss = 0.0
        valid_b = 0
        current_batch_size = img0.shape[0]
        
        for b in range(img0.shape[0]):
            lg_input_b = {
                'image0': {
                    'keypoints': k0_list[b].unsqueeze(0),
                    'descriptors': f0_list[b].unsqueeze(0),
                    'image_size': orig0[b].unsqueeze(0)
                },
                'image1': {
                    'keypoints': k1_list[b].unsqueeze(0),
                    'descriptors': f1_list[b].unsqueeze(0),
                    'image_size': orig1[b].unsqueeze(0)
                }
            }
            
            lg.train() 
            pred_b = lg(lg_input_b)
            
            log_assignment = pred_b.get('log_assignment', None)
            if log_assignment is None:
                if 'matching_scores0' in pred_b and pred_b['matching_scores0'].dim() >= 2:
                    log_assignment = pred_b['matching_scores0']
                else:
                    continue

            if isinstance(log_assignment, (list, tuple)):
                log_assignment = log_assignment[-1] 
            
            loss_b = compute_masked_match_loss(
                log_assignment, 
                [k0_list[b]], [k1_list[b]], [gt_list[b]], 
                [orig0[b]], [orig1[b]]
            )
            
            if loss_b is None:
                continue
                
            scaled_loss = loss_b / current_batch_size
            scaled_loss.backward()

            interval_loss_sum += scaled_loss.item() 
            valid_b += 1
            
            if b == 0:
                matches_b0 = torch.zeros((0, 2), dtype=torch.long, device=device)
                
                if 'matches' in pred_b:
                    raw_matches = pred_b['matches'][0]
                    if isinstance(raw_matches, torch.Tensor):
                        if raw_matches.dim() == 2:
                            matches_b0 = raw_matches
                        elif raw_matches.dim() == 1 and raw_matches.shape[0] == 2:
                            matches_b0 = raw_matches.unsqueeze(0) 
                    elif isinstance(raw_matches, (list, tuple)) and len(raw_matches) == 2:
                        try:
                            matches_b0 = torch.stack(raw_matches, dim=-1)
                        except Exception as e:
                            pass
                else:
                    assign = log_assignment[0].exp()
                    mvals, m1_idx = assign[:-1, :-1].max(dim=1)
                    m0_idx = torch.nonzero(mvals > 0.2).squeeze(-1)
                    if len(m0_idx) > 0:
                        matches_b0 = torch.stack([m0_idx, m1_idx[m0_idx]], dim=-1)
                        
                if matches_b0.dim() != 2 or matches_b0.shape[1] != 2:
                    matches_b0 = torch.zeros((0, 2), dtype=torch.long, device=device)
                
        if valid_b == 0:
            pbar.set_postfix({'Status': 'Skipped (valid_b=0)'})
            continue 
            
        torch.nn.utils.clip_grad_norm_(lg.parameters(), 1.0)
        optimizer.step()

        display_loss = total_loss / valid_b
        interval_loss_sum += display_loss

        pbar.set_postfix({'MatchLoss': f"{display_loss:.4f}"})

        # --- メトリクスと保存 ---
        if step % 50 == 0 and len(matches_b0) > 0:
            err, prec = compute_pose_metrics(matches_b0, k0_list[0], k1_list[0], T_rel[0], K[0])
            running_pose_errors.append(err)
            running_precisions.append(prec)

        if (step + 1) % eval_interval == 0:
            avg_interval_loss = interval_loss_sum / eval_interval
            if avg_interval_loss < best_loss:
                best_loss = avg_interval_loss
                best_path = os.path.join(args.output, 'lg_stage2_best.pth')
                torch.save(lg.state_dict(), best_path)
                print(f"\n🌟 Best LightGlue updated at Step {step+1} (Avg Match Loss: {best_loss:.4f})")
                
            interval_loss_sum = 0.0  

            if len(running_pose_errors) > 0 and not args.no_wandb:
                err_arr = np.array(running_pose_errors)
                auc5  = np.mean(err_arr < 5)  * 100
                auc10 = np.mean(err_arr < 10) * 100
                auc20 = np.mean(err_arr < 20) * 100
                avg_prec = np.mean(running_precisions) * 100

                wandb.log({
                    "metrics/PoseAUC@5": auc5,
                    "metrics/PoseAUC@10": auc10,
                    "metrics/PoseAUC@20": auc20,
                    "metrics/Precision": avg_prec
                }, step=step)

                running_pose_errors.clear()
                running_precisions.clear()
        
        if step % 10 == 0 and not args.no_wandb:
            wandb.log({"loss/match": display_loss}, step=step)
            
        if step % 500 == 0 and not args.no_wandb:
            img0_vis = _prepare_image_for_cv2(img0[0])
            img1_vis = _prepare_image_for_cv2(img1[0])
            
            img0_kpts = img0_vis.copy()
            kpts0_np = k0_list[0].cpu().numpy()
            for pt in kpts0_np:
                cv2.circle(img0_kpts, (int(pt[0]), int(pt[1])), 2, (0, 255, 0), -1)

            h0, w0, _ = img0_vis.shape
            h1, w1, _ = img1_vis.shape
            match_canvas = np.zeros((max(h0, h1), w0 + w1, 3), dtype=np.uint8)
            match_canvas[:h0, :w0] = img0_vis
            match_canvas[:h1, w0:w0+w1] = img1_vis

            if len(matches_b0) > 0:
                kpts1_np = k1_list[0].cpu().numpy()
                matches_np = matches_b0.cpu().numpy()
                for m in matches_np:
                    pt1 = (int(kpts0_np[m[0], 0]), int(kpts0_np[m[0], 1]))
                    pt2 = (int(kpts1_np[m[1], 0]) + w0, int(kpts1_np[m[1], 1]))
                    cv2.line(match_canvas, pt1, pt2, (0, 165, 255), 1)
                    cv2.circle(match_canvas, pt1, 2, (0, 255, 0), -1)
                    cv2.circle(match_canvas, pt2, 2, (0, 255, 0), -1)

            wandb.log({
                "Visuals/Keypoints (Image 0)": wandb.Image(img0_kpts, caption=f"Step {step} Keypoints"),
                "Visuals/LightGlue Matches": wandb.Image(match_canvas, caption=f"Step {step} Matches ({len(matches_b0)} pairs)")
            }, step=step)

    torch.save(lg.state_dict(), os.path.join(args.output, 'lg_stage2_final.pth'))
    print("✅ Stage 2 Training completed.")

if __name__ == '__main__':
    main()