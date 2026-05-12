"""
train_matcher_hybrid.py (Stage 2: Matcher Finetuning)

【設計思想】
1. Extractor (XFeat) の重みを凍結し、Matcher (LightGlue / GlueStick等) のAttentionと次元変換層のみを最適化する。
2. 複数の熱画像データセット (MS2, TartanRGBT等) を ConcatDataset で結合し、
   WeightedRandomSampler を用いてドメイン間の学習頻度を完全に均等化する。
3. 動的なGPU GTマッチ生成において、熱の滲み (ハロー効果) を考慮して dist_thresh を広げ、
   スパースな特徴点でも崩壊しない Robust な NLL Loss を計算する。
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
import wandb
import numpy as np

# --- 自作モジュールのインポート ---
from modules.dataset.thermal.sequential import (
    MS2SequentialDataset,
    VividSequentialDataset,
    SThErEOSequentialDataset
)

from modules.wireframe.thermal_wireframe_extractor import ThermalWireframeExtractor
from modules.xfeat import XFeat
from modules.matchers.factory import build_matcher
from modules.dataset.sampler_utils import build_weighted_sampler

# ==========================================
# 1. 損失関数 (NLL Loss with Dustbin)
# ==========================================
def compute_assignment_loss(log_assignment, gt_matches0):
    """
    log_assignment: [B, M+1, N+1] (M=画像0の点数, N=画像1の点数, 最後の行/列はダストビン)
    gt_matches0: [B, M] (画像0の点が画像1のどこに対応するか。Unmatchは -1)
    """
    B, M_plus_1, N_plus_1 = log_assignment.shape
    M, N = M_plus_1 - 1, N_plus_1 - 1
    
    loss = 0.0
    valid_batches = 0
    
    for b in range(B):
        # -1 (Unmatch/ダストビン行き) を N (最後の列のインデックス) に変換
        targets = gt_matches0[b].clone()
        targets[targets == -1] = N
        
        # ゼロパディングされたダミーノード (座標0,0等) をLossから除外するマスク
        # ※パディング実装に依存しますが、ここでは全ノードで計算する基本形
        row_indices = torch.arange(M, device=log_assignment.device)
        
        # NLL Loss: 正解ラベルの対数確率を最大化（= マイナスを最小化）
        nll = -log_assignment[b, row_indices, targets].mean()
        
        if not torch.isnan(nll):
            loss += nll
            valid_batches += 1
            
    return loss / max(valid_batches, 1)

# ==========================================
# 2. GPU対応 GTマッチング生成モジュール
# ==========================================
def generate_gt_matches_tensor(kpts0, kpts1, depth0, K, T_rel, dist_thresh=10.0):
    """
    深度マップと相対姿勢を用いて、バッチ単位でGPU上でGTを生成する
    ※熱の滲みによる「位置ズレ」を許容するため、dist_thresh を広めに設定 (デフォルト10.0)
    """
    B, N_kpts, _ = kpts0.shape
    device = kpts0.device
    gt_matches0 = torch.full((B, N_kpts), -1, dtype=torch.long, device=device)
    
    for b in range(B):
        u = kpts0[b, :, 0].round().long().clamp(0, depth0.shape[2]-1)
        v = kpts0[b, :, 1].round().long().clamp(0, depth0.shape[1]-1)
        z = depth0[b, v, u]
        
        valid_depth = (z > 0.5) & (z < 100.0)
        if not valid_depth.any(): continue
        
        cx, cy = K[b, 0, 2], K[b, 1, 2]
        fx, fy = K[b, 0, 0], K[b, 1, 1]
        
        # 逆投影 (Image 0 -> 3D)
        X0 = (kpts0[b, :, 0] - cx) * z / fx
        Y0 = (kpts0[b, :, 1] - cy) * z / fy
        P0 = torch.stack([X0, Y0, z, torch.ones_like(z)], dim=1) 
        
        # 相対姿勢で変換 (3D_0 -> 3D_1)
        P1 = (T_rel[b] @ P0.T).T 
        valid_z1 = P1[:, 2] > 0.1
        
        # 再投影 (3D_1 -> Image 1)
        u1 = (P1[:, 0] * fx / P1[:, 2]) + cx
        v1 = (P1[:, 1] * fy / P1[:, 2]) + cy
        proj_kpts0 = torch.stack([u1, v1], dim=1)
        
        # 距離計算と相互最近傍 (MNN)
        dist_matrix = torch.cdist(proj_kpts0, kpts1[b]) 
        dist_matrix[~valid_depth] = float('inf')
        dist_matrix[~valid_z1] = float('inf')
        
        min_dist_0to1, indices_0to1 = dist_matrix.min(dim=1)
        min_dist_1to0, indices_1to0 = dist_matrix.min(dim=0)
        
        mutual = indices_1to0[indices_0to1] == torch.arange(N_kpts, device=device)
        good_dist = min_dist_0to1 < dist_thresh
        
        valid_matches = mutual & good_dist
        gt_matches0[b, valid_matches] = indices_0to1[valid_matches]
        
    return gt_matches0

# ==========================================
# 3. メイン学習ループ
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    # オーケストレーターからの引数
    parser.add_argument('--config', type=str, default='config_master.yaml')
    parser.add_argument('--matcher_type', type=str, default='lightgluestick', choices=['lg', 'gluestick', 'lightgluestick'])
    parser.add_argument('--xfeat_weights', type=str, required=True, help="Stage 1 から渡される学習済み重み")
    parser.add_argument('--lg_weights', type=str, default=None, help="Matcherの再開用重み")
    
    parser.add_argument('--sthereo_root', type=str, default='')
    parser.add_argument('--ms2_root', type=str, default='')
    parser.add_argument('--vivid_root', type=str, default='')
    parser.add_argument('--tartanrgbt_root', type=str, default='')
    parser.add_argument('--freiburg_root', type=str, default='')
    
    parser.add_argument('--output', type=str, default='checkpoints/stage2')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=50, help="学習するエポック数")
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--wandb_run_name', type=str, default='stage2_matcher')
    parser.add_argument('--dataset_weights', type=str, default='', help="自動計算されたバランス重み")
    parser.add_argument('--no_wandb', action='store_true')
    parser.add_argument('--save_dataset_samples', action='store_true')
    
    # GT生成用ハイパーパラメータ
    parser.add_argument('--dist_thresh', type=float, default=10.0, help="再投影GTの許容ピクセル誤差")
    parser.add_argument('--max_keypoints', type=int, default=2048, help="抽出する特徴点の最大数")
    
    args = parser.parse_args()
    
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output, exist_ok=True)
    
    if not args.no_wandb:
        wandb.init(project="Thermal-SLAM-Hybrid", name=args.wandb_run_name, config=vars(args))

    datasets = []
    dataset_names = []
    
    # 🌟 1. MS2データセットの読み込み
    if args.ms2_root and os.path.exists(args.ms2_root):
        print(f"Loading MS2 from {args.ms2_root}")
        ds_ms2 = MS2SequentialDataset(data_root=args.ms2_root, split='train', stride=3)
        if len(ds_ms2) > 0:
            datasets.append(ds_ms2)
            dataset_names.append('ms2')

    # 🌟 2. SThErEOデータセットの読み込み
    if args.sthereo_root and os.path.exists(args.sthereo_root):
        print(f"Loading SThErEO from {args.sthereo_root}")
        ds_sthereo = SThErEOSequentialDataset(data_root=args.sthereo_root, split='train', stride=3)
        if len(ds_sthereo) > 0:
            datasets.append(ds_sthereo)
            dataset_names.append('sthereo')

    # 🌟 3. VIVIDデータセットの読み込み
    if args.vivid_root and os.path.exists(args.vivid_root):
        print(f"Loading VIVID from {args.vivid_root}")
        ds_vivid = VividSequentialDataset(data_root=args.vivid_root, split='train', stride=3)
        if len(ds_vivid) > 0:
            datasets.append(ds_vivid)
            dataset_names.append('vivid')

    if not datasets:
        raise RuntimeError("有効なデータセットがロードされませんでした。")

    concat_dataset = ConcatDataset(datasets)
    sampler = build_weighted_sampler(datasets, dataset_names, args.dataset_weights)

    if sampler is not None:
        print(f"🚀 Using WeightedRandomSampler with weights: {args.dataset_weights}")
        dataloader = DataLoader(concat_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=4, drop_last=True)
    else:
        print("⚠️ No dataset weights. Using standard random shuffle.")
        dataloader = DataLoader(concat_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)

    # --- 2. モデルの準備 ---
    print(f"Loading Extractor for {args.matcher_type}...")
    if args.matcher_type in ['lightgluestick', 'gluestick']:
        # 線分とJunctionを扱うハイブリッドExtractor
        extractor = ThermalWireframeExtractor(
            xfeat_weights=args.xfeat_weights, 
            max_keypoints=args.max_keypoints
        ).to(device).eval()
    else:
        # LightGlue用 (純粋な点のみ)
        extractor = XFeat(
            weights=args.xfeat_weights, 
            top_k=args.max_keypoints
        ).to(device).eval()
        
    # Extractor は Stage 2 では完全に凍結する
    for param in extractor.parameters():
        param.requires_grad = False

    # Matcher の構築
    matcher_config = {
        'name': args.matcher_type, 
        'input_dim': 64, 
        'descriptor_dim': 256,
        'filter_threshold': 0.1
    }
    matcher = build_matcher(matcher_config).to(device)
    
    if args.lg_weights and os.path.exists(args.lg_weights):
        matcher.load_state_dict(torch.load(args.lg_weights, map_location=device))
        print(f"Loaded Matcher weights from {args.lg_weights}")

    optimizer = torch.optim.Adam(matcher.parameters(), lr=args.lr)

    # --- 3. 学習ループ ---
    global_step = 0
    best_loss = float('inf')
    global_step = 0
    
    matcher.train()
    
    for epoch in range(1, args.epochs + 1):
        pbar = tqdm(dataloader, desc=f"Stage 2 (Epoch {epoch}/{args.epochs})")
        epoch_loss_sum = 0.0
        num_batches = 0
        
        for batch in pbar:
            img0 = batch['image0'][:, 0:1].to(device)
            img1 = batch['image1'][:, 0:1].to(device)
            depth0 = batch['depth0'].to(device)
            K = batch['K'].to(device)
            T_rel = batch['T_rel'].to(device)
            
            size0 = batch['orig_size0'].to(device)
            size1 = batch['orig_size1'].to(device)

            with torch.no_grad():
                data0 = extractor(img0)
                data1 = extractor(img1)
                
                matcher_input = {
                    'keypoints0': data0['keypoints'],
                    'keypoint_scores0': data0.get('keypoint_scores', data0.get('scores')), 
                    'descriptors0': data0['descriptors'],
                    'lines0': data0.get('lines'), 
                    'line_scores0': data0.get('line_scores'),     
                    'lines_junc_idx0': data0.get('lines_junc_idx'),
                    
                    'keypoints1': data1['keypoints'], 
                    'keypoint_scores1': data1.get('keypoint_scores', data1.get('scores')), 
                    'descriptors1': data1['descriptors'],
                    'lines1': data1.get('lines'), 
                    'line_scores1': data1.get('line_scores'),     
                    'lines_junc_idx1': data1.get('lines_junc_idx'),
                    
                    'view0': {'image_size': size0},
                    'view1': {'image_size': size1},
                }
                
                gt_matches0 = generate_gt_matches_tensor(
                    matcher_input['keypoints0'], matcher_input['keypoints1'], 
                    depth0, K, T_rel, dist_thresh=args.dist_thresh
                )

            optimizer.zero_grad()
            
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                pred = matcher(matcher_input) 
                log_assignment = pred['log_assignment']
                loss = compute_assignment_loss(log_assignment, gt_matches0)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(matcher.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss_sum += loss.item()
            num_batches += 1
            
            if not args.no_wandb and global_step % 10 == 0:
                avg_valid_gts = (gt_matches0 > -1).float().sum(dim=1).mean().item()
                wandb.log({
                    "Train/Matcher_Loss": loss.item(),
                    "Debug/Avg_Valid_GTs": avg_valid_gts
                }, step=global_step)
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            global_step += 1

        # ==========================================
        # 🌟 エポック終了時の処理 (重みの評価と保存)
        # ==========================================
        if num_batches > 0:
            avg_epoch_loss = epoch_loss_sum / num_batches
            
            # 1. Bestモデルの更新判定 (1エポックの平均Lossで評価)
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                save_path = os.path.join(args.output, "matcher_stage2_best.pth")
                torch.save(matcher.state_dict(), save_path)
                print(f"\n🌟 Best model updated at Epoch {epoch} (Avg Loss: {best_loss:.4f})")
                
                if not args.no_wandb:
                    wandb.run.summary["best_epoch"] = epoch
                    wandb.run.summary["best_loss"] = best_loss

        # 2. 定期的なモデルの保存 (10エポックごと)
        if epoch % 10 == 0:
            save_path = os.path.join(args.output, f"matcher_epoch_{epoch}.pth")
            torch.save(matcher.state_dict(), save_path)
            print(f"💾 Saved intermediate model at Epoch {epoch}")

    # 最終重みの保存
    torch.save(matcher.state_dict(), os.path.join(args.output, "matcher_stage2_final.pth"))
    if not args.no_wandb:
        wandb.finish()
    print("✅ Stage 2 Training Completed.")

if __name__ == "__main__":
    main()