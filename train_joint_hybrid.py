"""
train_joint_hybrid.py
Stage 3: Joint End-to-End Fine-tuning (XFeat + LightGlue)
"""
import argparse
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
import wandb
from tqdm import tqdm

from modules.model import XFeatModel
try:
    from lightglue import LightGlue
except ImportError:
    pass

from modules.dataset.thermal.stage23_geometry_datasets import Stage23_VIVIDDataset, Stage23_TartanRGBTDataset
from modules.training.visualization import log_stage23_geometry_to_wandb
# 🎯 高度なメトリクスと可視化モジュールのインポート
from modules.training.metrics_vis import compute_pose_metrics, log_matching_and_metrics

def save_geometry_dataset_checks(datasets, output_root, num_samples=5):
    import torchvision
    debug_dir = os.path.join(output_root, "dataset_checks")
    os.makedirs(debug_dir, exist_ok=True)
    
    print(f"\n🔍 Checking Geometry Datasets... saving {num_samples} sequential pairs per dataset to {debug_dir}")
    for idx, ds in enumerate(datasets):
        for i in range(min(num_samples, len(ds))):
            data = ds[i]
            img0, img1 = data['image0'], data['image1']
            ds_name = data.get('dataset_name', f'dataset_{idx}')
            comparison = torch.cat([img0, img1], dim=2)
            fname = f"sample_{ds_name}_seq_{i:03d}.png"
            torchvision.utils.save_image(comparison, os.path.join(debug_dir, fname))
            
    print(f"✅ Geometry Dataset check completed. Images saved in {debug_dir}\n")

# ==========================================
# 幾何学 & Loss ユーティリティ
# ==========================================
def compute_epipolar_gt(kpts0, kpts1, T_rel, K, th_pos=3.0):
    N, M = kpts0.shape[0], kpts1.shape[0]
    if N == 0 or M == 0: return torch.zeros((0, 2), dtype=torch.long, device=kpts0.device)
    K_np = K.cpu().numpy().astype(np.float64)
    T_np = T_rel.cpu().numpy().astype(np.float64)
    R, t = T_np[:3, :3], T_np[:3, 3]
    t_cross = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    E = t_cross @ R
    Ki = np.linalg.inv(K_np)
    F_mat = (Ki.T @ E @ Ki).astype(np.float32)
    p0h = torch.cat([kpts0, torch.ones_like(kpts0[:, :1])], dim=-1).cpu().numpy()
    p1h = torch.cat([kpts1, torch.ones_like(kpts1[:, :1])], dim=-1).cpu().numpy()
    lines = (F_mat @ p0h.T).T
    num = np.abs(p1h @ lines.T)
    denom = np.sqrt(lines[:, 0]**2 + lines[:, 1]**2)[None] + 1e-8
    dist = (num / denom).T
    min_idx = dist.argmin(axis=1)
    min_dist = dist[np.arange(N), min_idx]
    matches = [[i, min_idx[i]] for i in range(N) if min_dist[i] < th_pos]
    return torch.tensor(matches, dtype=torch.long, device=kpts0.device)

def compute_masked_match_loss(log_assignment, kpts0_list, kpts1_list, gt_matches_list, orig_size0, orig_size1):
    B = len(kpts0_list)
    total_loss, valid_b = 0.0, 0
    for b in range(B):
        w0, h0 = orig_size0[b][0].item(), orig_size0[b][1].item()
        w1, h1 = orig_size1[b][0].item(), orig_size1[b][1].item()
        valid_m0 = (kpts0_list[b][:, 0] < w0) & (kpts0_list[b][:, 1] < h0)
        valid_m1 = (kpts1_list[b][:, 0] < w1) & (kpts1_list[b][:, 1] < h1)
        matches = gt_matches_list[b]
        if len(matches) == 0: continue
        m0_idx, m1_idx = matches[:, 0], matches[:, 1]
        valid_match_mask = valid_m0[m0_idx] & valid_m1[m1_idx]
        f_m0, f_m1 = m0_idx[valid_match_mask], m1_idx[valid_match_mask]
        if len(f_m0) == 0: continue
        loss = -log_assignment[b, f_m0, f_m1].mean()
        total_loss += loss
        valid_b += 1
    return total_loss / valid_b if valid_b > 0 else torch.tensor(0.0, device=log_assignment.device, requires_grad=True)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--xfeat_weights', type=str, required=True)
    parser.add_argument('--lg_weights', type=str, required=True)
    parser.add_argument('--vivid_root', type=str, default='datasets/vivid')
    parser.add_argument('--tartanrgbt_root', type=str, default='datasets/tartanRGBT')
    parser.add_argument('--output', type=str, default='checkpoints/stage3_joint')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--n_steps', type=int, default=5000)
    parser.add_argument('--lambda_match', type=float, default=0.5)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--wandb_run_name', type=str, default='stage3_joint')
    parser.add_argument('--no_wandb', action='store_true')
    parser.add_argument('--save_debug_images', action='store_true', help='学習前に各データセットの時系列サンプル画像を保存して終了する')
    return parser.parse_args()

def main():
    args = get_args()
    os.makedirs(args.output, exist_ok=True)
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    wandb.init(project="thermal-xfeat-hybrid", name=args.wandb_run_name, mode="disabled" if args.no_wandb else "online")

    teacher_xfeat = XFeatModel().to(device).eval()
    student_xfeat = XFeatModel().to(device).train()
    lg = LightGlue(features=None, input_dim=64, filter_threshold=-1.0).to(device).train()

    teacher_xfeat.load_state_dict(torch.load(args.xfeat_weights, map_location=device))
    student_xfeat.load_state_dict(torch.load(args.xfeat_weights, map_location=device))
    lg.load_state_dict(torch.load(args.lg_weights, map_location=device))

    for p in teacher_xfeat.parameters(): p.requires_grad = False

    optimizer = torch.optim.Adam([
        {'params': student_xfeat.parameters(), 'lr': args.lr * 0.1},
        {'params': lg.parameters(), 'lr': args.lr}
    ])

    datasets = []
    if args.vivid_root: datasets.append(Stage23_VIVIDDataset(args.vivid_root, stride=5))
    if args.tartanrgbt_root: datasets.append(Stage23_TartanRGBTDataset(args.tartanrgbt_root, stride=5))
    
    if len(datasets) == 0:
        raise RuntimeError("❌ 学習データセットが見つかりません。")

    # 🎯 学習前のデータセット・サンプル保存と安全な終了
    if args.save_debug_images:
        save_geometry_dataset_checks(datasets, args.output, num_samples=5)
        print("🛑 連続フレーム画像の保存が完了しました。目視確認のため、ここでプログラムを終了します。")
        sys.exit(0)

    loader = DataLoader(ConcatDataset(datasets), batch_size=args.batch_size, shuffle=True, drop_last=True)
    loader_iter = iter(loader)

    best_loss = float('inf')
    interval_loss_sum = 0.0
    eval_interval = 500

    # 🎯 メトリクス蓄積用リスト
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

        optimizer.zero_grad()
        
        with torch.no_grad():
            t_f0_map, _, _ = teacher_xfeat(img0)
            t_f1_map, _, _ = teacher_xfeat(img1)

        s_f0_map, kpts0, sc0 = student_xfeat(img0)
        s_f1_map, kpts1, sc1 = student_xfeat(img1)

        loss_kd = F.mse_loss(F.normalize(s_f0_map, p=2, dim=1), F.normalize(t_f0_map, p=2, dim=1)) + \
                  F.mse_loss(F.normalize(s_f1_map, p=2, dim=1), F.normalize(t_f1_map, p=2, dim=1))

        k0_list, k1_list, gt_list = [], [], []
        f0_list, f1_list, sc0_list, sc1_list = [], [], [], []
        
        for b in range(img0.shape[0]):
            k0, k1 = kpts0[b][:512], kpts1[b][:512]
            k0_list.append(k0); k1_list.append(k1)
            f0_list.append(s_f0_map[b][:512]); f1_list.append(s_f1_map[b][:512])
            sc0_list.append(sc0[b][:512]); sc1_list.append(sc1[b][:512])
            gt_list.append(compute_epipolar_gt(k0, k1, T_rel[b], K[b]))

        lg_input = {
            'image0': {'keypoints': torch.stack(k0_list), 'descriptors': torch.stack(f0_list), 'keypoint_scores': torch.stack(sc0_list)},
            'image1': {'keypoints': torch.stack(k1_list), 'descriptors': torch.stack(f1_list), 'keypoint_scores': torch.stack(sc1_list)}
        }
        
        pred = lg(lg_input)
        loss_match = compute_masked_match_loss(pred['log_assignment'], k0_list, k1_list, gt_list, orig0, orig1)

        total_loss = loss_kd + args.lambda_match * loss_match
        total_loss.backward()

        interval_loss_sum += total_loss.item()

        # 🎯 予測マッチの取得 (Metrics計算用)
        matches_b0 = pred.get('matches0', pred.get('matches', [None]))[0]
        if matches_b0 is None and 'log_assignment' in pred:
            assign = pred['log_assignment'][0].exp()
            mvals, m1_idx = assign.max(dim=1)
            m0_idx = torch.nonzero(mvals > 0.2).squeeze(-1)
            matches_b0 = torch.stack([m0_idx, m1_idx[m0_idx]], dim=-1) if len(m0_idx) > 0 else torch.zeros((0, 2), dtype=torch.long, device=device)

        # 50ステップごとに誤差を計算・蓄積
        if step % 50 == 0 and len(matches_b0) > 0:
            err, prec = compute_pose_metrics(matches_b0, k0_list[0], k1_list[0], T_rel[0], K[0])
            running_pose_errors.append(err)
            running_precisions.append(prec)

        if (step + 1) % eval_interval == 0:
            avg_interval_loss = interval_loss_sum / eval_interval
            if avg_interval_loss < best_loss:
                best_loss = avg_interval_loss
                torch.save(student_xfeat.state_dict(), os.path.join(args.output, 'xfeat_stage3_best.pth'))
                torch.save(lg.state_dict(), os.path.join(args.output, 'lg_stage3_best.pth'))
                print(f"\n🌟 Best Joint Model updated at Step {step+1} (Avg Total Loss: {best_loss:.4f})")
                
            interval_loss_sum = 0.0  
            
            # 🎯 PoseAUC と Precision の計算・保存
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

        torch.nn.utils.clip_grad_norm_(student_xfeat.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(lg.parameters(), 1.0)
        optimizer.step()
        
        if step % 10 == 0 and not args.no_wandb:
            wandb.log({"loss/total": total_loss.item(), "loss/kd": loss_kd.item(), "loss/match": loss_match.item()}, step=step)
            
        if step % 500 == 0 and not args.no_wandb:
            log_stage23_geometry_to_wandb(img0[0], k0_list[0], sc0_list[0], orig0[0], step, "Stage3_Joint")
            # 🎯 マッチング可視化
            if len(matches_b0) > 0:
                log_matching_and_metrics(img0[0], img1[0], k0_list[0], k1_list[0], matches_b0, T_rel[0], K[0], step, "Stage3_Matching")

        pbar.set_postfix({'Tot': f"{total_loss.item():.4f}", 'Match': f"{loss_match.item():.4f}"})

    torch.save(student_xfeat.state_dict(), os.path.join(args.output, 'xfeat_stage3_final.pth'))
    torch.save(lg.state_dict(), os.path.join(args.output, 'lg_stage3_final.pth'))
    print("✅ Stage 3 Joint Training completed.")

if __name__ == '__main__':
    main()