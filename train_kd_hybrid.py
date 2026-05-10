# """
# train_kd_hybrid.py (Ablation & Diagnostic Edition)
# アブレーションスタディ、WandB詳細ログ、過学習テストに対応した決定版。
# """

# import argparse
# import os
# import cv2
# import numpy as np
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader, ConcatDataset
# import wandb
# from tqdm import tqdm

# from modules.model import XFeatModel
# from modules.training.losses_kd_hybrid import kd_feature_loss, hybrid_thermal_gradient_loss, spatial_entropy_loss
# from modules.dataset.thermal.sequential_hybrid import (
#     SThErEOSequentialDataset, MS2SequentialDataset, VividSequentialDataset,
#     TartanRGBTSequentialDataset, FreiburgSequentialDataset
# )

# def get_args():
#     parser = argparse.ArgumentParser(description="Hybrid Thermal-XFeat Training")
#     parser.add_argument('--sthereo_root', type=str, default='datasets/sthereo')
#     parser.add_argument('--ms2_root',     type=str, default='datasets/ms2')
#     parser.add_argument('--vivid_root',   type=str, default='datasets/vivid')
#     parser.add_argument('--tartanrgbt_root', type=str, default='datasets/tartanRGBT')
#     parser.add_argument('--freiburg_root',   type=str, default='datasets/freiburg')
#     parser.add_argument('--output',       type=str, default='checkpoints/diagnostic')
    
#     parser.add_argument('--epochs',     type=int,   default=10)
#     parser.add_argument('--batch_size', type=int,   default=8)
#     parser.add_argument('--lr',         type=float, default=1e-4)
#     parser.add_argument('--stride',     type=int,   default=3)
    
#     # ---------------------------------------------------------
#     # 🔬 アブレーション（損失関数）制御パラメータ
#     # ---------------------------------------------------------
#     parser.add_argument('--no_kd_loss',       action='store_true', help="KD Lossを無効化する")
#     parser.add_argument('--use_hybrid_loss',  action='store_true', help="Hybrid Thermal Gradient Lossを有効化する")
#     parser.add_argument('--use_spatial_loss', action='store_true', help="Spatial Entropy Lossを有効化する")
    
#     parser.add_argument('--lambda_kd',      type=float, default=1.0)
#     parser.add_argument('--lambda_hybrid',  type=float, default=0.5)
#     parser.add_argument('--lambda_spatial', type=float, default=0.1)
#     parser.add_argument('--tau_fixed',      type=float, default=200.0)
    
#     # 診断・過学習パラメータ
#     parser.add_argument('--dataset_choice', type=str, default='all', choices=['all', 'sthereo', 'ms2', 'vivid', 'tartanrgbt', 'freiburg'])
#     parser.add_argument('--debug_vis',      action='store_true')
#     parser.add_argument('--overfit_mode',   action='store_true')
#     parser.add_argument('--overfit_steps',  type=int, default=100)
    
#     # WandBとデバイス
#     parser.add_argument('--device',         type=str, default='0')
#     parser.add_argument('--no_wandb',       action='store_true')
#     parser.add_argument('--wandb_run_name', type=str, default='diagnostic_run', help="WandBの実行名を指定")
    
#     return parser.parse_args()

# def save_debug_visualization(img_8bit, img_raw, kpts_heatmap, batch_idx, epoch, out_dir="debug_vis"):
#     if torch.isnan(kpts_heatmap).any():
#         print(f"⚠️ [Warning] Model output contains NaN. Skipping visualization.")
#         return

#     os.makedirs(out_dir, exist_ok=True)
#     img = (img_8bit[0].permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
#     img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#     H, W = img_bgr.shape[:2]
    
#     raw = img_raw[0, 0].detach().cpu().numpy()
#     raw_norm = ((raw - raw.min()) / (raw.max() - raw.min() + 1e-5) * 255).astype(np.uint8)
#     raw_color = cv2.applyColorMap(raw_norm, cv2.COLORMAP_JET)

#     heatmap = kpts_heatmap[0, 0].detach().cpu().numpy()
#     heatmap_norm = ((heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8) * 255).astype(np.uint8)
#     heatmap_resized = cv2.resize(heatmap_norm, (W, H), interpolation=cv2.INTER_NEAREST)
#     heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_HOT)

#     overlay = cv2.addWeighted(img_bgr, 0.5, heatmap_color, 0.5, 0)
#     vis = np.hstack([img_bgr, overlay, raw_color])
#     save_path = os.path.join(out_dir, f"ep{epoch:03d}_b{batch_idx:03d}.png")
#     cv2.imwrite(save_path, vis)

# def main():
#     args = get_args()
#     os.makedirs(args.output, exist_ok=True)
#     device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

#     ds_dict = {}
#     anythermal_splits_base = 'third_party/anythermal/custom_datasets'

#     if args.dataset_choice in ['all', 'sthereo']:
#         ds_dict['SThErEO'] = SThErEOSequentialDataset(
#             data_root=args.sthereo_root,
#             # 🎯 修正: 'frame_lists' フォルダまで正確に指定する
#             splits_dir=os.path.join(anythermal_splits_base, 'sthereo', 'splits', 'frame_lists'),
#             split='train'
#         )
        
#     if args.dataset_choice in ['all', 'ms2']:
#         ds_dict['MS2'] = MS2SequentialDataset(
#             data_root=args.ms2_root,
#             splits_dir=os.path.join(anythermal_splits_base, 'ms2', 'splits'),
#             split='train'
#         )
        
#     if args.dataset_choice in ['all', 'vivid']:
#         ds_dict['Vivid'] = VividSequentialDataset(
#             data_root=args.vivid_root,
#             # 🎯 修正: 'splits' の下の 'frame_lists' まで正確にパスを指定する
#             splits_dir=os.path.join(anythermal_splits_base, 'vivid', 'splits', 'frame_lists'),
#             split='train'
#         )
        
#     if args.dataset_choice in ['all', 'tartanrgbt']:
#         ds_dict['TartanRGBT'] = TartanRGBTSequentialDataset(
#             data_root=args.tartanrgbt_root,
#             splits_dir=os.path.join(anythermal_splits_base, 'tartanRGBT', 'splits'),
#             split='train'
#         )
        
#     if args.dataset_choice in ['all', 'freiburg']:
#         ds_dict['Freiburg'] = FreiburgSequentialDataset(
#             data_root=args.freiburg_root,
#             # 🎯 念のため Freiburg も 'frame_list' まで指定しておくのが安全です
#             splits_dir=os.path.join(anythermal_splits_base, 'freiburg', 'splits', 'frame_list'),
#             split='train'
#         )
    
#     if len(ds_dict) == 0:
#         print("❌ Dataset not found.")
#         return
    
#     import torchvision
    
#     debug_dir = os.path.join(args.output, 'dataset_sanity_check')
#     os.makedirs(debug_dir, exist_ok=True)
#     print(f"\n=====================================================================")
#     print(f"📸 [Sanity Check] データセットの最初の1ペアを {debug_dir} に出力します")
    
#     for ds_name, ds in ds_dict.items():
#         print(f"  🔍 解析中: {ds_name} Dataset ...")
#         try:
#             # 最初の1ペアを取得
#             sample = ds[0] 
            
#             # 辞書の中に入っているすべてのテンソルを自動的に探索
#             found_images = False
#             for key, tensor in sample.items():
#                 # 画像テンソル (C, H, W) かどうかを判定
#                 if isinstance(tensor, torch.Tensor) and tensor.ndim == 3:
#                     found_images = True
                    
#                     # 保存用のテンソルを準備
#                     vis_tensor = tensor.clone()
                    
#                     # 16-bit Rawデータなど、値が [0, 1] を超えている場合の可視化用正規化
#                     if 'raw' in key.lower() or vis_tensor.max() > 1.0:
#                         t_min, t_max = vis_tensor.min(), vis_tensor.max()
#                         vis_tensor = (vis_tensor - t_min) / (t_max - t_min + 1e-8)
                    
#                     # 画像として保存
#                     save_name = f"{ds_name}_{key}.png"
#                     save_path = os.path.join(debug_dir, save_name)
#                     torchvision.utils.save_image(vis_tensor, save_path)
                    
#                     # ターミナルに形状とチャンネル数を詳細表示
#                     print(f"    ✅ [{key:^12}] 保存完了 | Shape: {list(tensor.shape)} | Min: {tensor.min():.2f}, Max: {tensor.max():.2f}")
            
#             if not found_images:
#                 print(f"    ⚠️ 画像テンソルが見つかりませんでした。辞書のキー: {sample.keys()}")
                
#         except Exception as e:
#             print(f"    🚨 {ds_name} の取得中にエラー発生: {e}")
            
#     print(f"=====================================================================\n")

#     dataset = ConcatDataset(list(ds_dict.values()))
#     print(f"Total training pairs: {len(dataset)}")

#     import torch.nn.functional as F
#     from torch.utils.data.dataloader import default_collate

#     def smart_collate_fn(batch):
#         """
#         バッチ内に異なるサイズの画像（MS2とFreiburgなど）が混在した場合、
#         バッチ内の「最大サイズ」に合わせて動的に右下パディングを行いスタック可能にする。
#         """
#         # バッチ内の最大の 高さ(H) と 幅(W) を取得
#         max_h = max(item['rgb_t'].shape[1] for item in batch)
#         max_w = max(item['rgb_t'].shape[2] for item in batch)

#         # 全てのアイテムを最大サイズに合わせてパディング
#         for item in batch:
#             for key in ['rgb_t', 'thr_t_8bit', 'thr_t_raw']:
#                 if key in item and isinstance(item[key], torch.Tensor) and item[key].ndim == 3:
#                     tensor = item[key]
#                     h, w = tensor.shape[1], tensor.shape[2]
                    
#                     pad_h = max_h - h
#                     pad_w = max_w - w
#                     if pad_h > 0 or pad_w > 0:
#                         # 4次元(1, C, H, W)に拡張してreplicateパディングし、3次元に戻す
#                         padded = F.pad(tensor.unsqueeze(0), (0, pad_w, 0, pad_h), mode='replicate')
#                         item[key] = padded.squeeze(0)

#         # サイズが完全に統一されたので、PyTorchのデフォルト関数で安全にスタックする
#         return default_collate(batch)

#     # =====================================================================
#     # 🎯 修正: DataLoader に collate_fn=smart_collate_fn を追加する
#     # =====================================================================
#     loader = DataLoader(
#         dataset, 
#         batch_size=args.batch_size, 
#         shuffle=True,
#         num_workers=4,  # ご自身の環境の設定に合わせてください
#         collate_fn=smart_collate_fn  # ← これがすべてを解決します
#     )

#     # loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

#     if args.overfit_mode:
#         print("\n⚠️ [OVERFIT MODE] Fetching a single batch to memorize...")
#         single_batch = next(iter(loader))
#         loader = [single_batch] * args.overfit_steps
#         args.epochs = 1 

#     if not args.no_wandb:
#         wandb.init(project="thermal-xfeat-hybrid", name=args.wandb_run_name, config=vars(args))

#     model = XFeatModel().to(device)
#     teacher = XFeatModel().to(device).eval()
    
#     if os.path.exists('weights/xfeat.pt'):
#         teacher.load_state_dict(torch.load('weights/xfeat.pt', map_location=device))
#     for p in teacher.parameters(): p.requires_grad = False
    
#     optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

#     for epoch in range(1, args.epochs + 1):
#         model.train()
#         pbar = tqdm(loader, desc=f"Epoch {epoch}" if not args.overfit_mode else "Overfitting")
        
#         for batch_idx, batch in enumerate(pbar):
#             img_rgb_t  = batch['rgb_t'].to(device)
#             img_8bit_t = batch['thr_t_8bit'].to(device)
#             img_raw_t  = batch['thr_t_raw'].to(device)
            
#             optimizer.zero_grad()
#             feats_s, kpts_s, scores_s = model(img_8bit_t)
            
#             with torch.no_grad():
#                 feats_t, _, _ = teacher(img_rgb_t)
            
#             # --- 🎯 損失関数のアブレーション制御 ---
#             loss_kd, loss_hybrid, loss_spatial = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
            
#             if not args.no_kd_loss:
#                 loss_kd = kd_feature_loss(feats_s, feats_t)
#             if args.use_hybrid_loss:
#                 loss_hybrid = hybrid_thermal_gradient_loss(kpts_s, scores_s, img_raw_t, tau_fixed=args.tau_fixed)
#             if args.use_spatial_loss:
#                 loss_spatial = spatial_entropy_loss(kpts_s, scores_s)

#             # 総合Loss
#             loss = (args.lambda_kd * loss_kd) + (args.lambda_hybrid * loss_hybrid) + (args.lambda_spatial * loss_spatial)
            
#             # NaN防壁と逆伝播
#             if torch.isnan(loss):
#                 print(f"🚨 [Error] Loss is NaN at batch {batch_idx}. Skipping update.")
#                 optimizer.zero_grad()
#                 continue
                
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#             optimizer.step()
            
#             # --- 📊 WandB への個別ロギング ---
#             if not args.no_wandb:
#                 wandb.log({
#                     "loss/total": loss.item(),
#                     "loss/kd": loss_kd.item() if not args.no_kd_loss else 0.0,
#                     "loss/hybrid": loss_hybrid.item() if args.use_hybrid_loss else 0.0,
#                     "loss/spatial": loss_spatial.item() if args.use_spatial_loss else 0.0,
#                     "epoch": epoch,
#                 })

#             pbar.set_postfix({'Total': f"{loss.item():.3f}", 'KD': f"{loss_kd.item():.3f}", 'Hyb': f"{loss_hybrid.item():.3f}"})

#             if args.debug_vis and batch_idx % 10 == 0:
#                 save_debug_visualization(img_8bit_t, img_raw_t, kpts_s, batch_idx, epoch)

#         if not args.overfit_mode and epoch % 10 == 0:
#             torch.save(model.state_dict(), os.path.join(args.output, f'model_epoch_{epoch}.pth'))

#     if not args.overfit_mode:
#         torch.save(model.state_dict(), os.path.join(args.output, 'diagnostic_model.pth'))
#         print("✅ Training completed.")
#     else:
#         print("✅ Overfitting test completed.")

# if __name__ == '__main__':
#     main()

"""
train_kd_hybrid.py
Stage 1: XFeat 知識蒸留 (KD) - 統合・完成版
"""
import argparse
import os
import sys
import glob
import torch
import numpy as np
import torchvision
from torch.utils.data import DataLoader, ConcatDataset
import wandb
from tqdm import tqdm

from modules.model import XFeatModel
from modules.training.losses_kd_hybrid import kd_feature_loss, hybrid_thermal_gradient_loss, spatial_entropy_loss
from modules.training.visualization import log_stage1_kd_to_wandb

# 🎯 Stage1専用ローダー
from modules.dataset.thermal.stage1_kd_datasets import (
    Stage1_VIVIDDataset, Stage1_MS2Dataset, Stage1_SThErEODataset,
    Stage1_FreiburgDataset, Stage1_CleanDataset, Stage1_KDAugmentWrapper
)

ANYTHERMAL_BASE = "third_party/anythermal/custom_datasets"

# =====================================================================
# ペア構築ヘルパー関数 (MS2, VIVID, TartanRGBT, SThErEO, Freiburg)
# =====================================================================
# def build_ms2_pairs(data_root: str) -> list:
#     train_seqs = [
#         "campus_1_2021-08-06-10-59-33", "campus_2_2021-08-06-11-23-45", "campus_3_2021-08-06-11-24-34",
#         "Road1_2021-02-26-10-58-10", "Road2_2021-02-26-11-30-49", "Road3_2021-02-26-15-46-51", "Road4_2021-02-26-15-58-00",
#         "residential_1_2021-02-26-10-58-10", "residential_2_2021-02-26-15-46-51",
#         "residential_3_2021-02-26-15-58-00", "residential_4_2021-08-06-11-24-34"
#     ]
#     pairs = []
#     for seq in train_seqs:
#         seq_dir = os.path.join(data_root, "sync_data", seq)
#         if not os.path.exists(seq_dir): continue
        
#         rgb_dir, thr_dir = os.path.join(seq_dir, "rgb", "img_left"), os.path.join(seq_dir, "thr", "img_left")
#         rgb_files = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))
#         thr_files = sorted(glob.glob(os.path.join(thr_dir, "*.png")))
        
#         rgb_basenames = {os.path.basename(f) for f in rgb_files}
#         valid_pairs = [(os.path.join(rgb_dir, os.path.basename(t)), t) for t in thr_files if os.path.basename(t) in rgb_basenames]
#         pairs.extend(valid_pairs[::10])
#     print(f"[MS2] Loaded {len(pairs)} train pairs.")
#     return pairs

# =====================================================================
# 1. MS2 のペア構築 (究極版: 大文字小文字の完全吸収)
# =====================================================================
def build_ms2_pairs(data_root: str) -> list:
    import os, glob
    # プレフィックス(campus_1_等)が実際のフォルダにあっても無くても、自動で吸収します
    train_seqs = [ '_2021-08-06-10-59-33', '_2021-08-06-17-44-55','_2021-08-13-17-06-04','_2021-08-13-21-18-04', #campus
                   # '_2021-08-06-17-21-04',  overlaps with VIVID city+urban
                   '_2021-08-13-16-50-57', #Road2
                   '_2021-08-06-16-59-13', '_2021-08-13-16-31-10', '_2021-08-13-22-16-02', #Road1
                   '_2021-08-13-16-08-46', '_2021-08-13-21-58-13',  #Road3
                   '_2021-08-13-22-36-41', #Road4
                   # '_2021-08-06-11-37-46', '_2021-08-06-16-19-00', '_2021-08-13-15-46-56', '_2021-08-13-21-36-10', #urban : overlaps with VIVID city+urban
                   ]
    
    sync_base = os.path.join(data_root, "sync_data")
    if not os.path.exists(sync_base):
        sync_base = os.path.join(data_root, "MS2", "sync_data")
        if not os.path.exists(sync_base): sync_base = data_root

    pairs = []
    loaded_folders = set() # 重複読み込み防止用

    if not os.path.exists(sync_base):
        print(f"⚠️ [MS2] sync_data folder not found.")
        return pairs

    for seq in train_seqs:
        # 🎯 文字列からタイムスタンプ(2021-...)だけを抽出
        parts = [p for p in seq.split('_') if len(p) >= 10 and '-' in p]
        if not parts: continue
        timestamp = parts[-1]

        # 🎯 sync_data の中のフォルダから、タイムスタンプが含まれるものを強引に探す
        matched_dir = None
        for d in os.listdir(sync_base):
            if timestamp in d and os.path.isdir(os.path.join(sync_base, d)):
                matched_dir = os.path.join(sync_base, d)
                break

        if not matched_dir:
            print(f"⚠️ [MS2] Folder matching timestamp {timestamp} not found.")
            continue
            
        if matched_dir in loaded_folders: continue
        loaded_folders.add(matched_dir)

        rgb_dir = next((os.path.join(matched_dir, d, "img_left") for d in os.listdir(matched_dir) if d.lower() == "rgb" and os.path.exists(os.path.join(matched_dir, d, "img_left"))), None)
        thr_dir = next((os.path.join(matched_dir, d, "img_left") for d in os.listdir(matched_dir) if d.lower() in ["thr", "ir"] and os.path.exists(os.path.join(matched_dir, d, "img_left"))), None)

        if not rgb_dir or not thr_dir: continue

        rgb_files = [f for f in glob.glob(os.path.join(rgb_dir, "*.*")) if f.lower().endswith(('.png', '.jpg'))]
        thr_files = [f for f in glob.glob(os.path.join(thr_dir, "*.*")) if f.lower().endswith(('.png', '.jpg'))]

        rgb_basenames = {os.path.basename(f) for f in rgb_files}
        valid_pairs = [(os.path.join(rgb_dir, os.path.basename(t)), t) for t in thr_files if os.path.basename(t) in rgb_basenames]
        pairs.extend(sorted(valid_pairs)[::10])
        
    print(f"[MS2] Loaded {len(pairs)} train pairs.")
    return pairs

# def build_vivid_pairs(data_root: str, anythermal_base: str = ANYTHERMAL_BASE) -> list:
#     splits_dir = os.path.join(anythermal_base, "VIVID", "splits", "frame_lists")
#     val_keywords = ["campus"]
#     pairs = []
#     for rgb_txt in glob.glob(os.path.join(splits_dir, "*/*/*rgb_framelist.txt")):
#         if any(vk in rgb_txt for vk in val_keywords): continue
#         thr_txt = rgb_txt.replace("rgb_framelist.txt", "thermal_framelist.txt")
#         if not os.path.exists(thr_txt): continue
            
#         with open(rgb_txt) as fr, open(thr_txt) as ft:
#             for r, t in zip(fr.readlines(), ft.readlines()):
#                 rp, tp = os.path.join(data_root, r.strip()), os.path.join(data_root, t.strip())
#                 if os.path.exists(rp) and os.path.exists(tp): pairs.append((rp, tp))
#     print(f"[VIVID] Loaded {len(pairs)} train pairs.")
#     return pairs

# =====================================================================
# 2. VIVID のペア構築 (修正版: 柔軟なパス探索)
# =====================================================================
def build_vivid_pairs(data_root: str, anythermal_base: str = ANYTHERMAL_BASE) -> list:
    import glob
    # 🎯 vivid と VIVID の両方を探す
    splits_dir = os.path.join(anythermal_base, "vivid", "splits", "frame_lists")
    if not os.path.exists(splits_dir):
        splits_dir = os.path.join(anythermal_base, "VIVID", "splits", "frame_lists")

    val_keywords = ["campus"]
    pairs = []
    
    # 階層の深さがブレても見つけられるように ** を使用
    rgb_txts = glob.glob(os.path.join(splits_dir, "**/*rgb_framelist.txt"), recursive=True)
    if not rgb_txts:
        print(f"⚠️ [VIVID] No split texts found in: {splits_dir}")
        
    for rgb_txt in rgb_txts:
        if any(vk in rgb_txt for vk in val_keywords): continue
        thr_txt = rgb_txt.replace("rgb_framelist.txt", "thermal_framelist.txt")
        if not os.path.exists(thr_txt): continue
            
        with open(rgb_txt) as fr, open(thr_txt) as ft:
            for r, t in zip(fr.readlines(), ft.readlines()):
                rp, tp = os.path.join(data_root, r.strip()), os.path.join(data_root, t.strip())
                if os.path.exists(rp) and os.path.exists(tp): 
                    pairs.append((rp, tp))
                    
    print(f"[VIVID] Loaded {len(pairs)} train pairs.")
    return pairs

def build_tartan_pairs(data_root: str) -> list:
    train_seqs = [
        'indoor_NSH_third_floor', 'indoor_NSH_fourth_floor', 'indoor_NSH_first_floor', 'indoor_SQH_office',
        'outdoor_campus_NSH_TO_CUT', 'outdoor_resedential_SQH_block', 'outdoor_urban_road_campus_to_marget_morrison',
        'indoor_GATES_garage_1', 'indoor_GATES_garage_3', 'indoor_GATES_seq_1', 'indoor_GATES_seq_2',
        'outdoor_urban_road_mill_19_seq_1', 'outdoor_urban_mill_19_circle_building_seq_1', 'outdoor_urban_road_mill_19_seq_2',
        'outdoor_urban_mill_19_circle_building_seq_2', 'indoor_outdoor_mill_19_indoor_to_outdoor_1',
        'outdoor_resedential_forbes_sqh_seq_1', 'outdoor_resedential_forbes_sqh_seq_2',
        'indoor_outdoor_mill_19_outdoor_to_indoor_1', 'indoor_outdoor_mill19_seq_3', 'indoor_CFA_seq_1',
        'park_frick_seq_1_falls_ravine_start_to_tranquil_trail', 'park_frick_seq_2_tranquil_trail',
        'park_frick_seq_3_tranquil_trail', 'park_frick_seq_5_falls_ravine', 'park_frick_seq_6_falls_ravine',
        'park_frick_seq_8_nine_mile_run_trail', 'park_frick_seq_9_nine_mile_run_trail',
        'park_frick_seq_10_nine_mile_to_commercial', 'park_frick_seq_11_commercial', 'park_frick_seq_12_commercial'
    ]
    pairs = []
    for day_dir in glob.glob(os.path.join(data_root, "day*")):
        for seq_dir in glob.glob(os.path.join(day_dir, "*")):
            if os.path.basename(seq_dir) not in train_seqs: continue
            rgb_dir, thr_dir = os.path.join(seq_dir, "RGB_aligned_with_thermal"), os.path.join(seq_dir, "thermal_left_rect_8")
            ffc_txt = os.path.join(seq_dir, "thermal_left_ffc", "data.txt")
            if not (os.path.exists(rgb_dir) and os.path.exists(thr_dir)): continue
                
            ffc_set = set()
            if os.path.exists(ffc_txt):
                with open(ffc_txt) as f:
                    for i, line in enumerate(f):
                        if int(line.strip()) == 1: ffc_set.add(i)
                        
            for i, fname in enumerate(sorted(os.listdir(rgb_dir))):
                if i in ffc_set: continue
                rp = os.path.join(rgb_dir, fname)
                tp = os.path.join(thr_dir, os.path.splitext(fname)[0].replace('_rgb_in_thermal', '') + ".png")
                if os.path.exists(tp): pairs.append((rp, tp))
    print(f"[TartanRGBT] Loaded {len(pairs)} train pairs.")
    return pairs

# def build_sthereo_pairs(data_root: str, anythermal_base: str = ANYTHERMAL_BASE) -> list:
#     train_seqs = ['snu_afternoon', 'snu_evening','snu_morning', 'valley_afternoon', 'valley_evening','valley_morning']
#     splits_dir = os.path.join(anythermal_base, "sthereo", "splits")
#     pairs = []
#     for seq in train_seqs:
#         txt_files = glob.glob(os.path.join(splits_dir, f"{seq.replace('_', '*')}*frame_pairs.txt"), recursive=True)
#         if not txt_files: continue
#         parts = seq.split('_')
#         rgb_dir = os.path.join(data_root, parts[0].upper(), parts[1].capitalize(), "image", "stereo_left")
#         thr_dir = os.path.join(data_root, parts[0].upper(), parts[1].capitalize(), "image", "thermal8_left_clahe")
        
#         with open(txt_files[0]) as f:
#             for line in f:
#                 p = line.strip().split()
#                 if len(p) < 2: continue
#                 rp, tp = os.path.join(rgb_dir, p[0]), os.path.join(thr_dir, p[1])
#                 if os.path.exists(rp) and os.path.exists(tp): pairs.append((rp, tp))
#     print(f"[SThErEO] Loaded {len(pairs)} train pairs.")
#     return pairs

# =====================================================================
# 4. SThErEO のペア構築 (修正版: 大文字小文字の吸収)
# =====================================================================
# =====================================================================
# 4. SThErEO のペア構築 (究極版: 構造非依存マッピング)
# =====================================================================
def build_sthereo_pairs(data_root: str, anythermal_base: str = ANYTHERMAL_BASE) -> list:
    import os
    train_seqs = ['snu_afternoon', 'snu_evening', 'snu_morning', 'valley_afternoon', 'valley_evening', 'valley_morning']
    
    splits_dirs_to_try = [
        os.path.join(anythermal_base, "sthereo", "splits"),
        os.path.join(anythermal_base, "SThErEO", "splits"),
        os.path.join(anythermal_base, "sthereo", "splits", "frame_lists")
    ]
    valid_split_dir = next((d for d in splits_dirs_to_try if os.path.exists(d)), None)
    
    pairs = []
    if not valid_split_dir:
        print(f"⚠️ [SThErEO] Splits directory not found.")
        return pairs

    # 🎯 魔法のロジック: フォルダ構造を無視して、data_root内の全画像のパスを記憶する
    rgb_map = {}
    thr_map = {}
    for dp, _, fn in os.walk(data_root):
        dl = dp.lower()
        is_thr = 'thermal' in dl or 'clahe' in dl or 'ir' in dl
        is_rgb = 'stereo' in dl or 'rgb' in dl
        
        for f in fn:
            if f.lower().endswith(('.png', '.jpg')):
                if is_thr:
                    thr_map[f.lower()] = os.path.join(dp, f)
                elif is_rgb:
                    rgb_map[f.lower()] = os.path.join(dp, f)

    for seq in train_seqs:
        txt_path = None
        search_key = seq.replace("_", "").lower()
        
        # txt ファイルを探す
        for root_dir, _, files in os.walk(valid_split_dir):
            for f in files:
                if search_key in f.replace("_", "").lower() and f.endswith(".txt"):
                    txt_path = os.path.join(root_dir, f)
                    break
            if txt_path: break

        if not txt_path: continue
            
        # 🎯 txtの中身を読み、記憶したマップから絶対パスを復元する
        with open(txt_path) as f:
            for line in f:
                p = line.strip().split()
                if len(p) < 2: continue
                r_name, t_name = p[0].lower(), p[1].lower()
                
                # フォルダ名がどうなっていようが、ファイル名が一致すればペアにする
                if r_name in rgb_map and t_name in thr_map:
                    pairs.append((rgb_map[r_name], thr_map[t_name]))
                    
    print(f"[SThErEO] Loaded {len(pairs)} train pairs.")
    return pairs

def build_freiburg_pairs(data_root: str, anythermal_base: str = ANYTHERMAL_BASE) -> list:
    splits_dir = os.path.join(anythermal_base, "freiburg", "splits", "frame_list")
    pairs = []
    for txt_path in glob.glob(os.path.join(splits_dir, "train_seq_*.txt")):
        fname = os.path.basename(txt_path)
        parts = fname.replace("train_", "").replace(".txt", "").split("_")
        base_dir = os.path.join(data_root, "train", "_".join(parts[:-1]), parts[-1])
        rgb_dir, thr_dir = os.path.join(base_dir, "fl_rgb"), os.path.join(base_dir, "thermal8_clahe")
        
        with open(txt_path) as f:
            for line in f:
                ts = line.strip()
                rp, tp = os.path.join(rgb_dir, f"fl_rgb_{ts}"), os.path.join(thr_dir, f"fl_ir_aligned_{ts}")
                if os.path.exists(rp) and os.path.exists(tp): pairs.append((rp, tp))
    print(f"[Freiburg] Loaded {len(pairs)} train pairs.")
    return pairs

# =====================================================================
# 🎯 データセットサンプル保存ロジック
# =====================================================================
def save_dataset_visual_checks(all_datasets, output_root, num_samples=5):
    """各データセットの Wrapper から画像を取り出し、ローカルに保存して目視確認を可能にする"""
    debug_dir = os.path.join(output_root, "dataset_checks")
    os.makedirs(debug_dir, exist_ok=True)
    
    print(f"\n🔍 Checking datasets... saving {num_samples} samples per dataset to {debug_dir}")
    
    for idx, ds in enumerate(all_datasets):
        for i in range(min(num_samples, len(ds))):
            data = ds[i]
            img0, img1 = data['image0'], data['image1'] # (3, H, W)
            
            # 辞書からデータセット名を取得（なければ連番）
            ds_name = data.get('dataset_name', f'dataset_{idx}')
            
            # RGBとThermalを横に結合
            comparison = torch.cat([img0, img1], dim=2)
            fname = f"sample_{ds_name}_{i:03d}.png"
            torchvision.utils.save_image(comparison, os.path.join(debug_dir, fname))
            
    print(f"✅ Dataset check completed. Images saved in {debug_dir}\n")


# ==========================================
# 🏋️ メイン学習スクリプト
# ==========================================

def get_args():
    parser = argparse.ArgumentParser()
    # パス設定
    parser.add_argument('--sthereo_root', type=str, default='datasets/sthereo')
    parser.add_argument('--ms2_root',     type=str, default='datasets/ms2')
    parser.add_argument('--vivid_root',   type=str, default='datasets/vivid')
    parser.add_argument('--tartanrgbt_root', type=str, default='datasets/tartanRGBT')
    parser.add_argument('--freiburg_root',   type=str, default='datasets/freiburg')
    parser.add_argument('--output',       type=str, default='checkpoints/stage1_kd')
    
    # 学習ハイパーパラメータ
    parser.add_argument('--epochs',     type=int,   default=100)
    parser.add_argument('--batch_size', type=int,   default=16)
    parser.add_argument('--lr',         type=float, default=1e-4)
    parser.add_argument('--device',     type=str,   default='0')
    parser.add_argument('--wandb_run_name', type=str, default='stage1_baseline')
    
    # アブレーション制御 (Loss)
    parser.add_argument('--use_kd_loss', action='store_true', help='KD Lossを有効にする')
    parser.add_argument('--use_hybrid_loss', action='store_true')
    parser.add_argument('--use_spatial_loss', action='store_true')
    parser.add_argument('--lambda_kd', type=float, default=1.0)
    parser.add_argument('--lambda_hybrid', type=float, default=0.1)
    parser.add_argument('--lambda_spatial', type=float, default=0.01)

    # データ拡張
    parser.add_argument('--aug_list',   type=str, default='')
    parser.add_argument('--crop_size',  type=str, default='256,256')

    # 🎯 運用機能フラグ
    parser.add_argument('--save_debug_images', action='store_true', help='学習前に各データセットのサンプル画像を保存して終了する')
    parser.add_argument('--teacher_weights', type=str, required=True, help='XFeat公式のRGB用事前学習済み重みへのパス')
    parser.add_argument('--no_wandb', action='store_true', help='WandBへのロギングを無効化する')
    
    return parser.parse_args()

def main():
    args = get_args()
    os.makedirs(args.output, exist_ok=True)
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

    aug_methods = args.aug_list.split(',') if args.aug_list else []
    crop_size = tuple(map(int, args.crop_size.split(',')))
    
    # -------------------------------------------------------------
    # 1. 本番用データセット構築
    # -------------------------------------------------------------
    all_datasets = []
    
    ms2_pairs = build_ms2_pairs(args.ms2_root)
    if ms2_pairs:
        all_datasets.append(Stage1_KDAugmentWrapper(Stage1_MS2Dataset(ms2_pairs), crop_size, True, aug_methods))
    
    # vivid_pairs = build_vivid_pairs(args.vivid_root, os.path.join(args.vivid_root, "frame_lists"))
    # if vivid_pairs:
    #     all_datasets.append(Stage1_KDAugmentWrapper(Stage1_VIVIDDataset(vivid_pairs), crop_size, True, aug_methods))

    vivid_pairs = []
    vivid_split_root = "third_party/anythermal/custom_datasets/vivid/splits/frame_lists"
    
    if os.path.exists(vivid_split_root):
        for root_dir, dirs, files in os.walk(vivid_split_root):
            # RGBとThermalの両方のテキストファイルが存在するフォルダを見つけたら
            if "rgb_framelist.txt" in files and "thermal_framelist.txt" in files:
                with open(os.path.join(root_dir, "rgb_framelist.txt")) as f:
                    rgb_lines = [line.strip() for line in f if line.strip()]
                with open(os.path.join(root_dir, "thermal_framelist.txt")) as f:
                    thr_lines = [line.strip() for line in f if line.strip()]
                
                # 行数が同じならペアにする
                if len(rgb_lines) == len(thr_lines):
                    for r, t in zip(rgb_lines, thr_lines):
                        rgb_p = os.path.join(args.vivid_root, r)
                        thr_p = os.path.join(args.vivid_root, t)
                        # 実体ファイルが存在するかチェック
                        if os.path.exists(rgb_p) and os.path.exists(thr_p):
                            vivid_pairs.append((rgb_p, thr_p))
    
    if vivid_pairs:
        print(f"✅ [VIVID] Successfully loaded {len(vivid_pairs)} pairs using os.walk!")
        all_datasets.append(Stage1_KDAugmentWrapper(Stage1_VIVIDDataset(vivid_pairs), crop_size, True, aug_methods))
    else:
        print("⚠️ [VIVID] Failed to construct pairs. Check the split text paths.")

    tartan_pairs = build_tartan_pairs(args.tartanrgbt_root)
    if tartan_pairs:
        all_datasets.append(Stage1_KDAugmentWrapper(Stage1_CleanDataset(tartan_pairs, 'tartan'), crop_size, True, aug_methods))
    
    sthereo_pairs = build_sthereo_pairs(args.sthereo_root)
    if sthereo_pairs:
        all_datasets.append(Stage1_KDAugmentWrapper(Stage1_SThErEODataset(sthereo_pairs), crop_size, True, aug_methods))

    freiburg_pairs = build_freiburg_pairs(args.freiburg_root)
    if freiburg_pairs:
        all_datasets.append(Stage1_KDAugmentWrapper(Stage1_FreiburgDataset(freiburg_pairs), crop_size, True, aug_methods))

    if not all_datasets:
        raise RuntimeError("❌ 学習データセットが1つも見つかりませんでした。パスを確認してください。")

    # 🎯 デバッグ画像のローカル保存と安全な停止
    if args.save_debug_images:
        save_dataset_visual_checks(all_datasets, args.output, num_samples=5)
        print("🛑 画像の保存が完了しました。目視確認のため、ここでプログラムを終了します。")
        sys.exit(0)

    combined_ds = ConcatDataset(all_datasets)
    loader = DataLoader(combined_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # -------------------------------------------------------------
    # 2. モデル & Optimizer 初期化
    # -------------------------------------------------------------
    wandb_mode = "disabled" if args.no_wandb else "online"
    wandb.init(project="thermal-xfeat-hybrid", name=args.wandb_run_name, config=vars(args), mode=wandb_mode)

    model = XFeatModel().to(device).train()
    teacher = XFeatModel().to(device).eval() 
    
    if not os.path.exists(args.teacher_weights):
        raise FileNotFoundError(f"❌ Teacherモデルの重みが見つかりません: {args.teacher_weights}")
    teacher.load_state_dict(torch.load(args.teacher_weights, map_location=device))
    print(f"✅ Teacher (RGB) weights loaded from: {args.teacher_weights}")
    
    for p in teacher.parameters(): p.requires_grad = False
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # -------------------------------------------------------------
    # 3. 学習ループ
    # -------------------------------------------------------------
    best_loss = float('inf')
    global_step = 0
    
    for epoch in range(1, args.epochs + 1):
        pbar = tqdm(loader, desc=f"Epoch {epoch}")
        epoch_loss_sum = 0.0
        num_batches = 0

        for batch in pbar:
            img_rgb = batch['image0'].to(device)
            img_thr = batch['image1'].to(device)
            img_raw = batch['image_raw'].to(device) # Raw Thermal (ダミー含む)
            has_raw = batch['has_raw'].to(device)   # (B,) のフラグ
            
            optimizer.zero_grad()

            with torch.no_grad():
                feats_t, _, _ = teacher(img_rgb)
            
            feats_s, kpts_s, scores_s = model(img_thr)

            log_dict = {}
            loss = torch.tensor(0.0, device=device, requires_grad=True)
            
            # 画像の周囲10ピクセルをLoss計算から除外
            feat_margin = 2  # 特徴マップ (32x32) において上下左右 4px 除外 (中央24x24を使用)
            img_margin = feat_margin * 8  # 元画像 (256x256) においては 8倍の 32px を除外
            if feat_margin > 0:
                feats_s_center = feats_s[:, :, feat_margin:-feat_margin, feat_margin:-feat_margin]
                feats_t_center = feats_t[:, :, feat_margin:-feat_margin, feat_margin:-feat_margin]
            else:
                feats_s_center = feats_s
                feats_t_center = feats_t
            
            if args.use_kd_loss:
                l_kd = kd_feature_loss(feats_s_center, feats_t_center)
                loss = loss + args.lambda_kd * l_kd
                l_kd_wandb = args.lambda_kd * l_kd
                log_dict["train/loss_kd"] = l_kd_wandb.item()
                
            if args.use_hybrid_loss:
                valid_idx = torch.nonzero(has_raw).squeeze(-1)
                if len(valid_idx) > 0:
                    # 🎯 feats_s は単一のテンソル (B, C, H, W) なので、そのまま valid_idx で抽出する
                    # feats_s_valid = feats_s[valid_idx]
                    # img_thr_valid = img_thr[valid_idx]
                    # img_raw_valid = img_raw[valid_idx]
                    feats_s_valid = feats_s_center[valid_idx]
                    
                    # 2. 画像のクロップ (margin が 0 の場合はスライスしない)
                    if img_margin > 0:
                        img_thr_valid = img_thr[valid_idx, :, img_margin:-img_margin, img_margin:-img_margin]
                        img_raw_valid = img_raw[valid_idx, :, img_margin:-img_margin, img_margin:-img_margin]
                    else:
                        img_thr_valid = img_thr[valid_idx]
                        img_raw_valid = img_raw[valid_idx]
                    
                    # 抽出したサンプルだけで Hybrid Loss を計算
                    l_hyb = (feats_s_valid, img_thr_valid, img_raw_valid) 
                    loss = loss + args.lambda_hybrid * l_hyb
                    l_hyb_wandb = args.lambda_hybrid * l_hyb
                    log_dict["train/loss_hybrid"] = l_hyb_wandb.item()
                else:
                    log_dict["train/loss_hybrid"] = 0.0


            if args.use_spatial_loss:
                if feat_margin > 0:
                    scores_s_center = scores_s[:, :, feat_margin:-feat_margin, feat_margin:-feat_margin]
                else:
                    scores_s_center = scores_s
                l_sp = spatial_entropy_loss(scores_s)
                loss = loss + args.lambda_spatial * l_sp
                log_dict["train/loss_spatial"] = l_sp.item()
            
            log_dict["train/loss_total"] = loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss_sum += loss.item()
            num_batches += 1

            if global_step % 10 == 0:
                wandb.log(log_dict, step=global_step)

            if global_step % 500 == 0:
                if img_margin > 0:
                    img_rgb_center = img_rgb[0, :, img_margin:-img_margin, img_margin:-img_margin]
                    img_thr_center = img_thr[0, :, img_margin:-img_margin, img_margin:-img_margin]
                else:
                    img_rgb_center = img_rgb[0]
                    img_thr_center = img_thr[0]
                
                log_stage1_kd_to_wandb(
                    rgb_img=img_rgb_center, 
                    thr_img=img_thr_center, 
                    teacher_heatmap=feats_t_center[0], # 上の処理で正しく分岐されたものをそのまま使う
                    student_heatmap=feats_s_center[0], # 上の処理で正しく分岐されたものをそのまま使う
                    step=global_step
                )
                # log_stage1_kd_to_wandb(
                #     rgb_img=img_rgb[0], 
                #     thr_img=img_thr[0], 
                #     teacher_heatmap=feats_t[0], 
                #     student_heatmap=feats_s[0], 
                #     step=global_step
                # )

            global_step += 1
            pbar.set_postfix({'Loss': f"{loss.item():.4f}"})

        # エポック終了時のBest判定
        avg_epoch_loss = epoch_loss_sum / num_batches
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_path = os.path.join(args.output, 'model_stage1_best.pth')
            torch.save(model.state_dict(), best_path)
            print(f"\n🌟 Best model updated at Epoch {epoch} (Avg Loss: {best_loss:.4f})")
            
            if not args.no_wandb:
                wandb.run.summary["best_epoch"] = epoch
                wandb.run.summary["best_loss"] = best_loss

        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(args.output, f'model_epoch_{epoch}.pth'))

    torch.save(model.state_dict(), os.path.join(args.output, 'final_model.pth'))
    print("✅ Stage 1 Training completed.")

if __name__ == '__main__':
    main()