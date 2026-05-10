"""
modules/training/visualization.py
W&Bへのヒートマップおよびキーポイント可視化モジュール
"""
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import wandb

def tensor_to_cv2(tensor_img):
    """(3, H, W) Tensor を (H, W, 3) の uint8 numpy 配列に変換"""
    img_np = tensor_img.detach().cpu().permute(1, 2, 0).numpy()
    img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
    return img_np

def overlay_heatmap(img_np, heatmap_tensor, alpha=0.5):
    """画像の上にヒートマップをカラーマップ(JET)で合成する"""
    hm_np = heatmap_tensor.detach().cpu().numpy()
    # 0~1 に正規化
    hm_np = (hm_np - hm_np.min()) / (hm_np.max() - hm_np.min() + 1e-8)
    
    # カラーマップ適用
    hm_color = cv2.applyColorMap((hm_np * 255).astype(np.uint8), cv2.COLORMAP_JET)
    hm_color = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)

    h, w = img_np.shape[:2]
    hm_color = cv2.resize(hm_color, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # ブレンド
    blended = cv2.addWeighted(img_np, 1 - alpha, hm_color, alpha, 0)
    return blended

def log_stage1_kd_to_wandb(rgb_img, thr_img, teacher_heatmap, student_heatmap, step):
    """
    [Stage 1用] 教師(RGB)と生徒(Thermal)のヒートマップを比較する
    """
    rgb_np = tensor_to_cv2(rgb_img)
    thr_np = tensor_to_cv2(thr_img)
    
    # C,H,W の特徴量マップからヒートマップ(空間的な強さ)を作る場合、チャネル平均やL2ノルムを使用
    if teacher_heatmap.ndim == 3: teacher_heatmap = torch.norm(teacher_heatmap, p=2, dim=0)
    if student_heatmap.ndim == 3: student_heatmap = torch.norm(student_heatmap, p=2, dim=0)

    t_overlay = overlay_heatmap(rgb_np, teacher_heatmap)
    s_overlay = overlay_heatmap(thr_np, student_heatmap)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(t_overlay)
    axes[0].set_title("Teacher (RGB) Heatmap")
    axes[0].axis('off')
    
    axes[1].imshow(s_overlay)
    axes[1].set_title("Student (Thermal) Heatmap")
    axes[1].axis('off')
    
    plt.tight_layout()
    wandb.log({"visuals/Stage1_KD_Heatmaps": wandb.Image(fig)}, step=step)
    plt.close(fig)

def log_stage23_geometry_to_wandb(thr_img, kpts, heatmap, orig_size, step, stage_name="Stage2"):
    """
    [Stage 2/3用] Thermal画像上のキーポイント分布とパディング境界(赤枠)を可視化する
    """
    thr_np = tensor_to_cv2(thr_img)
    k_np = kpts.detach().cpu().numpy()
    
    if heatmap.ndim == 3: heatmap = torch.norm(heatmap, p=2, dim=0)
    h_overlay = overlay_heatmap(thr_np, heatmap)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # --- 左: キーポイントとパディング枠 ---
    axes[0].imshow(thr_np)
    axes[0].scatter(k_np[:, 0], k_np[:, 1], c='lime', s=2, alpha=0.8) # キーポイント(緑)
    
    # パディング前の元のサイズを赤枠で描画
    w, h = orig_size[0].item(), orig_size[1].item()
    import matplotlib.patches as patches
    rect = patches.Rectangle((0, 0), w, h, linewidth=2, edgecolor='red', facecolor='none', linestyle='--')
    axes[0].add_patch(rect)
    axes[0].set_title(f"Keypoints (Red Box: Valid {w}x{h})")
    axes[0].axis('off')
    
    # --- 右: ヒートマップ ---
    axes[1].imshow(h_overlay)
    axes[1].set_title("XFeat Reliability Heatmap")
    axes[1].axis('off')
    
    plt.tight_layout()
    wandb.log({f"visuals/{stage_name}_Geometry": wandb.Image(fig)}, step=step)
    plt.close(fig)