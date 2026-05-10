import torch
import numpy as np
import cv2
import wandb
import matplotlib.pyplot as plt

def compute_pose_metrics(matches, kpts0, kpts1, T_rel, K):
    """RANSACを用いてポーズ誤差とPrecisionを計算する"""
    if len(matches) < 8:
        return 90.0, 0.0 # 誤差最大、Precision 0
    
    # Numpyへ変換
    p0 = kpts0[matches[:, 0]].cpu().numpy()
    p1 = kpts1[matches[:, 1]].cpu().numpy()
    K_np = K.cpu().numpy()
    T_gt = T_rel.cpu().numpy()
    
    # 1. 精度の計算 (RANSACのインライア率を指標とする)
    E, mask = cv2.findEssentialMat(p0, p1, K_np, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if E is None or mask is None: return 90.0, 0.0
    precision = np.mean(mask)
    
    # 2. 姿勢推定誤差の計算
    _, R_est, t_est, _ = cv2.recoverPose(E, p0, p1, K_np, mask=mask)
    
    # 回転誤差
    R_gt = T_gt[:3, :3]
    cos_theta = (np.trace(R_gt.T @ R_est) - 1) / 2
    err_R = np.rad2deg(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
    
    # 並進誤差 (方向のみ)
    t_gt = T_gt[:3, 3]
    t_gt = t_gt / (np.linalg.norm(t_gt) + 1e-8)
    cos_alpha = np.dot(t_gt.flatten(), t_est.flatten())
    err_t = np.rad2deg(np.arccos(np.clip(cos_alpha, -1.0, 1.0)))
    
    return max(err_R, err_t), precision

def log_matching_and_metrics(img0, img1, kpts0, kpts1, matches, T_rel, K, step, stage_name):
    """WandBにマッチング画像とPoseAUCを記録する"""
    error, precision = compute_pose_metrics(matches, kpts0, kpts1, T_rel, K)
    
    # 画像の可視化
    img0_np = (img0.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
    img1_np = (img1.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
    
    # マッチング線を引く (簡易版)
    vis_img = np.concatenate([img0_np, img1_np], axis=1)

    if vis_img.dtype != np.uint8:
        vis_img = (vis_img * 255).clip(0, 255).astype(np.uint8)
    
    vis_img = np.ascontiguousarray(vis_img)
    w = img0_np.shape[1]
    
    # 描画 (上位30本程度)
    for m in matches[:30]:
        pt0 = tuple(kpts0[m[0]].cpu().numpy().astype(int))
        pt1 = tuple(kpts1[m[1]].cpu().numpy().astype(int) + [w, 0])
        cv2.line(vis_img, pt0, pt1, (0, 255, 0), 1)

    wandb.log({
        f"visuals/{stage_name}_matching": wandb.Image(vis_img),
        f"metrics/pose_error": error,
        f"metrics/precision": precision,
    }, step=step)
    
    return error