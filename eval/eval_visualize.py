"""
evaluate/eval_visualize.py
評価結果の可視化モジュール。

evaluate.py が呼ぶ4関数を提供する：
    plot_auc_curves()     : モデル比較 AUC 曲線
    plot_error_histogram(): 再投影誤差ヒストグラム（空配列を許容）
    plot_summary_table()  : 結果比較表（PNG）
    save_match_images()   : 連続フレームのマッチング可視化

NOTE: evaluate/eval_matching.py の内部関数には依存しない独立実装。
      旧版は apply_homography / extract_features 等を import していたが
      それらは eval_matching.py に存在しないため、完全に独立した実装に変更。
"""

from __future__ import annotations

import os
import random
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _MPL_OK = True
except ImportError:
    _MPL_OK = False
    print("[Vis] WARNING: matplotlib not installed — figures will be skipped.")


# ---------------------------------------------------------------------------
# 内部ユーティリティ
# ---------------------------------------------------------------------------

def _imread(path: str, is_thermal: bool,
            size: Tuple[int, int]) -> Optional[np.ndarray]:
    """BGR ndarray を返す。16bit PNG にも対応。失敗時は None。"""
    if not os.path.isfile(path):
        return None
    if is_thermal:
        # 8bit グレースケールで試みる
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            # 16bit PNG フォールバック
            gray16 = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
            if gray16 is None:
                return None
            gray = cv2.normalize(gray16, None, 0, 255,
                                 cv2.NORM_MINMAX).astype(np.uint8)
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    else:
        bgr = cv2.imread(path)
        if bgr is None:
            return None
    return cv2.resize(bgr, size)


@torch.no_grad()
def _detect(model, bgr, device, max_kp):
    """
    XFeat のキーポイント検出。

    キーポイント位置の選択には kp_logits（検出ヘッド）を使う。
    hmap（信頼性マップ）は重みとして乗算する。

    combined_score = kp_score * hmap
      kp_score = softmax(kp_logits[:, :64], dim=1).max()  ← どこにキーポイントがあるか
      hmap                                                  ← どれだけ信頼できるか
    """
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    t = (torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
         ).unsqueeze(0).to(device)
    feats, kp_logits, hmap = model(t)
    feats = F.normalize(feats, dim=1)
    B, C, Hf, Wf = feats.shape
    H, W = t.shape[2], t.shape[3]

    # P(keypoint) = 1 - P(dustbin) : 65ch softmax で dustbin を正しく考慮
    probs    = F.softmax(kp_logits, dim=1)          # (B, 65, Hf, Wf)
    kp_score = probs[:, :64].sum(dim=1)             # (B, Hf, Wf)
    scores   = kp_score[0].cpu().numpy().flatten()  # (Hf*Wf,)

    feats_np = feats[0].reshape(C, -1).permute(1, 0).cpu().numpy()

    top  = np.argsort(scores)[::-1][:min(max_kp, len(scores))]
    ys   = (top // Wf).astype(np.float32) * (H / Hf)
    xs   = (top  % Wf).astype(np.float32) * (W / Wf)
    kpts = np.stack([xs, ys], axis=1)
    desc = feats_np[top].astype(np.float32)
    return kpts, desc


def _mutual_nn(d1, d2):
    if len(d1) == 0 or len(d2) == 0:
        return np.array([], np.int64), np.array([], np.int64)
    d1n = d1 / (np.linalg.norm(d1, axis=1, keepdims=True) + 1e-8)
    d2n = d2 / (np.linalg.norm(d2, axis=1, keepdims=True) + 1e-8)
    sim  = d1n @ d2n.T
    nn12 = np.argmax(sim, axis=1)
    nn21 = np.argmax(sim, axis=0)
    ids  = np.arange(len(d1))
    mask = nn21[nn12] == ids
    return ids[mask], nn12[mask]


def _classify_inliers(
    kpts1: np.ndarray,
    kpts2: np.ndarray,
    idx1:  np.ndarray,
    idx2:  np.ndarray,
    ransac_th: float = 5.0,
) -> np.ndarray:
    """
    RANSAC ホモグラフィー推定でインライア/アウトライアを判定する。

    Args:
        kpts1, kpts2 : (N, 2) キーポイント座標
        idx1, idx2   : マッチインデックス
        ransac_th    : インライア判定の再投影誤差閾値（画素）

    Returns:
        inlier_mask : (M,) bool array  True = インライア（正解マッチ）
    """
    n = len(idx1)
    if n < 4:
        return np.zeros(n, dtype=bool)

    pts1 = kpts1[idx1].reshape(-1, 1, 2).astype(np.float32)
    pts2 = kpts2[idx2].reshape(-1, 1, 2).astype(np.float32)

    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, ransac_th)
    if H is None or mask is None:
        return np.zeros(n, dtype=bool)

    return mask.ravel().astype(bool)


def _draw_matches_bgr(img1, img2, kpts1, kpts2, idx1, idx2,
                      color=(0, 255, 0), label="", max_draw=150,
                      show_inlier_color=True, ransac_th=5.0):
    """
    2枚を横並びにしてマッチング線を描く。

    Args:
        color            : show_inlier_color=False のときの単色
        show_inlier_color: True  → インライア=緑、アウトライア=赤
                           False → 全マッチを color で描画
        ransac_th        : RANSAC インライア判定閾値（画素）
    """
    H, W = img1.shape[:2]
    canvas = np.hstack([img1.copy(), img2.copy()])

    if len(idx1) == 0:
        if label:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(canvas, label, (8, 26), font, 0.6, (0, 0, 0), 3)
            cv2.putText(canvas, label, (8, 26), font, 0.6, (255, 255, 255), 1)
        return canvas

    # インライア判定
    if show_inlier_color and len(idx1) >= 4:
        inlier_mask = _classify_inliers(kpts1, kpts2, idx1, idx2, ransac_th)
    else:
        inlier_mask = np.ones(len(idx1), dtype=bool)

    n_inlier  = int(inlier_mask.sum())
    n_total   = len(idx1)

    # 描画数を制限（インライアを優先的に表示）
    inlier_idx  = np.where(inlier_mask)[0]
    outlier_idx = np.where(~inlier_mask)[0]
    rng = np.random.default_rng(42)

    n_draw_in  = min(max_draw * 2 // 3, len(inlier_idx))
    n_draw_out = min(max_draw // 3,     len(outlier_idx))

    draw_inlier  = (rng.choice(inlier_idx,  n_draw_in,  replace=False)
                    if n_draw_in  > 0 else np.array([], dtype=int))
    draw_outlier = (rng.choice(outlier_idx, n_draw_out, replace=False)
                    if n_draw_out > 0 else np.array([], dtype=int))

    # 色の定義
    COLOR_INLIER  = (0,   220,  0)    # 緑: インライア（正解）
    COLOR_OUTLIER = (0,   0,   220)   # 赤: アウトライア（不正解）

    def _draw_one(di, c):
        x1 = int(kpts1[idx1[di]][0])
        y1 = int(kpts1[idx1[di]][1])
        x2 = int(kpts2[idx2[di]][0]) + W
        y2 = int(kpts2[idx2[di]][1])
        cv2.line(canvas,   (x1, y1), (x2, y2), c, 1, cv2.LINE_AA)
        cv2.circle(canvas, (x1, y1), 3, c, -1)
        cv2.circle(canvas, (x2, y2), 3, c, -1)

    # アウトライア→インライアの順に描画（インライアが上に来る）
    for di in draw_outlier:
        c = COLOR_OUTLIER if show_inlier_color else color
        _draw_one(di, c)
    for di in draw_inlier:
        c = COLOR_INLIER if show_inlier_color else color
        _draw_one(di, c)

    # ラベル（インライア数 / 全マッチ数 を表示）
    if label:
        inlier_str = f"  {n_inlier}/{n_total}" if show_inlier_color else ""
        full_label = f"{label}{inlier_str}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(canvas, full_label, (8, 26), font, 0.55, (0, 0, 0),    3)
        cv2.putText(canvas, full_label, (8, 26), font, 0.55, (255, 255, 255), 1)

        # 凡例（右下）
        if show_inlier_color:
            lx, ly = canvas.shape[1] - 180, canvas.shape[0] - 30
            cv2.circle(canvas, (lx, ly), 5, COLOR_INLIER,  -1)
            cv2.putText(canvas, 'inlier',  (lx+10, ly+5), font, 0.45,
                        (255, 255, 255), 1)
            cv2.circle(canvas, (lx+80, ly), 5, COLOR_OUTLIER, -1)
            cv2.putText(canvas, 'outlier', (lx+90, ly+5), font, 0.45,
                        (255, 255, 255), 1)

    return canvas


# ---------------------------------------------------------------------------
# save_match_images
# ---------------------------------------------------------------------------

def save_match_images(models_cfg, models, pairs, device, output_dir,
                      dataset_name, cfg, n_samples=10, seed=42,
                      lightglue_model=None):
    """
    マッチング可視化画像を保存する。

    Args:
        lightglue_model: fine-tuning 済み LightGlue モデル。
                         None の場合は MNN で代替。
    """
    vis_dir = os.path.join(output_dir, 'match_images', dataset_name)
    os.makedirs(vis_dir, exist_ok=True)
    size   = (cfg.get('viz_width', 640), cfg.get('viz_height', 480))
    max_kp = cfg.get('max_keypoints', 2048)
    rng    = random.Random(seed)
    picks  = rng.sample(range(len(pairs) - 1), min(n_samples, len(pairs) - 1))
    colors_bgr = [(255, 80, 0), (0, 200, 80), (0, 80, 255), (180, 0, 180)]
    saved = 0
    H_img, W_img = size[1], size[0]

    use_lg = lightglue_model is not None
    if use_lg:
        from eval.eval_matching import match_lightglue
        print(f"[Vis] Using LightGlue for matching visualization")

    # 最初の1件でパスの存在を診断
    if pairs:
        rgb0, thr0 = pairs[0]
        print(f"[Vis] path check:")
        print(f"      rgb exists={os.path.isfile(rgb0)} → {rgb0}")
        print(f"      thr exists={os.path.isfile(thr0)} → {thr0}")

    for si, idx in enumerate(picks):
        rgb1_p, thr1_p = pairs[idx]
        rgb2_p, thr2_p = pairs[idx + 1]
        rows = []
        for mi, m_cfg in enumerate(models_cfg):
            model_name = m_cfg['name']
            modality   = m_cfg['modality']
            model      = models[model_name]
            is_thr     = (modality == 'thermal')
            p1 = thr1_p if is_thr else rgb1_p
            p2 = thr2_p if is_thr else rgb2_p
            bgr1 = _imread(p1, is_thr, size)
            bgr2 = _imread(p2, is_thr, size)
            if bgr1 is None or bgr2 is None:
                if si == 0:
                    print(f"[Vis] imread FAILED — model={model_name}")
                    print(f"      p1 exists={os.path.isfile(p1)} → {p1}")
                    print(f"      p2 exists={os.path.isfile(p2)} → {p2}")
                continue
            kpts1, desc1 = _detect(model, bgr1, device, max_kp)
            kpts2, desc2 = _detect(model, bgr2, device, max_kp)

            # LightGlue または MNN でマッチング
            if use_lg:
                idx1, idx2 = match_lightglue(
                    kpts1, desc1, kpts2, desc2,
                    image_size=(H_img, W_img),
                    device=device,
                    lightglue_model=lightglue_model,
                )
                matcher_tag = 'LG'
            else:
                idx1, idx2 = _mutual_nn(desc1, desc2)
                matcher_tag = 'MNN'

            color = colors_bgr[mi % len(colors_bgr)]
            label = f"{model_name} [{matcher_tag}]  matches={len(idx1)}"
            row   = _draw_matches_bgr(
                bgr1, bgr2, kpts1, kpts2, idx1, idx2,
                color=color,
                label=label,
                show_inlier_color=True,   # インライア=緑、アウトライア=赤
                ransac_th=cfg.get('ransac_th', 5.0),
            )
            rows.append(row)
        if not rows:
            continue
        w_max = max(r.shape[1] for r in rows)
        out_rows = []
        for r in rows:
            if r.shape[1] != w_max:
                h_new = int(r.shape[0] * w_max / r.shape[1])
                r = cv2.resize(r, (w_max, h_new))
            out_rows.append(r)
        canvas = np.vstack(out_rows)
        cv2.imwrite(os.path.join(vis_dir, f'sample_{si:04d}.jpg'), canvas)
        saved += 1
    print(f"[Vis] {saved} match images saved → {vis_dir}/")


# ---------------------------------------------------------------------------
# plot_auc_curves
# ---------------------------------------------------------------------------

def plot_auc_curves(results, output_dir, auc_thresholds):
    if not _MPL_OK:
        return
    fig_dir = os.path.join(output_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    dataset_names = list(results.keys())
    n_ds = len(dataset_names)
    fig, axes = plt.subplots(1, n_ds, figsize=(6 * n_ds, 5), squeeze=False)
    colors = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0', '#FF9800']
    for di, ds_name in enumerate(dataset_names):
        ax = axes[0][di]
        for mi, (model_name, metrics) in enumerate(results[ds_name].items()):
            auc_dict = metrics.auc if hasattr(metrics, 'auc') else {}
            thrs = sorted(auc_dict.keys())
            aucs = [auc_dict[t] for t in thrs]
            ax.plot(thrs, aucs, marker='o', linewidth=2, markersize=6,
                    color=colors[mi % len(colors)], label=model_name)
        ax.set_xlabel('Reprojection Error Threshold (px)', fontsize=12)
        ax.set_ylabel('AUC', fontsize=12)
        ax.set_title(ds_name, fontsize=13, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.set_xticks(auc_thresholds)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    plt.suptitle('Keypoint Matching AUC', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(fig_dir, 'auc_curves.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Vis] AUC curves → {save_path}")


# ---------------------------------------------------------------------------
# plot_error_histogram
# ---------------------------------------------------------------------------

def plot_error_histogram(all_errors, output_dir, max_error=20.0):
    if not _MPL_OK:
        return
    fig_dir = os.path.join(output_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    dataset_names = list(all_errors.keys())
    n_ds = len(dataset_names)
    fig, axes = plt.subplots(1, n_ds, figsize=(6 * n_ds, 4), squeeze=False)
    colors = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0', '#FF9800']
    bins = np.linspace(0, max_error, 50)
    for di, ds_name in enumerate(dataset_names):
        ax = axes[0][di]
        for mi, (model_name, errs) in enumerate(all_errors[ds_name].items()):
            if len(errs) == 0:
                continue
            ax.hist(np.clip(errs, 0, max_error), bins=bins, alpha=0.5,
                    color=colors[mi % len(colors)], label=model_name, density=True)
        ax.set_xlabel('Reprojection Error (px)', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(ds_name, fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    plt.suptitle('Reprojection Error Distribution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(fig_dir, 'error_histogram.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Vis] Error histogram → {save_path}")


# ---------------------------------------------------------------------------
# plot_summary_table
# ---------------------------------------------------------------------------

def plot_summary_table(results, output_dir, auc_thresholds):
    if not _MPL_OK:
        return
    fig_dir = os.path.join(output_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    headers = (['Model', 'Dataset'] +
               [f'AUC@{t}px' for t in sorted(auc_thresholds)] +
               ['MS(%)', 'kpts', 'time(ms)'])
    rows = []
    for ds_name, ds_results in results.items():
        for model_name, m in ds_results.items():
            row = [model_name, ds_name]
            for t in sorted(auc_thresholds):
                v = m.auc.get(t, 0) if hasattr(m, 'auc') else 0
                row.append(f"{v * 100:.1f}")
            ms = m.matching_score  if hasattr(m, 'matching_score')  else 0.0
            nk = m.mean_n_kpts     if hasattr(m, 'mean_n_kpts')     else 0.0
            ts = m.mean_time_sec   if hasattr(m, 'mean_time_sec')   else 0.0
            row += [f"{ms*100:.1f}", f"{nk:.0f}", f"{ts*1000:.1f}"]
            rows.append(row)
    n_rows = max(len(rows), 1)
    n_cols = len(headers)
    fig, ax = plt.subplots(figsize=(max(12, n_cols * 1.6),
                                    max(3, n_rows * 0.5 + 1.2)))
    ax.axis('off')
    table = ax.table(
        cellText  = rows if rows else [['—'] * n_cols],
        colLabels = headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    for j in range(n_cols):
        table[0, j].set_facecolor('#1565C0')
        table[0, j].set_text_props(color='white', fontweight='bold')
    plt.title('Matching Accuracy Summary', fontsize=13, fontweight='bold', pad=10)
    plt.tight_layout()
    save_path = os.path.join(fig_dir, 'summary_table.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Vis] Summary table → {save_path}")