"""
evaluate/vo_visualize.py
Visual Odometry 軌跡の可視化。

出力:
  - trajectory_top.png  : 上面図（X-Z平面）
  - trajectory_3d.png   : 3D 軌跡
  - error_plot.png      : フレームごとの位置誤差
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from mpl_toolkits.mplot3d import Axes3D
    _MPL_OK = True
except ImportError:
    _MPL_OK = False


# ---------------------------------------------------------------------------
# カラーパレット（モデルごとに色を割り当て）
# ---------------------------------------------------------------------------

_COLORS = [
    '#E24B4A',   # 赤: proposed
    '#378ADD',   # 青: baseline
    '#1D9E75',   # 緑
    '#EF9F27',   # 橙
    '#7F77DD',   # 紫
]
_GT_COLOR  = '#2C2C2A'    # 黒: GT
_GT_ALPHA  = 0.6


def _umeyama_align(
    est: np.ndarray,   # (N, 3)
    gt:  np.ndarray,   # (N, 3)
) -> np.ndarray:
    """Umeyama alignment で est を gt に揃える。"""
    n = len(est)
    mu_e, mu_g = est.mean(0), gt.mean(0)
    ec, gc     = est - mu_e, gt - mu_g
    sigma_e    = (ec ** 2).sum() / n
    H          = (gc.T @ ec) / n
    U, S, Vt   = np.linalg.svd(H)
    d          = np.linalg.det(U @ Vt)
    D          = np.diag([1., 1., d])
    R          = U @ D @ Vt
    scale      = float(S.sum() / sigma_e) if sigma_e > 1e-8 else 1.0
    return (scale * (R @ ec.T).T) + mu_g


def plot_trajectory_comparison(
    seq_name:    str,
    gt_traj:     np.ndarray,           # (N, 3) GT 軌跡
    est_trajs:   Dict[str, np.ndarray], # {model_name: (N, 3)}
    ate_dict:    Dict[str, float],      # {model_name: ATE_RMSE}
    output_dir:  str,
    dataset_name: str = '',
) -> str:
    """
    軌跡の比較プロットを生成して保存する。

    Args:
        seq_name:     シーケンス名
        gt_traj:      GT 軌跡 (N, 3)
        est_trajs:    {model_name: aligned_est (N, 3)}
        ate_dict:     {model_name: ATE_RMSE [m]}
        output_dir:   保存先ディレクトリ
        dataset_name: データセット名（タイトル用）

    Returns:
        保存したファイルのパス
    """
    if not _MPL_OK:
        print("[Vis] matplotlib not available")
        return ""

    os.makedirs(output_dir, exist_ok=True)

    # ── アライン済み軌跡を準備 ────────────────────────────────────────────
    aligned: Dict[str, np.ndarray] = {}
    for name, est in est_trajs.items():
        if len(est) < 3:
            continue
        n = min(len(est), len(gt_traj))
        aligned[name] = _umeyama_align(est[:n], gt_traj[:n])

    # ── 図の作成 ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 12))
    fig.suptitle(
        f'{dataset_name} / {seq_name}',
        fontsize=14, fontweight='bold', y=0.98)

    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.38, wspace=0.32)

    ax_top  = fig.add_subplot(gs[0, 0])                    # 上面図 (X-Z)
    ax_side = fig.add_subplot(gs[0, 1])                    # 側面図 (X-Y)
    ax_yz   = fig.add_subplot(gs[0, 2])                    # 正面図 (Y-Z)
    ax_3d   = fig.add_subplot(gs[0, 3], projection='3d')   # 3D
    ax_err  = fig.add_subplot(gs[1, :3])                   # 誤差プロット
    ax_info = fig.add_subplot(gs[1, 3])                    # テキスト情報

    def _draw_plane(ax, xi, yi, xlabel, ylabel, title):
        """共通の2D軌跡描画ヘルパー (xi, yi はカラム番号)"""
        ax.set_title(title, fontsize=11)
        ax.plot(gt_traj[:, xi], gt_traj[:, yi],
                color=_GT_COLOR, linewidth=2.0, alpha=_GT_ALPHA,
                label='GT', zorder=5)
        ax.plot(gt_traj[0, xi], gt_traj[0, yi],
                'o', color=_GT_COLOR, markersize=8, zorder=6)
        for ci, (name, traj) in enumerate(aligned.items()):
            color = _COLORS[ci % len(_COLORS)]
            ate   = ate_dict.get(name, float('inf'))
            ax.plot(traj[:, xi], traj[:, yi],
                    color=color, linewidth=1.5, alpha=0.85,
                    label=f"{name}\nATE={ate:.3f}m", zorder=4)
            ax.plot(traj[0, xi], traj[0, yi],
                    's', color=color, markersize=6, zorder=6)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.legend(fontsize=7, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='datalim')

    _draw_plane(ax_top,  0, 2, 'X [m]', 'Z [m]', 'Top view (X-Z)')
    _draw_plane(ax_side, 0, 1, 'X [m]', 'Y [m]', 'Side view (X-Y)')
    _draw_plane(ax_yz,   1, 2, 'Y [m]', 'Z [m]', 'Front view (Y-Z)')

    # ── 3D 軌跡 ─────────────────────────────────────────────────────────
    ax_3d.set_title('3D trajectory', fontsize=11)
    ax_3d.plot(gt_traj[:, 0], gt_traj[:, 2], gt_traj[:, 1],
               color=_GT_COLOR, linewidth=2.0, alpha=_GT_ALPHA,
               label='GT')
    for ci, (name, traj) in enumerate(aligned.items()):
        color = _COLORS[ci % len(_COLORS)]
        ax_3d.plot(traj[:, 0], traj[:, 2], traj[:, 1],
                   color=color, linewidth=1.5, alpha=0.85, label=name)
    ax_3d.set_xlabel('X'); ax_3d.set_ylabel('Z'); ax_3d.set_zlabel('Y')
    ax_3d.legend(fontsize=7, loc='best')

    # ── フレームごとの誤差 ────────────────────────────────────────────────
    ax_err.set_title('Per-frame position error (after alignment)', fontsize=11)
    ax_err.set_xlabel('Frame index')
    ax_err.set_ylabel('Error [m]')

    for ci, (name, traj) in enumerate(aligned.items()):
        color = _COLORS[ci % len(_COLORS)]
        n     = min(len(traj), len(gt_traj))
        errs  = np.linalg.norm(traj[:n] - gt_traj[:n], axis=1)
        ax_err.plot(errs, color=color, linewidth=1.0, alpha=0.8, label=name)
        # 移動平均（平滑化）
        if len(errs) > 20:
            kernel = np.ones(20) / 20
            smooth = np.convolve(errs, kernel, mode='valid')
            ax_err.plot(
                np.arange(len(smooth)) + 10,
                smooth, color=color, linewidth=2.0, alpha=1.0)

    ax_err.legend(fontsize=8, loc='best')
    ax_err.grid(True, alpha=0.3)
    ax_err.set_ylim(bottom=0)

    # ── テキスト情報 ─────────────────────────────────────────────────────
    ax_info.axis('off')
    lines = [f"Sequence: {seq_name}", f"Frames: {len(gt_traj)}", ""]

    for name in aligned:
        ate = ate_dict.get(name, float('inf'))
        lines.append(f"[{name}]")
        lines.append(f"  ATE RMSE = {ate:.4f} m")
        lines.append("")

    ax_info.text(0.05, 0.95, '\n'.join(lines),
                 transform=ax_info.transAxes,
                 fontsize=9, verticalalignment='top',
                 fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='#F1EFE8',
                           alpha=0.5))

    # ── 保存 ─────────────────────────────────────────────────────────────
    safe_name = seq_name.replace('/', '_').replace(' ', '_')
    out_path  = os.path.join(output_dir, f'traj_{safe_name}.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path


def plot_summary_bar(
    results:    Dict[str, Dict[str, float]],  # {ds/seq/model: ATE}
    output_dir: str,
) -> str:
    """
    全シーケンスの ATE を棒グラフで比較する。

    Args:
        results: {dataset_name: {model_name: avg_ate_rmse}}
    """
    if not _MPL_OK or not results:
        return ""

    os.makedirs(output_dir, exist_ok=True)

    datasets = list(results.keys())
    models   = list(list(results.values())[0].keys()) if results else []
    n_ds     = len(datasets)
    n_models = len(models)

    fig, axes = plt.subplots(1, n_ds, figsize=(6 * n_ds, 5))
    if n_ds == 1:
        axes = [axes]

    fig.suptitle('ATE RMSE Comparison [m]  (lower is better)',
                 fontsize=13, fontweight='bold')

    for ax, ds_name in zip(axes, datasets):
        ds_res = results[ds_name]
        vals   = [ds_res.get(m, float('inf')) for m in models]

        x      = np.arange(len(models))
        bars   = ax.bar(x, vals, color=_COLORS[:len(models)],
                        edgecolor='white', linewidth=0.5)

        # 値ラベル
        for bar, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f'{v:.3f}', ha='center', va='bottom',
                        fontsize=9, fontweight='bold')

        ax.set_title(ds_name, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [m.replace('_', '\n') for m in models],
            fontsize=8)
        ax.set_ylabel('ATE RMSE [m]')
        ax.set_ylim(bottom=0)
        ax.grid(True, axis='y', alpha=0.3)

    out_path = os.path.join(output_dir, 'ate_summary.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path