"""
eval/eval_fpn_noise.py
FPN（固定パターンノイズ）耐性評価スクリプト。

新規性②「物理考慮損失」の有効性を定量的に示す。

評価方法:
    ノイズなし熱画像 vs ノイズあり熱画像 のマッチング精度を
    sigma_levels（ノイズ強度）ごとに測定し、精度劣化曲線を描く。

    列方向固定ノイズ: col_noise = randn(W) * sigma  (各列に同じ値)
    sigma = [0, 2, 4, 6, 8, 12, 16] DN単位

比較対象:
    teacher_thr : XFeat（元モデル）に熱画像を入力  ← KD前のベースライン
    student_thr : 提案手法（KD済み）に熱画像を入力 ← 提案手法

    提案手法はノイズ強度が上がっても精度低下が少ないことを示す。

使用方法:
    python eval/eval_fpn_noise.py --config configs/eval_config.yaml
    python eval/eval_fpn_noise.py --config configs/eval_config.yaml \\
        --datasets freiburg --sigma_levels 0 2 4 8 16 --n_eval_pairs 200
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from modules.model import XFeatModel


# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Thermal XFeat — FPN Noise Robustness Evaluation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--config',       type=str, required=True)
    parser.add_argument('--student_weights', type=str, default=None)
    parser.add_argument('--teacher_weights', type=str, default=None)
    parser.add_argument('--datasets',     nargs='+', default=None)
    parser.add_argument('--split',        type=str,  default=None)
    parser.add_argument('--n_eval_pairs', type=int,  default=None,
                        help='1データセットあたり評価ペア数')
    parser.add_argument('--sigma_levels', nargs='+', type=float, default=None,
                        help='ノイズ強度リスト（DN単位）例: 0 2 4 8 16')
    parser.add_argument('--output_dir',   type=str,  default=None)
    parser.add_argument('--device_num',   type=str,  default=None)

    cli = parser.parse_args()
    if not os.path.isfile(cli.config):
        parser.error(f'--config not found: {cli.config!r}')

    with open(cli.config) as f:
        cfg = yaml.safe_load(f) or {}

    for k, v in vars(cli).items():
        if k != 'config' and v is not None:
            cfg[k] = v

    defaults = dict(
        datasets         = ['freiburg'],
        split            = 'val',
        n_eval_pairs     = 300,
        sigma_levels     = [0, 2, 4, 6, 8, 12, 16],
        auc_thresholds   = [3, 5, 10],
        matching_method  = 'mutual_nn',
        ratio_threshold  = 0.9,
        max_keypoints    = 2048,
        output_dir       = 'eval/results',
        viz_width        = 640,
        viz_height       = 480,
        seed             = 42,
        device_num       = '0',
    )
    for k, v in defaults.items():
        cfg.setdefault(k, v)

    return argparse.Namespace(**cfg)


# ---------------------------------------------------------------------------
# モデルロード
# ---------------------------------------------------------------------------

def load_models(args, device) -> Dict[str, torch.nn.Module]:
    models = {}
    for role, attr in [('teacher', 'teacher_weights'),
                       ('student', 'student_weights')]:
        m = XFeatModel().to(device).eval()
        w = getattr(args, attr, None)
        if w and os.path.isfile(w):
            m.load_state_dict(
                torch.load(w, map_location=device, weights_only=True))
            print(f"[FPN] {role}: {w}")
        else:
            print(f"[FPN] WARNING: {attr} not found")
        models[role] = m
    return models


# ---------------------------------------------------------------------------
# FPN ノイズ付加
# ---------------------------------------------------------------------------

def add_fpn_noise(img_t: torch.Tensor, sigma_dn: float) -> torch.Tensor:
    """
    列方向固定パターンノイズを付加する。
    sigma_dn: ノイズ強度（DN単位、0〜255スケール）

    実装: 訓練時の fpn_invariance_loss と同一の生成方式
    col_noise = randn(1, 1, 1, W) * (sigma_dn / 255)
    """
    if sigma_dn == 0:
        return img_t
    B, C, H, W = img_t.shape
    sigma = sigma_dn / 255.0
    col_noise = torch.randn(B, 1, 1, W, device=img_t.device) * sigma
    col_noise = col_noise.expand(B, C, H, W)
    return (img_t + col_noise).clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# キーポイント検出・マッチング（eval_matching.py と同一実装）
# ---------------------------------------------------------------------------

def imread_tensor(path, is_thermal, device, size):
    if is_thermal:
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(path)
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    else:
        bgr = cv2.imread(path)
        if bgr is None:
            raise FileNotFoundError(path)
    bgr = cv2.resize(bgr, size)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    t   = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    return t.unsqueeze(0).to(device), bgr


@torch.no_grad()
def detect(model, img_t, max_kp):
    feats, _, hmap = model(img_t)
    feats = F.normalize(feats, dim=1)
    B, C, Hf, Wf = feats.shape
    H, W = img_t.shape[2], img_t.shape[3]
    scores   = hmap[0, 0].cpu().numpy().flatten()
    feats_np = feats[0].reshape(C, -1).permute(1, 0).cpu().numpy()
    top_idx  = np.argsort(scores)[::-1][:min(max_kp, len(scores))]
    ys = (top_idx // Wf).astype(np.float32) * (H / Hf)
    xs = (top_idx %  Wf).astype(np.float32) * (W / Wf)
    return np.stack([xs, ys], axis=1), feats_np[top_idx].astype(np.float32)


def match_mutual_nn(d1, d2):
    d1n = d1 / (np.linalg.norm(d1, axis=1, keepdims=True) + 1e-8)
    d2n = d2 / (np.linalg.norm(d2, axis=1, keepdims=True) + 1e-8)
    sim  = d1n @ d2n.T
    nn12 = np.argmax(sim, axis=1)
    nn21 = np.argmax(sim, axis=0)
    ids  = np.arange(len(d1))
    mask = nn21[nn12] == ids
    return ids[mask], nn12[mask]


def homography_error(kpts1, kpts2, idx1, idx2, hw):
    if len(idx1) < 4:
        return float('inf')
    H_mat, _ = cv2.findHomography(
        kpts1[idx1].reshape(-1,1,2),
        kpts2[idx2].reshape(-1,1,2),
        cv2.RANSAC, 5.0)
    if H_mat is None:
        return float('inf')
    h, w = hw
    c = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
    p = cv2.perspectiveTransform(c, H_mat).reshape(-1,2)
    return float(np.mean(np.linalg.norm(c.reshape(-1,2) - p, axis=1)))


def auc_at(errors, thresholds):
    arr = np.array(errors)
    return {f'AUC@{t}px': float((arr <= t).mean()) for t in thresholds}


# ---------------------------------------------------------------------------
# メイン評価ループ
# ---------------------------------------------------------------------------

def evaluate_fpn_noise(
    name: str,
    pairs: List[Tuple[str, str]],
    models: Dict[str, torch.nn.Module],
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, List[Dict]]:
    """
    Returns:
        {
            'teacher_thr': [{'sigma': 0, 'AUC@3px': ..., ...}, ...],
            'student_thr': [...],
        }
    """
    size   = (args.viz_width, args.viz_height)
    max_kp = args.max_keypoints
    thrs   = args.auc_thresholds
    hw     = (args.viz_height, args.viz_width)

    model_names = ['teacher_thr', 'student_thr']
    # sigma → {model_name → errors リスト}
    all_errors = {
        sigma: {mn: [] for mn in model_names}
        for sigma in args.sigma_levels
    }

    for i, (rgb_p, thr_p) in enumerate(pairs):
        if (i+1) % 50 == 0:
            print(f"  [{name}] {i+1}/{len(pairs)} ...")
        try:
            _, _ = imread_tensor(rgb_p, False, device, size)
            thr_t, _ = imread_tensor(thr_p, True, device, size)
        except FileNotFoundError:
            continue

        # ノイズなし基準特徴（anchor）
        k_clean, d_clean_t = detect(models['teacher'], thr_t, max_kp)
        k_clean_s, d_clean_s = detect(models['student'], thr_t, max_kp)

        for sigma in args.sigma_levels:
            thr_noisy = add_fpn_noise(thr_t, sigma)

            # teacher_thr: 元のXFeatに noisy 熱画像 → clean 熱画像とマッチング
            k_n_t, d_n_t = detect(models['teacher'], thr_noisy, max_kp)
            i1, i2 = match_mutual_nn(d_clean_t, d_n_t)
            err_t   = homography_error(k_clean, k_n_t, i1, i2, hw)
            all_errors[sigma]['teacher_thr'].append(err_t)

            # student_thr: 提案手法に noisy 熱画像 → clean 熱画像とマッチング
            k_n_s, d_n_s = detect(models['student'], thr_noisy, max_kp)
            i1, i2 = match_mutual_nn(d_clean_s, d_n_s)
            err_s   = homography_error(k_clean_s, k_n_s, i1, i2, hw)
            all_errors[sigma]['student_thr'].append(err_s)

    # 集計
    results = {mn: [] for mn in model_names}
    for sigma in args.sigma_levels:
        for mn in model_names:
            errs = all_errors[sigma][mn]
            r    = auc_at(errs, thrs)
            r['sigma']   = sigma
            r['n_pairs'] = len(errs)
            results[mn].append(r)

    return results


# ---------------------------------------------------------------------------
# 可視化（折れ線グラフ）
# ---------------------------------------------------------------------------

def plot_noise_curves(
    ds_name: str,
    results: Dict[str, List[Dict]],
    args: argparse.Namespace,
    output_dir: str,
) -> None:
    """
    各 sigma における AUC@5px を折れ線グラフで描画する。
    横軸: FPN ノイズ強度（DN）
    縦軸: AUC@5px（%）
    """
    fig, axes = plt.subplots(1, len(args.auc_thresholds),
                             figsize=(5 * len(args.auc_thresholds), 4))
    if len(args.auc_thresholds) == 1:
        axes = [axes]

    colors = {'teacher_thr': '#E24B4A', 'student_thr': '#378ADD'}
    labels = {'teacher_thr': 'XFeat(Thr) [KD前]',
              'student_thr': 'Student(Thr) [提案手法]'}

    sigmas = [r['sigma'] for r in results['teacher_thr']]

    for ax, thr in zip(axes, args.auc_thresholds):
        key = f'AUC@{thr}px'
        for mn in ['teacher_thr', 'student_thr']:
            vals = [r[key] * 100 for r in results[mn]]
            ax.plot(sigmas, vals, 'o-', color=colors[mn],
                    label=labels[mn], linewidth=2, markersize=6)

        ax.set_title(f'AUC@{thr}px vs FPN Noise', fontsize=12)
        ax.set_xlabel('FPN Noise σ (DN)', fontsize=11)
        ax.set_ylabel('AUC (%)', fontsize=11)
        ax.set_ylim(0, 100)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        # 訓練時の sigma 範囲を灰色帯で示す
        ax.axvspan(2, 8, alpha=0.08, color='gray', label='training range')

    fig.suptitle(f'FPN Noise Robustness — {ds_name}', fontsize=13)
    plt.tight_layout()

    save_path = os.path.join(output_dir, ds_name, 'fpn_noise_curve.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [FPN] Plot saved: {save_path}")


def plot_noise_example(
    thr_path: str,
    sigma_levels: List[float],
    device: torch.device,
    output_dir: str,
    ds_name: str,
    size: Tuple[int, int],
) -> None:
    """各ノイズ強度での熱画像の見た目を横並びで保存する。"""
    try:
        gray = cv2.imread(thr_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            return
        gray = cv2.resize(gray, size)
    except Exception:
        return

    cols = []
    for sigma in sigma_levels:
        if sigma == 0:
            noisy = gray.copy()
        else:
            t = torch.from_numpy(gray).float().unsqueeze(0).unsqueeze(0) / 255.0
            t = t.to(device)
            col_noise = torch.randn(1, 1, 1, size[0], device=device) * (sigma/255.0)
            t_noisy = (t + col_noise.expand(1, 1, size[1], size[0])).clamp(0,1)
            noisy = (t_noisy[0, 0].cpu().numpy() * 255).astype(np.uint8)

        bgr = cv2.cvtColor(noisy, cv2.COLOR_GRAY2BGR)
        cv2.putText(bgr, f'sigma={int(sigma)}DN',
                    (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(bgr, f'sigma={int(sigma)}DN',
                    (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255,255,255), 1, cv2.LINE_AA)
        cols.append(bgr)

    canvas = np.hstack(cols)
    sp = os.path.join(output_dir, ds_name, 'fpn_noise_example.png')
    os.makedirs(os.path.dirname(sp), exist_ok=True)
    cv2.imwrite(sp, canvas)
    print(f"  [FPN] Noise example saved: {sp}")


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_arguments()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device_num)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[FPN] Device: {device}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    models = load_models(args, device)
    output_dir = os.path.join(args.output_dir, 'fpn_noise')
    os.makedirs(output_dir, exist_ok=True)

    all_results = {}

    for ds_name in args.datasets:
        print(f"\n[FPN] ========== {ds_name} ==========")

        from modules.dataset.thermal.loader import \
            _resolve_data_root, _resolve_splits_dir
        from modules.dataset.thermal.freiburg   import FreiburgDataset
        from modules.dataset.thermal.tartanrgbt import TartanRGBTDataset
        from modules.dataset.thermal.vivid      import VividDataset
        from modules.dataset.thermal.sthereo    import SthEreoDataset
        from modules.dataset.thermal.ms2        import MS2Dataset

        CLS = {'freiburg': FreiburgDataset, 'tartanrgbt': TartanRGBTDataset,
               'vivid': VividDataset, 'sthereo': SthEreoDataset, 'ms2': MS2Dataset}
        name_l = ds_name.lower()
        if name_l not in CLS:
            print(f"[FPN] Unknown dataset: {ds_name}, skipping.")
            continue

        try:
            ds = CLS[name_l](
                data_root  = _resolve_data_root(name_l, args),
                splits_dir = _resolve_splits_dir(name_l, args),
                split      = args.split,
                augment    = False, aug_list=None, p_diurnal_inversion=0.0,
            )
        except Exception as e:
            print(f"[FPN] {ds_name} skipped: {e}")
            continue

        pairs = list(ds._pairs)
        n = getattr(args, 'n_eval_pairs', None)
        if n and n < len(pairs):
            pairs = random.Random(args.seed).sample(pairs, n)
        print(f"[FPN] {ds_name}: {len(pairs)} pairs × {len(args.sigma_levels)} sigma levels")

        # ノイズ例画像を保存
        if pairs:
            plot_noise_example(
                pairs[0][1], args.sigma_levels, device,
                output_dir, ds_name, (args.viz_width, args.viz_height))

        # 評価
        results = evaluate_fpn_noise(ds_name, pairs, models, args, device)
        all_results[ds_name] = results

        # 結果表示
        print(f"\n  FPN Noise Robustness — {ds_name}")
        print(f"  {'Model':<28s}", end='')
        for sigma in args.sigma_levels:
            print(f"  σ={int(sigma):>2d}DN", end='')
        print()
        print(f"  {'-'*80}")

        labels = {'teacher_thr': 'XFeat(Thr) [KD前]',
                  'student_thr': 'Student(Thr) [提案]'}
        metric = f'AUC@5px'
        for mn, lb in labels.items():
            print(f"  {lb:<28s}", end='')
            for entry in results[mn]:
                print(f"  {entry[metric]*100:>5.1f}%", end='')
            print()

        # グラフ保存
        plot_noise_curves(ds_name, results, args, output_dir)

    # JSON 保存
    save_path = os.path.join(output_dir, 'fpn_noise_results.json')
    with open(save_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[FPN] Results saved → {save_path}")
    print("[FPN] Done.")


if __name__ == '__main__':
    main()