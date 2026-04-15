"""
eval/eval_homo_aug.py
ホモグラフィ変換による特徴マッチング精度評価。

【評価原理】
    同一の熱画像にランダムなホモグラフィ変換を適用し、
    変換前後の画像間でキーポイントマッチングを行う。
    真の変換行列 H_gt が既知なので AUC を厳密に計算できる。

    img_thr      ──→ detect ──→ kpts1, desc1
         ↓ H_gt                              ↓ match ──→ AUC
    img_warped   ──→ detect ──→ kpts2, desc2

【AUC計算の根拠】
    マッチしたペア (kpts1[i], kpts2[j]) に対して
    H_gt で kpts1[i] を変換した点と kpts2[j] の距離を再投影誤差とする。
    これが 3px / 5px / 10px 以内のペアの割合が AUC。

【比較対象】
    teacher_thr : XFeat(元モデル)に熱画像を入力  ← KD前のベースライン
    student_thr : 提案手法(KD済み)に熱画像を入力 ← 提案手法

    ※ XFeat(RGB) は熱画像での公平比較にならないため除外

【使用方法】
    python eval/eval_homo_aug.py --config configs/eval_config.yaml
    python eval/eval_homo_aug.py --config configs/eval_config.yaml \\
        --datasets freiburg sthereo --n_eval_pairs 1000 --n_viz 10
    python eval/eval_homo_aug.py --config configs/eval_config.yaml \\
        --matching_method lightglue
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
        description='Thermal XFeat — Homography Augmentation Evaluation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--config',          type=str, required=True)
    parser.add_argument('--student_weights', type=str, default=None)
    parser.add_argument('--teacher_weights', type=str, default=None)
    parser.add_argument('--datasets',        nargs='+', default=None)
    parser.add_argument('--split',           type=str, default=None)
    parser.add_argument('--n_eval_pairs',    type=int, default=None,
                        help='1データセットあたり評価枚数（1枚→1ホモグラフィペア）')
    parser.add_argument('--n_viz',           type=int, default=None)
    parser.add_argument('--output_dir',      type=str, default=None)
    parser.add_argument('--max_keypoints',   type=int, default=None)
    parser.add_argument('--matching_method', type=str, default=None,
                        choices=['mutual_nn', 'ratio_test', 'lightglue'])
    parser.add_argument('--device_num',      type=str, default=None)
    # ホモグラフィ変換パラメータ
    parser.add_argument('--homo_perspective', type=float, default=None,
                        help='透視変換の強さ (0.0〜0.2)')
    parser.add_argument('--homo_scale_min',   type=float, default=None)
    parser.add_argument('--homo_scale_max',   type=float, default=None)
    parser.add_argument('--homo_rotation',    type=float, default=None,
                        help='回転角度の最大値（度）')
    parser.add_argument('--homo_translation', type=float, default=None,
                        help='並進量の最大値（画像幅に対する割合）')

    cli = parser.parse_args()
    if not os.path.isfile(cli.config):
        parser.error(f'--config not found: {cli.config!r}')

    with open(cli.config) as f:
        cfg = yaml.safe_load(f) or {}

    for k, v in vars(cli).items():
        if k != 'config' and v is not None:
            cfg[k] = v

    defaults = dict(
        datasets          = ['freiburg'],
        split             = 'val',
        n_eval_pairs      = 1000,
        n_viz             = 5,
        auc_thresholds    = [3, 5, 10],
        matching_method   = 'mutual_nn',
        ratio_threshold   = 0.9,
        max_keypoints     = 2048,
        output_dir        = 'eval/results',
        viz_width         = 640,
        viz_height        = 480,
        seed              = 42,
        device_num        = '0',
        # ホモグラフィ変換パラメータ
        # XFeat 論文・SuperPoint 論文の標準設定に準拠
        homo_perspective  = 0.10,   # 透視変換の強さ
        homo_scale_min    = 0.85,   # スケール下限
        homo_scale_max    = 1.15,   # スケール上限
        homo_rotation     = 30.0,   # 最大回転角（度）
        homo_translation  = 0.15,   # 最大並進（画像幅の割合）
    )
    for k, v in defaults.items():
        cfg.setdefault(k, v)

    return argparse.Namespace(**cfg)


# ---------------------------------------------------------------------------
# ホモグラフィ生成
# ---------------------------------------------------------------------------

def random_homography(
    H: int,
    W: int,
    perspective: float = 0.10,
    scale_min: float   = 0.85,
    scale_max: float   = 1.15,
    rotation_deg: float = 30.0,
    translation: float  = 0.15,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    ランダムなホモグラフィ行列を生成する。

    SuperPoint / XFeat 論文の標準設定に準拠した変換を使用する。
    変換の種類:
        1. 透視変換（4コーナーのランダム摂動）
        2. アフィン変換（スケール・回転・並進）

    Args:
        H, W:        画像の高さ・幅
        perspective: コーナー摂動の強さ（0=なし、0.1=標準）
        scale_min/max: スケール変換の範囲
        rotation_deg:  回転角度の最大値（度）
        translation:   並進量の最大値（画像幅に対する割合）

    Returns:
        H_mat: (3, 3) float64 ホモグラフィ行列
    """
    if rng is None:
        rng = np.random.default_rng()

    # ── 1. 透視変換（コーナー摂動）──────────────────────────────────────
    margin = perspective
    pts_src = np.float32([
        [0, 0], [W, 0], [W, H], [0, H]
    ])
    # 各コーナーをランダムに摂動
    pts_dst = pts_src + rng.uniform(
        -margin * min(H, W),
         margin * min(H, W),
        size=(4, 2)
    ).astype(np.float32)
    H_persp = cv2.getPerspectiveTransform(pts_src, pts_dst)

    # ── 2. アフィン変換（スケール・回転・並進）──────────────────────────
    scale  = rng.uniform(scale_min, scale_max)
    angle  = rng.uniform(-rotation_deg, rotation_deg)
    tx     = rng.uniform(-translation * W, translation * W)
    ty     = rng.uniform(-translation * H, translation * H)

    # 画像中心を回転の軸にする
    cx, cy = W / 2.0, H / 2.0
    cos_a = np.cos(np.radians(angle)) * scale
    sin_a = np.sin(np.radians(angle)) * scale
    H_affine = np.array([
        [cos_a, -sin_a, (1 - cos_a) * cx + sin_a * cy + tx],
        [sin_a,  cos_a, (1 - cos_a) * cy - sin_a * cx + ty],
        [0,      0,     1]
    ], dtype=np.float64)

    # ── 合成: 透視変換 → アフィン変換 ──────────────────────────────────
    H_mat = H_affine @ H_persp
    return H_mat


def apply_homography(
    img_bgr: np.ndarray,
    H_mat: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    画像にホモグラフィ変換を適用する。

    Returns:
        warped:   変換後の画像（元と同サイズ）
        mask:     変換後に有効なピクセル領域マスク（uint8）
    """
    H, W = img_bgr.shape[:2]
    warped = cv2.warpPerspective(
        img_bgr, H_mat, (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    # 有効領域マスク（元画像の全ピクセルが変換後に含まれる領域）
    ones  = np.ones((H, W), dtype=np.uint8) * 255
    mask  = cv2.warpPerspective(
        ones, H_mat, (W, H),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return warped, mask


# ---------------------------------------------------------------------------
# 再投影誤差（AUC 計算の核心）
# ---------------------------------------------------------------------------

def reprojection_errors(
    kpts1: np.ndarray,
    kpts2: np.ndarray,
    idx1: np.ndarray,
    idx2: np.ndarray,
    H_gt: np.ndarray,
) -> np.ndarray:
    """
    各マッチペアの再投影誤差を計算する。

    kpts1[i] を H_gt で変換した点と kpts2[j] のユークリッド距離。

    真の H_gt が既知なので、RANSAC は不要。
    全マッチペアに対して厳密な誤差を計算できる。

    Args:
        kpts1, kpts2: (N, 2) キーポイント座標
        idx1, idx2:   マッチインデックス
        H_gt:         (3, 3) 真のホモグラフィ行列

    Returns:
        errors: (M,) float32 各マッチペアの再投影誤差（ピクセル）
    """
    if len(idx1) == 0:
        return np.array([], dtype=np.float32)

    pts1 = kpts1[idx1]   # (M, 2)

    # 同次座標に変換して H_gt を適用
    pts1_h = np.hstack([pts1, np.ones((len(pts1), 1))])  # (M, 3)
    pts1_t = (H_gt @ pts1_h.T).T                          # (M, 3)
    # 同次座標を正規化
    pts1_t = pts1_t[:, :2] / (pts1_t[:, 2:3] + 1e-8)    # (M, 2)

    pts2 = kpts2[idx2]   # (M, 2)
    errors = np.linalg.norm(pts1_t - pts2, axis=1).astype(np.float32)
    return errors


def auc_at(errors: np.ndarray, thresholds: List[int]) -> Dict[str, float]:
    """各閾値での AUC を計算する。"""
    if len(errors) == 0:
        return {f'AUC@{t}px': 0.0 for t in thresholds}
    return {f'AUC@{t}px': float((errors <= t).mean()) for t in thresholds}


# ---------------------------------------------------------------------------
# モデルロード・検出・マッチング
# ---------------------------------------------------------------------------

def load_models(args, device):
    models = {}
    for role, attr in [('teacher', 'teacher_weights'),
                       ('student', 'student_weights')]:
        m = XFeatModel().to(device).eval()
        w = getattr(args, attr, None)
        if w and os.path.isfile(w):
            m.load_state_dict(torch.load(w, map_location=device, weights_only=True))
            print(f"[HomoAug] {role}: {w}")
        else:
            print(f"[HomoAug] WARNING: {attr} not found")
        models[role] = m
    return models


def bgr_to_tensor(bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    """BGR numpy を (1, 3, H, W) float [0,1] Tensor に変換。"""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    t   = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    return t.unsqueeze(0).to(device)


@torch.no_grad()
def detect(model: torch.nn.Module,
           img_t: torch.Tensor,
           max_kp: int) -> Tuple[np.ndarray, np.ndarray]:
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


def mutual_nn(d1, d2):
    if len(d1) == 0 or len(d2) == 0:
        return np.array([], np.int64), np.array([], np.int64)
    d1n = d1 / (np.linalg.norm(d1, axis=1, keepdims=True) + 1e-8)
    d2n = d2 / (np.linalg.norm(d2, axis=1, keepdims=True) + 1e-8)
    sim  = d1n @ d2n.T
    nn12 = np.argmax(sim, axis=1)
    nn21 = np.argmax(sim, axis=0)
    ids  = np.arange(len(d1))
    mask = nn21[nn12] == ids
    return ids[mask].astype(np.int64), nn12[mask].astype(np.int64)


def ratio_test(d1, d2, ratio_thr):
    if len(d1) == 0 or len(d2) == 0 or d2.shape[0] < 2:
        return np.array([], np.int64), np.array([], np.int64)
    d1n = d1 / (np.linalg.norm(d1, axis=1, keepdims=True) + 1e-8)
    d2n = d2 / (np.linalg.norm(d2, axis=1, keepdims=True) + 1e-8)
    sim  = d1n @ d2n.T
    order = np.argsort(-sim, axis=1)
    best1 = sim[np.arange(len(d1)), order[:, 0]]
    best2 = sim[np.arange(len(d1)), order[:, 1]]
    mask  = (best2 / (best1 + 1e-8)) < ratio_thr
    idx1  = np.where(mask)[0]
    return idx1.astype(np.int64), order[idx1, 0].astype(np.int64)


def do_match(k1, d1, k2, d2, method, ratio_thr, hw, device):
    if method == 'lightglue':
        try:
            from lightglue import LightGlue
            if not hasattr(do_match, '_lg'):
                do_match._lg = LightGlue(features='xfeat').eval().to(device)
            H, W = hw
            def fmt(k, d):
                kn = torch.from_numpy(k).float().unsqueeze(0).to(device)
                dn = torch.from_numpy(d).float().unsqueeze(0).to(device)
                kn[..., 0] = (kn[..., 0] / W) * 2 - 1
                kn[..., 1] = (kn[..., 1] / H) * 2 - 1
                return {'keypoints': kn, 'descriptors': dn,
                        'image_size': torch.tensor([[H, W]], device=device)}
            with torch.no_grad():
                pred = do_match._lg({'image0': fmt(k1, d1), 'image1': fmt(k2, d2)})
            m = pred['matches'][0].cpu().numpy()
            v = m >= 0
            return np.where(v)[0].astype(np.int64), m[v].astype(np.int64)
        except Exception as e:
            print(f"  [LightGlue] fallback: {e}")
    if method == 'ratio_test':
        return ratio_test(d1, d2, ratio_thr)
    return mutual_nn(d1, d2)


# ---------------------------------------------------------------------------
# データセット読み込み（熱画像パスのみ）
# ---------------------------------------------------------------------------

def load_thr_paths(name: str, args, split: str) -> List[str]:
    """
    データセットの熱画像パスリストを返す。
    同一シーン内の連続性は問わない（ホモグラフィ変換なので不要）。
    """
    from modules.dataset.thermal.loader    import _resolve_data_root, _resolve_splits_dir
    from modules.dataset.thermal.freiburg   import FreiburgDataset
    from modules.dataset.thermal.tartanrgbt import TartanRGBTDataset
    from modules.dataset.thermal.vivid      import VividDataset
    from modules.dataset.thermal.sthereo    import SthEreoDataset
    from modules.dataset.thermal.ms2        import MS2Dataset

    CLS = {'freiburg': FreiburgDataset, 'tartanrgbt': TartanRGBTDataset,
           'vivid': VividDataset, 'sthereo': SthEreoDataset, 'ms2': MS2Dataset}
    name_l = name.lower()
    if name_l not in CLS:
        raise ValueError(f"Unknown dataset: {name!r}")

    ds = CLS[name_l](
        data_root  = _resolve_data_root(name_l, args),
        splits_dir = _resolve_splits_dir(name_l, args),
        split=split, augment=False, aug_list=None, p_diurnal_inversion=0.0,
    )
    thr_paths = [thr for _, thr in ds._pairs]
    print(f"[HomoAug] {name} ({split}): {len(thr_paths)} thermal images")

    n = getattr(args, 'n_eval_pairs', None)
    if n and n < len(thr_paths):
        thr_paths = random.Random(args.seed).sample(thr_paths, n)
        print(f"[HomoAug] {name}: subsampled → {len(thr_paths)} images")
    return thr_paths


# ---------------------------------------------------------------------------
# 1画像の評価
# ---------------------------------------------------------------------------

def evaluate_image(
    thr_path: str,
    models: Dict,
    args,
    device: torch.device,
    rng: np.random.Generator,
) -> Dict[str, Dict]:
    """
    1枚の熱画像にホモグラフィ変換を適用してマッチング精度を評価する。

    Returns:
        {'teacher_thr': {'errors': [...], 'n_matches': int},
         'student_thr': {'errors': [...], 'n_matches': int}}
    """
    H_px, W_px = args.viz_height, args.viz_width

    # 熱画像を読み込み
    gray = cv2.imread(thr_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return {}
    gray = cv2.resize(gray, (W_px, H_px))
    bgr  = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # ホモグラフィ生成・適用
    H_gt = random_homography(
        H_px, W_px,
        perspective  = args.homo_perspective,
        scale_min    = args.homo_scale_min,
        scale_max    = args.homo_scale_max,
        rotation_deg = args.homo_rotation,
        translation  = args.homo_translation,
        rng          = rng,
    )
    warped, mask = apply_homography(bgr, H_gt)

    # Tensor に変換
    orig_t   = bgr_to_tensor(bgr,    device)
    warped_t = bgr_to_tensor(warped, device)

    results = {}
    hw = (H_px, W_px)

    for mn, mdl in [('teacher_thr', models['teacher']),
                    ('student_thr', models['student'])]:
        # 元画像と変換後画像のキーポイント検出
        k1, d1 = detect(mdl, orig_t,   args.max_keypoints)
        k2, d2 = detect(mdl, warped_t, args.max_keypoints)

        # マスク外のキーポイントを除去（変換後に画像外に出た点）
        if len(k2) > 0:
            valid = (mask[k2[:, 1].astype(int).clip(0, H_px-1),
                         k2[:, 0].astype(int).clip(0, W_px-1)] > 0)
            k2 = k2[valid]
            d2 = d2[valid]

        if len(k1) == 0 or len(k2) == 0:
            results[mn] = {'errors': np.array([float('inf')]), 'n_matches': 0}
            continue

        # マッチング
        i1, i2 = do_match(k1, d1, k2, d2,
                           args.matching_method, args.ratio_threshold, hw, device)

        # 再投影誤差（H_gt が既知なので厳密に計算）
        errors = reprojection_errors(k1, k2, i1, i2, H_gt)

        results[mn] = {
            'errors':    errors,
            'n_matches': len(i1),
            'n_kpts1':   len(k1),
            'n_kpts2':   len(k2),
        }
    return results


# ---------------------------------------------------------------------------
# 定量評価（1データセット）
# ---------------------------------------------------------------------------

def evaluate_dataset(
    name: str,
    thr_paths: List[str],
    models: Dict,
    args,
    device: torch.device,
) -> Dict[str, Dict]:
    rng = np.random.default_rng(args.seed)

    buf = {
        'teacher_thr': {'errors': [], 'n_matches': [], 'n_kpts1': [], 'n_kpts2': []},
        'student_thr': {'errors': [], 'n_matches': [], 'n_kpts1': [], 'n_kpts2': []},
    }

    for i, thr_p in enumerate(thr_paths):
        if (i+1) % 100 == 0:
            print(f"  [{name}] {i+1}/{len(thr_paths)} ...")
        r = evaluate_image(thr_p, models, args, device, rng)
        for mn in buf:
            if mn in r:
                buf[mn]['errors'].append(r[mn]['errors'])
                buf[mn]['n_matches'].append(r[mn].get('n_matches', 0))
                buf[mn]['n_kpts1'].append(r[mn].get('n_kpts1', 0))
                buf[mn]['n_kpts2'].append(r[mn].get('n_kpts2', 0))

    summary = {}
    labels  = {
        'teacher_thr': 'XFeat(Thr) [KD前]',
        'student_thr': 'Student(Thr) [提案手法]',
    }
    for mn in buf:
        b = buf[mn]
        all_errors   = np.concatenate(b['errors']) if b['errors'] else np.array([float('inf')])
        n_matches    = np.array(b['n_matches'], dtype=np.float32)
        n_kpts1      = np.array(b['n_kpts1'],  dtype=np.float32)
        n_kpts2      = np.array(b['n_kpts2'],  dtype=np.float32)
        n_min        = np.minimum(n_kpts1, n_kpts2)

        # AUC: 再投影誤差が閾値以内のマッチの割合（全マッチに対して）
        r = auc_at(all_errors, args.auc_thresholds)

        # MS（Matching Score）: マッチ数 / min(kpts1数, kpts2数)
        # eval_matching.py の matching_score() と同じ定義
        ms_per_image = np.where(n_min > 0, n_matches / n_min, 0.0)
        r['MS']           = float(ms_per_image.mean()) if len(ms_per_image) > 0 else 0.0

        # 参考情報（論文には載せないが診断に使う）
        r['mean_n_matches'] = float(n_matches.mean()) if len(n_matches) > 0 else 0.0
        r['mean_n_kpts']    = float(n_kpts1.mean())   if len(n_kpts1)   > 0 else 0.0
        r['n_images']       = len(b['errors'])
        r['label']          = labels.get(mn, mn)
        summary[mn] = r
    return summary


# ---------------------------------------------------------------------------
# 定性評価（可視化）
# ---------------------------------------------------------------------------

def visualize_homo(
    thr_path: str,
    models: Dict,
    args,
    device: torch.device,
    save_path: str,
    rng: np.random.Generator,
) -> None:
    """
    1枚の熱画像に対するホモグラフィ変換評価の可視化。

    レイアウト:
        上段: 元の熱画像（kpts付き）  |  変換後の熱画像（kpts付き）
        下段左:  XFeat(Thr)   のマッチング結果（赤→再投影点）
        下段右:  Student(Thr) のマッチング結果（橙→再投影点）

    緑の×印 = H_gt で変換した「正解の位置」
    色付き点 = マッチングした点
    → 点と×印が近いほど精度が高い
    """
    H_px, W_px = args.viz_height, args.viz_width

    gray = cv2.imread(thr_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return
    gray   = cv2.resize(gray, (W_px, H_px))
    bgr    = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    H_gt   = random_homography(H_px, W_px,
                                args.homo_perspective, args.homo_scale_min,
                                args.homo_scale_max, args.homo_rotation,
                                args.homo_translation, rng)
    warped, mask = apply_homography(bgr, H_gt)
    orig_t   = bgr_to_tensor(bgr,    device)
    warped_t = bgr_to_tensor(warped, device)

    colors   = {'teacher_thr': (0, 0, 220),   'student_thr': (20, 140, 255)}
    labels_m = {'teacher_thr': 'XFeat(Thr) [KD前]',
                'student_thr': 'Student(Thr) [提案]'}
    font = cv2.FONT_HERSHEY_SIMPLEX

    rows = []
    for mn, mdl in [('teacher_thr', models['teacher']),
                    ('student_thr', models['student'])]:
        k1, d1 = detect(mdl, orig_t,   args.max_keypoints)
        k2, d2 = detect(mdl, warped_t, args.max_keypoints)

        if len(k2) > 0:
            valid = (mask[k2[:, 1].astype(int).clip(0, H_px-1),
                         k2[:, 0].astype(int).clip(0, W_px-1)] > 0)
            k2 = k2[valid]; d2 = d2[valid]

        clr = colors[mn]
        left  = bgr.copy()
        right = warped.copy()

        i1, i2 = (np.array([], np.int64), np.array([], np.int64))
        if len(k1) > 0 and len(k2) > 0:
            i1, i2 = do_match(k1, d1, k2, d2,
                               args.matching_method, args.ratio_threshold,
                               (H_px, W_px), device)

        # 元画像にキーポイントを描画
        for x, y in k1[:200]:
            cv2.circle(left, (int(x), int(y)), 2, clr, -1)

        # 変換後画像に検出点と「正解位置（×印）」を描画
        errors = reprojection_errors(k1, k2, i1, i2, H_gt)
        for idx, (a, b) in enumerate(zip(i1[:80], i2[:80])):
            x2, y2 = int(k2[b][0]), int(k2[b][1])
            cv2.circle(right, (x2, y2), 3, clr, -1)
            # H_gt で正解位置を計算して ×印 で表示
            pt = np.array([[k1[a][0], k1[a][1], 1.0]])
            pt_t = (H_gt @ pt.T).flatten()
            gx = int(pt_t[0] / (pt_t[2] + 1e-8))
            gy = int(pt_t[1] / (pt_t[2] + 1e-8))
            # 誤差に応じて×印の色を変える（緑=正解、赤=誤り）
            err_val = errors[idx] if idx < len(errors) else float('inf')
            cross_clr = (0, 200, 0) if err_val <= 5 else (0, 0, 200)
            size = 5
            cv2.line(right, (gx-size, gy-size), (gx+size, gy+size), cross_clr, 1)
            cv2.line(right, (gx+size, gy-size), (gx-size, gy+size), cross_clr, 1)

        n_correct = int((errors <= 5).sum()) if len(errors) > 0 else 0
        row = np.hstack([left, right])
        lbl = f"{labels_m[mn]}  matches={len(i1)}  correct@5px={n_correct}"
        cv2.putText(row, lbl, (8, 26), font, 0.55, (0,0,0),    3, cv2.LINE_AA)
        cv2.putText(row, lbl, (8, 26), font, 0.55, (255,255,255),1, cv2.LINE_AA)
        rows.append(row)

    # 凡例
    legend_h = 32
    legend   = np.zeros((legend_h, rows[0].shape[1], 3), dtype=np.uint8)
    cv2.putText(legend, 'Green x = correct position (H_gt)  |  Colored dot = matched keypoint  |  Red x = error > 5px',
                (8, 20), font, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    canvas = np.vstack(rows + [legend])
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    cv2.imwrite(save_path, canvas)
    print(f"  [HomoAug VIZ] Saved: {save_path}")


# ---------------------------------------------------------------------------
# 結果表示・保存
# ---------------------------------------------------------------------------

def print_results(all_results: Dict, method: str) -> None:
    print()
    print('=' * 92)
    print(f'  HOMOGRAPHY AUGMENTATION RESULTS  [matcher: {method}]')
    print('  ※ AUC の計算は H_gt（真のホモグラフィ）による再投影誤差を使用')
    print('=' * 92)

    for ds, res in all_results.items():
        print(f"\n  Dataset: {ds}")
        print(f"  {'Model':<38s} {'AUC@3px':>8s} {'AUC@5px':>8s} {'AUC@10px':>9s}"
              f" {'MS':>6s} {'matches':>8s} {'kpts':>6s} {'images':>7s}")
        print(f"  {'-'*90}")
        for mn in ['teacher_thr', 'student_thr']:
            if mn not in res:
                continue
            r   = res[mn]
            lbl = r.get('label', mn)
            print(
                f"  {lbl:<38s}"
                f" {r.get('AUC@3px',0)*100:>7.2f}%"
                f" {r.get('AUC@5px',0)*100:>7.2f}%"
                f" {r.get('AUC@10px',0)*100:>8.2f}%"
                f" {r.get('MS',0)*100:>5.1f}%"
                f" {r.get('mean_n_matches',0):>8.1f}"
                f" {r.get('mean_n_kpts',0):>6.0f}"
                f" {r.get('n_images',0):>7d}"
            )
        if 'teacher_thr' in res and 'student_thr' in res:
            t_auc = res['teacher_thr'].get('AUC@5px', 0) * 100
            s_auc = res['student_thr'].get('AUC@5px', 0) * 100
            t_kpt = res['teacher_thr'].get('mean_n_kpts', 0)
            s_kpt = res['student_thr'].get('mean_n_kpts', 0)
            t_mat = res['teacher_thr'].get('mean_n_matches', 0)
            s_mat = res['student_thr'].get('mean_n_matches', 0)
            sign  = '✅' if s_auc > t_auc else '❌'
            print(f"\n  {sign} AUC@5px: {t_auc:.2f}% → {s_auc:.2f}%  (Δ={s_auc-t_auc:+.2f}%)")
            print(f"  キーポイント数: XFeat={t_kpt:.0f}  Student={s_kpt:.0f}"
                  f"  (比率={s_kpt/max(t_kpt,1)*100:.1f}%)")
            print(f"  マッチ数:       XFeat={t_mat:.1f}  Student={s_mat:.1f}"
                  f"  (比率={s_mat/max(t_mat,1)*100:.1f}%)")


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_arguments()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device_num)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[HomoAug] Device: {device}")
    if device.type == 'cuda':
        print(f"[HomoAug] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[HomoAug] Matcher: {args.matching_method}")
    print(f"[HomoAug] Homo params: perspective={args.homo_perspective} "
          f"scale=[{args.homo_scale_min},{args.homo_scale_max}] "
          f"rot={args.homo_rotation}deg trans={args.homo_translation}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    models     = load_models(args, device)
    output_dir = os.path.join(args.output_dir, 'homo_aug')
    os.makedirs(output_dir, exist_ok=True)

    all_results = {}

    for ds_name in args.datasets:
        print(f"\n[HomoAug] ========== {ds_name} ==========")
        try:
            thr_paths = load_thr_paths(ds_name, args, args.split)
        except Exception as e:
            print(f"[HomoAug] {ds_name} skipped: {e}")
            continue

        # 定量評価
        print(f"[HomoAug] Evaluating {len(thr_paths)} images ...")
        all_results[ds_name] = evaluate_dataset(
            ds_name, thr_paths, models, args, device)

        # 定性評価
        n_viz = args.n_viz
        if n_viz > 0:
            viz_rng = np.random.default_rng(args.seed + 77)
            picks   = random.Random(args.seed).sample(
                thr_paths, min(n_viz, len(thr_paths)))
            for vi, thr_p in enumerate(picks):
                sp = os.path.join(output_dir, ds_name, f'viz_{vi+1:03d}.png')
                visualize_homo(thr_p, models, args, device, sp, viz_rng)

    print_results(all_results, args.matching_method)

    save_path = os.path.join(output_dir, 'homo_aug_results.json')
    with open(save_path, 'w') as f:
        json.dump(
            {ds: {mn: {k: float(v) if isinstance(v, (np.floating, float)) else v
                       for k, v in r.items() if k != 'errors'}
                  for mn, r in res.items()}
             for ds, res in all_results.items()},
            f, indent=2)
    print(f"\n[HomoAug] Results saved → {save_path}")
    print("[HomoAug] Done.")


if __name__ == '__main__':
    main()