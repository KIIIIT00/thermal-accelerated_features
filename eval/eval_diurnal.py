"""
eval/eval_diurnal.py
昼夜クロスマッチング評価スクリプト。

新規性③「昼夜ループ閉じ込め」の有効性を定量的に示す。

評価方法:
    同一シーンの昼画像 → 夜画像 のマッチング精度を測定する。
    Freiburg データセットは昼夜シーケンスが分かれているため最適。

比較対象:
    teacher_thr : XFeat（元モデル）に熱画像を入力  ← diurnal_inversion なし
    student_thr : 提案手法（KD済み）に熱画像を入力 ← diurnal_inversion あり

    提案手法は昼夜ペアでも teacher_thr より高い精度を示すことを期待する。

使用方法:
    python eval/eval_diurnal.py --config configs/eval_config.yaml
    python eval/eval_diurnal.py --config configs/eval_config.yaml \\
        --day_split train --night_split val --n_eval_pairs 200
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
        description='Thermal XFeat — Diurnal (Day/Night) Cross-Matching Evaluation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--config',          type=str, required=True)
    parser.add_argument('--student_weights', type=str, default=None)
    parser.add_argument('--teacher_weights', type=str, default=None)
    parser.add_argument('--n_eval_pairs',    type=int, default=None,
                        help='昼夜クロスペアの評価数')
    parser.add_argument('--output_dir',      type=str, default=None)
    parser.add_argument('--n_viz',           type=int, default=None)
    parser.add_argument('--device_num',      type=str, default=None)

    cli = parser.parse_args()
    if not os.path.isfile(cli.config):
        parser.error(f'--config not found: {cli.config!r}')

    with open(cli.config) as f:
        cfg = yaml.safe_load(f) or {}

    for k, v in vars(cli).items():
        if k != 'config' and v is not None:
            cfg[k] = v

    defaults = dict(
        n_eval_pairs    = 200,
        n_viz           = 5,
        auc_thresholds  = [3, 5, 10],
        matching_method = 'mutual_nn',
        ratio_threshold = 0.9,
        max_keypoints   = 2048,
        output_dir      = 'eval/results',
        viz_width       = 640,
        viz_height      = 480,
        seed            = 42,
        device_num      = '0',
    )
    for k, v in defaults.items():
        cfg.setdefault(k, v)

    return argparse.Namespace(**cfg)


# ---------------------------------------------------------------------------
# モデルロード
# ---------------------------------------------------------------------------

def load_models(args, device):
    models = {}
    for role, attr in [('teacher', 'teacher_weights'),
                       ('student', 'student_weights')]:
        m = XFeatModel().to(device).eval()
        w = getattr(args, attr, None)
        if w and os.path.isfile(w):
            m.load_state_dict(torch.load(w, map_location=device, weights_only=True))
            print(f"[Diurnal] {role}: {w}")
        else:
            print(f"[Diurnal] WARNING: {attr} not found")
        models[role] = m
    return models


# ---------------------------------------------------------------------------
# Freiburg 昼夜ペア構築
# ---------------------------------------------------------------------------

def build_diurnal_pairs(
    args: argparse.Namespace,
) -> Dict[str, List[Tuple[str, str, str, str]]]:
    """
    Freiburg データセットから昼夜クロスペアを構築する。

    同一シーケンス番号の昼画像と夜画像をランダムにペアリングする。
    returns: {
        'day_day':   [(thr_day1_path, thr_day2_path), ...]   ← 同一時間帯（ベースライン）
        'day_night': [(thr_day_path,  thr_night_path), ...]  ← 昼夜クロス（本評価）
    }
    """
    from modules.dataset.thermal.loader    import _resolve_data_root, _resolve_splits_dir
    from modules.dataset.thermal.freiburg   import FreiburgDataset

    name_l = 'freiburg'
    data_root  = _resolve_data_root(name_l, args)
    splits_dir = _resolve_splits_dir(name_l, args)

    # train (day) と val (night/day2) を両方ロード
    ds_train = FreiburgDataset(data_root=data_root, splits_dir=splits_dir,
                                split='train', augment=False,
                                aug_list=None, p_diurnal_inversion=0.0)
    ds_val   = FreiburgDataset(data_root=data_root, splits_dir=splits_dir,
                                split='val',   augment=False,
                                aug_list=None, p_diurnal_inversion=0.0)

    # 熱画像パスのみ抽出
    day_thr_paths   = [thr for _, thr in ds_train._pairs]
    night_thr_paths = [thr for _, thr in ds_val._pairs
                       if 'night' in thr.lower()]

    print(f"[Diurnal] Freiburg day thermal pairs:   {len(day_thr_paths)}")
    print(f"[Diurnal] Freiburg night thermal paths: {len(night_thr_paths)}")

    rng = random.Random(args.seed)
    n   = getattr(args, 'n_eval_pairs', 200)

    # 昼-昼ペア（同一時間帯の比較基準）
    day_day = []
    for _ in range(min(n, len(day_thr_paths) // 2)):
        p1, p2 = rng.sample(day_thr_paths, 2)
        day_day.append((p1, p2))

    # 昼-夜ペア（本評価）
    day_night = []
    for _ in range(min(n, min(len(day_thr_paths), len(night_thr_paths)))):
        p_day   = rng.choice(day_thr_paths)
        p_night = rng.choice(night_thr_paths)
        day_night.append((p_day, p_night))

    print(f"[Diurnal] day-day pairs:   {len(day_day)}")
    print(f"[Diurnal] day-night pairs: {len(day_night)}")

    return {'day_day': day_day, 'day_night': day_night}


# ---------------------------------------------------------------------------
# 共通ユーティリティ
# ---------------------------------------------------------------------------

def imread_tensor(path, device, size):
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(path)
    bgr = cv2.cvtColor(cv2.resize(gray, size), cv2.COLOR_GRAY2BGR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    t   = torch.from_numpy(rgb).permute(2,0,1).float() / 255.0
    return t.unsqueeze(0).to(device), bgr


@torch.no_grad()
def detect(model, img_t, max_kp):
    feats, _, hmap = model(img_t)
    feats = F.normalize(feats, dim=1)
    B, C, Hf, Wf = feats.shape
    H, W = img_t.shape[2], img_t.shape[3]
    scores   = hmap[0, 0].cpu().numpy().flatten()
    feats_np = feats[0].reshape(C,-1).permute(1,0).cpu().numpy()
    top_idx  = np.argsort(scores)[::-1][:min(max_kp, len(scores))]
    ys = (top_idx // Wf).astype(np.float32) * (H / Hf)
    xs = (top_idx %  Wf).astype(np.float32) * (W / Wf)
    return np.stack([xs,ys],axis=1), feats_np[top_idx].astype(np.float32)


def match_mutual(d1, d2):
    d1n = d1 / (np.linalg.norm(d1, axis=1, keepdims=True) + 1e-8)
    d2n = d2 / (np.linalg.norm(d2, axis=1, keepdims=True) + 1e-8)
    sim  = d1n @ d2n.T
    nn12 = np.argmax(sim, axis=1)
    nn21 = np.argmax(sim, axis=0)
    ids  = np.arange(len(d1))
    mask = nn21[nn12] == ids
    return ids[mask], nn12[mask]


def homo_error(k1, k2, i1, i2, hw):
    if len(i1) < 4:
        return float('inf')
    H, _ = cv2.findHomography(
        k1[i1].reshape(-1,1,2), k2[i2].reshape(-1,1,2), cv2.RANSAC, 5.0)
    if H is None:
        return float('inf')
    h, w = hw
    c = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
    p = cv2.perspectiveTransform(c, H).reshape(-1,2)
    return float(np.mean(np.linalg.norm(c.reshape(-1,2)-p, axis=1)))


def auc_at(errors, thresholds):
    arr = np.array(errors)
    return {f'AUC@{t}px': float((arr <= t).mean()) for t in thresholds}


# ---------------------------------------------------------------------------
# 評価ループ
# ---------------------------------------------------------------------------

def evaluate_pairs(
    pair_list: List[Tuple[str, str]],
    models: Dict[str, torch.nn.Module],
    args: argparse.Namespace,
    device: torch.device,
    pair_label: str,
) -> Dict[str, Dict]:
    """
    pair_list の各ペアに対して teacher_thr / student_thr の AUC を計算する。
    pair_label: 'day_day' or 'day_night'（ログ表示用）
    """
    size = (args.viz_width, args.viz_height)
    hw   = (args.viz_height, args.viz_width)

    buf = {'teacher_thr': [], 'student_thr': []}

    for i, (p1, p2) in enumerate(pair_list):
        if (i+1) % 50 == 0:
            print(f"  [{pair_label}] {i+1}/{len(pair_list)} ...")
        try:
            t1_t, _ = imread_tensor(p1, device, size)
            t2_t, _ = imread_tensor(p2, device, size)
        except FileNotFoundError:
            continue

        for mn, mdl in [('teacher_thr', models['teacher']),
                        ('student_thr', models['student'])]:
            k1, d1 = detect(mdl, t1_t, args.max_keypoints)
            k2, d2 = detect(mdl, t2_t, args.max_keypoints)
            i1, i2 = match_mutual(d1, d2)
            err    = homo_error(k1, k2, i1, i2, hw)
            buf[mn].append(err)

    summary = {}
    for mn, errs in buf.items():
        r = auc_at(errs, args.auc_thresholds)
        r['n_pairs'] = len(errs)
        summary[mn]  = r
    return summary


# ---------------------------------------------------------------------------
# 可視化
# ---------------------------------------------------------------------------

def visualize_diurnal(
    pair_list: List[Tuple[str, str]],
    models: Dict,
    args: argparse.Namespace,
    device: torch.device,
    output_dir: str,
    label: str,
) -> None:
    """昼夜ペアのマッチング結果を可視化する。"""
    size = (args.viz_width, args.viz_height)
    n    = min(args.n_viz, len(pair_list))
    rng  = random.Random(args.seed + 200)
    picks = rng.sample(pair_list, n)

    for vi, (p1, p2) in enumerate(picks):
        try:
            t1_t, bgr1 = imread_tensor(p1, device, size)
            t2_t, bgr2 = imread_tensor(p2, device, size)
        except FileNotFoundError:
            continue

        W = args.viz_width
        rows = []
        font = cv2.FONT_HERSHEY_SIMPLEX

        for mn, mdl, clr, lbl in [
            ('teacher_thr', models['teacher'], (0,0,255),   'XFeat(Thr) [KD前]'),
            ('student_thr', models['student'], (255,128,0), 'Student(Thr) [提案]'),
        ]:
            k1, d1 = detect(mdl, t1_t, args.max_keypoints)
            k2, d2 = detect(mdl, t2_t, args.max_keypoints)
            i1, i2 = match_mutual(d1, d2)

            canvas = np.hstack([bgr1.copy(), bgr2.copy()])
            for a, b in zip(i1[:80], i2[:80]):
                x1,y1 = int(k1[a][0]), int(k1[a][1])
                x2,y2 = int(k2[b][0])+W, int(k2[b][1])
                cv2.line(canvas, (x1,y1),(x2,y2),(200,200,0),1,cv2.LINE_AA)
                cv2.circle(canvas,(x1,y1),3,clr,-1)
                cv2.circle(canvas,(x2,y2),3,clr,-1)

            txt = f'{lbl}  matches={len(i1)}'
            cv2.putText(canvas,txt,(8,26),font,0.60,(0,0,0),3,cv2.LINE_AA)
            cv2.putText(canvas,txt,(8,26),font,0.60,(255,255,255),1,cv2.LINE_AA)
            rows.append(canvas)

        final = np.vstack(rows)
        sp = os.path.join(output_dir, f'{label}_viz_{vi+1:03d}.png')
        os.makedirs(os.path.dirname(sp), exist_ok=True)
        cv2.imwrite(sp, final)
        print(f"  [Diurnal] Saved: {sp}")


# ---------------------------------------------------------------------------
# 結果表示
# ---------------------------------------------------------------------------

def print_results(results: Dict) -> None:
    labels = {'teacher_thr': 'XFeat(Thr) [KD前]',
              'student_thr': 'Student(Thr) [提案手法]'}

    print()
    print('=' * 75)
    print('  DIURNAL (DAY/NIGHT) EVALUATION RESULTS')
    print('=' * 75)

    for condition in ['day_day', 'day_night']:
        if condition not in results:
            continue
        title = '同一時間帯（day-day）' if condition == 'day_day' \
                else '昼夜クロス（day-night）【本評価】'
        print(f"\n  {title}")
        print(f"  {'Model':<36s} {'AUC@3px':>8s} {'AUC@5px':>8s} {'AUC@10px':>9s} {'pairs':>6s}")
        print(f"  {'-'*72}")
        for mn, lb in labels.items():
            r = results[condition].get(mn, {})
            print(
                f"  {lb:<36s}"
                f" {r.get('AUC@3px',0)*100:>7.2f}%"
                f" {r.get('AUC@5px',0)*100:>7.2f}%"
                f" {r.get('AUC@10px',0)*100:>8.2f}%"
                f" {r.get('n_pairs',0):>6d}"
            )

    # 昼夜クロスでの改善率を表示
    if 'day_night' in results:
        print(f"\n  改善率 (Student vs XFeat, AUC@5px):")
        t_auc = results['day_night'].get('teacher_thr', {}).get('AUC@5px', 0) * 100
        s_auc = results['day_night'].get('student_thr', {}).get('AUC@5px', 0) * 100
        diff  = s_auc - t_auc
        print(f"    XFeat(Thr): {t_auc:.2f}%  →  Student(Thr): {s_auc:.2f}%"
              f"  (Δ = {diff:+.2f}%)")


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_arguments()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device_num)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Diurnal] Device: {device}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    models = load_models(args, device)

    output_dir = os.path.join(args.output_dir, 'diurnal')
    os.makedirs(output_dir, exist_ok=True)

    # Freiburg 昼夜ペア構築
    try:
        pair_sets = build_diurnal_pairs(args)
    except Exception as e:
        print(f"[Diurnal] Failed to build pairs: {e}")
        return

    results = {}

    for condition, pair_list in pair_sets.items():
        print(f"\n[Diurnal] Evaluating: {condition} ({len(pair_list)} pairs)")
        results[condition] = evaluate_pairs(
            pair_list, models, args, device, condition)

        if args.n_viz > 0:
            visualize_diurnal(
                pair_list, models, args, device,
                output_dir, condition)

    print_results(results)

    save_path = os.path.join(output_dir, 'diurnal_results.json')
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[Diurnal] Results saved → {save_path}")
    print("[Diurnal] Done.")


if __name__ == '__main__':
    main()