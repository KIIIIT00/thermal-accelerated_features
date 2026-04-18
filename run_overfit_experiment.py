"""
run_overfit_experiment.py
問題切り分け用の過学習実験スクリプト。

1シーケンスのみで500エポック学習し，同じシーケンスで評価する。
目的: 「モデルが学習できるか」「どの損失が問題か」を特定する。

使用方法:
    # 実験A: KD損失のみ
    python run_overfit_experiment.py \
        --seq kaist_morning \
        --loss kd_only \
        --epochs 500

    # 実験B: repeatability損失のみ
    python run_overfit_experiment.py \
        --seq kaist_morning \
        --loss rep_only \
        --epochs 500

    # 実験C: 幾何整合損失のみ
    python run_overfit_experiment.py \
        --seq kaist_morning \
        --loss geo_only \
        --epochs 500

    # 実験D: 全損失（現在の設定）
    python run_overfit_experiment.py \
        --seq kaist_morning \
        --loss all \
        --epochs 500
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# 引数
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--seq',    default='kaist_morning',
                   help='過学習させるシーケンス名')
    p.add_argument('--sthereo_root', default='datasets/sthereo')
    p.add_argument('--loss',   default='all',
                   choices=['kd_only', 'rep_only', 'geo_only', 'all'],
                   help='使用する損失の組み合わせ')
    p.add_argument('--epochs', type=int, default=500)
    p.add_argument('--max_pairs', type=int, default=500,
                   help='過学習に使うペア数（少ないほど過学習しやすい）')
    p.add_argument('--lr',     type=float, default=1e-4)
    p.add_argument('--batch',  type=int, default=4)
    p.add_argument('--device', default='0')
    p.add_argument('--weights_proposed',
                   default='checkpoints/post_kd/default/post_kd_s2_final.pth',
                   help='評価のベースとなる提案手法の重み')
    p.add_argument('--eval_interval', type=int, default=50)
    p.add_argument('--output_dir',
                   default='checkpoints/overfit_experiment')
    p.add_argument('--n_vis', type=int, default=5,
                   help='可視化する画像ペア数（0 で可視化なし）')
    p.add_argument('--seed', type=int, default=42,
                   help='固定シード（再現性確保）')
    p.add_argument('--eval_all_pairs', action='store_true',
                   help='全ペアで評価（ランダムサンプリングなし）')
    p.add_argument('--eval_only', action='store_true',
                   help='学習せず評価のみ実行')
    return p.parse_args()


# ---------------------------------------------------------------------------
# ペア生成（SThErEO 1シーケンスから連続フレームペアを取得）
# ---------------------------------------------------------------------------

def make_pairs_from_sequence(
    seq_dir: str,
    max_pairs: int = 500,
    stride: int = 3,
) -> List[Tuple[str, str, np.ndarray, np.ndarray]]:
    """
    SThErEO の 1 シーケンスから (img_t, img_t1, T_rel, K) を返す。
    """
    from modules.dataset.thermal.sequential import (
        _load_sthereo_K,
        _load_sthereo_poses,
        _nearest_pose_idx,
    )

    K     = _load_sthereo_K(
        os.path.join(seq_dir, 'calibration', 'thermal_14bit_left.yaml'))
    poses = _load_sthereo_poses(
        os.path.join(seq_dir, 'pose', 'global_pose.csv'))
    if not poses:
        raise RuntimeError(f"No poses in {seq_dir}")

    pose_ts = [p[0] for p in poses]
    pose_Ts = [p[1] for p in poses]

    img_dir = os.path.join(seq_dir, 'image', 'thermal8_left_clahe')
    if not os.path.isdir(img_dir):
        img_dir = os.path.join(seq_dir, 'image', 'thermal8_left')

    img_files = sorted(f for f in os.listdir(img_dir) if f.endswith('.png'))

    matched: List[Tuple[str, np.ndarray]] = []
    for fname in img_files:
        try:
            ts_ns = int(fname.split('.')[0])
        except ValueError:
            continue
        idx = _nearest_pose_idx(ts_ns, pose_ts)
        if abs(pose_ts[idx] - ts_ns) < 250_000_000:
            matched.append((os.path.join(img_dir, fname), pose_Ts[idx]))

    pairs = []
    for i in range(0, len(matched) - stride, stride):
        j = i + stride
        p_t,  T_t  = matched[i]
        p_t1, T_t1 = matched[j]
        T_rel = np.linalg.inv(T_t) @ T_t1
        pairs.append((p_t, p_t1, T_rel, K))
        if len(pairs) >= max_pairs:
            break

    print(f"  {os.path.basename(seq_dir)}: {len(pairs)} pairs")
    return pairs


# ---------------------------------------------------------------------------
# 損失計算
# ---------------------------------------------------------------------------

def kd_loss_fn(
    student_feats: torch.Tensor,
    teacher_feats: torch.Tensor,
) -> torch.Tensor:
    """特徴マップの L2 蒸留損失（hmap + desc）"""
    return F.mse_loss(
        F.normalize(student_feats, dim=1),
        F.normalize(teacher_feats,  dim=1),
    )


def repeatability_loss_fn(
    kpts1: np.ndarray,
    kpts2: np.ndarray,
    H:     np.ndarray,
    scores1: torch.Tensor,
    hw:    Tuple[int, int],
    threshold: float = 3.0,
) -> torch.Tensor:
    """
    Repeatability 損失: homography で対応づけた点の検出スコアが一致するか。
    """
    if len(kpts1) == 0 or len(kpts2) == 0:
        return torch.tensor(0.0)

    pts1 = np.array(kpts1, dtype=np.float32).reshape(-1, 1, 2)
    pts1_warped = cv2.perspectiveTransform(pts1, H).reshape(-1, 2)

    # warp 後に画像内に収まる点のみ
    H_img, W_img = hw
    valid = (
        (pts1_warped[:, 0] >= 0) & (pts1_warped[:, 0] < W_img) &
        (pts1_warped[:, 1] >= 0) & (pts1_warped[:, 1] < H_img)
    )
    if not valid.any():
        return torch.tensor(0.0)

    pts1_w = pts1_warped[valid]
    dists  = np.linalg.norm(
        pts1_w[:, None] - np.array(kpts2)[None], axis=-1)
    min_dists = dists.min(axis=1)
    inlier_mask = min_dists < threshold

    if inlier_mask.sum() == 0:
        return torch.tensor(0.0)

    # インライアの検出スコアの最大化
    # torch でスコアを取り出して損失化
    inlier_idx = np.where(valid)[0][inlier_mask]
    scores = scores1[inlier_idx]
    # スコアが 1 に近づくよう学習
    loss = (1.0 - scores).mean()
    return loss


# ---------------------------------------------------------------------------
# 公式 LightGlue ロード
# ---------------------------------------------------------------------------

def load_official_lightglue(
    device: torch.device,
    features: str = 'xfeat',
    checkpoint_path: Optional[str] = None,
) -> Optional[Any]:
    """
    XFeat 64 次元対応の fine-tuned LightGlue を読み込む。

    重要: 公式 glue-factory の XFeat LightGlue は input_dim=256 のため
    我々の XFeat (64次元) とは互換性がない → AssertionError になる。
    そのため fine-tuned チェックポイントを使用する。

    Recall の定義:
        分子 = LightGlue が残したマッチのうち GT dist < τ のもの
        分母 = MNN 全マッチのうち GT dist < τ のもの
        → LG の rejection（信頼度）が正解を正しく通過させるかを測定
    """
    from eval.eval_matching import load_lightglue as _load_lg

    # fine-tuned LG のパスを検索
    candidates = [
        checkpoint_path,
        'third_party/glue-factory/outputs/training/thermal_xfeat_lg_v3/checkpoint_best.tar',
        'third_party/glue-factory/outputs/training/thermal_xfeat_lg_v2/checkpoint_best.tar',
        'third_party/glue-factory/outputs/training/thermal_xfeat_lg/checkpoint_best.tar',
    ]
    ckpt = None
    for c in candidates:
        if c and os.path.isfile(c):
            ckpt = c
            break

    if ckpt is None:
        print("[LG] fine-tuned checkpoint not found → Recall は ratio_test で代替")
        return None

    try:
        lg = _load_lg(ckpt, device)
        if lg is None:
            return None
        print(f"[LG] fine-tuned LightGlue loaded: {ckpt}")
        return ('finetuned', lg)
    except Exception as e:
        print(f"[LG] load failed: {e}")
        return None



def run_lightglue(
    lg_handle:  Any,
    kpts1:      np.ndarray,
    descs1:     np.ndarray,
    kpts2:      np.ndarray,
    descs2:     np.ndarray,
    image_size: Tuple[int, int],
    device:     torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    公式 LightGlue でマッチングを実行。

    glue-factory の XFeat 用 LightGlue はフラット形式を期待する:
        {'keypoints0':   (1,N,2) 正規化座標 [-1,1],
         'descriptors0': (1,N,D),
         'image_size0':  (1,2) [H,W],
         'keypoints1':   ...,
         'descriptors1': ...,
         'image_size1':  ...}
    """
    if lg_handle is None:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    _, lg_model = lg_handle
    H, W = image_size

    def to_tensor(kpts, descs):
        """ピクセル座標 → 正規化座標 [-1,1] に変換"""
        k = torch.from_numpy(kpts).float().unsqueeze(0).to(device)
        d = torch.from_numpy(descs).float().unsqueeze(0).to(device)
        k_n = k.clone()
        k_n[..., 0] = (k[..., 0] / W) * 2.0 - 1.0
        k_n[..., 1] = (k[..., 1] / H) * 2.0 - 1.0
        return k_n, d

    source, lg_model = lg_handle

    if source == 'finetuned':
        # eval_matching.py の match_lightglue と同じインターフェースを使用
        # fine-tuned LG は 64 次元 XFeat 記述子に対応している
        from eval.eval_matching import match_lightglue as _mlg
        return _mlg(
            kpts1, descs1, kpts2, descs2,
            image_size=image_size,
            device=device,
            lightglue_model=lg_model,
        )

    # 以下は参考実装（finetuned 以外のソースの場合）
    k1_px = torch.from_numpy(kpts1).float().unsqueeze(0).to(device)
    k2_px = torch.from_numpy(kpts2).float().unsqueeze(0).to(device)
    d1    = torch.from_numpy(descs1).float().unsqueeze(0).to(device)
    d2    = torch.from_numpy(descs2).float().unsqueeze(0).to(device)
    sz    = torch.tensor([[H, W]], device=device)

    try:
        with torch.no_grad():
            pred = lg_model({
                'keypoints0':   k1_px,
                'descriptors0': d1,
                'keypoints1':   k2_px,
                'descriptors1': d2,
                'view0': {'image_size': sz},
                'view1': {'image_size': sz},
            })
        m = pred['matches0'].squeeze(0).cpu().numpy()
        valid = m >= 0
        return np.where(valid)[0].astype(np.int64), m[valid].astype(np.int64)

    except Exception as e:
        import traceback
        print(f"  [LG] run error: {type(e).__name__}: {e}")
        traceback.print_exc()
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)


# ---------------------------------------------------------------------------
# マッチング可視化（3モデル比較）
# ---------------------------------------------------------------------------

def _draw_matches_on_canvas(
    img1_bgr:  np.ndarray,
    img2_bgr:  np.ndarray,
    kpts1:     np.ndarray,
    kpts2:     np.ndarray,
    idx1:      np.ndarray,
    idx2:      np.ndarray,
    F_gt:      Optional[np.ndarray],
    epi_thr:   float = 3.0,
) -> np.ndarray:
    """
    1モデル分のマッチングを横並び画像に描画して返す。

    色分け:
        緑: 正解マッチ（エピポーラ距離 < epi_thr px）
        赤: 誤マッチ
    """
    from eval.eval_matching import _sym_epi_dist

    H, W = img1_bgr.shape[:2]
    gap  = 8
    canvas = np.zeros((H, W * 2 + gap, 3), dtype=np.uint8)
    canvas[:, :W]        = img1_bgr
    canvas[:, W+gap:]    = img2_bgr
    canvas[:, W:W+gap]   = 50   # 区切り

    if len(idx1) == 0:
        return canvas

    # エピポーラ距離で色分け
    if F_gt is not None:
        epi = _sym_epi_dist(
            kpts1[idx1].astype(np.float32),
            kpts2[idx2].astype(np.float32),
            F_gt,
        )
        correct_mask = epi < epi_thr
    else:
        correct_mask = np.ones(len(idx1), dtype=bool)

    # 誤マッチを先に（緑が上に重なるよう）
    for i, (i1, i2) in enumerate(zip(idx1, idx2)):
        if correct_mask[i]:
            continue
        x1, y1 = int(kpts1[i1][0]),        int(kpts1[i1][1])
        x2, y2 = int(kpts2[i2][0])+W+gap,  int(kpts2[i2][1])
        cv2.line(canvas, (x1,y1), (x2,y2), (30,30,200), 1, cv2.LINE_AA)

    # 正解マッチ
    for i, (i1, i2) in enumerate(zip(idx1, idx2)):
        if not correct_mask[i]:
            continue
        x1, y1 = int(kpts1[i1][0]),        int(kpts1[i1][1])
        x2, y2 = int(kpts2[i2][0])+W+gap,  int(kpts2[i2][1])
        cv2.line(canvas, (x1,y1), (x2,y2), (30,200,30), 1, cv2.LINE_AA)

    # キーポイント
    for i1 in idx1:
        cv2.circle(canvas, (int(kpts1[i1][0]), int(kpts1[i1][1])),
                   3, (0, 220, 220), -1)
    for i2 in idx2:
        cv2.circle(canvas,
                   (int(kpts2[i2][0])+W+gap, int(kpts2[i2][1])),
                   3, (0, 220, 220), -1)

    return canvas


def visualize_matches_compare(
    models:     Dict[str, nn.Module],
    pairs:      List[Tuple],
    device:     torch.device,
    output_dir: str,
    n_vis:      int = 5,
    epi_thr:    float = 3.0,
) -> None:
    """
    baseline / proposed(init) / proposed(final) を1枚に並べて保存する。

    出力レイアウト（縦に3モデルを並べる）:
    ┌─────────────────────────────────┐
    │  baseline: [img1]──[img2]       │  統計
    ├─────────────────────────────────┤
    │  proposed(init): [img1]──[img2] │  統計
    ├─────────────────────────────────┤
    │  proposed(final): [img1]──[img2]│  統計
    └─────────────────────────────────┘

    Args:
        models:     {'baseline': model, 'proposed_init': model, 'proposed_final': model}
        n_vis:      保存する画像ペア数（コマンドライン引数で指定可能）
        epi_thr:    正解判定のエピポーラ距離閾値 [px]
    """
    from eval.eval_matching import (
        detect, match, imread_tensor, _compute_F_gt)

    os.makedirs(output_dir, exist_ok=True)

    size   = (640, 480)
    max_kp = 1024
    gap_h  = 12   # モデル間の縦方向マージン
    label_h = 32  # ラベル行の高さ

    model_names = list(models.keys())
    print(f"  [Vis] {n_vis} ペアを可視化 ({len(model_names)} モデル比較) → {output_dir}/")

    for vi, (path_t, path_t1, T_rel, K) in enumerate(pairs[:n_vis]):
        try:
            img_t,  _ = imread_tensor(path_t,  True, device, size)
            img_t1, _ = imread_tensor(path_t1, True, device, size)
        except FileNotFoundError:
            continue

        # GT F 行列
        try:
            F_gt = _compute_F_gt(T_rel, K).astype(np.float32)
        except Exception:
            F_gt = None

        # 画像読み込み
        img1_bgr = cv2.imread(path_t)
        img2_bgr = cv2.imread(path_t1)
        if img1_bgr is None or img2_bgr is None:
            continue
        img1_bgr = cv2.resize(img1_bgr, (size[0], size[1]))
        img2_bgr = cv2.resize(img2_bgr, (size[0], size[1]))
        H, W = size[1], size[0]

        # 各モデルの行を生成
        rows = []
        for m_name, model in models.items():
            model.eval()
            with torch.no_grad():
                kpts1, descs1 = detect(model, img_t,  max_kp)
                kpts2, descs2 = detect(model, img_t1, max_kp)

            if len(kpts1) == 0 or len(kpts2) == 0:
                row = np.zeros((H + label_h, W*2+8, 3), dtype=np.uint8)
            else:
                idx1, idx2 = match(descs1, descs2, 'mutual_nn', ratio_thr=0.9)
                row_img = _draw_matches_on_canvas(
                    img1_bgr, img2_bgr, kpts1, kpts2,
                    idx1, idx2, F_gt, epi_thr)

                # ラベル行
                label_row = np.full((label_h, W*2+8, 3), 30, dtype=np.uint8)

                # 統計テキスト
                n_m = len(idx1)
                if F_gt is not None and n_m > 0:
                    from eval.eval_matching import _sym_epi_dist
                    epi = _sym_epi_dist(
                        kpts1[idx1].astype(np.float32),
                        kpts2[idx2].astype(np.float32), F_gt)
                    n_correct = int((epi < epi_thr).sum())
                    prec = n_correct / max(n_m, 1)
                    stat = (f"{m_name}  |  "
                            f"matches={n_m}  "
                            f"Prec@{epi_thr:.0f}px={prec*100:.1f}%  "
                            f"correct={n_correct}")
                else:
                    stat = f"{m_name}  |  matches={n_m}"

                # モデル名（色で識別）
                label_colors = {
                    'baseline':       (180, 180, 180),
                    'proposed_init':  (80,  200, 80),
                    'proposed_final': (80,  80,  220),
                }
                c = label_colors.get(m_name, (200, 200, 200))
                cv2.putText(label_row, stat, (8, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 1,
                            cv2.LINE_AA)

                row = np.vstack([label_row, row_img])

            rows.append(row)

        # 縦に連結（モデル間に区切り線）
        total_h = sum(r.shape[0] for r in rows) + gap_h * (len(rows)-1)
        full_w  = rows[0].shape[1] if rows else W*2+8
        canvas  = np.zeros((total_h, full_w, 3), dtype=np.uint8)

        y = 0
        for ri, row in enumerate(rows):
            h = row.shape[0]
            canvas[y:y+h, :full_w] = row[:, :full_w]
            y += h
            if ri < len(rows) - 1:
                canvas[y:y+gap_h] = 20  # 区切り線
                y += gap_h

        # ヘッダー（ペア番号）
        cv2.putText(canvas,
                    f"pair {vi+1}/{n_vis}  |  green=correct  red=wrong",
                    (8, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (200, 200, 200), 1, cv2.LINE_AA)

        out_path = os.path.join(output_dir, f"compare_pair{vi+1:02d}.png")
        cv2.imwrite(out_path, canvas)
        print(f"  [Vis] → {out_path}")


def geometry_loss_fn(
    kpts1: np.ndarray,
    kpts2: np.ndarray,
    idx1:  np.ndarray,
    idx2:  np.ndarray,
    T_rel: np.ndarray,
    K:     np.ndarray,
) -> torch.Tensor:
    """
    幾何整合損失: GT Fundamental Matrix によるエピポーラ距離。
    """
    from eval.eval_matching import _compute_F_gt, _sym_epi_dist

    if len(idx1) < 8:
        return torch.tensor(0.0)

    try:
        F_gt = _compute_F_gt(T_rel, K).astype(np.float32)
        epi_dists = _sym_epi_dist(
            kpts1[idx1].astype(np.float32),
            kpts2[idx2].astype(np.float32),
            F_gt,
        )
        # エピポーラ距離の平均を損失化（小さいほど良い）
        return torch.tensor(float(np.mean(epi_dists)))
    except Exception:
        return torch.tensor(0.0)


# ---------------------------------------------------------------------------
# 評価（MS・PoseAUC・n_matches）
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model:      nn.Module,
    pairs:      List[Tuple],
    device:     torch.device,
    n_pairs:    int = 100,
    lg_handle:  Any = None,
) -> Dict[str, float]:
    """
    論文標準の指標を計算する。

    Precision@τ [SuperGlue/LightGlue 準拠]:
        「実際のマッチのうち GT エピポーラ距離 < τ の割合」
        = TP / (TP + FP)
        = 質の指標（マッチの正確さ）

    Recall@τ [LightGlue 論文準拠]:
        「MNN で取れる全正解マッチのうち実際に取れた割合」
        = TP / (TP + FN)
        分母 = MNN(全KP1, 全KP2)で得られる全マッチのうち dist < τ のもの
        = 量の指標（見逃しがないか）

        ※ 従来の MS（Matching Score = matches/min(kp1,kp2)）は
          精度を考慮しないため、Recall とは別物。

    F1@τ: 2*P*R / (P+R)  Precision と Recall の調和平均

    MS: matches / min(kp1, kp2)  量の指標（精度非考慮）
    n_matches: 平均マッチ数（絶対値）
    """
    from eval.eval_matching import (
        detect, match, imread_tensor, _compute_F_gt, _sym_epi_dist)

    model.eval()
    size   = (640, 480)
    max_kp = 1024
    method = 'mutual_nn'
    thrs   = [1, 3, 5]   # px 閾値

    ms_list         = []
    n_match_list    = []
    pose_errs       = []
    prec_vals       = {t: [] for t in thrs}
    # Recall の分子・分母をペアごとに累積
    recall_tp       = {t: [] for t in thrs}   # TP: 実際のマッチで dist < τ
    recall_denom    = {t: [] for t in thrs}   # TP+FN: MNN全点で dist < τ

    pairs_eval = pairs[:n_pairs]

    for path_t, path_t1, T_rel, K in pairs_eval:
        try:
            img_t,  _ = imread_tensor(path_t,  True, device, size)
            img_t1, _ = imread_tensor(path_t1, True, device, size)
        except FileNotFoundError:
            continue

        kpts1, descs1 = detect(model, img_t,  max_kp)
        kpts2, descs2 = detect(model, img_t1, max_kp)
        if len(kpts1) == 0 or len(kpts2) == 0:
            continue

        # ── GT F 行列を計算（Precision/Recall に使用）──────────────────
        try:
            F_gt = _compute_F_gt(T_rel, K).astype(np.float32)
            has_gt_f = True
        except Exception:
            has_gt_f = False

        # ── 実際のマッチング ─────────────────────────────────────────────
        idx1, idx2 = match(descs1, descs2, method, ratio_thr=0.9)
        n_m = len(idx1)

        ms = n_m / max(min(len(kpts1), len(kpts2)), 1)
        ms_list.append(ms)
        n_match_list.append(n_m)

        if has_gt_f and n_m > 0:
            # ── Precision@τ ─────────────────────────────────────────────
            # 定義: 実際のマッチのうち GT エピポーラ距離 < τ の割合
            # = TP / (TP + FP)
            epi_actual = _sym_epi_dist(
                kpts1[idx1].astype(np.float32),
                kpts2[idx2].astype(np.float32),
                F_gt,
            )
            for t in thrs:
                prec = float((epi_actual < t).mean())
                prec_vals[t].append(prec)
                tp = int((epi_actual < t).sum())
                recall_tp[t].append(tp)

            # ── Recall@τ（公式 LightGlue ベース）───────────────────────────
            # 設計:
            #   分母 = MNN(全KP) の正解数（rejection なし・取れる上限）
            #   分子 = LG が残したマッチの正解数（rejection あり・実際に取れた）
            #
            # LG の信頼度 threshold が rejection 機構なので
            # 分子 ≦ 分母 が保証され、Recall が [0,1] に収まる。
            # LG が使えない場合は ratio_test で代替。

            if len(kpts1) > 0 and len(kpts2) > 0:
                d1 = descs1 / (np.linalg.norm(descs1, axis=1, keepdims=True) + 1e-8)
                d2 = descs2 / (np.linalg.norm(descs2, axis=1, keepdims=True) + 1e-8)
                sim = d1 @ d2.T   # (N1, N2)

                # ── 分母: MNN 全点での正解数（棄却なし）─────────────────────
                nn12 = np.argmax(sim, axis=1)
                nn21 = np.argmax(sim, axis=0)
                mnn_mask = nn21[nn12] == np.arange(len(kpts1))
                mnn_idx1 = np.where(mnn_mask)[0]
                mnn_idx2 = nn12[mnn_mask]

                if len(mnn_idx1) > 0:
                    epi_mnn = _sym_epi_dist(
                        kpts1[mnn_idx1].astype(np.float32),
                        kpts2[mnn_idx2].astype(np.float32),
                        F_gt,
                    )
                    for t in thrs:
                        denom = int((epi_mnn < t).sum())
                        recall_denom[t].append(max(denom, 1))

                # ── 分子: LG（または ratio_test）での正解数 ──────────────────
                if lg_handle is not None:
                    # 公式 LightGlue でマッチング → rejection あり
                    lg_i1, lg_i2 = run_lightglue(
                        lg_handle, kpts1, descs1, kpts2, descs2,
                        image_size=(size[1], size[0]), device=device)

                    if len(lg_i1) > 0:
                        epi_lg = _sym_epi_dist(
                            kpts1[lg_i1].astype(np.float32),
                            kpts2[lg_i2].astype(np.float32),
                            F_gt,
                        )
                        for t in thrs:
                            recall_tp[t].append(int((epi_lg < t).sum()))
                    else:
                        for t in thrs:
                            recall_tp[t].append(0)
                else:
                    # LG 未使用時: ratio_test（次善策）
                    nn12_s = np.argsort(-sim, axis=1)
                    if sim.shape[1] >= 2:
                        top1 = sim[np.arange(len(kpts1)), nn12_s[:, 0]]
                        top2 = sim[np.arange(len(kpts1)), nn12_s[:, 1]]
                        rt_mask = (top1 / (top2 + 1e-8)) > 0.9
                        rt_idx1 = np.where(rt_mask & mnn_mask)[0]
                        rt_idx2 = nn12[rt_idx1]
                        if len(rt_idx1) > 0:
                            epi_rt = _sym_epi_dist(
                                kpts1[rt_idx1].astype(np.float32),
                                kpts2[rt_idx2].astype(np.float32),
                                F_gt,
                            )
                            for t in thrs:
                                recall_tp[t].append(int((epi_rt < t).sum()))
                        else:
                            for t in thrs:
                                recall_tp[t].append(0)
                    else:
                        for t in thrs:
                            recall_tp[t].append(0)

        if n_m < 8:
            continue

        # ── Pose AUC ─────────────────────────────────────────────────────
        K_np = np.array(K, dtype=np.float64)
        pts1_r = kpts1[idx1].astype(np.float32).reshape(-1, 1, 2)
        pts2_r = kpts2[idx2].astype(np.float32).reshape(-1, 1, 2)

        E, e_mask = cv2.findEssentialMat(
            pts1_r, pts2_r, K_np, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None or e_mask is None:
            continue
        if E.shape[0] > 3:
            E = E[:3, :]
        if int(e_mask.sum()) < 5:
            continue

        _, R_est, t_est, _ = cv2.recoverPose(E, pts1_r, pts2_r, K_np, mask=e_mask)
        T_np = np.array(T_rel)
        R_gt, t_gt = T_np[:3, :3], T_np[:3, 3]

        R_rel = R_est @ R_gt.T
        trace = float(np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0))
        R_err = float(np.degrees(np.arccos(trace)))

        if np.linalg.norm(t_gt) < 1e-4:
            continue
        t_e = t_est.ravel() / (np.linalg.norm(t_est) + 1e-8)
        t_g = t_gt / np.linalg.norm(t_gt)
        t_err = float(np.degrees(
            np.arccos(abs(float(np.clip(np.dot(t_e, t_g), -1, 1))))))
        pose_errs.append(max(R_err, t_err))

    # ── 集計 ─────────────────────────────────────────────────────────────
    n = len(ms_list)
    pose_arr = np.array(pose_errs) if pose_errs else np.array([180.0])

    # Precision@τ（全ペアの平均）
    prec_out = {}
    for t in thrs:
        vals = prec_vals[t]
        prec_out[f'Prec@{t}px'] = float(np.mean(vals)) if vals else 0.0

    # Recall@τ = Σ TP / Σ (TP+FN)  （ペアをプールして計算）
    recall_out = {}
    for t in thrs:
        tp_list    = recall_tp[t]
        denom_list = recall_denom[t]
        n_common   = min(len(tp_list), len(denom_list))
        if n_common > 0:
            total_tp    = sum(tp_list[:n_common])
            total_denom = sum(denom_list[:n_common])
            recall_out[f'Rec@{t}px'] = (total_tp / total_denom
                                          if total_denom > 0 else 0.0)
        else:
            recall_out[f'Rec@{t}px'] = 0.0

    # F1@τ = 2*P*R / (P+R)
    f1_out = {}
    for t in thrs:
        p = prec_out.get(f'Prec@{t}px', 0.0)
        r = recall_out.get(f'Rec@{t}px', 0.0)
        f1_out[f'F1@{t}px'] = (2*p*r/(p+r)) if (p+r) > 1e-8 else 0.0

    return {
        'MS':         float(np.mean(ms_list))       if ms_list     else 0.0,
        'n_matches':  float(np.mean(n_match_list))  if n_match_list else 0.0,
        'PoseAUC@5':  float((pose_arr < 5).mean()),
        'PoseAUC@10': float((pose_arr < 10).mean()),
        'PoseAUC@20': float((pose_arr < 20).mean()),
        'n_pairs':    n,
        **prec_out,
        **recall_out,
        **f1_out,
    }


# ---------------------------------------------------------------------------
# 学習ループ
# ---------------------------------------------------------------------------

def main():
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    exp_name = f"overfit_{args.seq}_{args.loss}_{args.epochs}ep"
    out_dir  = os.path.join(args.output_dir, exp_name)
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, 'log.csv')

    print(f"\n{'='*60}")
    print(f"  過学習実験: {exp_name}")
    print(f"  Loss: {args.loss}, Epochs: {args.epochs}")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    # ── ペア生成 ────────────────────────────────────────────────────────
    seq_dir = os.path.join(args.sthereo_root, args.seq)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    pairs   = make_pairs_from_sequence(seq_dir, args.max_pairs, stride=3)

    # ── 教師モデル（RGB XFeat, 固定）────────────────────────────────────
    from modules.model import XFeatModel
    teacher = XFeatModel().to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    print("Teacher (RGB XFeat): loaded")

    # ── 生徒モデル（提案手法の重みで初期化）────────────────────────────
    student = XFeatModel().to(device).train()
    if os.path.isfile(args.weights_proposed):
        state = torch.load(args.weights_proposed,
                           map_location=device, weights_only=True)
        student.load_state_dict(state)
        print(f"Student initialized from: {args.weights_proposed}")
    else:
        print("Student initialized from scratch (default RGB weights)")

    optimizer = torch.optim.Adam(student.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    # ── 公式 LightGlue をロード（Recall の分子計算に使用）────────────
    print("\n[LightGlue ロード（公式 XFeat 用）]")
    lg_handle = load_official_lightglue(device, features='xfeat')
    if lg_handle is None:
        print("  LightGlue 未使用 → Recall は ratio_test で代替")
    else:
        print(f"  LightGlue 使用: {lg_handle[0]}")

    # ── Baseline 評価 ──────────────────────────────────────────────────
    print("\n[Baseline 評価]")
    baseline_model = XFeatModel().to(device).eval()
    n_eval   = len(pairs) if args.eval_all_pairs else min(100, len(pairs))
    print(f'  評価ペア数: {n_eval}/{len(pairs)}')
    baseline = evaluate(baseline_model, pairs, device,
                        n_pairs=n_eval, lg_handle=lg_handle)
    print(f"  baseline: MS={baseline['MS']*100:.1f}%  "
          f"PoseAUC@5={baseline['PoseAUC@5']*100:.1f}%  "
          f"n_matches={baseline['n_matches']:.0f}")

    print("\n[提案手法 評価（学習前）]")
    proposed_init = evaluate(student, pairs, device,
                              n_pairs=n_eval, lg_handle=lg_handle)
    print(f"  proposed (init): MS={proposed_init['MS']*100:.1f}%  "
          f"PoseAUC@5={proposed_init['PoseAUC@5']*100:.1f}%  "
          f"n_matches={proposed_init['n_matches']:.0f}")

    # ── ログヘッダ ─────────────────────────────────────────────────────
    def _fmt_row(tag: str, loss: str, m: dict) -> str:
        return (
            f"{tag},{loss},"
            f"{m['MS']*100:.1f},{m['n_matches']:.0f},"
            f"{m.get('Prec@1px',0)*100:.1f},{m.get('Rec@1px',0)*100:.1f},{m.get('F1@1px',0)*100:.1f},"
            f"{m.get('Prec@3px',0)*100:.1f},{m.get('Rec@3px',0)*100:.1f},{m.get('F1@3px',0)*100:.1f},"
            f"{m.get('Prec@5px',0)*100:.1f},{m.get('Rec@5px',0)*100:.1f},{m.get('F1@5px',0)*100:.1f},"
            f"{m['PoseAUC@5']*100:.1f},{m['PoseAUC@10']*100:.1f},{m['PoseAUC@20']*100:.1f}"
        )

    with open(log_path, 'w') as f:
        f.write("epoch,loss_total,"
                "MS,n_matches,"
                "Prec@1px,Rec@1px,F1@1px,"
                "Prec@3px,Rec@3px,F1@3px,"
                "Prec@5px,Rec@5px,F1@5px,"
                "PoseAUC@5,PoseAUC@10,PoseAUC@20\n")
        f.write(_fmt_row("0(baseline)", "-", baseline) + "\n")
        f.write(_fmt_row("0(init)",     "-", proposed_init) + "\n")

    # ── 損失の設定 ────────────────────────────────────────────────────
    use_kd  = args.loss in ('kd_only',  'all')
    use_rep = args.loss in ('rep_only', 'all')
    use_geo = args.loss in ('geo_only', 'all')
    print(f"\n  使用する損失: kd={use_kd}, rep={use_rep}, geo={use_geo}\n")

    from eval.eval_matching import imread_tensor, detect, match

    size   = (640, 480)
    max_kp = 1024

    # ── 学習ループ ───────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        student.train()
        epoch_losses = []

        # ランダムにバッチを選択
        idx = np.random.permutation(len(pairs))[:args.batch]

        for i in idx:
            path_t, path_t1, T_rel, K = pairs[i]
            try:
                img_t,  _ = imread_tensor(path_t,  True, device, size)
                img_t1, _ = imread_tensor(path_t1, True, device, size)
            except FileNotFoundError:
                continue

            loss = torch.tensor(0.0, requires_grad=True, device=device)

            # ── KD 損失（記述子空間の蒸留）──────────────────────────
            # XFeatModel.forward() の出力から特徴マップを取り出し
            # teacher-student 間の L2 距離を最小化する。
            # detect() は @torch.no_grad() のため直接 forward() を呼ぶ。
            if use_kd:
                try:
                    with torch.no_grad():
                        t_out = teacher(img_t)
                    s_out = student(img_t)
                    # forward の出力形式を動的に判定
                    # XFeat は (feats, keypoints, heatmap) のタプル or dict を返す
                    if isinstance(t_out, (tuple, list)):
                        t_feat = t_out[0]  # 最初の要素が特徴マップ
                        s_feat = s_out[0]
                    elif isinstance(t_out, dict):
                        key = 'feats' if 'feats' in t_out else list(t_out.keys())[0]
                        t_feat = t_out[key]
                        s_feat = s_out[key]
                    else:
                        t_feat = t_out
                        s_feat = s_out
                    l_kd = kd_loss_fn(s_feat, t_feat.detach())
                    loss = loss + l_kd
                except Exception as e:
                    print(f"  [KD] forward error: {e} → skip")

            # ── Repeatability 損失 ──────────────────────────────────
            if use_rep:
                with torch.no_grad():
                    kpts1, _ = detect(student, img_t,  max_kp)
                    kpts2, _ = detect(student, img_t1, max_kp)
                # homography ペアでの repeatability
                # SThErEO では pose があるので F 行列で代用
                # ここでは簡易的に連続フレームの kpts 数の差で評価
                n1, n2 = len(kpts1), len(kpts2)
                # KP 数が均等に出るよう学習（不安定な点を除去）
                if n1 > 0 and n2 > 0:
                    ratio_kp = min(n1, n2) / max(n1, n2)
                    l_rep = torch.tensor(1.0 - ratio_kp, device=device)
                    loss = loss + l_rep

            # ── 幾何整合損失 ────────────────────────────────────────
            if use_geo:
                with torch.no_grad():
                    kpts1, descs1 = detect(student, img_t,  max_kp)
                    kpts2, descs2 = detect(student, img_t1, max_kp)
                    idx1, idx2 = match(descs1, descs2, 'mutual_nn', 0.9)
                l_geo = geometry_loss_fn(
                    kpts1, kpts2, idx1, idx2, T_rel, K)
                loss = loss + l_geo.to(device)

            if loss.requires_grad:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                optimizer.step()
                epoch_losses.append(float(loss.detach()))

        scheduler.step()

        # ── 定期評価 ─────────────────────────────────────────────────
        if epoch % args.eval_interval == 0 or epoch == args.epochs:
            metrics = evaluate(student, pairs, device,
                                  n_pairs=n_eval, lg_handle=lg_handle)
            avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0

            print(f"  Epoch {epoch:4d}/{args.epochs}  loss={avg_loss:.4f}")
            print(f"    MS={metrics['MS']*100:.1f}%  n_match={metrics['n_matches']:.0f}")
            print(f"    Prec@3px={metrics.get('Prec@3px',0)*100:.1f}%  "
                  f"Rec@3px={metrics.get('Rec@3px',0)*100:.1f}%  "
                  f"F1@3px={metrics.get('F1@3px',0)*100:.1f}%")
            print(f"    PoseAUC@5={metrics['PoseAUC@5']*100:.1f}%  "
                  f"PoseAUC@10={metrics['PoseAUC@10']*100:.1f}%")

            with open(log_path, 'a') as f:
                f.write(_fmt_row(str(epoch), f"{avg_loss:.4f}", metrics) + "\n")

    # ── 最終結果のまとめ ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  最終結果 (loss={args.loss}, {args.epochs} epochs)")
    print(f"{'='*60}")
    final = evaluate(student, pairs, device,
                     n_pairs=n_eval, lg_handle=lg_handle)
    def _show(tag: str, m: dict):
        print(f"  {tag:<20s} "
              f"MS={m['MS']*100:.1f}%  n_match={m['n_matches']:.0f}")
        print(f"  {'':20s} "
              f"Prec@3px={m.get('Prec@3px',0)*100:.1f}%  "
              f"Rec@3px={m.get('Rec@3px',0)*100:.1f}%  "
              f"F1@3px={m.get('F1@3px',0)*100:.1f}%")
        print(f"  {'':20s} "
              f"PoseAUC@5={m['PoseAUC@5']*100:.1f}%  "
              f"PoseAUC@10={m['PoseAUC@10']*100:.1f}%")

    _show("baseline:",       baseline)
    _show("proposed(init):", proposed_init)
    _show("proposed(final):", final)

    # ── 判断 ─────────────────────────────────────────────────────────
    print(f"\n  [判断]")

    prec_final    = final.get('Prec@3px', 0.0)
    prec_baseline = baseline.get('Prec@3px', 0.0)
    prec_init     = proposed_init.get('Prec@3px', 0.0)
    pose_final    = final['PoseAUC@5']
    pose_baseline = baseline['PoseAUC@5']
    pose_init     = proposed_init['PoseAUC@5']

    # 矛盾パターンの検出
    if pose_final < pose_baseline * 0.8 and prec_final > prec_baseline:
        print(f"  ⚠️  [矛盾検出] Prec 最高 ({prec_final*100:.1f}%) なのに")
        print(f"      PoseAUC が baseline 以下 ({pose_final*100:.1f}% < {pose_baseline*100:.1f}%)")
        print(f"")
        print(f"  原因の可能性:")
        print(f"  (1) KP が特定領域に集中（空間分布の退化）")
        print(f"      → E行列推定に必要な5点分散が不十分")
        print(f"      → 確認: 可視化でKPが画像全体に分散しているか確認")
        print(f"  (2) 繰り返しパターンを記憶（テクスチャの曖昧性）")
        print(f"      → 道路・壁などで複数の「正解に見える」マッチが生成")
        print(f"      → E行列が一意に定まらない（degenerate configuration）")
        print(f"  (3) 過学習によるドメイン固有記憶")
        print(f"      → 特定フレームのテクスチャを記憶しただけで")
        print(f"         幾何的整合性が学習されていない")
        print(f"")
        print(f"  対策: PoseAUC に幾何整合損失を直接追加して学習")

    elif final['MS'] >= baseline['MS'] * 0.9 and pose_final >= pose_baseline:
        print(f"  ✅ MS と PoseAUC が両方 baseline 水準に回復")
        print(f"     → 汎化問題（データ不足）が主因")
        print(f"     → 全シーケンスで再学習することで改善する")

    elif pose_final >= baseline['PoseAUC@5'] * 1.1 and final['MS'] < baseline['MS'] * 0.8:
        print(f"  ⚠️  PoseAUC は改善するが MS は低いまま")
        print(f"     → 幾何精度は学習できるが KP 選択に問題")
        print(f"     → 対策: 温度勾配ベース KP 選択を追加")
    else:
        print(f"  ❌ 過学習でも PoseAUC が改善しない")
        print(f"     → 損失設計の根本的な問題の可能性")
        print(f"     → kd_only で再実験して KD 損失の影響を切り分け")

    torch.save(student.state_dict(),
               os.path.join(out_dir, 'model_final.pth'))

    # ── マッチング可視化（3モデル比較）────────────────────────────────
    if args.n_vis > 0:
        vis_dir = os.path.join(out_dir, 'vis_matches')
        print(f"\n  [可視化] {args.n_vis} ペアを3モデル比較で生成中 → {vis_dir}/")

        # proposed_init の重みを読み込む
        proposed_init_model = XFeatModel().to(device).eval()
        if os.path.isfile(args.weights_proposed):
            state = torch.load(args.weights_proposed,
                               map_location=device, weights_only=True)
            proposed_init_model.load_state_dict(state)

        student.eval()
        visualize_matches_compare(
            models={
                'baseline':       XFeatModel().to(device).eval(),
                'proposed_init':  proposed_init_model,
                'proposed_final': student,
            },
            pairs      = pairs,
            device     = device,
            output_dir = vis_dir,
            n_vis      = args.n_vis,
            epi_thr    = 3.0,
        )
    else:
        print("  [可視化] --n_vis 0 のためスキップ")

    print(f"\n  ログ:   {log_path}")
    print(f"  モデル: {out_dir}/model_final.pth")
    print(f"  可視化: {vis_dir}/")


if __name__ == '__main__':
    main()