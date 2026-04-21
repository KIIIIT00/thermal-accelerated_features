"""
evaluate_vo.py
Visual Odometry 評価エントリーポイント。

ThermalXFeat / XFeat を特徴抽出器として使用したモノキュラー VO を
SThErEO / VIVID で実行し、ATE・RPE を GT と比較する。

使用方法:
    python evaluate_vo.py --config configs/vo_config.yaml
    python evaluate_vo.py --config configs/vo_config.yaml \
        --dataset sthereo --n_seqs 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import yaml

# evaluate/ をモジュールとして認識させる
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)


# ---------------------------------------------------------------------------
# 設定読み込み
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


# ---------------------------------------------------------------------------
# 姿勢ユーティリティ
# ---------------------------------------------------------------------------

def rotation_angle(R: np.ndarray) -> float:
    """回転行列 → 回転角 [degrees]"""
    trace = float(np.trace(R))
    return float(np.degrees(np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))))


def relative_pose_error(
    T_est: np.ndarray,
    T_gt:  np.ndarray,
) -> Tuple[float, float]:
    """
    1ペアの Relative Pose Error を計算する。

    Args:
        T_est: 推定相対姿勢 (4×4)
        T_gt:  GT 相対姿勢 (4×4)

    Returns:
        r_err [deg], t_err [deg] (translation 方向誤差)
    """
    R_est, t_est = T_est[:3, :3], T_est[:3, 3]
    R_gt,  t_gt  = T_gt[:3, :3],  T_gt[:3, 3]

    # Rotation error
    R_rel = R_est @ R_gt.T
    r_err = rotation_angle(R_rel)

    # Translation direction error
    t_est_n = t_est / (np.linalg.norm(t_est) + 1e-8)
    t_gt_n  = t_gt  / (np.linalg.norm(t_gt)  + 1e-8)
    dot     = float(np.clip(np.dot(t_est_n, t_gt_n), -1.0, 1.0))
    t_err   = float(np.degrees(np.arccos(abs(dot))))

    return r_err, t_err


def align_trajectories(
    est_traj: List[np.ndarray],
    gt_traj:  List[np.ndarray],
) -> Tuple[np.ndarray, float]:
    """
    Umeyama alignment: est → gt のスケール・回転・平行移動を求める。

    Args:
        est_traj: [(3,) position, ...] 推定軌跡
        gt_traj:  [(3,) position, ...] GT 軌跡

    Returns:
        aligned_est: アライン済み推定軌跡 (N, 3)
        scale:       スケール係数
    """
    est = np.array(est_traj)   # (N, 3)
    gt  = np.array(gt_traj)    # (N, 3)
    n   = len(est)

    mu_est = est.mean(0)
    mu_gt  = gt.mean(0)
    est_c  = est - mu_est
    gt_c   = gt  - mu_gt

    sigma_est = (est_c ** 2).sum() / n
    H         = (gt_c.T @ est_c) / n
    U, S, Vt  = np.linalg.svd(H)

    d = np.linalg.det(U @ Vt)
    D = np.diag([1.0, 1.0, d])
    R_align = U @ D @ Vt
    scale   = float(S.sum() / sigma_est) if sigma_est > 1e-8 else 1.0
    t_align = mu_gt - scale * R_align @ mu_est

    aligned = (scale * (R_align @ est_c.T).T) + mu_gt
    return aligned, scale


def compute_ate(
    est_traj: List[np.ndarray],
    gt_traj:  List[np.ndarray],
) -> Dict[str, float]:
    """
    ATE (Absolute Trajectory Error) を計算する。

    Umeyama alignment 後の RMSE と平均誤差を返す。
    """
    if len(est_traj) < 3:
        return {'ate_rmse': float('inf'), 'ate_mean': float('inf')}

    aligned, scale = align_trajectories(est_traj, gt_traj)
    gt_arr = np.array(gt_traj)
    diff   = aligned - gt_arr
    dist   = np.linalg.norm(diff, axis=1)
    return {
        'ate_rmse':  float(np.sqrt((dist ** 2).mean())),
        'ate_mean':  float(dist.mean()),
        'ate_max':   float(dist.max()),
        'scale':     scale,
    }


def compute_rpe(
    est_poses: List[np.ndarray],
    gt_poses:  List[np.ndarray],
    delta:     int = 1,
) -> Dict[str, float]:
    """
    RPE (Relative Pose Error) を計算する。

    Args:
        est_poses: [(4×4) T_world, ...] 推定絶対姿勢
        gt_poses:  [(4×4) T_world, ...] GT 絶対姿勢
        delta:     間隔（1 = 隣接フレーム）
    """
    r_errs, t_errs = [], []
    n = len(est_poses)

    for i in range(n - delta):
        j = i + delta
        T_est_rel = np.linalg.inv(est_poses[i]) @ est_poses[j]
        T_gt_rel  = np.linalg.inv(gt_poses[i])  @ gt_poses[j]
        r_err, t_err = relative_pose_error(T_est_rel, T_gt_rel)
        r_errs.append(r_err)
        t_errs.append(t_err)

    if not r_errs:
        return {'rpe_r_mean': float('inf'), 'rpe_t_mean': float('inf')}

    return {
        'rpe_r_mean': float(np.mean(r_errs)),
        'rpe_r_rmse': float(np.sqrt(np.mean(np.array(r_errs) ** 2))),
        'rpe_t_mean': float(np.mean(t_errs)),
        'rpe_t_rmse': float(np.sqrt(np.mean(np.array(t_errs) ** 2))),
    }


# ---------------------------------------------------------------------------
# モデルロード
# ---------------------------------------------------------------------------

def load_model(weights_path: Optional[str], device: torch.device):
    from modules.model import XFeatModel
    model = XFeatModel().to(device).eval()
    if weights_path and os.path.isfile(weights_path):
        state = torch.load(weights_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        print(f"  [Model] Loaded: {weights_path}")
    else:
        print(f"  [Model] Default weights")
    return model


def load_lightglue(weights_path: Optional[str],
                   device: torch.device):
    """fine-tuning 済み LightGlue を読み込む（オプション）"""
    try:
        from eval.eval_matching import load_lightglue as _load_lg
        lg = _load_lg(weights_path, device)
        return lg
    except Exception as e:
        print(f"  [LightGlue] load failed: {e}")
        return None


# ---------------------------------------------------------------------------
# 特徴抽出・マッチング
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract(model, img_bgr: np.ndarray,
            max_kp: int, device: torch.device,
            size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """BGR 画像から kpts (N,2), descs (N,64) を抽出する。"""
    from eval.eval_matching import imread_tensor, detect

    # BGR → tensor
    h, w = size[1], size[0]
    img_r = cv2.resize(img_bgr, (w, h))
    gray  = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
    gray3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    t = torch.from_numpy(gray3).permute(2, 0, 1).float() / 255.0
    t = t.unsqueeze(0).to(device)

    from eval.eval_matching import detect as _detect
    return _detect(model, t, max_kp)


def match_mnn(descs1: np.ndarray,
              descs2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """相互最近傍マッチング"""
    if len(descs1) == 0 or len(descs2) == 0:
        return np.array([], np.int64), np.array([], np.int64)
    d1 = descs1 / (np.linalg.norm(descs1, axis=1, keepdims=True) + 1e-8)
    d2 = descs2 / (np.linalg.norm(descs2, axis=1, keepdims=True) + 1e-8)
    sim  = d1 @ d2.T
    nn12 = np.argmax(sim, axis=1)
    nn21 = np.argmax(sim, axis=0)
    ids  = np.arange(len(descs1))
    mask = nn21[nn12] == ids
    return ids[mask], nn12[mask]


def match_lightglue_vo(
    kpts1: np.ndarray, descs1: np.ndarray,
    kpts2: np.ndarray, descs2: np.ndarray,
    image_size: Tuple[int, int],
    device: torch.device,
    lg_model,
) -> Tuple[np.ndarray, np.ndarray]:
    """LightGlue マッチング"""
    try:
        from eval.eval_matching import match_lightglue
        return match_lightglue(
            kpts1, descs1, kpts2, descs2,
            image_size=image_size,
            device=device,
            lightglue_model=lg_model,
        )
    except Exception as e:
        print(f"  [LG] error: {e} → MNN fallback")
        return match_mnn(descs1, descs2)


# ---------------------------------------------------------------------------
# VO パイプライン（1シーケンス）
# ---------------------------------------------------------------------------

def run_vo_sequence(
    img_pose_list: List[Tuple[str, np.ndarray]],
    K:             np.ndarray,
    model,
    device:        torch.device,
    cfg:           dict,
    lg_model       = None,
) -> Dict[str, Any]:
    """
    1シーケンスに対して VO を実行する。

    Args:
        img_pose_list: [(img_path, T_world(4×4)), ...] タイムスタンプ順
        K:             カメラ内部パラメータ (3×3)
        model:         特徴抽出モデル
        device:        torch.device
        cfg:           設定辞書
        lg_model:      LightGlue モデル（None の場合は MNN）

    Returns:
        {'ate': {...}, 'rpe': {...}, 'n_frames': int, 'n_matched_pairs': int}
    """
    max_kp   = cfg.get('max_keypoints', 512)
    size     = tuple(cfg.get('image_size', [640, 480]))  # (W, H)
    min_inliers = cfg.get('min_inliers', 15)
    use_lg   = lg_model is not None and cfg.get('matching_method', 'mnn') == 'lightglue'

    K_f = K.astype(np.float64)

    # 推定姿勢（累積）
    # est_traj と gt_traj は常に同じ長さを保つ
    # マッチング失敗フレームは「前の推定姿勢を維持」として記録する
    est_poses:  List[np.ndarray] = []
    gt_poses:   List[np.ndarray] = []
    est_traj:   List[np.ndarray] = []
    gt_traj:    List[np.ndarray] = []

    n_matched    = 0
    n_failed     = 0
    prev_kpts    = prev_descs = None
    T_cum        = np.eye(4)          # 累積変換（ワールド座標）
    T_cum_prev   = np.eye(4)          # 前フレームの T_cum

    for i, (img_path, T_gt) in enumerate(img_pose_list):
        bgr = cv2.imread(img_path)
        if bgr is None:
            continue

        kpts, descs = extract(model, bgr, max_kp, device, size)
        if len(kpts) == 0:
            continue

        if prev_kpts is None:
            # 最初のフレーム: 推定 = 原点、GT = 実際の位置
            prev_kpts, prev_descs = kpts, descs
            gt_poses.append(T_gt)
            gt_traj.append(T_gt[:3, 3].copy())
            est_poses.append(T_cum.copy())
            est_traj.append(T_cum[:3, 3].copy())
            continue

        # ── マッチング ───────────────────────────────────────────────────
        if use_lg:
            idx1, idx2 = match_lightglue_vo(
                prev_kpts, prev_descs, kpts, descs,
                image_size=(size[1], size[0]),
                device=device, lg_model=lg_model)
        else:
            idx1, idx2 = match_mnn(prev_descs, descs)

        n_matches = len(idx1)
        match_ok  = False

        if n_matches >= 8:
            pts1 = prev_kpts[idx1].astype(np.float32).reshape(-1, 1, 2)
            pts2 = kpts[idx2].astype(np.float32).reshape(-1, 1, 2)

            E, e_mask = cv2.findEssentialMat(
                pts1, pts2, K_f,
                method=cv2.RANSAC, prob=0.999, threshold=1.0)

            if E is not None and e_mask is not None:
                n_inliers = int(e_mask.sum())
                if n_inliers >= min_inliers:
                    if E.shape[0] > 3:
                        E = E[:3, :]
                    _, R, t, _ = cv2.recoverPose(
                        E, pts1, pts2, K_f, mask=e_mask)

                    # GT スケールで t を正規化
                    gt_t_norm = np.linalg.norm(
                        T_gt[:3, 3] - gt_poses[-1][:3, 3])
                    t_scaled = t.ravel() * (gt_t_norm if gt_t_norm > 1e-4 else 1.0)

                    T_rel = np.eye(4)
                    T_rel[:3, :3] = R
                    T_rel[:3,  3] = t_scaled
                    T_cum = T_cum @ T_rel
                    match_ok = True
                    n_matched += 1

        if not match_ok:
            # マッチング失敗: 前フレームの姿勢を維持（ゼロ速度仮定）
            n_failed += 1

        # 常に両方に追加して長さを一致させる
        est_poses.append(T_cum.copy())
        gt_poses.append(T_gt)
        est_traj.append(T_cum[:3, 3].copy())
        gt_traj.append(T_gt[:3, 3].copy())

        prev_kpts, prev_descs = kpts, descs

    if len(est_traj) < 3:
        return {
            'ate': {'ate_rmse': float('inf'), 'ate_mean': float('inf')},
            'rpe': {'rpe_r_mean': float('inf'), 'rpe_t_mean': float('inf')},
            'n_frames':        i + 1,
            'n_matched_pairs': n_matched,
            'n_failed':        0,
            'gt_traj':         np.array(gt_traj)  if gt_traj  else np.zeros((0,3)),
            'est_traj':        np.array(est_traj) if est_traj else np.zeros((0,3)),
        }

    # ATE / RPE
    ate = compute_ate(est_traj, gt_traj)
    rpe = compute_rpe(est_poses[1:], gt_poses[1:])   # ペア単位

    return {
        'ate':             ate,
        'rpe':             rpe,
        'n_frames':        len(gt_traj),
        'n_matched_pairs': n_matched,
        'n_failed':        n_failed,
        'gt_traj':         np.array(gt_traj),    # (N, 3) 可視化用
        'est_traj':        np.array(est_traj),   # (N, 3) 可視化用
    }


# ---------------------------------------------------------------------------
# シーケンスリストの取得
# ---------------------------------------------------------------------------

def get_sthereo_sequences(
    data_root:  str,
    split:      str = 'val',
    max_frames: int = 500,
) -> List[Tuple[str, List[Tuple[str, np.ndarray]], np.ndarray]]:
    """
    SThErEO のシーケンスリストを返す。

    Returns:
        [(seq_name, [(img_path, T_world), ...], K), ...]
    """
    from modules.dataset.thermal.sequential import (
        SThErEOSequentialDataset,
        _load_sthereo_K,
        _load_sthereo_poses,
        _nearest_pose_idx,
    )

    VAL_SEQS = {'snu_afternoon', 'kaist_morning', 'valley_afternoon'}
    result = []

    for seq_name in sorted(os.listdir(data_root)):
        seq_dir = os.path.join(data_root, seq_name)
        if not os.path.isdir(seq_dir):
            continue
        is_val = seq_name in VAL_SEQS
        if split == 'val'   and not is_val: continue
        if split == 'train' and is_val:     continue

        K     = _load_sthereo_K(os.path.join(
            seq_dir, 'calibration', 'thermal_14bit_left.yaml'))
        poses = _load_sthereo_poses(os.path.join(
            seq_dir, 'pose', 'global_pose.csv'))
        if not poses:
            continue
        pose_ts = [p[0] for p in poses]
        pose_Ts = [p[1] for p in poses]

        img_dir = os.path.join(seq_dir, 'image', 'thermal8_left_clahe')
        if not os.path.isdir(img_dir):
            img_dir = os.path.join(seq_dir, 'image', 'thermal8_left')
        if not os.path.isdir(img_dir):
            continue

        img_files = sorted(f for f in os.listdir(img_dir) if f.endswith('.png'))

        # サンプリング戦略:
        #   stride=vo_stride (デフォルト=3) で間引きして連続フレームとして使用
        #   1秒間隔の均一サンプリングではなくフレーム間運動量が小さい連続フレームを使う
        #   stride=3 → 10Hz/3 ≈ 3.3Hz、フレーム間 0.3秒、移動量 ~1m
        vo_stride = 3   # フレーム間隔（10Hzカメラで3コマとばし = ~3Hz）
        sampled   = img_files[::vo_stride][:max_frames]
        total     = len(img_files)

        seq_list = []
        for fname in sampled:
            try:
                ts_ns = int(fname.split('.')[0])
            except ValueError:
                continue
            idx = _nearest_pose_idx(ts_ns, pose_ts)
            if abs(pose_ts[idx] - ts_ns) < 250_000_000:
                seq_list.append((
                    os.path.join(img_dir, fname),
                    pose_Ts[idx],
                ))

        if len(seq_list) >= 10:
            result.append((seq_name, seq_list, K))
            print(f"  SThErEO/{seq_name}: {len(seq_list)} frames "
                  f"(stride={vo_stride}, total={total})")

    return result


def get_vivid_sequences(
    data_root:  str,
    split:      str = 'val',
    max_frames: int = 500,
) -> List[Tuple[str, List[Tuple[str, np.ndarray]], np.ndarray]]:
    """
    VIVID のシーケンスリストを返す（extracted_data から）。
    """
    from modules.dataset.thermal.sequential import (
        VividSequentialDataset,
        _VIVID_K_DEFAULT,
    )

    extracted = os.path.join(data_root, 'extracted_data')
    if not os.path.isdir(extracted):
        print(f"[VIVID] extracted_data not found: {extracted}")
        return []

    VAL_PATTERN = 'campus'
    result = []

    for seq_name in sorted(os.listdir(extracted)):
        seq_dir = os.path.join(extracted, seq_name)
        if not os.path.isdir(seq_dir):
            continue
        is_val = VAL_PATTERN in seq_name.lower()
        if split == 'val'   and not is_val: continue
        if split == 'train' and is_val:     continue

        # 熱画像ディレクトリを探す
        thr_dir = None
        for cand in ['thermal', 'thermal8', 'thermal8_clahe']:
            d = os.path.join(seq_dir, cand)
            if os.path.isdir(d) and any(f.endswith('.png') for f in os.listdir(d)):
                thr_dir = d
                break
        if thr_dir is None:
            continue

        img_files = sorted(f for f in os.listdir(thr_dir) if f.endswith('.png'))

        # GT ポーズ
        from modules.dataset.thermal.sequential import VividSequentialDataset as VSD
        poses_t, K = VSD._get_gt(data_root, seq_name)
        if not poses_t:
            continue

        p_times = [t for t, _ in poses_t]
        p_Ts    = [T for _, T in poses_t]

        seq_list = []
        for fname in img_files[:max_frames]:
            first = fname.split('.')[0]
            if len(first) >= 15 and first.isdigit():
                ts_sec = int(first) / 1e9
            elif first.replace('.', '', 1).isdigit():
                ts_sec = float(first)
            else:
                continue
            diffs = [abs(ts_sec - pt) for pt in p_times]
            idx   = int(np.argmin(diffs))
            if diffs[idx] < 0.1:
                seq_list.append((os.path.join(thr_dir, fname), p_Ts[idx]))

        if len(seq_list) >= 10:
            result.append((seq_name, seq_list, K))
            print(f"  VIVID/{seq_name}: {len(seq_list)} frames")

    return result


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Thermal VO Evaluation')
    p.add_argument('--config', default='configs/vo_config.yaml')
    p.add_argument('--dataset', nargs='+', default=None,
                   choices=['sthereo', 'vivid'],
                   help='評価するデータセット（デフォルト: config から）')
    p.add_argument('--n_seqs',   type=int, default=None,
                   help='評価するシーケンス数の上限')
    p.add_argument('--n_frames', type=int, default=None,
                   help='シーケンスあたりのフレーム数の上限')
    p.add_argument('--device_num', type=str, default=None)
    p.add_argument('--output_dir', type=str, default=None)
    p.add_argument('--no_vis', action='store_true')
    return p


def main():
    args   = build_parser().parse_args()
    cfg    = load_config(args.config)

    device_num = args.device_num or str(cfg.get('device_num', '0'))
    os.environ['CUDA_VISIBLE_DEVICES'] = device_num
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[VO] Device: {device}")

    output_dir = args.output_dir or cfg.get('output_dir', 'evaluate/vo_results')
    os.makedirs(output_dir, exist_ok=True)

    datasets   = args.dataset or cfg.get('eval_dataset', ['sthereo', 'vivid'])
    n_seqs     = args.n_seqs   or cfg.get('n_seqs',   None)
    n_frames   = args.n_frames or cfg.get('n_frames', 500)
    split      = cfg.get('split', 'val')

    # ── モデルのロード ──────────────────────────────────────────────────────
    models_cfg = cfg.get('models', [])
    print(f"\n[VO] Loading {len(models_cfg)} models ...")

    models = {}
    for m in models_cfg:
        print(f"  {m['name']} ...")
        models[m['name']] = load_model(m.get('weights'), device)

    # LightGlue（オプション）
    lg_model = None
    if cfg.get('matching_method', 'mnn') == 'lightglue':
        from eval.eval_matching import load_lightglue
        lg_model = load_lightglue(cfg.get('lightglue_weights'), device)

    # ── 評価ループ ─────────────────────────────────────────────────────────
    all_results = {}

    for ds_name in datasets:
        print(f"\n{'='*60}")
        print(f"  Dataset: {ds_name}")
        print(f"{'='*60}")

        data_root = cfg.get('data_roots', {}).get(ds_name)
        if not data_root or not os.path.isdir(data_root):
            print(f"  [Skip] data_root not found: {data_root}")
            continue

        # シーケンスリストを取得
        if ds_name == 'sthereo':
            seqs = get_sthereo_sequences(data_root, split, n_frames)
        elif ds_name == 'vivid':
            seqs = get_vivid_sequences(data_root, split, n_frames)
        else:
            continue

        if not seqs:
            print(f"  [Skip] no sequences found")
            continue

        if n_seqs:
            seqs = seqs[:n_seqs]

        all_results[ds_name] = {}

        for m_name, model in models.items():
            print(f"\n  -- {m_name} --")
            seq_results = []

            for seq_name, img_pose_list, K in seqs:
                print(f"    Seq: {seq_name} ({len(img_pose_list)} frames) ...",
                      end='', flush=True)
                t0 = time.perf_counter()

                res = run_vo_sequence(
                    img_pose_list, K, model, device, cfg, lg_model)

                elapsed = time.perf_counter() - t0
                ate = res['ate']
                rpe = res['rpe']
                print(f" ATE={ate.get('ate_rmse', float('inf')):.3f}m  "
                      f"RPE_r={rpe.get('rpe_r_mean', float('inf')):.2f}°  "
                      f"({elapsed:.1f}s)")
                seq_results.append({'seq': seq_name, **res})

            # シーケンス平均
            valid = [r for r in seq_results
                     if np.isfinite(r['ate'].get('ate_rmse', float('inf')))]
            if valid:
                avg_ate  = np.mean([r['ate']['ate_rmse'] for r in valid])
                avg_rpe_r = np.mean([r['rpe']['rpe_r_mean'] for r in valid])
                avg_rpe_t = np.mean([r['rpe']['rpe_t_mean'] for r in valid])
                print(f"\n  [{ds_name}] {m_name}: "
                      f"ATE={avg_ate:.3f}m  "
                      f"RPE_r={avg_rpe_r:.2f}°  "
                      f"RPE_t={avg_rpe_t:.2f}°  "
                      f"({len(valid)}/{len(seq_results)} seqs)")
            else:
                print(f"\n  [{ds_name}] {m_name}: no valid sequences")

            all_results[ds_name][m_name] = {
                'sequences':  seq_results,
                'avg_ate_rmse': float(np.mean([r['ate']['ate_rmse'] for r in valid]))
                                if valid else float('inf'),
                'avg_rpe_r_mean': float(np.mean([r['rpe']['rpe_r_mean'] for r in valid]))
                                  if valid else float('inf'),
                'avg_rpe_t_mean': float(np.mean([r['rpe']['rpe_t_mean'] for r in valid]))
                                  if valid else float('inf'),
                'n_valid_seqs': len(valid),
            }

    # ── 結果の保存（JSON）────────────────────────────────────────────────
    # ndarray は JSON 非対応のため軌跡データを除いて保存
    def _strip_arrays(obj):
        if isinstance(obj, dict):
            return {k: _strip_arrays(v) for k, v in obj.items()
                    if k not in ('gt_traj', 'est_traj')}
        if isinstance(obj, list):
            return [_strip_arrays(x) for x in obj]
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    out_path = os.path.join(output_dir, 'vo_results.json')
    with open(out_path, 'w') as f:
        json.dump(_strip_arrays(all_results), f, indent=2, default=str)
    print(f"\n[VO] Results saved: {out_path}")

    # ── 軌跡の可視化 ─────────────────────────────────────────────────────
    # all_results[ds][m]['sequences'] から gt_traj / est_traj を取り出して描画
    if not args.no_vis:
        try:
            import importlib.util as _ilu
            _vv_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'eval', 'vo_visualize.py')
            if not os.path.isfile(_vv_path):
                raise FileNotFoundError(f"vo_visualize.py not found: {_vv_path}")
            _spec = _ilu.spec_from_file_location('vo_visualize', _vv_path)
            _vv   = _ilu.module_from_spec(_spec)
            _spec.loader.exec_module(_vv)
            plot_trajectory_comparison = _vv.plot_trajectory_comparison
            plot_summary_bar           = _vv.plot_summary_bar
            print("\n[Vis] vo_visualize loaded OK")
        except Exception as e:
            print(f"[Vis] ERROR loading vo_visualize: {e}")
            plot_trajectory_comparison = None
            plot_summary_bar = None

    if not args.no_vis and plot_trajectory_comparison is not None:
        vis_dir = os.path.join(output_dir, 'trajectories')
        os.makedirs(vis_dir, exist_ok=True)
        print(f"[Vis] output dir: {vis_dir}")

        for ds_name, ds_res in all_results.items():
            model_names = list(ds_res.keys())
            if not model_names:
                continue

            # 全シーケンス名を収集
            first_model = ds_res[model_names[0]]
            seq_list = first_model.get('sequences', [])
            print(f"[Vis] {ds_name}: {len(seq_list)} sequences, "
                  f"models={model_names}")

            seq_names = []
            for s in seq_list:
                if s.get('seq') not in seq_names:
                    seq_names.append(s.get('seq'))

            for seq_name in seq_names:
                gt_traj_arr = None
                est_trajs   = {}
                ate_dict    = {}

                for m_name in model_names:
                    for s in ds_res[m_name].get('sequences', []):
                        if s.get('seq') != seq_name:
                            continue
                        gt  = s.get('gt_traj')
                        est = s.get('est_traj')
                        print(f"[Vis]   {m_name}/{seq_name}: "
                              f"gt={type(gt).__name__}, est={type(est).__name__}")
                        if gt is None or est is None:
                            print(f"[Vis]   WARNING: gt_traj or est_traj is None!")
                            continue
                        gt_arr  = np.array(gt)
                        est_arr = np.array(est)
                        if len(gt_arr) < 3:
                            print(f"[Vis]   WARNING: too short ({len(gt_arr)} frames)")
                            continue
                        gt_traj_arr       = gt_arr
                        est_trajs[m_name] = est_arr
                        ate_dict[m_name]  = s['ate'].get('ate_rmse', float('inf'))

                if gt_traj_arr is None or not est_trajs:
                    print(f"[Vis]   SKIP {seq_name}: no valid trajectories")
                    continue

                try:
                    saved = plot_trajectory_comparison(
                        seq_name     = seq_name,
                        gt_traj      = gt_traj_arr,
                        est_trajs    = est_trajs,
                        ate_dict     = ate_dict,
                        output_dir   = os.path.join(vis_dir, ds_name),
                        dataset_name = ds_name,
                    )
                    if saved:
                        print(f"[Vis] Saved: {saved}")
                    else:
                        print(f"[Vis] WARNING: plot_trajectory_comparison returned empty")
                except Exception as e:
                    print(f"[Vis] ERROR plotting {seq_name}: {e}")
                    import traceback; traceback.print_exc()

        # ATE サマリー棒グラフ
        if plot_summary_bar is not None:
            try:
                summary = {
                    ds: {m: ds_res[m].get('avg_ate_rmse', float('inf'))
                         for m in ds_res}
                    for ds, ds_res in all_results.items()
                }
                bar_path = plot_summary_bar(summary, output_dir)
                if bar_path:
                    print(f"[Vis] Summary bar: {bar_path}")
            except Exception as e:
                print(f"[Vis] ERROR summary bar: {e}")

    # ── サマリー表示 ───────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  VO Results Summary")
    print(f"{'='*60}")
    for ds, ds_res in all_results.items():
        for m_name, res in ds_res.items():
            print(f"  [{ds}] {m_name:<35s} "
                  f"ATE={res.get('avg_ate_rmse', float('inf')):.3f}m  "
                  f"RPE_r={res.get('avg_rpe_r_mean', float('inf')):.2f}°  "
                  f"RPE_t={res.get('avg_rpe_t_mean', float('inf')):.2f}°  "
                  f"n={res.get('n_valid_seqs', 0)}")


if __name__ == '__main__':
    main()