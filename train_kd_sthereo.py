"""
train_kd_sthereo.py
SThErEO + VIVID で KD 学習を実行するスクリプト。

設計方針（実験結果に基づく）:
  - kd_only が最良（rep/geo 損失は -22.4pt の劣化を引き起こす）
  - 評価は SThErEO と VIVID を分けて報告（データセット特性が異なるため）
  - best.pth の選択基準: SThErEO PoseAUC@5（主結果）

使用方法:
    python train_kd_sthereo.py \
        --sthereo_root datasets/sthereo \
        --vivid_root   datasets/vivid \
        --output       checkpoints/pipeline/stage1_kd \
        --epochs       100 \
        --batch_size   16 \
        --weights_init checkpoints/post_kd/default/post_kd_s2_final.pth
"""

from __future__ import annotations

import argparse, os, sys
from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from modules.training.losses_kd import spatial_entropy_loss, thermal_gradient_loss

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# wandb ユーティリティ
# ---------------------------------------------------------------------------

def init_wandb(args: Any) -> bool:
    """wandb を初期化する。失敗時は False を返してログなしで継続。"""
    if getattr(args, 'no_wandb', False):
        print("[wandb] disabled (--no_wandb)")
        return False
    try:
        import wandb
        wandb.init(
            project = getattr(args, 'wandb_project', 'thermal-xfeat-kd'),
            name    = getattr(args, 'wandb_run_name', None),
            group   = getattr(args, 'wandb_group',   'pipeline'),
            tags    = getattr(args, 'wandb_tags',    []),
            config  = {k: v for k, v in vars(args).items()
                       if isinstance(v, (int, float, str, bool, type(None)))},
            resume  = 'allow',
        )
        print(f"[wandb] {wandb.run.url}")
        return True
    except Exception as e:
        print(f"[wandb] init failed ({e}) → CSV ログのみ")
        return False


def wandb_log(metrics: dict, step: int, use_wandb: bool) -> None:
    """wandb と標準出力の両方にログを送る。"""
    if use_wandb:
        try:
            import wandb
            wandb.log(metrics, step=step)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# CLI 引数
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser()
    # データセット
    p.add_argument('--sthereo_root',      default='datasets/sthereo')
    p.add_argument('--vivid_root',        default=None,
                   help='VIVID ルート（None の場合は SThErEO のみ）')
    p.add_argument('--ms2_root',          default=None,
                   help='MS2 ルート（None の場合は MS2 を使用しない）')
    p.add_argument('--ms2_stride',        type=int, default=3)
    p.add_argument('--split',             default='all',
                   choices=['train', 'val', 'all'])
    p.add_argument('--max_pairs_per_seq', type=int, default=2000)
    p.add_argument('--stride',            type=int, default=3,
                   help='SThErEO フレーム間隔')
    p.add_argument('--vivid_stride',      type=int, default=2,
                   help='VIVID フレーム間隔（stride=5 は移動が大きすぎる）')
    
    # 初期評価の有無
    p.add_argument('--skip_initial_eval', action='store_true', default=False)

    # データ拡張
    p.add_argument('--p_flip',        type=float, default=0.5)
    p.add_argument('--p_brightness',  type=float, default=0.5)
    p.add_argument('--p_contrast',    type=float, default=0.5)
    p.add_argument('--p_fpn',         type=float, default=0.4)
    p.add_argument('--p_vignetting',  type=float, default=0.3)
    p.add_argument('--p_motion_blur', type=float, default=0.2)
    p.add_argument('--p_gaussian',    type=float, default=0.3)
    p.add_argument('--p_rain',        type=float, default=0.15)
    p.add_argument('--p_clahe_rand',  type=float, default=0.4)

    # 評価設定
    p.add_argument('--n_eval_pairs',      type=int, default=200,
                   help='SThErEO/VIVID それぞれの評価ペア数')
    p.add_argument('--eval_interval',     type=int, default=10)

    # 学習設定
    p.add_argument('--output',            default='checkpoints/kd_sthereo')
    p.add_argument('--epochs',            type=int, default=100)
    p.add_argument('--batch_size',        type=int, default=16)
    p.add_argument('--lr',               type=float, default=1e-4)
    p.add_argument('--weights_init',      default=None)
    p.add_argument('--device',            default='0')
    p.add_argument('--seed',              type=int, default=42)

    # 損失関数
    p.add_argument('--lambda_spatial', type=float, default=0.0)
    p.add_argument('--lambda_thermal', type=float, default=0.0)

    # best.pth の選択基準
    p.add_argument('--best_metric',       default='sthereo_PoseAUC@5',
                   help='best.pth を保存する指標（sthereo_PoseAUC@5 / vivid_PoseAUC@5 / avg_PoseAUC@5）')
    # wandb
    p.add_argument('--no_wandb',          action='store_true', default=False)
    p.add_argument('--wandb_project',     default='thermal-xfeat-kd')
    p.add_argument('--wandb_run_name',    default=None)
    p.add_argument('--wandb_group',       default='pipeline')
    p.add_argument('--wandb_tags',        nargs='+', default=[])
    return p.parse_args()


# ---------------------------------------------------------------------------
# データ収集
# ---------------------------------------------------------------------------

def make_sthereo_pairs(sthereo_root: str, stride: int,
                       max_per_seq: int, split: str) -> List[Tuple]:
    """SThErEO の全シーケンスからペアを収集する。"""
    from modules.dataset.thermal.sequential import (
        _load_sthereo_K, _load_sthereo_poses, _nearest_pose_idx)

    VAL_SEQS = {'snu_afternoon', 'kaist_morning', 'valley_afternoon'}
    all_pairs = []

    for seq_name in sorted(os.listdir(sthereo_root)):
        seq_dir = os.path.join(sthereo_root, seq_name)
        if not os.path.isdir(seq_dir):
            continue
        is_val = seq_name in VAL_SEQS
        if split == 'train' and is_val:     continue
        if split == 'val'   and not is_val: continue

        K     = _load_sthereo_K(
            os.path.join(seq_dir, 'calibration', 'thermal_14bit_left.yaml'))
        poses = _load_sthereo_poses(
            os.path.join(seq_dir, 'pose', 'global_pose.csv'))
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
        matched = []
        for fname in img_files:
            try:
                ts_ns = int(fname.split('.')[0])
            except ValueError:
                continue
            idx = _nearest_pose_idx(ts_ns, pose_ts)
            if abs(pose_ts[idx] - ts_ns) < 250_000_000:
                matched.append((os.path.join(img_dir, fname), pose_Ts[idx]))

        n_added = 0
        for i in range(0, len(matched) - stride, stride):
            j = i + stride
            p_t, T_t   = matched[i]
            p_t1, T_t1 = matched[j]
            T_rel = np.linalg.inv(T_t) @ T_t1
            all_pairs.append((p_t, p_t1, T_rel, K))
            n_added += 1
            if n_added >= max_per_seq:
                break

        print(f"  [SThErEO] {seq_name}: {n_added} pairs")

    return all_pairs


def make_ms2_pairs(ms2_root: str, stride: int,
                   max_per_seq: int, split: str) -> List[Tuple]:
    """MS2 の連続フレームペアを収集する（GT pose 付き）。"""
    try:
        from modules.dataset.thermal.sequential import MS2SequentialDataset
        ds = MS2SequentialDataset(
            data_root=ms2_root, stride=stride, split=split,
            max_pairs_per_seq=max_per_seq, apply_clahe=True,
            clahe_clip_range=(1.5, 3.0),
        )
        print(f"  [MS2 all] {len(ds._pairs)} pairs (stride={stride})")
        return list(ds._pairs)
    except Exception as e:
        print(f"  [MS2] ロード失敗: {e}")
        return []


def make_ms2_val_pairs(ms2_root: str, stride: int,
                       max_per_seq: int) -> List[Tuple]:
    """MS2 val ペアを収集する（GPS/IMU GT・雨天夜間評価用）。"""
    try:
        from modules.dataset.thermal.sequential import MS2SequentialDataset
        ds = MS2SequentialDataset(
            data_root=ms2_root, stride=stride, split='val',
            max_pairs_per_seq=max_per_seq, apply_clahe=True,
            clahe_clip_range=(2.0, 2.0),   # val は固定 clipLimit
        )
        print(f"  [MS2 val] {len(ds._pairs)} pairs")
        return list(ds._pairs)
    except Exception as e:
        print(f"  [MS2 val] ロード失敗: {e}")
        return []


def make_vivid_pairs(vivid_root: str, stride: int,
                     max_per_seq: int, split: str) -> List[Tuple]:
    """VIVID のシーケンスからペアを収集する。"""
    try:
        from modules.dataset.thermal.sequential import VividSequentialDataset
        ds = VividSequentialDataset(
            data_root=vivid_root,
            stride=stride,
            split=split,
            max_pairs_per_seq=max_per_seq,
        )
        print(f"  [VIVID] total: {len(ds._pairs)} pairs (stride={stride})")
        return list(ds._pairs)
    except Exception as e:
        print(f"  [VIVID] ロード失敗: {e}")
        return []


# ---------------------------------------------------------------------------
# 画像ロード
# ---------------------------------------------------------------------------

def load_img(path: str, size=(640, 480)) -> Optional[torch.Tensor]:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, size)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return torch.from_numpy(img).permute(2, 0, 1).float().div(255.0).unsqueeze(0)


# ---------------------------------------------------------------------------
# KD 損失
# ---------------------------------------------------------------------------

def kd_loss(student_out, teacher_out) -> torch.Tensor:
    def _feat(out):
        if isinstance(out, (tuple, list)): return out[0]
        if isinstance(out, dict):
            k = 'feats' if 'feats' in out else list(out.keys())[0]
            return out[k]
        return out
    s = F.normalize(_feat(student_out), dim=1)
    t = F.normalize(_feat(teacher_out), dim=1)
    return F.mse_loss(s, t.detach())


# ---------------------------------------------------------------------------
# 評価関数（SThErEO と VIVID で共通・データセット名を返す）
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_dataset(
    model:   torch.nn.Module,
    pairs:   List[Tuple],
    device:  torch.device,
    n_pairs: int,
    ds_name: str,
) -> Dict[str, float]:
    """
    1 データセットの評価を実行し、指標辞書を返す。
    キーは ds_name プレフィックス付き（例: sthereo_PoseAUC@5）。
    """
    from eval.eval_matching import (
        detect, match, imread_tensor, _compute_F_gt, _sym_epi_dist)

    size   = (640, 480)
    max_kp = 1024
    ms_list, prec_list, n_match_list, pose_errs = [], [], [], []

    eval_pairs = pairs[:n_pairs]
    if not eval_pairs:
        return {f'{ds_name}_MS': 0.0, f'{ds_name}_Prec@3px': 0.0,
                f'{ds_name}_n_match': 0.0, f'{ds_name}_PoseAUC@5': 0.0,
                f'{ds_name}_PoseAUC@10': 0.0}

    model.eval()
    for path_t, path_t1, T_rel, K in eval_pairs:
        try:
            img_t,  _ = imread_tensor(path_t,  True, device, size)
            img_t1, _ = imread_tensor(path_t1, True, device, size)
        except Exception:
            continue

        kpts1, descs1 = detect(model, img_t,  max_kp)
        kpts2, descs2 = detect(model, img_t1, max_kp)
        if len(kpts1) == 0 or len(kpts2) == 0:
            continue

        idx1, idx2 = match(descs1, descs2, 'mutual_nn', ratio_thr=0.9)
        n_m = len(idx1)
        ms  = n_m / max(min(len(kpts1), len(kpts2)), 1)
        ms_list.append(ms)
        n_match_list.append(n_m)

        # Precision@3px
        if n_m > 0:
            try:
                F_gt = _compute_F_gt(T_rel, K).astype(np.float32)
                epi  = _sym_epi_dist(
                    kpts1[idx1].astype(np.float32),
                    kpts2[idx2].astype(np.float32), F_gt)
                prec_list.append(float((epi < 3.0).mean()))
            except Exception:
                pass

        # PoseAUC
        if n_m >= 8:
            try:
                K_np  = np.array(K, dtype=np.float64)
                pts1  = kpts1[idx1].astype(np.float32).reshape(-1, 1, 2)
                pts2  = kpts2[idx2].astype(np.float32).reshape(-1, 1, 2)
                E, msk = cv2.findEssentialMat(
                    pts1, pts2, K_np, method=cv2.RANSAC,
                    prob=0.999, threshold=1.0)
                if E is None or msk is None or int(msk.sum()) < 5:
                    continue
                if E.shape[0] > 3:
                    E = E[:3]
                _, R_est, t_est, _ = cv2.recoverPose(
                    E, pts1, pts2, K_np, mask=msk)
                T_np = np.array(T_rel, dtype=np.float64)
                R_gt, t_gt = T_np[:3, :3], T_np[:3, 3]
                R_rel = R_est @ R_gt.T
                trace = float(np.clip((np.trace(R_rel)-1)/2, -1, 1))
                R_err = float(np.degrees(np.arccos(trace)))
                t_gt_n = np.linalg.norm(t_gt)
                if t_gt_n < 1e-4:
                    continue
                t_e = t_est.ravel() / (np.linalg.norm(t_est) + 1e-8)
                t_g = t_gt / t_gt_n
                t_err = float(np.degrees(
                    np.arccos(abs(float(np.clip(np.dot(t_e, t_g), -1, 1))))))
                pose_errs.append(max(R_err, t_err))
            except Exception:
                pass

    pose_arr = np.array(pose_errs) if pose_errs else np.array([180.0])
    p = ds_name  # プレフィックス
    return {
        f'{p}_MS':         float(np.mean(ms_list))     if ms_list     else 0.0,
        f'{p}_Prec@3px':   float(np.mean(prec_list))   if prec_list   else 0.0,
        f'{p}_n_match':    float(np.mean(n_match_list)) if n_match_list else 0.0,
        f'{p}_PoseAUC@5':  float((pose_arr < 5).mean()),
        f'{p}_PoseAUC@10': float((pose_arr < 10).mean()),
    }


def print_metrics(tag: str, m: Dict[str, float], ds_name: str) -> None:
    """評価結果を整形して表示する。"""
    p = ds_name
    print(f"  [{ds_name}] {tag}")
    print(f"    MS={m[f'{p}_MS']*100:.1f}%  "
          f"Prec@3px={m[f'{p}_Prec@3px']*100:.1f}%  "
          f"n_match={m[f'{p}_n_match']:.0f}")
    print(f"    PoseAUC@5={m[f'{p}_PoseAUC@5']*100:.1f}%  "
          f"PoseAUC@10={m[f'{p}_PoseAUC@10']*100:.1f}%")


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------

def main():
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output, exist_ok=True)

    log_path  = os.path.join(args.output, 'train_log.csv')
    use_wandb = init_wandb(args)

    print(f"\n{'='*60}")
    print(f"  KD Training")
    print(f"  SThErEO: {args.sthereo_root}")
    print(f"  VIVID:   {args.vivid_root or '（使用しない）'}")
    print(f"  epochs={args.epochs}  batch={args.batch_size}  lr={args.lr}")
    print(f"  device={device}  split={args.split}")
    print(f"  best_metric={args.best_metric}")
    print(f"{'='*60}\n")

    # ── データ収集 ─────────────────────────────────────────────────────
    print("[データ収集]")
    sthereo_pairs = make_sthereo_pairs(
        args.sthereo_root, args.stride, args.max_pairs_per_seq, args.split)

    vivid_pairs = []
    if args.vivid_root:
        vivid_pairs = make_vivid_pairs(
            args.vivid_root, args.vivid_stride, args.max_pairs_per_seq, args.split)

    ms2_all_pairs, ms2_val_pairs = [], []
    if getattr(args, 'ms2_root', None):
        ms2_all_pairs  = make_ms2_pairs(
            args.ms2_root, getattr(args, 'ms2_stride', 3),
            args.max_pairs_per_seq, 'all')
        ms2_val_pairs  = make_ms2_val_pairs(
            args.ms2_root, getattr(args, 'ms2_stride', 3),
            args.n_eval_pairs)

    # ── 評価用ペアを分離（SThErEO と VIVID それぞれ独立に分離）─────────
    np.random.shuffle(sthereo_pairs)
    n_eval_s  = min(args.n_eval_pairs, len(sthereo_pairs) // 5)
    sthereo_eval  = sthereo_pairs[:n_eval_s]
    sthereo_train = sthereo_pairs[n_eval_s:]

    vivid_eval  = []
    vivid_train = []
    if vivid_pairs:
        np.random.shuffle(vivid_pairs)
        n_eval_v  = min(args.n_eval_pairs, len(vivid_pairs) // 5)
        vivid_eval  = vivid_pairs[:n_eval_v]
        vivid_train = vivid_pairs[n_eval_v:]

    # 学習ペアを統合してシャッフル
    train_pairs = sthereo_train + vivid_train + ms2_all_pairs
    np.random.shuffle(train_pairs)

    print(f"\n  [SThErEO] train={len(sthereo_train)} / eval={len(sthereo_eval)}")
    if vivid_pairs:
        print(f"  [VIVID]   train={len(vivid_train)} / eval={len(vivid_eval)}")
    if ms2_all_pairs:
        print(f"  [MS2]     train={len(ms2_all_pairs)} / val_eval={len(ms2_val_pairs)}")
    print(f"  [合計]    train={len(train_pairs)}\n")

    # ── モデル ─────────────────────────────────────────────────────────
    from modules.model import XFeatModel
    teacher = XFeatModel().to(device).eval()
    for param in teacher.parameters():
        param.requires_grad_(False)

    student = XFeatModel().to(device).train()
    if args.weights_init:
        # パスが渡されている場合は常に表示する
        if os.path.isfile(args.weights_init):
            state = torch.load(args.weights_init, map_location=device, weights_only=True)
            student.load_state_dict(state)
            print(f"[Student] Loaded weights from: {args.weights_init}")
        else:
            # ファイルが見つからない場合は警告を出す
            print(f"[WARNING] Weight file not found: {args.weights_init}")
            print("[Student] Falling back to default initial weights")
    else:
        print("[Student] No weights specified: using default initial weights")

    print("[Student] モデルをコンパイルしています (torch.compile)...")
    # student = torch.compile(student)

    optimizer = torch.optim.Adam(student.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    # 初期評価変数の初期化
    init_s, init_v = None, None

    # ── ベースライン評価 ───────────────────────────────────────────────
    baseline = XFeatModel().to(device).eval()
    print("\n[Baseline 評価]")
    base_s = evaluate_dataset(baseline, sthereo_eval, device,
                               args.n_eval_pairs, 'sthereo')
    print_metrics("RGB XFeat", base_s, 'sthereo')
    if vivid_eval:
        base_v = evaluate_dataset(baseline, vivid_eval, device,
                                   args.n_eval_pairs, 'vivid')
        print_metrics("RGB XFeat", base_v, 'vivid')

    if not args.skip_initial_eval:
        print("\n[Student 初期評価]")
        init_s = evaluate_dataset(student, sthereo_eval, device,
                               args.n_eval_pairs, 'sthereo')
        print_metrics("proposed(init)", init_s, 'sthereo')
        if vivid_eval:
            init_v = evaluate_dataset(student, vivid_eval, device,
                                    args.n_eval_pairs, 'vivid')
            print_metrics("proposed(init)", init_v, 'vivid')
    else:
        print("\n[Student 初期評価] skipped")

    # ── CSV ログ ───────────────────────────────────────────────────────
    header = ("epoch,loss,"
              "sthereo_MS,sthereo_Prec@3px,sthereo_n_match,"
              "sthereo_PoseAUC@5,sthereo_PoseAUC@10")
    if vivid_eval:
        header += (",vivid_MS,vivid_Prec@3px,vivid_n_match,"
                   "vivid_PoseAUC@5,vivid_PoseAUC@10")

    def fmt_row(epoch, loss, ms, mv=None):
        row = (f"{epoch},{loss},"
               f"{ms['sthereo_MS']*100:.1f},"
               f"{ms['sthereo_Prec@3px']*100:.1f},"
               f"{ms['sthereo_n_match']:.0f},"
               f"{ms['sthereo_PoseAUC@5']*100:.1f},"
               f"{ms['sthereo_PoseAUC@10']*100:.1f}")
        if mv:
            row += (f",{mv['vivid_MS']*100:.1f},"
                    f"{mv['vivid_Prec@3px']*100:.1f},"
                    f"{mv['vivid_n_match']:.0f},"
                    f"{mv['vivid_PoseAUC@5']*100:.1f},"
                    f"{mv['vivid_PoseAUC@10']*100:.1f}")
        return row

    with open(log_path, 'w') as f:
        f.write(header + '\n')
        f.write(fmt_row('baseline', '-', base_s,
                        base_v if vivid_eval else None) + '\n')
        if init_s is not None:
            f.write(fmt_row('init', '-', init_s,
                            init_v if vivid_eval else None) + '\n')

    # ── DataLoader の構築 ──────────────────────────────────────────────
    # 1 epoch = 全学習ペアを1周する（通常の定義）
    # steps_per_epoch = len(train_pairs) // batch_size

    from torch.utils.data import Dataset as TorchDataset, DataLoader
    from modules.training.thermal_augmentation import ThermalAugmentation

    class PairDataset(TorchDataset):
        """学習ペアを DataLoader で扱うための Dataset ラッパー。"""
        def __init__(self, pairs):
            self.pairs = pairs
        def __len__(self):
            return len(self.pairs)
        def __getitem__(self, idx):
            path_t, path_t1, *_ = self.pairs[idx]
            return path_t  # 画像パスのみ返す（load_img でロード）

    custom_augmentation = ThermalAugmentation(
        p_flip        = getattr(args, 'p_flip', 0.5),
        p_brightness  = getattr(args, 'p_brightness', 0.5),
        p_contrast    = getattr(args, 'p_contrast', 0.5),
        p_fpn         = getattr(args, 'p_fpn', 0.4),
        p_vignetting  = getattr(args, 'p_vignetting', 0.3),
        p_motion_blur = getattr(args, 'p_motion_blur', 0.2),
        p_gaussian    = getattr(args, 'p_gaussian', 0.3),
        p_rain        = getattr(args, 'p_rain', 0.15),
        p_clahe_rand  = getattr(args, 'p_clahe_rand', 0.4)
    )

    def collate_fn(paths):
        """
        画像パスリスト → (clean_batch, aug_batch) のタプル。
        clean: teacher 用（拡張なし）
        aug:   student 用（データ拡張あり）
        """
        cleans, augs = [], []
        for p in paths:
            t = load_img(p, (640, 480))
            if t is None:
                continue
            cleans.append(t)
            
            # 2. 学生モデル用（aug）の画像生成
            # config で拡張が有効化されている場合のみカスタム設定を適用する
            if getattr(args, 'aug_enabled', True):
                # DEFAULT_AUGMENTATION ではなく、上記で生成したインスタンスを使用
                aug = custom_augmentation(t.squeeze(0)).unsqueeze(0)
            else:
                aug = t.clone()
            augs.append(aug)

        if not cleans:
            return None
        return torch.cat(cleans, dim=0), torch.cat(augs, dim=0)

    train_ds     = PairDataset(train_pairs)
    train_loader = DataLoader(
        train_ds,
        batch_size  = args.batch_size,
        shuffle     = True,
        num_workers = 2,
        collate_fn  = collate_fn,
        drop_last   = True,
    )

    steps_per_epoch = len(train_loader)
    total_steps     = steps_per_epoch * args.epochs
    print(f"  train: {len(train_pairs):,} ペア → "
          f"{steps_per_epoch:,} steps/epoch × {args.epochs} epochs "
          f"= {total_steps:,} steps")

    # ── 学習ループ（1 epoch = 全データ1周）─────────────────────────────
    size = (640, 480)
    best_val = 0.0
    global_step = 0

    epoch_losses = []
    epoch_kd_losses = []
    epoch_spatial_losses = []
    epoch_thermal_losses = []

    print("\n[学習開始] 最初のバッチの処理には数分かかる場合があります (torch.compile)...")
    for epoch in range(1, args.epochs + 1):
        student.train()
        # print(f"\n[DEBUG] Starting Epoch {epoch}")
        epoch_losses.clear()
        epoch_kd_losses.clear()
        epoch_spatial_losses.clear()
        epoch_thermal_losses.clear()

        for i, batch in enumerate(train_loader):
            # print(f"[DEBUG] Batch {i} loading...")
            if batch is None:
                # print(f"[DEBUG] Batch {i} is None, skipping...")
                continue

            imgs_clean, imgs_aug = batch
            imgs_clean = imgs_clean.to(device)
            imgs_aug   = imgs_aug.to(device)
            
            # print(f"[DEBUG] Model forward start...")
            with torch.no_grad():
                t_out = teacher(imgs_clean)  # 教師: clean（拡張なし）
            s_out = student(imgs_aug)        # 学生: 拡張画像
            # print(f"[DEBUG] Model forward end.")
            s_feats, s_kpts, s_hmap = s_out

            # 知識蒸留損失
            l_kd  = kd_loss(s_out, t_out)
            current_loss = l_kd

            # 空間分散損失
            l_spatial_val = 0.0
            if args.lambda_spatial > 0:
                l_spatial = torch.stack([
                spatial_entropy_loss(s_kpts[b].T, s_hmap[b].flatten(), (480, 640)) 
                for b in range(args.batch_size)
                ]).mean()
                current_loss = current_loss + args.lambda_spatial * l_spatial
                l_spatial_val = l_spatial.item()

            # 温度勾配誘導損失
            l_thermal_val = 0.0
            if args.lambda_thermal > 0:
                l_thermal = torch.stack([
                thermal_gradient_loss(s_kpts[b].T, s_hmap[b].flatten(), imgs_aug[b:b+1]) 
                for b in range(args.batch_size)
                ]).mean()
                current_loss = current_loss + args.lambda_thermal * l_thermal
                l_thermal_val = l_thermal.item()

            optimizer.zero_grad()
            current_loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()

            epoch_losses.append(current_loss.item())
            epoch_kd_losses.append(l_kd.item())
            epoch_spatial_losses.append(l_spatial_val)
            epoch_thermal_losses.append(l_thermal_val)
            global_step += 1
            
            if i % 10 == 0:
                print(f"  Step [{i:3d}/{steps_per_epoch}] "
                      f"Loss: {current_loss.item():.4f} "
                      f"(KD: {l_kd.item():.4f}, SP: {l_spatial_val:.4f})", 
                      end='\r', flush=True)

        scheduler.step()

        if epoch % args.eval_interval == 0 or epoch == args.epochs:
            avg_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0

            log_dict = {
                'epoch': epoch,
                'train/loss_total':   avg_loss,
                'train/loss_kd':      np.mean(epoch_kd_losses),
                'train/loss_spatial': np.mean(epoch_spatial_losses),
                'train/loss_thermal': np.mean(epoch_thermal_losses),
            }

            # SThErEO 評価
            m_s = evaluate_dataset(student, sthereo_eval, device,
                                    args.n_eval_pairs, 'sthereo')
            # VIVID 評価（独立）
            m_v = None
            if vivid_eval:
                m_v = evaluate_dataset(student, vivid_eval, device,
                                        args.n_eval_pairs, 'vivid')
            # MS2 val 評価（GPS/IMU GT・雨天夜間の堅牢性確認）
            m_ms2v = None
            if ms2_val_pairs:
                m_ms2v = evaluate_dataset(student, ms2_val_pairs, device,
                                           args.n_eval_pairs, 'ms2_val')

            print(f"\n  Epoch {epoch:4d}/{args.epochs}  loss={avg_loss:.5f}", flush=True)
            print_metrics("proposed", m_s, 'sthereo')
            if m_v:
                print_metrics("proposed", m_v, 'vivid')
            if m_ms2v:
                print_metrics("proposed", m_ms2v, 'ms2_val')

            # CSV 書き込み
            with open(log_path, 'a') as f:
                f.write(fmt_row(epoch, f'{avg_loss:.5f}',
                                m_s, m_v) + '\n')

            # wandb ログ
            # log_dict = {'epoch': epoch, 'train/loss': avg_loss}
            log_dict.update({f'sthereo/{k.replace("sthereo_","")}': v
                             for k, v in m_s.items()})
            if m_v:
                log_dict.update({f'vivid/{k.replace("vivid_","")}': v
                                 for k, v in m_v.items()})
            wandb_log(log_dict, step=global_step, use_wandb=use_wandb)

            # best.pth の選択（指定指標で）
            key = args.best_metric
            if key in m_s:
                val = m_s[key]
            elif m_v and key in m_v:
                val = m_v[key]
            elif key == 'avg_PoseAUC@5':
                vals = [m_s['sthereo_PoseAUC@5']]
                if m_v:
                    vals.append(m_v['vivid_PoseAUC@5'])
                val = float(np.mean(vals))
            else:
                val = m_s.get('sthereo_PoseAUC@5', 0.0)

            if val > best_val:
                best_val = val
                torch.save(student.state_dict(),
                           os.path.join(args.output, 'best.pth'))
                print(f"    → best.pth 更新  {key}={best_val*100:.1f}%")

            student.train()

    # 最終モデルを保存
    torch.save(student.state_dict(),
               os.path.join(args.output, 'final.pth'))

    print(f"\n{'='*60}")
    print(f"  学習完了")
    print(f"  log:   {log_path}")
    print(f"  best:  {args.output}/best.pth  ({args.best_metric}={best_val*100:.1f}%)")
    print(f"  final: {args.output}/final.pth")

    if use_wandb:
        try:
            import wandb
            wandb.finish()
        except Exception:
            pass


if __name__ == '__main__':
    main()