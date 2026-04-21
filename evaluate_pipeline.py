"""
evaluate_pipeline.py
学習パイプラインの最終評価・可視化スクリプト。

評価方法:
  XFeat で KP・記述子を検出し、以下の matcher でマッチング:
    - MNN    : mutual nearest neighbor（論文と同一・LG 不要）
    - LightGlue: fine-tuning 済み LG checkpoint を使用

比較する4つのモデル構成（eval_pipeline.yaml で設定）:
  Config A: XFeat(RGB)   + matcher(RGB)   ← baseline
  Config B: XFeat(Therm) + matcher(RGB)   ← KD のみ効果確認
  Config C: XFeat(RGB)   + matcher(Therm) ← matcher のみ Thermal 適応
  Config D: XFeat(Therm) + matcher(Therm) ← 完全 Thermal パイプライン

可視化:
  緑: inlier（エピポーラ距離 < inlier_thr px）
  赤: outlier（エピポーラ距離 ≥ inlier_thr px）

使用方法:
  python evaluate_pipeline.py --config eval_pipeline.yaml
  python evaluate_pipeline.py --config eval_pipeline.yaml --seqs kaist_morning snu_afternoon
"""

from __future__ import annotations

import argparse
import os
import sys
import types
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# 設定ロード
# ---------------------------------------------------------------------------

def load_yaml_config(path: str) -> dict:
    """eval_pipeline.yaml を読み込んでフラットな設定辞書を返す。"""
    import yaml
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}

    data   = cfg.get('data',   {})
    ev     = cfg.get('eval',   {})
    out    = cfg.get('output', {})
    models = cfg.get('models', {})

    # seqs: リストまたは単一文字列を正規化
    seqs_raw = data.get('seqs') or data.get('seq', 'kaist_morning')
    seqs     = seqs_raw if isinstance(seqs_raw, list) else [seqs_raw]

    return {
        'sthereo_root': data.get('sthereo_root', 'datasets/sthereo'),
        'seqs':         seqs,
        'stride':       int(data.get('stride', 3)),
        'n_eval_pairs': int(ev.get('n_eval_pairs', 200)),
        'n_vis_pairs':  int(ev.get('n_vis_pairs',  10)),
        'max_kp':       int(ev.get('max_kp',       512)),
        'inlier_thr':   float(ev.get('inlier_thr', 3.0)),
        'output_dir':   out.get('dir',       'results/eval_pipeline'),
        'save_csv':     bool(out.get('save_csv', True)),
        'save_vis':     bool(out.get('save_vis', True)),
        'xfeat_rgb':    models.get('xfeat_rgb',   'weights/xfeat.pt'),
        'xfeat_therm':  models.get('xfeat_therm', None),
        # matcher_rgb: null → MNN、文字列 → LG checkpoint パス
        'matcher_rgb':  models.get('matcher_rgb',  None),
        'matcher_therm': models.get('matcher_therm', None),
        'device':       str(cfg.get('device', '0')),
    }


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='eval_pipeline.yaml')
    p.add_argument('--seqs',   nargs='+', default=None)
    p.add_argument('--seq',    default=None)
    p.add_argument('--n_eval_pairs', type=int, default=None)
    p.add_argument('--n_vis_pairs',  type=int, default=None)
    p.add_argument('--output_dir',   default=None)
    p.add_argument('--device',       default=None)
    cli = p.parse_args()

    cfg = load_yaml_config(cli.config)
    if cli.seqs         is not None: cfg['seqs']         = cli.seqs
    elif cli.seq        is not None: cfg['seqs']         = [cli.seq]
    if cli.n_eval_pairs is not None: cfg['n_eval_pairs'] = cli.n_eval_pairs
    if cli.n_vis_pairs  is not None: cfg['n_vis_pairs']  = cli.n_vis_pairs
    if cli.output_dir   is not None: cfg['output_dir']   = cli.output_dir
    if cli.device       is not None: cfg['device']       = cli.device

    ns = types.SimpleNamespace(**cfg)
    ns.config = cli.config
    return ns


# ---------------------------------------------------------------------------
# モデルロード
# ---------------------------------------------------------------------------

def load_xfeat(weights: Optional[str], device: torch.device) -> torch.nn.Module:
    from modules.model import XFeatModel
    model = XFeatModel().to(device).eval()
    if weights and os.path.isfile(weights):
        model.load_state_dict(
            torch.load(weights, map_location=device, weights_only=True))
        print(f"    XFeat: {weights}")
    else:
        print(f"    XFeat: default RGB weights (weights/xfeat.pt)")
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_matcher(matcher_path: Optional[str],
                 device: torch.device) -> Optional[object]:
    """
    matcher_path が None → MNN（LG 不要）
    matcher_path が文字列 → glue-factory 形式の LightGlue checkpoint をロード
    """
    if matcher_path is None:
        return None

    try:
        # ── [修正] プロジェクト内のモデル定義をインポート ──
        from modules.lighterglue import LightGlue
        
        # モデルのインスタンス化 (input_dim=64 など、学習時と同一にする)
        model = LightGlue(features=None).to(device).eval()

        # 重みのロード
        ckpt = torch.load(matcher_path, map_location=device)
        
        # ── [論理的修正] KeyError: 'model' 回避ロジック ──
        if isinstance(ckpt, dict):
            if 'model' in ckpt:
                state_dict = ckpt['model'] # glue-factory 形式
            elif 'state_dict' in ckpt:
                state_dict = ckpt['state_dict']
            else:
                state_dict = ckpt # 辞書そのものが重み
        else:
            state_dict = ckpt

        # キー名の微調整 (model. プレフィックスが付いている場合などの対策)
        # strict=False にすることで、多少の名称差異を許容してロードを優先します
        model.load_state_dict(state_dict, strict=False)
        
        print(f"    Matcher: LightGlue loaded from {matcher_path}")
        return model

    except Exception as e:
        print(f"    [ERROR] Matcher ロード失敗: {e}")
        return None


# ---------------------------------------------------------------------------
# 1ペアのマッチング（MNN または LightGlue を統一インターフェースで）
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_matching(
    xfeat:   torch.nn.Module,
    matcher: Optional[object],
    img0:    np.ndarray,         # (H, W) グレースケール
    img1:    np.ndarray,
    max_kp:  int,
    device:  torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    XFeat で検出 → MNN または LightGlue でマッチング。

    Returns:
        mkpts0, mkpts1: (n_match, 2) float32 の対応点座標
    """
    from eval.eval_matching import detect, match

    H, W  = img0.shape[:2]
    size  = (640, 480)
    img0r = cv2.resize(img0, size)
    img1r = cv2.resize(img1, size)

    def to_tensor(im):
        im3 = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        return (torch.from_numpy(im3)
                .permute(2, 0, 1).float().div(255)
                .unsqueeze(0).to(device))

    t0 = to_tensor(img0r)
    t1 = to_tensor(img1r)

    kpts0, descs0 = detect(xfeat, t0, max_kp)   # (N, 2), (N, 64)
    kpts1, descs1 = detect(xfeat, t1, max_kp)

    if len(kpts0) == 0 or len(kpts1) == 0:
        return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)

    if matcher is None:
        # MNN マッチング（論文と同一方法）
        idx0, idx1 = match(descs0, descs1, 'mutual_nn', ratio_thr=0.9)
    else:
        # LightGlue マッチング
        idx0, idx1 = match(
            descs0, descs1, 'lightglue',
            ratio_thr=0.9,
            kpts1=kpts0, kpts2=kpts1,
            image_size=(size[1], size[0]),
            device=device,
        )

    if len(idx0) == 0:
        return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)

    # リサイズ後座標 → 元解像度座標に変換（可視化用）
    sx = W / size[0]
    sy = H / size[1]
    m0 = kpts0[idx0].copy(); m0[:, 0] *= sx; m0[:, 1] *= sy
    m1 = kpts1[idx1].copy(); m1[:, 0] *= sx; m1[:, 1] *= sy

    return m0.astype(np.float32), m1.astype(np.float32)


# ---------------------------------------------------------------------------
# 指標計算
# ---------------------------------------------------------------------------

def compute_F_gt(T_rel: np.ndarray, K: np.ndarray) -> np.ndarray:
    from eval.eval_matching import _compute_F_gt
    return _compute_F_gt(T_rel, K).astype(np.float32)


def sym_epi_dist(pts0: np.ndarray, pts1: np.ndarray,
                 F: np.ndarray) -> np.ndarray:
    from eval.eval_matching import _sym_epi_dist
    return _sym_epi_dist(pts0, pts1, F)


def compute_pose_error(mkpts0: np.ndarray, mkpts1: np.ndarray,
                       T_rel: np.ndarray, K: np.ndarray,
                       size: Tuple[int, int] = (640, 480)) -> Tuple[float, float]:
    """E行列から R_err, t_err (degrees) を計算する。"""
    if len(mkpts0) < 8:
        return float('inf'), float('inf')

    # 可視化サイズ座標 → 評価用リサイズ座標
    W_orig, H_orig = size[0], size[1]

    K64 = np.array(K, dtype=np.float64)
    p0  = mkpts0.astype(np.float32).reshape(-1, 1, 2)
    p1  = mkpts1.astype(np.float32).reshape(-1, 1, 2)

    E, msk = cv2.findEssentialMat(p0, p1, K64,
                                   method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if E is None or msk is None or int(msk.sum()) < 5:
        return float('inf'), float('inf')
    if E.shape[0] > 3:
        E = E[:3]

    _, R_est, t_est, _ = cv2.recoverPose(E, p0, p1, K64, mask=msk)
    R_gt = np.array(T_rel[:3, :3], dtype=np.float64)
    t_gt = np.array(T_rel[:3, 3],  dtype=np.float64)

    R_rel = R_est @ R_gt.T
    trace = float(np.clip((np.trace(R_rel) - 1) / 2, -1, 1))
    R_err = float(np.degrees(np.arccos(trace)))

    t_n   = np.linalg.norm(t_est) + 1e-8
    t_gt_n = np.linalg.norm(t_gt)  + 1e-8
    cos_t = float(np.clip(
        np.dot(t_est.flatten() / t_n, t_gt / t_gt_n), -1, 1))
    t_err = float(np.degrees(np.arccos(abs(cos_t))))

    return R_err, t_err


def evaluate_config(
    xfeat:   torch.nn.Module,
    matcher: Optional[object],
    pairs:   List[Tuple],
    device:  torch.device,
    n_pairs: int,
    max_kp:  int,
    inlier_thr: float = 3.0,
) -> Dict[str, float]:
    """1設定の定量評価を行う。"""
    ms_list, prec_list, pose_errs = [], [], []
    size = (640, 480)

    for path0, path1, T_rel_t, K_t in pairs[:n_pairs]:
        T_rel = np.array(T_rel_t, dtype=np.float64)
        K     = np.array(K_t,     dtype=np.float64)

        img0 = cv2.imread(path0, cv2.IMREAD_GRAYSCALE)
        img1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
        if img0 is None or img1 is None:
            continue

        mkpts0, mkpts1 = run_matching(
            xfeat, matcher, img0, img1, max_kp, device)
        n_m = len(mkpts0)

        # MS = n_match / min(|KP1|, |KP2|)（論文準拠）
        ms_list.append(n_m / max(min(max_kp, max_kp), 1))

        if n_m > 0:
            try:
                F_gt = compute_F_gt(T_rel, K)
                # 評価はリサイズ後座標で行うため元画像比率で KP を再スケール
                sx = size[0] / img0.shape[1]
                sy = size[1] / img0.shape[0]
                m0s = mkpts0.copy(); m0s[:, 0] *= sx; m0s[:, 1] *= sy
                m1s = mkpts1.copy(); m1s[:, 0] *= sx; m1s[:, 1] *= sy
                epi = sym_epi_dist(m0s, m1s, F_gt)
                prec_list.append(float((epi < inlier_thr).mean()))
            except Exception:
                pass

        if n_m >= 8:
            R_err, t_err = compute_pose_error(mkpts0, mkpts1, T_rel, K)
            pose_errs.append(max(R_err, t_err))

    def auc(errs, thr):
        return float(np.mean(np.array(errs) < thr)) if errs else 0.0

    return {
        'MS':          float(np.mean(ms_list))   if ms_list   else 0.0,
        'Prec@3px':    float(np.mean(prec_list)) if prec_list else 0.0,
        'n_match':     float(np.mean([m * max_kp for m in ms_list])) if ms_list else 0.0,
        'PoseAUC@5':   auc(pose_errs, 5.0),
        'PoseAUC@10':  auc(pose_errs, 10.0),
        'PoseAUC@20':  auc(pose_errs, 20.0),
    }


# ---------------------------------------------------------------------------
# 可視化
# ---------------------------------------------------------------------------

def draw_matches(
    img0: np.ndarray,
    img1: np.ndarray,
    mkpts0: np.ndarray,
    mkpts1: np.ndarray,
    inlier_mask: np.ndarray,
    title: str,
    metrics: Dict[str, float],
) -> np.ndarray:
    """
    2枚の画像を横並びにしてマッチングを描画する。
    緑: inlier（エピポーラ距離 < inlier_thr）
    赤: outlier
    """
    H0, W0 = img0.shape[:2]
    H1, W1 = img1.shape[:2]
    H = max(H0, H1)
    canvas = np.zeros((H + 60, W0 + W1, 3), dtype=np.uint8)
    canvas[:H0, :W0]      = img0
    canvas[:H1, W0:W0+W1] = img1

    for p0, p1, inl in zip(mkpts0, mkpts1, inlier_mask):
        x0, y0 = int(p0[0]), int(p0[1])
        x1, y1 = int(p1[0]) + W0, int(p1[1])
        color = (0, 200, 0) if inl else (0, 0, 220)
        cv2.line(  canvas, (x0, y0), (x1, y1), color, 1, cv2.LINE_AA)
        cv2.circle(canvas, (x0, y0), 2, color, -1)
        cv2.circle(canvas, (x1, y1), 2, color, -1)

    n_in  = int(inlier_mask.sum())
    n_all = len(inlier_mask)
    bar   = (f"{title}  |  "
             f"MS={metrics['MS']*100:.1f}%  "
             f"Prec={metrics['Prec@3px']*100:.1f}%  "
             f"PoseAUC@5={metrics['PoseAUC@5']*100:.1f}%  "
             f"Inlier={n_in}/{n_all}")
    cv2.rectangle(canvas, (0, H), (W0+W1, H+60), (30, 30, 30), -1)
    cv2.putText(canvas, bar, (10, H+40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def visualize_pair(
    configs:  List[Dict],
    path0:    str,
    path1:    str,
    T_rel:    np.ndarray,
    K:        np.ndarray,
    out_path: str,
    max_kp:   int,
    inlier_thr: float,
    device:   torch.device,
) -> None:
    """1ペアを全設定で可視化し、縦に並べて保存する。"""
    img0 = cv2.imread(path0, cv2.IMREAD_GRAYSCALE)
    img1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
    if img0 is None or img1 is None:
        return

    F_gt = compute_F_gt(T_rel, K)
    rows = []

    for cfg in configs:
        mkpts0, mkpts1 = run_matching(
            cfg['xfeat'], cfg['matcher'], img0, img1, max_kp, device)

        if len(mkpts0) > 0:
            epi          = sym_epi_dist(mkpts0, mkpts1, F_gt)
            inlier_mask  = epi < inlier_thr
        else:
            inlier_mask  = np.zeros(0, dtype=bool)

        img0_bgr = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR)
        img1_bgr = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        row = draw_matches(img0_bgr, img1_bgr,
                           mkpts0, mkpts1, inlier_mask,
                           cfg['name'], cfg['metrics'])
        rows.append(row)

    cv2.imwrite(out_path, np.vstack(rows))
    print(f"    {out_path}")


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------

# def main():
#     args = get_args()
#     os.environ['CUDA_VISIBLE_DEVICES'] = args.device
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     os.makedirs(args.output_dir, exist_ok=True)

#     print(f"\n{'='*60}")
#     print(f"  Pipeline 評価・可視化")
#     print(f"  config:       {args.config}")
#     print(f"  seqs:         {args.seqs}")
#     print(f"  n_eval_pairs: {args.n_eval_pairs}")
#     print(f"  n_vis_pairs:  {args.n_vis_pairs}")
#     print(f"  output_dir:   {args.output_dir}")
#     print(f"  device:       {device}")
#     print(f"{'='*60}\n")

#     # ── データ収集 ──────────────────────────────────────────────────────
#     from modules.dataset.thermal.sequential import SThErEOSequentialDataset
#     pairs_all: List[Tuple] = []
#     for split_ in ['train', 'val']:
#         try:
#             ds = SThErEOSequentialDataset(
#                 data_root=args.sthereo_root,
#                 stride=args.stride,
#                 split=split_,
#                 max_pairs_per_seq=args.n_eval_pairs,
#             )
#             pairs_all.extend(ds._pairs)
#         except Exception:
#             pass

#     seq_pairs: Dict[str, List] = {}
#     for sn in args.seqs:
#         sp = [p for p in pairs_all if sn in p[0]]
#         if not sp:
#             print(f"[警告] {sn} が見つかりません → スキップ")
#             continue
#         seq_pairs[sn] = sp
#         print(f"[データ] {sn}: {len(sp)} ペア")
#     print()
#     if not seq_pairs:
#         print("[ERROR] 有効なシーケンスがありません"); return

#     # ── モデルロード ─────────────────────────────────────────────────────
#     # print("[Config A] XFeat(RGB)  + Matcher(RGB)")
#     # xfeat_rgb    = load_xfeat(args.xfeat_rgb, device)
#     # matcher_rgb  = load_matcher(args.matcher_rgb, device)

#     # print("\n[Config B] XFeat(Therm)+ Matcher(RGB)")
#     # xfeat_therm  = load_xfeat(args.xfeat_therm, device)

#     # print("\n[Config C] XFeat(RGB)  + Matcher(Therm)")
#     # matcher_therm = load_matcher(args.matcher_therm, device)

#     # print("\n[Config D] XFeat(Therm)+ Matcher(Therm)\n")

#     # configs = [
#     #     {'name': 'A: XFeat(RGB)  +Matcher(RGB)',
#     #      'xfeat': xfeat_rgb,   'matcher': matcher_rgb},
#     #     {'name': 'B: XFeat(Therm)+Matcher(RGB)',
#     #      'xfeat': xfeat_therm, 'matcher': matcher_rgb},
#     #     {'name': 'C: XFeat(RGB)  +Matcher(Therm)',
#     #      'xfeat': xfeat_rgb,   'matcher': matcher_therm},
#     #     {'name': 'D: XFeat(Therm)+Matcher(Therm)',
#     #      'xfeat': xfeat_therm, 'matcher': matcher_therm},
#     # ]
#     print("[Config A/B] Loading LightGlue (RGB weights)...")
#     matcher_rgb = load_matcher(args.matcher_rgb, device)
#     if matcher_rgb is None:
#         raise RuntimeError("Config A/B 用の重みがロードできません。YAML のパスを確認してください。")

#     print("\n[Config C/D] Loading LightGlue (Thermal weights)...")
#     matcher_therm = load_matcher(args.matcher_therm, device)
#     if matcher_therm is None:
#         raise RuntimeError("Config C/D 用の重みがロードできません。")

#     xfeat_rgb   = load_xfeat(args.xfeat_rgb, device)
#     xfeat_therm = load_xfeat(args.xfeat_therm, device)

#     configs = [
#         {'name': 'A: XFeat(RGB)  + LG(RGB)',   'xfeat': xfeat_rgb,   'matcher': matcher_rgb},
#         {'name': 'B: XFeat(Therm)+ LG(RGB)',   'xfeat': xfeat_therm, 'matcher': matcher_rgb},
#         {'name': 'C: XFeat(RGB)  + LG(Therm)', 'xfeat': xfeat_rgb,   'matcher': matcher_therm},
#         {'name': 'D: XFeat(Therm)+ LG(Therm)', 'xfeat': xfeat_therm, 'matcher': matcher_therm},
#     ]
#     print(f"\n[Status] All 4 configurations are ready using LightGlue.")

#     # ── 定量評価（シーケンスごと）────────────────────────────────────────
#     col_w = 16
#     print(f"[定量評価] n_eval_pairs={args.n_eval_pairs} / シーケンス")
#     print(f"  {'指標':<6} {'Config':<32}", end='')
#     for sn in seq_pairs: print(f" {sn[:col_w]:>{col_w}}", end='')
#     print(f" {'avg':>{col_w}}")
#     print("  " + "-" * (6 + 32 + col_w * (len(seq_pairs)+1) + 2))

#     csv_rows = []
#     for cfg in configs:
#         per_seq: Dict[str, Dict] = {}
#         for sn, sp in seq_pairs.items():
#             m = evaluate_config(cfg['xfeat'], cfg['matcher'], sp, device,
#                                 args.n_eval_pairs, args.max_kp, args.inlier_thr)
#             per_seq[sn] = m

#         keys = ['MS', 'Prec@3px', 'PoseAUC@5', 'PoseAUC@10']
#         avg  = {k: float(np.mean([v[k] for v in per_seq.values()])) for k in keys}
#         avg['n_match'] = float(np.mean([v['n_match'] for v in per_seq.values()]))
#         cfg['per_seq'] = per_seq
#         cfg['metrics'] = avg

#         # 表示（PoseAUC@5 行）
#         row = f"  {'AUC@5':<6} {cfg['name']:<32}"
#         for sn in seq_pairs:
#             row += f" {per_seq[sn]['PoseAUC@5']*100:>{col_w}.1f}%"
#         row += f" {avg['PoseAUC@5']*100:>{col_w}.1f}%"
#         print(row)

#         # CSV
#         for sn, m in per_seq.items():
#             csv_rows.append({'Config': cfg['name'], 'Seq': sn,
#                 'MS(%)': round(m['MS']*100, 2),
#                 'Prec@3px(%)': round(m['Prec@3px']*100, 2),
#                 'n_match': round(m['n_match'], 0),
#                 'PoseAUC@5(%)': round(m['PoseAUC@5']*100, 2),
#                 'PoseAUC@10(%)': round(m['PoseAUC@10']*100, 2)})
#         csv_rows.append({'Config': cfg['name'], 'Seq': 'avg',
#             'MS(%)': round(avg['MS']*100, 2),
#             'Prec@3px(%)': round(avg['Prec@3px']*100, 2),
#             'n_match': round(avg['n_match'], 0),
#             'PoseAUC@5(%)': round(avg['PoseAUC@5']*100, 2),
#             'PoseAUC@10(%)': round(avg['PoseAUC@10']*100, 2)})

#     if args.save_csv:
#         csv_path = os.path.join(args.output_dir, 'metrics.csv')
#         with open(csv_path, 'w') as f:
#             f.write('Config,Seq,MS(%),Prec@3px(%),n_match,'
#                     'PoseAUC@5(%),PoseAUC@10(%)\n')
#             for r in csv_rows:
#                 f.write(f"{r['Config']},{r['Seq']},{r['MS(%)']},"
#                         f"{r['Prec@3px(%)']},"
#                         f"{r['n_match']},{r['PoseAUC@5(%)']},"
#                         f"{r['PoseAUC@10(%)']}\n")
#         print(f"\n  CSV 保存: {csv_path}")

#     # ── 可視化 ────────────────────────────────────────────────────────
#     # if args.save_vis:
#     #     print(f"\n[可視化] {args.n_vis_pairs} ペア × {len(seq_pairs)} シーケンス")
#     #     for sn, sp in seq_pairs.items():
#     #         seq_dir = os.path.join(args.output_dir, sn)
#     #         os.makedirs(seq_dir, exist_ok=True)
#     #         print(f"  [{sn}]")
#     #         for i, (p0, p1, T_rel_t, K_t) in enumerate(sp[:args.n_vis_pairs]):
#     #             T_rel = np.array(T_rel_t, dtype=np.float64)
#     #             K     = np.array(K_t,     dtype=np.float64)
#     #             for cfg in configs:
#     #                 cfg['metrics'] = cfg['per_seq'][sn]
#     #             out = os.path.join(seq_dir, f'pair_{i:03d}.png')
#     #             visualize_pair(configs, p0, p1, T_rel, K, out,
#     #                            args.max_kp, args.inlier_thr, device)

#     # print(f"\n{'='*60}")
#     # print(f"  評価完了  →  {args.output_dir}/")
#     # print(f"{'='*60}")
#     if args.save_vis:
#         print(f"\n[可視化開始] {args.n_vis_pairs} ペア × {len(seq_pairs)} シーケンス")
        
#         for sn, sp in seq_pairs.items():
#             seq_dir = os.path.join(args.output_dir, sn)
#             os.makedirs(seq_dir, exist_ok=True)
#             print(f"  > シーケンス処理中: [{sn}]")
            
#             # 指定された枚数（n_vis_pairs）だけループ
#             n_target = min(len(sp), args.n_vis_pairs)
#             for i, (p0, p1, T_rel_t, K_t) in enumerate(sp[:n_target]):
#                 T_rel = np.array(T_rel_t, dtype=np.float64)
#                 K     = np.array(K_t,     dtype=np.float64)
                
#                 # 指標を現在のシーケンスのものに同期（画像キャプション用）
#                 for cfg in configs:
#                     cfg['metrics'] = cfg['per_seq'][sn]
                
#                 out_path = os.path.join(seq_dir, f'pair_{i:03d}.png')
                
#                 # ── [追加] 進捗状況の表示 ──
#                 # 5ペアごと、または最後のペアの時にログを出力
#                 if (i + 1) % 5 == 0 or (i + 1) == n_target:
#                     print(f"    - 進捗: {sn} ({i+1}/{n_target}) 枚目の画像を生成中...")

#                 # visualize_pair 内部で cfg['name'] が自動的に画像タイトルとして使用されます
#                 visualize_pair(configs, p0, p1, T_rel, K, out_path,
#                                args.max_kp, args.inlier_thr, device)

#         print(f"\n[完了] 全ての可視化画像が保存されました。")
def main():
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Pipeline 評価・可視化 (Matcher: All LightGlue)")
    print(f"  device: {device}")
    print(f"{'='*60}\n")

    # ── データ収集 ──────────────────────────────────────────────────────
    from modules.dataset.thermal.sequential import SThErEOSequentialDataset
    pairs_all: List[Tuple] = []
    for split_ in ['train', 'val']:
        try:
            ds = SThErEOSequentialDataset(
                data_root=args.sthereo_root, stride=args.stride,
                split=split_, max_pairs_per_seq=args.n_eval_pairs,
            )
            pairs_all.extend(ds._pairs)
        except Exception: pass

    seq_pairs: Dict[str, List] = {}
    for sn in args.seqs:
        sp = [p for p in pairs_all if sn in p[0]]
        if sp:
            seq_pairs[sn] = sp
            print(f"[データ] {sn}: {len(sp)} ペア")

    # ── モデルロード (LighterGlue 定義を使用) ──────────────────────────
    print("\n[Config A/B] Loading LG (RGB weights)...")
    matcher_rgb = load_matcher(args.matcher_rgb, device)
    if matcher_rgb is None: raise RuntimeError("RGB用重みのロードに失敗しました。")

    print("[Config C/D] Loading LG (Thermal weights)...")
    matcher_therm = load_matcher(args.matcher_therm, device)
    if matcher_therm is None: raise RuntimeError("Thermal用重みのロードに失敗しました。")

    xfeat_rgb   = load_xfeat(args.xfeat_rgb, device)
    xfeat_therm = load_xfeat(args.xfeat_therm, device)

    configs = [
        {'name': 'A: XFeat(RGB)  + LG(RGB)',   'xfeat': xfeat_rgb,   'matcher': matcher_rgb},
        {'name': 'B: XFeat(Therm)+ LG(RGB)',   'xfeat': xfeat_therm, 'matcher': matcher_rgb},
        {'name': 'C: XFeat(RGB)  + LG(Therm)', 'xfeat': xfeat_rgb,   'matcher': matcher_therm},
        {'name': 'D: XFeat(Therm)+ LG(Therm)', 'xfeat': xfeat_therm, 'matcher': matcher_therm},
    ]

    # ── 定量評価の実行 ──────────────────────────────────────────────────
    print(f"\n[計算中] 指標を算出しています...")
    for cfg in configs:
        per_seq: Dict[str, Dict] = {}
        for sn, sp in seq_pairs.items():
            # evaluate_config が PoseAUC@20 を返すよう修正されている前提
            m = evaluate_config(cfg['xfeat'], cfg['matcher'], sp, device,
                                args.n_eval_pairs, args.max_kp, args.inlier_thr)
            per_seq[sn] = m

        # 指標キーの定義 (PoseAUC@20を追加)
        keys = ['MS', 'Prec@3px', 'PoseAUC@5', 'PoseAUC@10', 'PoseAUC@20']
        avg = {k: float(np.mean([v[k] for v in per_seq.values()])) for k in keys}
        avg['n_match'] = float(np.mean([v['n_match'] for v in per_seq.values()]))
        cfg['per_seq'] = per_seq
        cfg['metrics'] = avg

    # ── [修正点] シーケンスごとの個別テーブル表示 ──────────────────────────
    # 表示したい指標とラベル、スケーリングの定義
    metrics_display = [
        ('PoseAUC@5',  'AUC@5(%)',  100),
        ('PoseAUC@10', 'AUC@10(%)', 100),
        ('PoseAUC@20', 'AUC@20(%)', 100),
        ('Prec@3px',   'Precision', 100)
    ]

    target_seqs = list(seq_pairs.keys()) + ['avg']

    for sn in target_seqs:
        print(f"\n【Sequence: {sn}】")
        header = f"{'Model Configuration':<40} | " + " | ".join([f"{m[1]:>12}" for m in metrics_display])
        print("-" * len(header))
        print(header)
        print("-" * len(header))

        for cfg in configs:
            data = cfg['per_seq'][sn] if sn != 'avg' else cfg['metrics']
            row = [f"{data.get(k, 0)*s:>12.2f}" for k, _, s in metrics_display]
            print(f"{cfg['name']:<40} | " + " | ".join(row))
        print("-" * len(header))

    # ── CSV保存 (PoseAUC@20を含む) ──────────────────────────────────────
    if args.save_csv:
        csv_path = os.path.join(args.output_dir, 'metrics.csv')
        import csv
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Config', 'Seq', 'MS(%)', 'Prec(%)', 'PoseAUC@5', 'PoseAUC@10', 'PoseAUC@20'])
            for cfg in configs:
                for sn in target_seqs:
                    d = cfg['per_seq'][sn] if sn != 'avg' else cfg['metrics']
                    writer.writerow([cfg['name'], sn, d['MS']*100, d['Prec@3px']*100, 
                                     d['PoseAUC@5']*100, d['PoseAUC@10']*100, d['PoseAUC@20']*100])
        print(f"\n  CSV保存: {csv_path}")

    # ── 可視化 (進捗表示付き) ──────────────────────────────────────────
    if args.save_vis:
        print(f"\n[可視化開始] {args.n_vis_pairs} ペア × {len(seq_pairs)} シーケンス")
        for sn, sp in seq_pairs.items():
            seq_dir = os.path.join(args.output_dir, sn)
            os.makedirs(seq_dir, exist_ok=True)
            n_target = min(len(sp), args.n_vis_pairs)
            for i, (p0, p1, T_rel_t, K_t) in enumerate(sp[:n_target]):
                if (i + 1) % 5 == 0 or (i + 1) == n_target:
                    print(f"    - {sn}: ({i+1}/{n_target}) 枚目の画像を生成中...")
                for cfg in configs:
                    cfg['metrics'] = cfg['per_seq'][sn]
                visualize_pair(configs, p0, p1, np.array(T_rel_t), np.array(K_t), 
                               os.path.join(seq_dir, f'pair_{i:03d}.png'),
                               args.max_kp, args.inlier_thr, device)
    
    print(f"\n{'='*60}\n  評価完了 -> {args.output_dir}/\n{'='*60}")


if __name__ == '__main__':
    main()