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

# def load_yaml_config(path: str) -> dict:
#     """eval_pipeline.yaml を読み込んでフラットな設定辞書を返す。"""
#     import yaml
#     with open(path) as f:
#         cfg = yaml.safe_load(f) or {}

#     data   = cfg.get('data',   {})
#     ev     = cfg.get('eval',   {})
#     out    = cfg.get('output', {})
#     models = cfg.get('models', {})

#     # seqs: リストまたは単一文字列を正規化
#     seqs_raw = data.get('seqs') or data.get('seq', 'kaist_morning')
#     seqs     = seqs_raw if isinstance(seqs_raw, list) else [seqs_raw]

#     return {
#         'sthereo_root': data.get('sthereo_root', 'datasets/sthereo'),
#         'seqs':         seqs,
#         'stride':       int(data.get('stride', 3)),
#         'n_eval_pairs': int(ev.get('n_eval_pairs', 200)),
#         'n_vis_pairs':  int(ev.get('n_vis_pairs',  10)),
#         'max_kp':       int(ev.get('max_kp',       512)),
#         'inlier_thr':   float(ev.get('inlier_thr', 3.0)),
#         'output_dir':   out.get('dir',       'results/eval_pipeline'),
#         'save_csv':     bool(out.get('save_csv', True)),
#         'save_vis':     bool(out.get('save_vis', True)),
#         'xfeat_rgb':    models.get('xfeat_rgb',   'weights/xfeat.pt'),
#         'xfeat_therm':  models.get('xfeat_therm', None),
#         'matcher_rgb':  models.get('matcher_rgb',  None),
#         'matcher_therm': models.get('matcher_therm', None),
#         'compare_pipeline': models.get('compare_pipeline', False),
#         'xfeat_pipe_therm': models.get('xfeat_pipe_therm', None),
#         'matcher_pipe_therm': models.get('matcher_pipe_therm', None),
#         'device':       str(cfg.get('device', '0')),
#     }


# def get_args():
#     p = argparse.ArgumentParser()
#     p.add_argument('--config', default='eval_pipeline.yaml')
#     p.add_argument('--seqs',   nargs='+', default=None)
#     p.add_argument('--seq',    default=None)
#     p.add_argument('--n_eval_pairs', type=int, default=None)
#     p.add_argument('--n_vis_pairs',  type=int, default=None)
#     p.add_argument('--output_dir',   default=None)
#     p.add_argument('--device',       default=None)
#     cli = p.parse_args()

#     cfg = load_yaml_config(cli.config)
#     if cli.seqs         is not None: cfg['seqs']         = cli.seqs
#     elif cli.seq        is not None: cfg['seqs']         = [cli.seq]
#     if cli.n_eval_pairs is not None: cfg['n_eval_pairs'] = cli.n_eval_pairs
#     if cli.n_vis_pairs  is not None: cfg['n_vis_pairs']  = cli.n_vis_pairs
#     if cli.output_dir   is not None: cfg['output_dir']   = cli.output_dir
#     if cli.device       is not None: cfg['device']       = cli.device

#     ns = types.SimpleNamespace(**cfg)
#     ns.config = cli.config
#     return ns


# # ---------------------------------------------------------------------------
# # モデルロード
# # ---------------------------------------------------------------------------

# def load_xfeat(weights: Optional[str], device: torch.device) -> torch.nn.Module:
#     from modules.model import XFeatModel
#     model = XFeatModel().to(device).eval()
#     if weights and os.path.isfile(weights):
#         model.load_state_dict(
#             torch.load(weights, map_location=device, weights_only=True))
#         print(f"    XFeat: {weights}")
#     else:
#         print(f"    XFeat: default RGB weights (weights/xfeat.pt)")
#     for p in model.parameters():
#         p.requires_grad_(False)
#     return model


# def load_matcher(matcher_path: Optional[str],
#                  device: torch.device) -> Optional[object]:
#     """
#     matcher_path が None → MNN（LG 不要）
#     matcher_path が文字列 → glue-factory 形式の LightGlue checkpoint をロード
#     """
#     if matcher_path is None:
#         print(f"    Matcher: MNN (mutual nearest neighbor)")
#         return None

#     # evaluate/eval_matching.py の load_lightglue を使用
#     from eval.eval_matching import load_lightglue as _load_lg
#     lg = _load_lg(weights_path=matcher_path, device=device)
#     if lg is not None:
#         print(f"    Matcher: LightGlue ← {matcher_path}")
#     else:
#         print(f"    Matcher: LightGlue ロード失敗 → MNN にフォールバック")
#     return lg


# # ---------------------------------------------------------------------------
# # 1ペアのマッチング（MNN または LightGlue を統一インターフェースで）
# # ---------------------------------------------------------------------------

# @torch.no_grad()
# def run_matching(
#     xfeat:   torch.nn.Module,
#     matcher: Optional[object],
#     img0:    np.ndarray,         # (H, W) グレースケール
#     img1:    np.ndarray,
#     max_kp:  int,
#     device:  torch.device,
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     XFeat で検出 → MNN または LightGlue でマッチング。

#     Returns:
#         mkpts0, mkpts1: (n_match, 2) float32 の対応点座標
#     """
#     from eval.eval_matching import detect, match

#     H, W  = img0.shape[:2]
#     size  = (640, 480)
#     img0r = cv2.resize(img0, size)
#     img1r = cv2.resize(img1, size)

#     def to_tensor(im):
#         im3 = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
#         return (torch.from_numpy(im3)
#                 .permute(2, 0, 1).float().div(255)
#                 .unsqueeze(0).to(device))

#     t0 = to_tensor(img0r)
#     t1 = to_tensor(img1r)

#     kpts0, descs0 = detect(xfeat, t0, max_kp)   # (N, 2), (N, 64)
#     kpts1, descs1 = detect(xfeat, t1, max_kp)

#     if len(kpts0) == 0 or len(kpts1) == 0:
#         return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)

#     if matcher is None:
#         # MNN マッチング（論文と同一方法）
#         idx0, idx1 = match(descs0, descs1, 'mutual_nn', ratio_thr=0.9)
#     else:
#         # LightGlue マッチング
#         idx0, idx1 = match(
#             descs0, descs1, 'lightglue',
#             ratio_thr=0.9,
#             kpts1=kpts0, kpts2=kpts1,
#             image_size=(size[1], size[0]),
#             device=device,
#         )

#     if len(idx0) == 0:
#         return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)

#     # リサイズ後座標 → 元解像度座標に変換（可視化用）
#     sx = W / size[0]
#     sy = H / size[1]
#     m0 = kpts0[idx0].copy(); m0[:, 0] *= sx; m0[:, 1] *= sy
#     m1 = kpts1[idx1].copy(); m1[:, 0] *= sx; m1[:, 1] *= sy

#     return m0.astype(np.float32), m1.astype(np.float32)


# # ---------------------------------------------------------------------------
# # 指標計算
# # ---------------------------------------------------------------------------

# def compute_F_gt(T_rel: np.ndarray, K: np.ndarray) -> np.ndarray:
#     from eval.eval_matching import _compute_F_gt
#     return _compute_F_gt(T_rel, K).astype(np.float32)


# def sym_epi_dist(pts0: np.ndarray, pts1: np.ndarray,
#                  F: np.ndarray) -> np.ndarray:
#     from eval.eval_matching import _sym_epi_dist
#     return _sym_epi_dist(pts0, pts1, F)


# def compute_pose_error(mkpts0: np.ndarray, mkpts1: np.ndarray,
#                        T_rel: np.ndarray, K: np.ndarray,
#                        size: Tuple[int, int] = (640, 480)) -> Tuple[float, float]:
#     """E行列から R_err, t_err (degrees) を計算する。"""
#     if len(mkpts0) < 8:
#         return float('inf'), float('inf')

#     # 可視化サイズ座標 → 評価用リサイズ座標
#     W_orig, H_orig = size[0], size[1]

#     K64 = np.array(K, dtype=np.float64)
#     p0  = mkpts0.astype(np.float32).reshape(-1, 1, 2)
#     p1  = mkpts1.astype(np.float32).reshape(-1, 1, 2)

#     E, msk = cv2.findEssentialMat(p0, p1, K64,
#                                    method=cv2.RANSAC, prob=0.999, threshold=1.0)
#     if E is None or msk is None or int(msk.sum()) < 5:
#         return float('inf'), float('inf')
#     if E.shape[0] > 3:
#         E = E[:3]

#     _, R_est, t_est, _ = cv2.recoverPose(E, p0, p1, K64, mask=msk)
#     R_gt = np.array(T_rel[:3, :3], dtype=np.float64)
#     t_gt = np.array(T_rel[:3, 3],  dtype=np.float64)

#     R_rel = R_est @ R_gt.T
#     trace = float(np.clip((np.trace(R_rel) - 1) / 2, -1, 1))
#     R_err = float(np.degrees(np.arccos(trace)))

#     t_n   = np.linalg.norm(t_est) + 1e-8
#     t_gt_n = np.linalg.norm(t_gt)  + 1e-8
#     cos_t = float(np.clip(
#         np.dot(t_est.flatten() / t_n, t_gt / t_gt_n), -1, 1))
#     t_err = float(np.degrees(np.arccos(abs(cos_t))))

#     return R_err, t_err


# def evaluate_config(
#     xfeat:   torch.nn.Module,
#     matcher: Optional[object],
#     pairs:   List[Tuple],
#     device:  torch.device,
#     n_pairs: int,
#     max_kp:  int,
#     inlier_thr: float = 3.0,
# ) -> Dict[str, float]:
#     """1設定の定量評価を行う。"""
#     ms_list, prec_list, pose_errs = [], [], []
#     size = (640, 480)

#     for path0, path1, T_rel_t, K_t in pairs[:n_pairs]:
#         T_rel = np.array(T_rel_t, dtype=np.float64)
#         K     = np.array(K_t,     dtype=np.float64)

#         img0 = cv2.imread(path0, cv2.IMREAD_GRAYSCALE)
#         img1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
#         if img0 is None or img1 is None:
#             continue

#         mkpts0, mkpts1 = run_matching(
#             xfeat, matcher, img0, img1, max_kp, device)
#         n_m = len(mkpts0)

#         # MS = n_match / min(|KP1|, |KP2|)（論文準拠）
#         ms_list.append(n_m / max(min(max_kp, max_kp), 1))

#         if n_m > 0:
#             try:
#                 F_gt = compute_F_gt(T_rel, K)
#                 # 評価はリサイズ後座標で行うため元画像比率で KP を再スケール
#                 sx = size[0] / img0.shape[1]
#                 sy = size[1] / img0.shape[0]
#                 m0s = mkpts0.copy(); m0s[:, 0] *= sx; m0s[:, 1] *= sy
#                 m1s = mkpts1.copy(); m1s[:, 0] *= sx; m1s[:, 1] *= sy
#                 epi = sym_epi_dist(m0s, m1s, F_gt)
#                 prec_list.append(float((epi < inlier_thr).mean()))
#             except Exception:
#                 pass

#         if n_m >= 8:
#             R_err, t_err = compute_pose_error(mkpts0, mkpts1, T_rel, K)
#             pose_errs.append(max(R_err, t_err))

#     def auc(errs, thr):
#         return float(np.mean(np.array(errs) < thr)) if errs else 0.0

#     return {
#         'MS':          float(np.mean(ms_list))   if ms_list   else 0.0,
#         'Prec@3px':    float(np.mean(prec_list)) if prec_list else 0.0,
#         'n_match':     float(np.mean([m * max_kp for m in ms_list])) if ms_list else 0.0,
#         'PoseAUC@5':   auc(pose_errs, 5.0),
#         'PoseAUC@10':  auc(pose_errs, 10.0),
#     }


# # ---------------------------------------------------------------------------
# # 可視化
# # ---------------------------------------------------------------------------

# def draw_matches(
#     img0: np.ndarray,
#     img1: np.ndarray,
#     mkpts0: np.ndarray,
#     mkpts1: np.ndarray,
#     inlier_mask: np.ndarray,
#     title: str,
#     metrics: Dict[str, float],
# ) -> np.ndarray:
#     """
#     2枚の画像を横並びにしてマッチングを描画する。
#     緑: inlier（エピポーラ距離 < inlier_thr）
#     赤: outlier
#     """
#     H0, W0 = img0.shape[:2]
#     H1, W1 = img1.shape[:2]
#     H = max(H0, H1)
#     canvas = np.zeros((H + 60, W0 + W1, 3), dtype=np.uint8)
#     canvas[:H0, :W0]      = img0
#     canvas[:H1, W0:W0+W1] = img1

#     for p0, p1, inl in zip(mkpts0, mkpts1, inlier_mask):
#         x0, y0 = int(p0[0]), int(p0[1])
#         x1, y1 = int(p1[0]) + W0, int(p1[1])
#         color = (0, 200, 0) if inl else (0, 0, 220)
#         cv2.line(  canvas, (x0, y0), (x1, y1), color, 1, cv2.LINE_AA)
#         cv2.circle(canvas, (x0, y0), 2, color, -1)
#         cv2.circle(canvas, (x1, y1), 2, color, -1)

#     n_in  = int(inlier_mask.sum())
#     n_all = len(inlier_mask)
#     bar   = (f"{title}  |  "
#              f"MS={metrics['MS']*100:.1f}%  "
#              f"Prec={metrics['Prec@3px']*100:.1f}%  "
#              f"PoseAUC@5={metrics['PoseAUC@5']*100:.1f}%  "
#              f"Inlier={n_in}/{n_all}")
#     cv2.rectangle(canvas, (0, H), (W0+W1, H+60), (30, 30, 30), -1)
#     cv2.putText(canvas, bar, (10, H+40),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)
#     return canvas


# def visualize_pair(
#     configs:  List[Dict],
#     path0:    str,
#     path1:    str,
#     T_rel:    np.ndarray,
#     K:        np.ndarray,
#     out_path: str,
#     max_kp:   int,
#     inlier_thr: float,
#     device:   torch.device,
# ) -> None:
#     """1ペアを全設定で可視化し、縦に並べて保存する。"""
#     img0 = cv2.imread(path0, cv2.IMREAD_GRAYSCALE)
#     img1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
#     if img0 is None or img1 is None:
#         return

#     F_gt = compute_F_gt(T_rel, K)
#     rows = []

#     for cfg in configs:
#         mkpts0, mkpts1 = run_matching(
#             cfg['xfeat'], cfg['matcher'], img0, img1, max_kp, device)

#         if len(mkpts0) > 0:
#             epi          = sym_epi_dist(mkpts0, mkpts1, F_gt)
#             inlier_mask  = epi < inlier_thr
#         else:
#             inlier_mask  = np.zeros(0, dtype=bool)

#         img0_bgr = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR)
#         img1_bgr = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
#         row = draw_matches(img0_bgr, img1_bgr,
#                            mkpts0, mkpts1, inlier_mask,
#                            cfg['name'], cfg['metrics'])
#         rows.append(row)

#     cv2.imwrite(out_path, np.vstack(rows))
#     print(f"    {out_path}")


# # ---------------------------------------------------------------------------
# # メイン
# # ---------------------------------------------------------------------------

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
#     print("[Config A] XFeat(RGB)  + Matcher(RGB)")
#     xfeat_rgb    = load_xfeat(args.xfeat_rgb, device)
#     matcher_rgb  = load_matcher(args.matcher_rgb, device)

#     print("\n[Config B] XFeat(Therm)+ Matcher(RGB)")
#     xfeat_therm  = load_xfeat(args.xfeat_therm, device)

#     print("\n[Config C] XFeat(RGB)  + Matcher(Therm)")
#     matcher_therm = load_matcher(args.matcher_therm, device)

#     print("\n[Config D] XFeat(Therm)+ Matcher(Therm)\n")

#     configs = [
#         {'name': 'A: XFeat(RGB)  +Matcher(RGB)',
#          'xfeat': xfeat_rgb,   'matcher': matcher_rgb},
#         {'name': 'B: XFeat(Therm)+Matcher(RGB)',
#          'xfeat': xfeat_therm, 'matcher': matcher_rgb},
#         {'name': 'C: XFeat(RGB)  +Matcher(Therm)',
#          'xfeat': xfeat_rgb,   'matcher': matcher_therm},
#         {'name': 'D: XFeat(Therm)+Matcher(Therm)',
#          'xfeat': xfeat_therm, 'matcher': matcher_therm},
#     ]

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
#     if args.save_vis:
#         print(f"\n[可視化] {args.n_vis_pairs} ペア × {len(seq_pairs)} シーケンス")
#         for sn, sp in seq_pairs.items():
#             seq_dir = os.path.join(args.output_dir, sn)
#             os.makedirs(seq_dir, exist_ok=True)
#             print(f"  [{sn}]")
#             for i, (p0, p1, T_rel_t, K_t) in enumerate(sp[:args.n_vis_pairs]):
#                 T_rel = np.array(T_rel_t, dtype=np.float64)
#                 K     = np.array(K_t,     dtype=np.float64)
#                 for cfg in configs:
#                     cfg['metrics'] = cfg['per_seq'][sn]
#                 out = os.path.join(seq_dir, f'pair_{i:03d}.png')
#                 visualize_pair(configs, p0, p1, T_rel, K, out,
#                                args.max_kp, args.inlier_thr, device)

#     print(f"\n{'='*60}")
#     print(f"  評価完了  →  {args.output_dir}/")
#     print(f"{'='*60}")


# if __name__ == '__main__':
#     main()


# ---------------------------------------------------------------------------
# 設定ロード
# ---------------------------------------------------------------------------

def load_yaml_config(path: str) -> dict:
    import yaml
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}

    data   = cfg.get('data',   {})
    ev     = cfg.get('eval',   {})
    out    = cfg.get('output', {})
    models = cfg.get('models', {})

    # 'seqs' があればそれを、なければ 'seq' を取得
    seqs_raw = data.get('seqs') or data.get('seq', ['kaist_morning'])
    
    # 古い設定（リスト形式）の場合は 'sthereo' のシーケンスとして互換性を持たせる
    if isinstance(seqs_raw, list):
        seqs = {'sthereo': seqs_raw}
    elif isinstance(seqs_raw, dict):
        seqs = seqs_raw
    else:
        seqs = {'sthereo': [seqs_raw]}

    # データセットのルートパス取得
    datasets = data.get('datasets', {})
    if not datasets and 'sthereo_root' in data:
        datasets = {'sthereo': data.get('sthereo_root')}

    return {
        'data_roots':   datasets,
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
        'matcher_rgb':  models.get('matcher_rgb',  None),
        'matcher_therm': models.get('matcher_therm', None),
        'compare_pipeline': models.get('compare_pipeline', False),
        'xfeat_pipe_therm': models.get('xfeat_pipe_therm', None),
        'matcher_pipe_therm': models.get('matcher_pipe_therm', None),
        'device':       str(cfg.get('device', '0')),
    }


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/eval_pipeline.yaml')
    p.add_argument('--device', default=None)
    cli = p.parse_args()

    cfg = load_yaml_config(cli.config)
    if cli.device is not None: cfg['device'] = cli.device

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


def load_matcher(matcher_path: Optional[str], device: torch.device) -> Optional[object]:
    if matcher_path is None:
        print(f"    Matcher: MNN (mutual nearest neighbor)")
        return None

    ckpt = torch.load(matcher_path, map_location='cpu', weights_only=False)
    state_dict = ckpt.get('model', ckpt)
    
    is_large_model = False
    for k, v in state_dict.items():
        if 'input_proj.weight' in k:
            if v.shape[0] == 256: 
                is_large_model = True
            break

    if not is_large_model:
        from modules.lighterglue import LighterGlue
        print(f"    Matcher: LighterGlue(Small) ← {matcher_path}")
        lg = LighterGlue(weights=matcher_path).to(device)
        lg.eval()
        return lg
    else:
        print(f"    Matcher: LightGlue(Large/GF) ← {matcher_path}")
        _THIS = os.path.dirname(os.path.abspath(__file__))
        _GF   = os.path.join(_THIS, 'third_party', 'glue-factory')
        if os.path.isdir(_GF) and _GF not in sys.path:
            sys.path.insert(0, _GF)
            
        from gluefactory.models.matchers.lightglue import LightGlue
        from omegaconf import OmegaConf
        
        lg_conf = OmegaConf.create({
            'name':             'matchers.lightglue',
            'features':         None,
            'input_dim':        64,
            'descriptor_dim':   256,
            'n_layers':         9,
            'num_heads':        4,
            'flash':            False,
            'mp':               False,
            'checkpointed':     False,
            'depth_confidence': -1,
            'width_confidence': -1,
            'filter_threshold': 0.1,
            'weights':          None,
        })
        lg = LightGlue(lg_conf).to(device).eval()
        
        matcher_state = {
            k[len('matcher.'):] if k.startswith('matcher.') else k: v
            for k, v in state_dict.items()
        }
        
        lg.load_state_dict(matcher_state, strict=False)
        return lg


# ---------------------------------------------------------------------------
# 1ペアのマッチング
# ---------------------------------------------------------------------------

def resize_with_pad(img: np.ndarray, target_size: Tuple[int, int] = (640, 480)) -> Tuple[np.ndarray, Tuple[float, float, int, int]]:
    """
    アスペクト比を維持してリサイズし、黒でパディングする (Letterbox)。
    戻り値: パディング済み画像, (スケールX, スケールY, pad_x, pad_y)
    """
    h, w = img.shape[:2]
    tw, th = target_size
    scale = min(tw / w, th / h)
    
    nw, nh = int(w * scale), int(h * scale)
    img_resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    
    pad_w = tw - nw
    pad_h = th - nh
    top, bottom = pad_h // 2, pad_h - (pad_h // 2)
    left, right = pad_w // 2, pad_w - (pad_w // 2)
    
    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    
    # 座標を元に戻すための情報を返す
    return img_padded, (w / nw, h / nh, left, top)
    
# @torch.no_grad()
# def run_matching(
#     xfeat:   torch.nn.Module,
#     matcher: Optional[object],
#     img0:    np.ndarray,
#     img1:    np.ndarray,
#     max_kp:  int,
#     device:  torch.device,
# ) -> Tuple[np.ndarray, np.ndarray]:
#     from eval.eval_matching import detect, match

#     H, W  = img0.shape[:2]
#     size  = (640, 480)
#     img0r = cv2.resize(img0, size)
#     img1r = cv2.resize(img1, size)

#     def to_tensor(im):
#         im3 = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
#         return (torch.from_numpy(im3)
#                 .permute(2, 0, 1).float().div(255)
#                 .unsqueeze(0).to(device))

#     t0 = to_tensor(img0r)
#     t1 = to_tensor(img1r)

#     kpts0, descs0 = detect(xfeat, t0, max_kp)
#     kpts1, descs1 = detect(xfeat, t1, max_kp)

#     if len(kpts0) == 0 or len(kpts1) == 0:
#         return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)

#     if matcher is None:
#         idx0, idx1 = match(descs0, descs1, 'mutual_nn', ratio_thr=0.9)
#     else:
#         if matcher.__class__.__name__ == 'LighterGlue':
#             kpts0_t = torch.from_numpy(kpts0).unsqueeze(0).to(device)
#             descs0_t = torch.from_numpy(descs0).unsqueeze(0).to(device)
#             size0_t = torch.tensor([[size[0], size[1]]], device=device)
            
#             kpts1_t = torch.from_numpy(kpts1).unsqueeze(0).to(device)
#             descs1_t = torch.from_numpy(descs1).unsqueeze(0).to(device)
#             size1_t = torch.tensor([[size[0], size[1]]], device=device)

#             data = {
#                 'keypoints0': kpts0_t,
#                 'descriptors0': descs0_t,
#                 'image_size0': size0_t,
#                 'keypoints1': kpts1_t,
#                 'descriptors1': descs1_t,
#                 'image_size1': size1_t,
#             }
            
#             res = matcher(data)
#             if 'matches' in res:
#                 m = res['matches'][0].cpu().numpy()
#                 idx0 = m[:, 0].astype(np.int64)
#                 idx1 = m[:, 1].astype(np.int64)
#             elif 'matches0' in res:
#                 m = res['matches0'][0].cpu().numpy()
#                 valid = m >= 0
#                 idx0 = np.where(valid)[0]
#                 idx1 = m[valid]
#             else:
#                 idx0, idx1 = match(descs0, descs1, 'mutual_nn', ratio_thr=0.9)
#         else:
#             from eval.eval_matching import match_lightglue
#             idx0, idx1 = match_lightglue(
#                 kpts0, descs0, kpts1, descs1,
#                 image_size=(size[1], size[0]),
#                 device=device,
#                 lightglue_model=matcher
#             )

#     if len(idx0) == 0:
#         return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)

#     sx = W / size[0]
#     sy = H / size[1]
#     m0 = kpts0[idx0].copy(); m0[:, 0] *= sx; m0[:, 1] *= sy
#     m1 = kpts1[idx1].copy(); m1[:, 0] *= sx; m1[:, 1] *= sy

#     return m0.astype(np.float32), m1.astype(np.float32)

def read_image_for_eval(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None

    if 'ms2' in path.lower() or 'sync_data' in path.lower():
        img_float = img.astype(np.float32)
        if img_float.ndim == 3:
            img_float = cv2.cvtColor(img_float, cv2.COLOR_BGR2GRAY)
            
        # 1. AnyThermal方式の hist_99 正規化
        im_srt = np.sort(img_float.reshape(-1))
        upper_bound = im_srt[round(len(im_srt) * 0.99) - 1]
        lower_bound = im_srt[round(len(im_srt) * 0.01)]

        img_float[img_float < lower_bound] = lower_bound
        img_float[img_float > upper_bound] = upper_bound
        
        if upper_bound - lower_bound > 1e-5:
            image_out = ((img_float - lower_bound) / (upper_bound - lower_bound)) * 255.0
        else:
            image_out = img_float * 0
            
        image_out = image_out.astype(np.uint8)

        # 2. AnyThermal方式の enhance_image (CLAHE + Bilateral)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_img = clahe.apply(image_out)
        img_final = cv2.bilateralFilter(clahe_img, 5, 20, 15)
        
        # 3. 【最重要】AnyThermal方式の Crop (静的ノイズ領域の除去)
        h, w = img_final.shape[:2]
        crop_top, crop_bottom = 9, 35
        crop_left, crop_right = 28, 34
        img_final = img_final[crop_top:h - crop_bottom, crop_left:w - crop_right]
        
    else:
        # SThErEO等の処理済みデータセット用
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img.dtype == np.uint16:
            img_uint8 = (img / 256).astype(np.uint8)
        else:
            img_uint8 = img.astype(np.uint8)
        img_final = img_uint8

    return img_final

@torch.no_grad()
def run_matching(
    xfeat:   torch.nn.Module,
    matcher: Optional[object],
    img0:    np.ndarray,
    img1:    np.ndarray,
    max_kp:  int,
    device:  torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    from eval.eval_matching import detect, match

    # --- 修正: アスペクト比を維持したリサイズ＆パディング ---
    size = (640, 480)
    img0r, meta0 = resize_with_pad(img0, size)
    img1r, meta1 = resize_with_pad(img1, size)

    def to_tensor(im):
        im3 = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        return (torch.from_numpy(im3).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device))

    t0 = to_tensor(img0r)
    t1 = to_tensor(img1r)

    kpts0, descs0 = detect(xfeat, t0, max_kp)
    kpts1, descs1 = detect(xfeat, t1, max_kp)

    if len(kpts0) == 0 or len(kpts1) == 0:
        return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)

    if matcher is None:
        idx0, idx1 = match(descs0, descs1, 'mutual_nn', ratio_thr=0.9)
    else:
        if matcher.__class__.__name__ == 'LighterGlue':
            kpts0_t = torch.from_numpy(kpts0).unsqueeze(0).to(device)
            descs0_t = torch.from_numpy(descs0).unsqueeze(0).to(device)
            size0_t = torch.tensor([[size[0], size[1]]], device=device)
            kpts1_t = torch.from_numpy(kpts1).unsqueeze(0).to(device)
            descs1_t = torch.from_numpy(descs1).unsqueeze(0).to(device)
            size1_t = torch.tensor([[size[0], size[1]]], device=device)
            data = {'keypoints0': kpts0_t, 'descriptors0': descs0_t, 'image_size0': size0_t,
                    'keypoints1': kpts1_t, 'descriptors1': descs1_t, 'image_size1': size1_t}
            res = matcher(data)
            if 'matches' in res:
                m = res['matches'][0].cpu().numpy()
                idx0, idx1 = m[:, 0].astype(np.int64), m[:, 1].astype(np.int64)
            elif 'matches0' in res:
                m = res['matches0'][0].cpu().numpy()
                valid = m >= 0
                idx0, idx1 = np.where(valid)[0], m[valid]
            else:
                idx0, idx1 = match(descs0, descs1, 'mutual_nn', ratio_thr=0.9)
        else:
            from eval.eval_matching import match_lightglue
            idx0, idx1 = match_lightglue(kpts0, descs0, kpts1, descs1, image_size=(size[1], size[0]), device=device, lightglue_model=matcher)

    if len(idx0) == 0:
        return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)

    # --- 修正: パディングとスケールを考慮して元の画像座標に逆変換 ---
    m0 = kpts0[idx0].copy()
    sx0, sy0, pad_x0, pad_y0 = meta0
    m0[:, 0] = (m0[:, 0] - pad_x0) * sx0
    m0[:, 1] = (m0[:, 1] - pad_y0) * sy0

    m1 = kpts1[idx1].copy()
    sx1, sy1, pad_x1, pad_y1 = meta1
    m1[:, 0] = (m1[:, 0] - pad_x1) * sx1
    m1[:, 1] = (m1[:, 1] - pad_y1) * sy1

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
    if len(mkpts0) < 8:
        return float('inf'), float('inf')

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
    ms_list, prec_list, pose_errs = [], [], []
    size = (640, 480)

    for path0, path1, T_rel_t, K_t in pairs[:n_pairs]:
        T_rel = np.array(T_rel_t, dtype=np.float64)
        K     = np.array(K_t,     dtype=np.float64)

        # img0 = cv2.imread(path0, cv2.IMREAD_GRAYSCALE)
        # img1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
        img0 = read_image_for_eval(path0)
        img1 = read_image_for_eval(path1)
        if img0 is None or img1 is None:
            continue

        mkpts0, mkpts1 = run_matching(
            xfeat, matcher, img0, img1, max_kp, device)
        n_m = len(mkpts0)

        ms_list.append(n_m / max(min(max_kp, max_kp), 1))

        if n_m > 0:
            try:
                F_gt = compute_F_gt(T_rel, K)
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
        'MS_mean':     float(np.mean(ms_list))   if ms_list   else 0.0,
        'MS_min':      float(np.min(ms_list))    if ms_list   else 0.0,
        'MS_max':      float(np.max(ms_list))    if ms_list   else 0.0,
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
             f"MS={metrics['MS_mean']*100:.1f}%  "
             f"Prec={metrics['Prec@3px']*100:.1f}%  "
             f"PoseAUC@5={metrics['PoseAUC@5']*100:.1f}%  "
             f"Inlier={n_in}/{n_all}")
    cv2.rectangle(canvas, (0, H), (W0+W1, H+60), (30, 30, 30), -1)
    cv2.putText(canvas, bar, (10, H+40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas

# _DEBUG_MS2_SAVE_COUNT = 5
def read_image_for_eval(path: str) -> np.ndarray:
    """評価パイプライン専用の画像読み込み関数"""
    global _DEBUG_MS2_SAVE_COUNT
    
    # 1. 16-bit 情報を保持して読み込む
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None

    # MS2 の場合のみ特別な前処理を適用
    if 'ms2' in path.lower() or 'sync_data' in path.lower():
        img_float = img.astype(np.float32)
        if img_float.ndim == 3:
            img_float = cv2.cvtColor(img_float, cv2.COLOR_BGR2GRAY)
            
        im_srt = np.sort(img_float.reshape(-1))
        upper_bound = im_srt[round(len(im_srt) * 0.99) - 1]
        lower_bound = im_srt[round(len(im_srt) * 0.01)]

        img_float[img_float < lower_bound] = lower_bound
        img_float[img_float > upper_bound] = upper_bound
        
        if upper_bound - lower_bound > 1e-5:
            image_out = ((img_float - lower_bound) / (upper_bound - lower_bound)) * 255.0
        else:
            image_out = img_float * 0
            
        image_out = image_out.astype(np.uint8)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_img = clahe.apply(image_out)
        img_final = cv2.bilateralFilter(clahe_img, 5, 20, 15)

        h, w = img_final.shape[:2]
        crop_top, crop_bottom = 9, 35
        crop_left, crop_right = 28, 34
        img_final = img_final[crop_top:h - crop_bottom, crop_left:w - crop_right]

        # ----------------------------------------------------------
        # # デバッグ用：前処理が完了した画像を最初の10枚だけディスクに保存
        # if _DEBUG_MS2_SAVE_COUNT < 10:
        #     import os
        #     save_dir = "debug_ms2_eval"
        #     os.makedirs(save_dir, exist_ok=True)
        #     original_name = os.path.basename(path)
        #     save_path = os.path.join(save_dir, f"eval_prep_{_DEBUG_MS2_SAVE_COUNT:02d}_{original_name}")
        #     cv2.imwrite(save_path, img_final)
        #     print(f"[DEBUG] MS2前処理確認用画像を保存しました: {save_path}")
        #     _DEBUG_MS2_SAVE_COUNT += 1
        # ----------------------------------------------------------
        return img_final

    else:
        # SThErEO や VIVID など既に前処理済みのデータセット用
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img.dtype == np.uint16:
            return (img / 256).astype(np.uint8)
        return img.astype(np.uint8)

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
    # img0 = cv2.imread(path0, cv2.IMREAD_GRAYSCALE)
    # img1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
    img0 = read_image_for_eval(path0)
    img1 = read_image_for_eval(path1)
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


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------

def main():
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Pipeline 評価・可視化 (Multi-Dataset)")
    print(f"  config:       {args.config}")
    print(f"  n_eval_pairs: {args.n_eval_pairs}")
    print(f"  output_dir:   {args.output_dir}")
    print(f"  device:       {device}")
    print(f"{'='*60}\n")

    # ── データ収集 ──────────────────────────────────────────────────────
    from modules.dataset.thermal.sequential import (
        SThErEOSequentialDataset,
        VividSequentialDataset,
        MS2SequentialDataset,
        TartanRGBTSequentialDataset
    )

    seq_pairs: Dict[str, List] = {}
    
    # ユーザーが要求したすべてのシーケンス名をリスト化
    all_requested_seqs = []
    for ds_name, s_list in args.seqs.items():
        if isinstance(s_list, list):
            all_requested_seqs.extend(s_list)

    print("[データ収集を開始します...]")

    # 1. SThErEO の収集
    if 'sthereo' in args.data_roots and args.seqs.get('sthereo'):
        ds_root = args.data_roots['sthereo']
        if os.path.exists(ds_root):
            for split_ in ['train', 'val']:
                try:
                    ds = SThErEOSequentialDataset(
                        data_root=ds_root, stride=args.stride, 
                        split=split_, max_pairs_per_seq=args.n_eval_pairs)
                    for target_seq in args.seqs['sthereo']:
                        sp = [p for p in ds._pairs if target_seq in p[0]]
                        if sp:
                            seq_pairs[f"[SThErEO] {target_seq}"] = sp
                except Exception:
                    pass

    # 2. VIVID の収集
    if 'vivid' in args.data_roots and args.seqs.get('vivid'):
        ds_root = args.data_roots['vivid']
        if os.path.exists(ds_root):
            for split_ in ['train', 'val']:
                try:
                    ds = VividSequentialDataset(
                        data_root=ds_root, stride=args.stride, 
                        split=split_, max_pairs_per_seq=args.n_eval_pairs)
                    for target_seq in args.seqs['vivid']:
                        sp = [p for p in ds._pairs if target_seq in p[0]]
                        if sp:
                            seq_pairs[f"[VIVID] {target_seq}"] = sp
                except Exception:
                    pass

    # 3. MS2 の収集
    if 'ms2' in args.data_roots and args.seqs.get('ms2'):
        ds_root = args.data_roots['ms2']
        if os.path.exists(ds_root):
            for split_ in ['train', 'val']:
                try:
                    ds = MS2SequentialDataset(
                        data_root=ds_root, stride=args.stride, 
                        split=split_, max_pairs_per_seq=args.n_eval_pairs)
                    for target_seq in args.seqs['ms2']:
                        sp = [p for p in ds._pairs if target_seq in p[0]]
                        if sp:
                            seq_pairs[f"[MS2] {target_seq}"] = sp
                except Exception:
                    pass

    # 4. TartanRGBT の収集
    if 'tartanrgbt' in args.data_roots and args.seqs.get('tartanrgbt'):
        ds_root = args.data_roots['tartanrgbt']
        if os.path.exists(ds_root):
            try:
                ds = TartanRGBTSequentialDataset(
                    data_root=ds_root, stride=args.stride, 
                    max_pairs_per_seq=args.n_eval_pairs)
                for target_seq in args.seqs['tartanrgbt']:
                    sp = [p for p in ds._pairs if target_seq in p[0]]
                    if sp:
                        seq_pairs[f"[Tartan] {target_seq}"] = sp
            except Exception:
                pass

    print()
    for name, sp in seq_pairs.items():
        print(f"  {name}: {len(sp)} ペア抽出完了")
        
    if not seq_pairs:
        print("\n[ERROR] 有効なシーケンスがありません。パス設定と seqs 設定を確認してください。"); return

    # ── モデルロード ─────────────────────────────────────────────────────
    print("\n[Config A] XFeat(RGB)  + Matcher(RGB)")
    xfeat_rgb    = load_xfeat(args.xfeat_rgb, device)
    matcher_rgb  = load_matcher(args.matcher_rgb, device)

    print("\n[Config B] XFeat(Therm)+ Matcher(RGB)")
    print(f"DEBUG: Loading Thermal XFeat from {args.xfeat_therm}")
    xfeat_therm  = load_xfeat(args.xfeat_therm, device)

    print("\n[Config C] XFeat(RGB)  + Matcher(Therm)")
    matcher_therm = load_matcher(args.matcher_therm, device)

    print("\n[Config D] XFeat(Therm)+ Matcher(Therm)\n")

    configs = [
        {'name': 'A: XFeat(RGB)+LG(RGB)',
         'xfeat': xfeat_rgb,   'matcher': matcher_rgb},
        {'name': 'B: XFeat(Therm)+LG(RGB)',
         'xfeat': xfeat_therm, 'matcher': matcher_rgb},
        {'name': 'C: XFeat(RGB)+LG(Therm)',
         'xfeat': xfeat_rgb,   'matcher': matcher_therm},
        {'name': 'D: XFeat(Therm)+LG(Therm)',
         'xfeat': xfeat_therm, 'matcher': matcher_therm},
    ]

    if args.compare_pipeline:
        print("\n[Config E] XFeat(Pipe) + Matcher(Pipe)\n")
        xfeat_pipe  = load_xfeat(args.xfeat_pipe_therm, device)
        matcher_pipe = load_matcher(args.matcher_pipe_therm, device)
        configs.append({
            'name': 'E: XFeat(Pipe)+LG(Pipe)',
            'xfeat': xfeat_pipe, 'matcher': matcher_pipe
        })

    # ── 定量評価（シーケンスごと）────────────────────────────────────────
    print(f"[定量評価] n_eval_pairs={args.n_eval_pairs} / シーケンス\n")
    
    header = f"  {'Model':<25} {'Seq':<30} {'MS(Mean)':>8} {'MS(Min)':>8} {'MS(Max)':>8} {'AUC@5':>8} {'AUC@10':>8} {'AUC@20':>8} {'Prec':>8}"
    print(header)

    for cfg in configs:
        per_seq: Dict[str, Dict] = {}
        for sn, sp in seq_pairs.items():
            m = evaluate_config(cfg['xfeat'], cfg['matcher'], sp, device,
                                args.n_eval_pairs, args.max_kp, args.inlier_thr)
            per_seq[sn] = m

        keys = ['MS_mean', 'MS_min', 'MS_max', 'Prec@3px', 'PoseAUC@5', 'PoseAUC@10', 'PoseAUC@20']
        avg  = {k: float(np.mean([v[k] for v in per_seq.values()])) for k in keys}
        avg['n_match'] = float(np.mean([v['n_match'] for v in per_seq.values()]))
        cfg['per_seq'] = per_seq
        cfg['metrics'] = avg

    all_seqs = list(seq_pairs.keys()) + ['avg']
    csv_rows = []

    for sn in all_seqs:
        print("  " + "-" * 118)
        for cfg in configs:
            m = cfg['metrics'] if sn == 'avg' else cfg['per_seq'][sn]
            row = f"  {cfg['name'][:25]:<25} {sn[:30]:<30} {m['MS_mean']*100:>7.1f}% {m['MS_min']*100:>7.1f}% {m['MS_max']*100:>7.1f}% {m['PoseAUC@5']*100:>7.1f}% {m['PoseAUC@10']*100:>7.1f}% {m['PoseAUC@20']*100:>7.1f}% {m['Prec@3px']*100:>7.1f}%"
            print(row)
            
            csv_rows.append({
                'Model': cfg['name'], 'Seq': sn,
                'MS_mean(%)': round(m['MS_mean']*100, 2),
                'MS_min(%)': round(m['MS_min']*100, 2),
                'MS_max(%)': round(m['MS_max']*100, 2),
                'Prec@3px(%)': round(m['Prec@3px']*100, 2),
                'n_match': round(m['n_match'], 0),
                'PoseAUC@5(%)': round(m['PoseAUC@5']*100, 2),
                'PoseAUC@10(%)': round(m['PoseAUC@10']*100, 2),
                'PoseAUC@20(%)': round(m['PoseAUC@20']*100, 2)
            })
    print("  " + "-" * 118)

    if args.save_csv:
        csv_path = os.path.join(args.output_dir, 'metrics.csv')
        with open(csv_path, 'w') as f:
            f.write('Model,Seq,MS_mean(%),MS_min(%),MS_max(%),Prec@3px(%),n_match,'
                    'PoseAUC@5(%),PoseAUC@10(%),PoseAUC@20(%)\n')
            for r in csv_rows:
                f.write(f"{r['Model']},{r['Seq']},{r['MS_mean(%)']},{r['MS_min(%)']},{r['MS_max(%)']},"
                        f"{r['Prec@3px(%)']},"
                        f"{r['n_match']},{r['PoseAUC@5(%)']},"
                        f"{r['PoseAUC@10(%)']},{r['PoseAUC@20(%)']}\n")
        print(f"\n  CSV 保存: {csv_path}")

    # ── 可視化 ────────────────────────────────────────────────────────
    if args.save_vis:
        print(f"\n[可視化] {args.n_vis_pairs} ペア × {len(seq_pairs)} シーケンス")
        for sn, sp in seq_pairs.items():
            # ファイルパスとして安全な名前に置換
            safe_sn = sn.replace('[', '').replace('] ', '_')
            seq_dir = os.path.join(args.output_dir, safe_sn)
            os.makedirs(seq_dir, exist_ok=True)
            print(f"  [{sn}]")
            for i, (p0, p1, T_rel_t, K_t) in enumerate(sp[:args.n_vis_pairs]):
                T_rel = np.array(T_rel_t, dtype=np.float64)
                K     = np.array(K_t,     dtype=np.float64)
                for cfg in configs:
                    cfg['metrics'] = cfg['per_seq'][sn]
                out = os.path.join(seq_dir, f'pair_{i:03d}.png')
                visualize_pair(configs, p0, p1, T_rel, K, out,
                               args.max_kp, args.inlier_thr, device)

    print(f"\n{'='*60}")
    print(f"  評価完了  →  {args.output_dir}/")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()