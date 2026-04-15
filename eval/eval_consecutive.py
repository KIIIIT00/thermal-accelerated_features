"""
eval/eval_consecutive.py
同一シーン内の連続フレームペアによるマッチング精度評価。

【評価原理】
    データセットのシーケンス構造を正確に解析し、
    同一シーケンス内の連続フレームペア (frame_n, frame_n+1) を構築する。

    異なるシーケンスのフレームを組み合わせると：
        → ホモグラフィが存在しない（シーンが違う）
        → 誤差が常に inf → AUC が不正確
    のため、シーケンス内の連続性を厳密に保証する。

【各データセットのシーケンス区切り方】
    Freiburg:   splits_dir 内の各 .txt ファイル = 1サブシーケンス
                txt ファイル名: train_seq_00_day_00.txt
                → seq_00_day/00/ フォルダ内の連続フレーム

    STheReO:    splits_dir 内の各 _frame_pairs.txt = 1シーケンス
                kaist_afternoon_frame_pairs.txt → KAIST/Afternoon/
                → txt 内の行順が時系列順

    TartanRGBT: sequence.yaml の各ラベル = 1シーケンス
                indoor_SQH_office/ 内の RGB_aligned/ → ファイル名ソート順

    VIVID:      splits_dir 内の各サブディレクトリ = 1シーケンス
                frame_lists/driving_full/campus_day1/ → rgb_framelist.txt の行順

    MS2:        sync_data/{seq}/ 内の各シーケンス
                → ファイル名ソート順

【使用方法】
    python eval/eval_consecutive.py --config configs/eval_config.yaml
    python eval/eval_consecutive.py --config configs/eval_config.yaml \\
        --datasets sthereo freiburg --stride 1 --n_viz 10
    python eval/eval_consecutive.py --config configs/eval_config.yaml \\
        --matching_method lightglue --max_pairs_per_seq 200
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
        description='Thermal XFeat — Consecutive Frame Evaluation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--config',             type=str, required=True)
    parser.add_argument('--student_weights',    type=str, default=None)
    parser.add_argument('--teacher_weights',    type=str, default=None)
    parser.add_argument('--datasets',           nargs='+', default=None)
    parser.add_argument('--split',              type=str, default=None)
    parser.add_argument('--stride',             type=int, default=None,
                        help='フレーム間隔（1=直前フレーム、5=5フレーム後）')
    parser.add_argument('--max_pairs_per_seq',  type=int, default=None,
                        help='1シーケンスあたりの最大評価ペア数')
    parser.add_argument('--n_viz',              type=int, default=None)
    parser.add_argument('--output_dir',         type=str, default=None)
    parser.add_argument('--max_keypoints',      type=int, default=None)
    parser.add_argument('--matching_method',    type=str, default=None,
                        choices=['mutual_nn', 'ratio_test', 'lightglue'])
    parser.add_argument('--device_num',         type=str, default=None)

    cli = parser.parse_args()
    if not os.path.isfile(cli.config):
        parser.error(f'--config not found: {cli.config!r}')

    with open(cli.config) as f:
        cfg = yaml.safe_load(f) or {}

    for k, v in vars(cli).items():
        if k != 'config' and v is not None:
            cfg[k] = v

    defaults = dict(
        datasets          = ['sthereo'],
        split             = 'val',
        stride            = 1,
        max_pairs_per_seq = 200,
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
    )
    for k, v in defaults.items():
        cfg.setdefault(k, v)

    return argparse.Namespace(**cfg)


# ---------------------------------------------------------------------------
# シーケンス区切りを厳密に管理する SequencePairBuilder
# ---------------------------------------------------------------------------

class SequencePairBuilder:
    """
    データセットのシーケンス構造を解析して
    同一シーケンス内の連続フレームペアを構築する。

    返す形式:
        sequences: List[List[Tuple[str, str]]]
            = [ [(rgb1, thr1), (rgb2, thr2), ...],  ← シーケンス1
                [(rgb1, thr1), (rgb2, thr2), ...],  ← シーケンス2
                ... ]
    """

    @staticmethod
    def build(name: str, args, split: str) -> List[List[Tuple[str, str]]]:
        name_l = name.lower()
        method_map = {
            'freiburg':   SequencePairBuilder._freiburg,
            'sthereo':    SequencePairBuilder._sthereo,
            'tartanrgbt': SequencePairBuilder._tartanrgbt,
            'vivid':      SequencePairBuilder._vivid,
            'ms2':        SequencePairBuilder._ms2,
        }
        if name_l not in method_map:
            raise ValueError(f"Unknown dataset: {name!r}")
        return method_map[name_l](args, split)

    @staticmethod
    def _get_roots(name, args):
        from modules.dataset.thermal.loader import _resolve_data_root, _resolve_splits_dir
        return (_resolve_data_root(name, args),
                _resolve_splits_dir(name, args))

    # ── Freiburg ────────────────────────────────────────────────────────────
    @staticmethod
    def _freiburg(args, split: str) -> List[List[Tuple[str, str]]]:
        """
        splits_dir の各 .txt ファイル = 1サブシーケンス
        txt ファイル内の行順 = 時系列順（タイムスタンプ順にソート済み）

        区切りの根拠:
            train_seq_00_day_00.txt と train_seq_00_day_01.txt は
            同じ seq_00_day でも異なる走行区間なのでシーケンスを分ける。
        """
        data_root, splits_dir = SequencePairBuilder._get_roots('freiburg', args)

        _VAL_PREFIXES = ('train_seq_01_night', 'train_seq_02_day')

        sequences = []
        for txt_name in sorted(os.listdir(splits_dir)):
            if not txt_name.endswith('.txt'):
                continue
            stem   = txt_name[:-4]
            is_val = any(stem.startswith(p) for p in _VAL_PREFIXES)
            if (split == 'val') != is_val:
                continue

            items    = stem.split('_')
            split_d  = items[0]
            seq_name = '_'.join(items[1:-1])
            subseq   = items[-1]
            seq_dir  = os.path.join(data_root, split_d, seq_name, subseq)
            rgb_dir  = os.path.join(seq_dir, 'fl_rgb')
            thr_dir  = os.path.join(seq_dir, 'thermal8_clahe')

            if not os.path.isdir(rgb_dir) or not os.path.isdir(thr_dir):
                continue

            seq_pairs = []
            with open(os.path.join(splits_dir, txt_name)) as f:
                for line in f:
                    frame = line.strip()
                    if not frame:
                        continue
                    rp = os.path.join(rgb_dir, f'fl_rgb_{frame}')
                    tp = os.path.join(thr_dir, f'fl_ir_aligned_{frame}')
                    if os.path.isfile(rp) and os.path.isfile(tp):
                        seq_pairs.append((rp, tp))

            if seq_pairs:
                sequences.append(seq_pairs)
                print(f"  [Freiburg] seq={stem}: {len(seq_pairs)} frames")

        return sequences

    # ── STheReO ─────────────────────────────────────────────────────────────
    @staticmethod
    def _sthereo(args, split: str) -> List[List[Tuple[str, str]]]:
        """
        splits_dir の各 _frame_pairs.txt = 1シーケンス
        txt ファイル内の行順 = 時系列順

        区切りの根拠:
            kaist_afternoon と kaist_morning は
            異なる時間帯の撮影なのでシーケンスを分ける。
        """
        data_root, splits_dir = SequencePairBuilder._get_roots('sthereo', args)

        _SEQ_NAME_TO_DIR = {
            'snu_morning':      'SNU/Morning',    'snu_afternoon':    'SNU/Afternoon',
            'snu_evening':      'SNU/Evening',    'valley_morning':   'Valley/Morning',
            'valley_afternoon': 'Valley/Afternoon','valley_evening':  'Valley/Evening',
            'kaist_morning':    'KAIST/Morning',  'kaist_afternoon':  'KAIST/Afternoon',
            'kaist_evening':    'KAIST/Evening',
        }
        _TRAIN = ('snu_', 'valley_')
        _VAL   = ('kaist_',)

        sequences = []
        for fname in sorted(os.listdir(splits_dir)):
            if not fname.endswith('_frame_pairs.txt'):
                continue
            seq_name = fname[:-len('_frame_pairs.txt')]
            is_val   = any(seq_name.startswith(p) for p in _VAL)
            if (split == 'val') != is_val:
                continue

            seq_dir = _SEQ_NAME_TO_DIR.get(seq_name, '')
            if not seq_dir:
                continue
            rgb_dir = os.path.join(data_root, seq_dir, 'image', 'stereo_left')
            thr_dir = os.path.join(data_root, seq_dir, 'image', 'thermal8_left_clahe')

            if not os.path.isdir(rgb_dir) or not os.path.isdir(thr_dir):
                continue

            seq_pairs = []
            with open(os.path.join(splits_dir, fname)) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 2:
                        continue
                    rp = os.path.join(rgb_dir, parts[0])
                    tp = os.path.join(thr_dir, parts[1])
                    if os.path.isfile(rp) and os.path.isfile(tp):
                        seq_pairs.append((rp, tp))

            if seq_pairs:
                sequences.append(seq_pairs)
                print(f"  [STheReO] seq={seq_name}: {len(seq_pairs)} frames")

        return sequences

    # ── TartanRGBT ──────────────────────────────────────────────────────────
    @staticmethod
    def _tartanrgbt(args, split: str) -> List[List[Tuple[str, str]]]:
        """
        sequence.yaml の各ラベル = 1シーケンス
        フレームはファイル名のソート順（ゼロパディング番号順）

        区切りの根拠:
            indoor_SQH_office と outdoor_campus_NSH_TO_CUT は
            完全に異なるシーンなのでシーケンスを分ける。
        """
        import yaml as _yaml
        data_root, splits_dir = SequencePairBuilder._get_roots('tartanrgbt', args)

        _VAL_LABELS = frozenset([
            'indoor_outdoor_mill19_building_interior_exterior', 'indoor_CFA_seq_2',
            'urban_resedential_frick_park',
            'park_frick_seq_4_return_tranquil_trail_start_to_falls_ravine',
            'park_frick_seq_7_deer_creek_trail_and_nine_mile_start',
            'offroad_turnpike_seq_1', 'offroad_turnpike_seq_4',
            'offroad_turnpike_seq_3', 'offroad_turnpike_seq_2',
        ])

        yaml_path = os.path.join(splits_dir, 'sequence.yaml')
        with open(yaml_path) as f:
            raw = _yaml.safe_load(f)
        seq_map = raw.get('traj_list', raw)

        sequences = []
        for key, label in seq_map.items():
            if not isinstance(label, str):
                continue
            is_val = label in _VAL_LABELS
            if (split == 'val') != is_val:
                continue

            day_prefix = key.split('/')[0]
            seq_dir = os.path.join(data_root, day_prefix, label)
            if not os.path.isdir(seq_dir):
                seq_dir = os.path.join(data_root, key)
            if not os.path.isdir(seq_dir):
                continue

            # RGB ディレクトリを探す
            rgb_dir = None
            for cand in ('RGB_aligned_with_thermal', 'rgb_in_thermal'):
                c = os.path.join(seq_dir, cand)
                if os.path.isdir(c):
                    rgb_dir = c
                    break
            thr_dir = os.path.join(seq_dir, 'thermal_left_rect_8')
            if rgb_dir is None or not os.path.isdir(thr_dir):
                continue

            # FFC フレームをスキップ
            ffc_set = set()
            ffc_path = os.path.join(seq_dir, 'thermal_left_ffc', 'data.txt')
            if os.path.isfile(ffc_path):
                with open(ffc_path) as f:
                    for i, ln in enumerate(f):
                        if ln.strip() == '1':
                            ffc_set.add(i)

            rgb_files = sorted(
                f for f in os.listdir(rgb_dir)
                if f.lower().endswith(('.png', '.jpg'))
            )
            seq_pairs = []
            for i, fname in enumerate(rgb_files):
                if i in ffc_set:
                    continue
                stem     = os.path.splitext(fname)[0]
                thr_stem = stem.replace('_rgb_in_thermal', '')
                thr_fn   = thr_stem + os.path.splitext(fname)[1]
                rp = os.path.join(rgb_dir, fname)
                tp = os.path.join(thr_dir, thr_fn)
                if os.path.isfile(rp) and os.path.isfile(tp):
                    seq_pairs.append((rp, tp))

            if seq_pairs:
                sequences.append(seq_pairs)
                print(f"  [TartanRGBT] seq={label}: {len(seq_pairs)} frames")

        return sequences

    # ── VIVID ────────────────────────────────────────────────────────────────
    @staticmethod
    def _vivid(args, split: str) -> List[List[Tuple[str, str]]]:
        """
        splits_dir の各サブディレクトリ（group/seq）= 1シーケンス
        rgb_framelist.txt の行順 = 時系列順

        区切りの根拠:
            campus_day1 と campus_day2 は別日の撮影なので分ける。
            campus_day1 内の時系列連続性は framelist の行順で保証。
        """
        data_root, splits_dir = SequencePairBuilder._get_roots('vivid', args)

        sequences = []
        for group in sorted(os.listdir(splits_dir)):
            group_dir = os.path.join(splits_dir, group)
            if not os.path.isdir(group_dir):
                continue
            for seq in sorted(os.listdir(group_dir)):
                seq_dir = os.path.join(group_dir, seq)
                if not os.path.isdir(seq_dir):
                    continue

                is_val = 'campus' in f"{group}/{seq}"
                if (split == 'val') != is_val:
                    continue

                rgb_list = os.path.join(seq_dir, 'rgb_framelist.txt')
                thr_list = os.path.join(seq_dir, 'thermal_framelist.txt')
                if not os.path.isfile(rgb_list) or not os.path.isfile(thr_list):
                    continue

                with open(rgb_list) as f:
                    rgb_rels = [l.strip() for l in f if l.strip()]
                with open(thr_list) as f:
                    thr_rels = [l.strip() for l in f if l.strip()]

                seq_pairs = []
                for rr, tr in zip(rgb_rels, thr_rels):
                    rp = os.path.join(data_root, rr)
                    tp = os.path.join(data_root, tr)
                    if os.path.isfile(rp) and os.path.isfile(tp):
                        seq_pairs.append((rp, tp))

                if seq_pairs:
                    sequences.append(seq_pairs)
                    print(f"  [VIVID] seq={group}/{seq}: {len(seq_pairs)} frames")

        return sequences

    # ── MS2 ─────────────────────────────────────────────────────────────────
    @staticmethod
    def _ms2(args, split: str) -> List[List[Tuple[str, str]]]:
        """
        sync_data/{seq}/ 内の各シーケンス = 1シーケンス
        ファイル名ソート順 = 時系列順（ゼロパディング）

        区切りの根拠:
            residential_day と residential_night は
            異なる時間帯の走行なので分ける。
        """
        data_root, _ = SequencePairBuilder._get_roots('ms2', args)

        _VAL_SEQS = [
            'residential_day1', 'residential_day2',
            'residential_night1', 'residential_night2',
        ]
        _ALL_SEQS = [
            'urban_day1', 'urban_day2', 'urban_night1',
            'campus_day1', 'campus_day2', 'campus_night1',
            'residential_day1', 'residential_day2',
            'residential_night1', 'residential_night2',
        ]

        sequences = []
        for seq in _ALL_SEQS:
            is_val = seq in _VAL_SEQS
            if (split == 'val') != is_val:
                continue

            rgb_dir = os.path.join(data_root, 'sync_data', seq, 'rgb',  'img_left')
            thr_dir = os.path.join(data_root, 'sync_data', seq, 'thr',  'img_left')
            if not os.path.isdir(rgb_dir) or not os.path.isdir(thr_dir):
                continue

            rgb_files = sorted(
                f for f in os.listdir(rgb_dir)
                if f.lower().endswith('.png'))
            seq_pairs = []
            for fname in rgb_files:
                rp = os.path.join(rgb_dir, fname)
                tp = os.path.join(thr_dir, fname)
                if os.path.isfile(rp) and os.path.isfile(tp):
                    seq_pairs.append((rp, tp))

            if seq_pairs:
                sequences.append(seq_pairs)
                print(f"  [MS2] seq={seq}: {len(seq_pairs)} frames")

        return sequences


# ---------------------------------------------------------------------------
# 連続フレームペアを生成
# ---------------------------------------------------------------------------

def make_consecutive_pairs_from_sequences(
    sequences: List[List[Tuple[str, str]]],
    stride: int,
    max_per_seq: Optional[int],
    seed: int,
) -> List[Tuple[Tuple[str, str], Tuple[str, str], str]]:
    """
    各シーケンスから (frame_n, frame_n+stride) のペアを生成する。

    Returns:
        [(frame_a, frame_b, seq_id), ...]
        ※ frame_a と frame_b は同一シーケンス内のみ
    """
    all_pairs = []
    for seq_idx, seq in enumerate(sequences):
        seq_id = f"seq_{seq_idx:03d}"
        pairs  = []
        for i in range(len(seq) - stride):
            pairs.append((seq[i], seq[i + stride], seq_id))

        if max_per_seq and len(pairs) > max_per_seq:
            # ランダムサブサンプリング（シーケンス内の連続性は保たれる）
            rng = random.Random(seed + seq_idx)
            pairs = rng.sample(pairs, max_per_seq)

        all_pairs.extend(pairs)

    return all_pairs


# ---------------------------------------------------------------------------
# モデルロード・検出・マッチング（eval_homo_aug.py と共通）
# ---------------------------------------------------------------------------

def load_models(args, device):
    models = {}
    for role, attr in [('teacher', 'teacher_weights'),
                       ('student', 'student_weights')]:
        m = XFeatModel().to(device).eval()
        w = getattr(args, attr, None)
        if w and os.path.isfile(w):
            m.load_state_dict(torch.load(w, map_location=device, weights_only=True))
            print(f"[Consec] {role}: {w}")
        else:
            print(f"[Consec] WARNING: {attr} not found")
        models[role] = m
    return models


def imread_thermal_tensor(path, device, size):
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(path)
    gray = cv2.resize(gray, size)
    bgr  = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    t    = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
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

    # mutual_nn
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


def homography_error_ransac(k1, k2, i1, i2, hw):
    """RANSAC でホモグラフィを推定して再投影誤差を計算。"""
    if len(i1) < 4:
        return float('inf')
    H_mat, mask = cv2.findHomography(
        k1[i1].reshape(-1, 1, 2),
        k2[i2].reshape(-1, 1, 2),
        cv2.RANSAC, 5.0)
    if H_mat is None:
        return float('inf')
    h, w = hw
    c = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
    p = cv2.perspectiveTransform(c, H_mat).reshape(-1, 2)
    return float(np.mean(np.linalg.norm(c.reshape(-1,2) - p, axis=1)))


def auc_at(errors, thresholds):
    arr = np.array(errors)
    return {f'AUC@{t}px': float((arr <= t).mean()) for t in thresholds}


# ---------------------------------------------------------------------------
# 定量評価
# ---------------------------------------------------------------------------

def evaluate_dataset(
    name: str,
    pair_list: List[Tuple[Tuple[str,str], Tuple[str,str], str]],
    models: Dict,
    args,
    device: torch.device,
) -> Dict[str, Dict]:
    size = (args.viz_width, args.viz_height)
    hw   = (args.viz_height, args.viz_width)

    buf = {'teacher_thr': {'errors': [], 'ms': []},
           'student_thr': {'errors': [], 'ms': []}}

    for i, (frame_a, frame_b, seq_id) in enumerate(pair_list):
        if (i+1) % 200 == 0:
            print(f"  [{name}] {i+1}/{len(pair_list)} ...")
        try:
            _, thr_a_bgr = imread_thermal_tensor(frame_a[1], device, size)
            _, thr_b_bgr = imread_thermal_tensor(frame_b[1], device, size)
            t_a = torch.from_numpy(
                cv2.cvtColor(thr_a_bgr, cv2.COLOR_BGR2RGB)
            ).permute(2,0,1).float().unsqueeze(0).to(device) / 255.0
            t_b = torch.from_numpy(
                cv2.cvtColor(thr_b_bgr, cv2.COLOR_BGR2RGB)
            ).permute(2,0,1).float().unsqueeze(0).to(device) / 255.0
        except FileNotFoundError:
            continue

        for mn, mdl in [('teacher_thr', models['teacher']),
                        ('student_thr', models['student'])]:
            k1, d1 = detect(mdl, t_a, args.max_keypoints)
            k2, d2 = detect(mdl, t_b, args.max_keypoints)
            i1, i2 = do_match(k1, d1, k2, d2,
                               args.matching_method, args.ratio_threshold, hw, device)
            err = homography_error_ransac(k1, k2, i1, i2, hw)
            ms  = float(len(i1) / max(min(len(k1), len(k2)), 1))
            buf[mn]['errors'].append(err)
            buf[mn]['ms'].append(ms)

    summary = {}
    labels  = {'teacher_thr': 'XFeat(Thr) [KD前]',
               'student_thr': 'Student(Thr) [提案手法]'}
    for mn in buf:
        errs = buf[mn]['errors']
        mss  = buf[mn]['ms']
        r    = auc_at(errs, args.auc_thresholds)
        r['MS']      = float(np.mean(mss)) if mss else 0.0
        r['n_pairs'] = len(errs)
        r['label']   = labels[mn]
        summary[mn]  = r
    return summary


# ---------------------------------------------------------------------------
# 定性評価（可視化）
# ---------------------------------------------------------------------------

def visualize_consecutive(
    frame_a: Tuple[str, str],
    frame_b: Tuple[str, str],
    seq_id: str,
    models: Dict,
    args,
    device: torch.device,
    save_path: str,
) -> None:
    """
    連続フレームのマッチング結果を可視化する。

    レイアウト:
        上段: frame_a（熱）  |  frame_b（熱）
        下段左:  XFeat(Thr)  a↔b マッチング
        下段右:  Student(Thr) a↔b マッチング
    """
    size = (args.viz_width, args.viz_height)
    hw   = (args.viz_height, args.viz_width)
    font = cv2.FONT_HERSHEY_SIMPLEX

    try:
        _, bgr_a = imread_thermal_tensor(frame_a[1], device, size)
        _, bgr_b = imread_thermal_tensor(frame_b[1], device, size)
        t_a = torch.from_numpy(cv2.cvtColor(bgr_a, cv2.COLOR_BGR2RGB)
            ).permute(2,0,1).float().unsqueeze(0).to(device) / 255.0
        t_b = torch.from_numpy(cv2.cvtColor(bgr_b, cv2.COLOR_BGR2RGB)
            ).permute(2,0,1).float().unsqueeze(0).to(device) / 255.0
    except FileNotFoundError as e:
        print(f"  [VIZ SKIP] {e}")
        return

    W = args.viz_width
    configs = [
        ('teacher_thr', models['teacher'], (0, 0, 200),   'XFeat(Thr) [KD前]'),
        ('student_thr', models['student'], (20, 140, 255), 'Student(Thr) [提案]'),
    ]

    rows = []
    for mn, mdl, clr, lbl in configs:
        k1, d1 = detect(mdl, t_a, args.max_keypoints)
        k2, d2 = detect(mdl, t_b, args.max_keypoints)
        i1, i2 = do_match(k1, d1, k2, d2,
                           args.matching_method, args.ratio_threshold, hw, device)

        canvas = np.hstack([bgr_a.copy(), bgr_b.copy()])
        for a, b in zip(i1[:100], i2[:100]):
            x1,y1 = int(k1[a][0]), int(k1[a][1])
            x2,y2 = int(k2[b][0])+W, int(k2[b][1])
            cv2.line(canvas,(x1,y1),(x2,y2),(180,180,0),1,cv2.LINE_AA)
            cv2.circle(canvas,(x1,y1),2,clr,-1)
            cv2.circle(canvas,(x2,y2),2,clr,-1)

        label_txt = f"{lbl}  seq={seq_id}  matches={len(i1)}"
        cv2.putText(canvas,label_txt,(8,26),font,0.55,(0,0,0),   3,cv2.LINE_AA)
        cv2.putText(canvas,label_txt,(8,26),font,0.55,(255,255,255),1,cv2.LINE_AA)
        rows.append(canvas)

    canvas = np.vstack(rows)
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    cv2.imwrite(save_path, canvas)
    print(f"  [VIZ] Saved: {save_path}")


# ---------------------------------------------------------------------------
# 結果表示・保存
# ---------------------------------------------------------------------------

def print_results(all_results: Dict, method: str, stride: int) -> None:
    print()
    print('=' * 80)
    print(f'  CONSECUTIVE FRAME RESULTS  [matcher: {method}  stride: {stride}]')
    print('  ※ 同一シーケンス内の連続フレームペアのみを使用')
    print('=' * 80)
    for ds, res in all_results.items():
        print(f"\n  Dataset: {ds}")
        print(f"  {'Model':<38s} {'AUC@3px':>8s} {'AUC@5px':>8s} {'AUC@10px':>9s} {'MS':>7s} {'pairs':>6s}")
        print(f"  {'-'*80}")
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
                f" {r.get('MS',0)*100:>6.2f}%"
                f" {r.get('n_pairs',0):>6d}"
            )
        if 'teacher_thr' in res and 'student_thr' in res:
            t = res['teacher_thr'].get('AUC@5px', 0) * 100
            s = res['student_thr'].get('AUC@5px', 0) * 100
            sign = '✅' if s > t else '❌'
            print(f"\n  {sign} AUC@5px 改善率: {t:.2f}% → {s:.2f}%  (Δ={s-t:+.2f}%)")


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_arguments()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device_num)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Consec] Device: {device}")
    if device.type == 'cuda':
        print(f"[Consec] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[Consec] Matcher: {args.matching_method}  Stride: {args.stride}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    models     = load_models(args, device)
    output_dir = os.path.join(args.output_dir, 'consecutive')
    os.makedirs(output_dir, exist_ok=True)

    all_results = {}

    for ds_name in args.datasets:
        print(f"\n[Consec] ========== {ds_name} ==========")
        try:
            sequences = SequencePairBuilder.build(ds_name, args, args.split)
        except Exception as e:
            print(f"[Consec] {ds_name} skipped: {e}")
            continue

        if not sequences:
            print(f"[Consec] {ds_name}: No sequences found.")
            continue

        n_seqs  = len(sequences)
        n_total = sum(max(0, len(s) - args.stride) for s in sequences)
        print(f"[Consec] {ds_name}: {n_seqs} sequences, ~{n_total} consecutive pairs")

        pair_list = make_consecutive_pairs_from_sequences(
            sequences, args.stride, args.max_pairs_per_seq, args.seed)
        print(f"[Consec] {ds_name}: Using {len(pair_list)} pairs after subsampling")

        # 定量評価
        all_results[ds_name] = evaluate_dataset(
            ds_name, pair_list, models, args, device)

        # 定性評価
        n_viz = args.n_viz
        if n_viz > 0:
            picks = random.Random(args.seed + 7).sample(
                pair_list, min(n_viz, len(pair_list)))
            for vi, (fa, fb, sid) in enumerate(picks):
                sp = os.path.join(output_dir, ds_name, f'viz_{vi+1:03d}.png')
                visualize_consecutive(fa, fb, sid, models, args, device, sp)

    print_results(all_results, args.matching_method, args.stride)

    save_path = os.path.join(output_dir, 'consecutive_results.json')
    with open(save_path, 'w') as f:
        json.dump(
            {ds: {mn: {k: float(v) if isinstance(v, (float, np.floating)) else v
                       for k, v in r.items()}
                  for mn, r in res.items()}
             for ds, res in all_results.items()},
            f, indent=2)
    print(f"\n[Consec] Results saved → {save_path}")
    print("[Consec] Done.")


if __name__ == '__main__':
    main()