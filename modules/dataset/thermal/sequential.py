"""
modules/dataset/thermal/sequential.py
Stage 2 Post-KD 用: 相対姿勢付き連続熱画像フレームデータセット。

設計思想:
  - TartanRGBT / Freiburg のシーケンスデータから
    連続フレームペア (I_t, I_{t+1}) と相対姿勢 T_rel を取得する
  - カメラ内部パラメータ K も提供し、再投影誤差・エピポーラ損失を計算可能にする

出力フォーマット:
  {
    'thr_t'  : (3, H, W)   フレーム t
    'thr_t1' : (3, H, W)   フレーム t+1
    'T_rel'  : (4, 4)      相対姿勢 T_{t→t+1}（カメラ座標系）
    'K'      : (3, 3)      カメラ内部パラメータ
    'valid'  : bool        有効なペアかどうか
  }

対応データセット:
  - TartanRGBT: pose_{left,right}_rect.txt からポーズ取得
  - Freiburg:   シーケンス連続フレームを近似姿勢として使用
                （厳密な姿勢なし → valid=False, エピポーラ損失は基本行列で代替）
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import yaml
from torch import Tensor
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# 姿勢ユーティリティ
# ---------------------------------------------------------------------------

def _pose_vec_to_mat(pose_vec: np.ndarray) -> np.ndarray:
    """
    TartanRGBT 形式の pose ベクトル [tx, ty, tz, qx, qy, qz, qw] を
    4×4 変換行列に変換する（カメラ→ワールド）。
    """
    tx, ty, tz, qx, qy, qz, qw = pose_vec
    # クォータニオン → 回転行列
    R = np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),     1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw),     1 - 2*(qx**2 + qy**2)],
    ], dtype=np.float64)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3,  3] = [tx, ty, tz]
    return T


def _relative_pose(T_a: np.ndarray, T_b: np.ndarray) -> np.ndarray:
    """
    T_a, T_b: カメラ→ワールド変換行列
    Returns: T_{a→b} (フレームa から見たフレームb の変換) 4×4
    """
    return np.linalg.inv(T_b) @ T_a


# TartanRGBT の thermal_left カメラ内部パラメータ（論文記載値）
# 実際は sequence ごとの calib.yaml を参照することが望ましい
_TARTANRGBT_K_DEFAULT = np.array([
    [421.23237248, 0.0,          317.55165969],
    [0.0,          420.80872096, 255.54588954],
    [0.0,          0.0,          1.0         ]
], dtype=np.float64)

# Freiburg thermal の近似内部パラメータ（文献値）
_FREIBURG_K_THERMAL = np.array([
    [381.35, 0.0,   329.35],
    [0.0,   381.35, 205.37],
    [0.0,    0.0,     1.0 ],
], dtype=np.float64)


# ---------------------------------------------------------------------------
# TartanRGBT シーケンスデータセット
# ---------------------------------------------------------------------------
class TartanRGBTSequentialDataset(Dataset):
    """
    TartanRGBT の連続フレームペア + 相対姿勢データセット。

    Args:
        data_root:   TARTANRGBT_ROOT
        splits_dir:  sequence.yaml が置かれたディレクトリ（None → data_root/splits/）
        stride:      フレーム間隔（1=直接隣接フレーム）
        max_pairs_per_seq: シーケンスあたりの最大ペア数（メモリ節約）
    """

    def __init__(
        self,
        data_root: str,
        splits_dir: Optional[str] = None,
        stride: int = 5,
        max_pairs_per_seq: int = 500,
    ):
        self.data_root = data_root
        self.stride    = stride

        self._pairs: List[Tuple[str, str, np.ndarray, np.ndarray]] = []

        # splits_dir が未設定の場合は AnyThermal の公式パスを優先する
        if splits_dir is None:
            anythermal_splits = os.path.join(
                'third_party', 'anythermal',
                'custom_datasets', 'tartanRGBT', 'splits')
            splits_dir = anythermal_splits if os.path.isdir(anythermal_splits) else None

        seq_dirs = self._find_seq_dirs(data_root, splits_dir)
        print(f"[SeqTartanRGBT] {len(seq_dirs)} sequences found")

        for seq_dir in sorted(seq_dirs):
            thr_dir    = os.path.join(seq_dir, 'thermal_left_rect_8')
            ffc_path   = os.path.join(seq_dir, 'thermal_left_ffc', 'data.txt')
            calib_path = os.path.join(seq_dir, 'calib.yaml')
            
            # 【修正ポイント2】正しい姿勢ファイルのパス (odometry/poses.npy)
            odom_path  = os.path.join(seq_dir, 'stereo_depth', 'poses.npy')
            print(f"\n--- [DEBUG] Checking sequence: {seq_dir} ---")
            print(f"  odom_path exists?: {os.path.isfile(odom_path)}")

            if not os.path.isdir(thr_dir):
                continue

            # FFC フレーム除外
            ffc_set: set = set()
            if os.path.isfile(ffc_path):
                with open(ffc_path) as f:
                    for idx_l, line in enumerate(f):
                        if line.strip() == '1':
                            ffc_set.add(idx_l)

            thr_files = sorted(
                f for f in os.listdir(thr_dir)
                if f.lower().endswith(('.png', '.jpg'))
            )
            if len(thr_files) < 2:
                continue

            # 【修正ポイント3】poses.npy のロード処理
            poses: Optional[List[np.ndarray]] = None
            if os.path.isfile(odom_path):
                try:
                    # np.load で .npy ファイルを読み込む
                    raw_poses = np.load(odom_path)
                    print(f"\n--- DEBUG: {seq_dir} ---")
                    print(f"raw_poses.shape: {raw_poses.shape}")
                    print(f"raw_poses[0]: {raw_poses[0]}")

                    ts_path = os.path.join(seq_dir, 'target_timestamps.txt')
                    if os.path.isfile(ts_path):
                        with open(ts_path) as f:
                            ts_lines = f.readlines()
                        print(f"target_timestamps lines: {len(ts_lines)}")
                    else:
                        print("target_timestamps.txt NOT FOUND!")
                        
                    print(f"thermal images count: {len(thr_files)}")
                    print("--------------------------\n")
                    poses = []
                    for row in raw_poses:
                        # TartanRGBTのposes.npyは [tx, ty, tz, qx, qy, qz, qw] の7要素
                        # (万が一タイムスタンプが含まれる8要素の場合は row[1:8] を取得)
                        if len(row) == 7:
                            pose_vec = row
                        elif len(row) >= 8:
                            pose_vec = row[1:8]
                        else:
                            continue
                        poses.append(_pose_vec_to_mat(pose_vec))
                except Exception as e:
                    print(f"[SeqTartanRGBT] Failed to load poses.npy in {seq_dir}: {e}")

            # カメラ行列
            K = _TARTANRGBT_K_DEFAULT.copy()
            if os.path.isfile(calib_path):
                try:
                    with open(calib_path) as f:
                        calib = yaml.safe_load(f)
                    if isinstance(calib, dict) and 'thermal_left' in calib:
                        km = calib['thermal_left'].get('K', None)
                        if km:
                            K = np.array(km, dtype=np.float64).reshape(3, 3)
                except Exception:
                    pass

            count = 0
            for i in range(len(thr_files) - stride):
                j = i + stride
                if i in ffc_set or j in ffc_set:
                    continue

                tp_t  = os.path.join(thr_dir, thr_files[i])
                tp_t1 = os.path.join(thr_dir, thr_files[j])
                if not (os.path.isfile(tp_t) and os.path.isfile(tp_t1)):
                    continue

                T_rel = np.eye(4, dtype=np.float64)
                if poses and i < len(poses) and j < len(poses):
                    T_rel = _relative_pose(poses[i], poses[j])
                    
                    # 【修正ポイント4】並進移動が 0.1m 未満（ホバリング・静止状態）ならペアから除外
                    if np.linalg.norm(T_rel[:3, 3]) < 0.1:
                        continue

                self._pairs.append((tp_t, tp_t1, T_rel, K))
                count += 1
                if count >= max_pairs_per_seq:
                    break

            if count > 0:
                seq_name = os.path.basename(seq_dir)
                print(f"  {seq_name}: {count} pairs (pose={'✓' if poses else '×'})")
    @staticmethod
    def _find_seq_dirs(data_root: str,
                       splits_dir: Optional[str]) -> List[str]:
        """
        thermal_left_rect_8 ディレクトリを持つシーケンスを収集する。
        sequence.yaml がある場合はそれを優先する。
        """
        # sequence.yaml を探す
        for sd in [splits_dir, os.path.join(data_root, 'splits')]:
            if sd and os.path.isfile(os.path.join(sd, 'sequence.yaml')):
                try:
                    with open(os.path.join(sd, 'sequence.yaml')) as f:
                        seq_map = yaml.safe_load(f)
                    traj_map = seq_map.get('traj_list', seq_map)
                    dirs = []
                    for key, label in traj_map.items():
                        if not isinstance(label, str):
                            continue
                        day_prefix = key.split('/')[0]
                        seq_dir = os.path.join(data_root, day_prefix, label)
                        if not os.path.isdir(seq_dir):
                            seq_dir = os.path.join(data_root, key)
                        if os.path.isdir(seq_dir):
                            dirs.append(seq_dir)
                    if dirs:
                        return dirs
                except Exception:
                    pass

        # sequence.yaml がない → directory walk で探す
        # TartanRGBT の構造:
        #   data_root/
        #     {scene_name}/
        #       thermal_left_rect_8/   ← このディレクトリがあればシーケンス
        #       pose_left_rect.txt
        result = []
        for name in os.listdir(data_root):
            d = os.path.join(data_root, name)
            if not os.path.isdir(d):
                continue
            # 直下に thermal_left_rect_8 がある場合
            if os.path.isdir(os.path.join(d, 'thermal_left_rect_8')):
                result.append(d)
                continue
            # 1階層下に thermal_left_rect_8 がある場合
            for sub in os.listdir(d):
                sd = os.path.join(d, sub)
                if os.path.isdir(sd) and os.path.isdir(
                        os.path.join(sd, 'thermal_left_rect_8')):
                    result.append(sd)
        return result

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        tp_t, tp_t1, T_rel, K = self._pairs[idx]
        thr_t  = self._read_thr(tp_t)
        thr_t1 = self._read_thr(tp_t1)
        return {
            'thr_t'  : thr_t,
            'thr_t1' : thr_t1,
            'T_rel'  : torch.from_numpy(T_rel).float(),
            'K'      : torch.from_numpy(K).float(),
            'valid'  : torch.tensor(True),
        }

    @staticmethod
    def _read_thr(path: str) -> Tensor:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"[SeqTartanRGBT] not found: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    
# class TartanRGBTSequentialDataset(Dataset):
#     """
#     TartanRGBT の連続フレームペア + 相対姿勢データセット。

#     Args:
#         data_root:   TARTANRGBT_ROOT
#         splits_dir:  sequence.yaml が置かれたディレクトリ（None → data_root/splits/）
#         stride:      フレーム間隔（1=直接隣接フレーム）
#         max_pairs_per_seq: シーケンスあたりの最大ペア数（メモリ節約）
#     """

#     def __init__(
#         self,
#         data_root: str,
#         splits_dir: Optional[str] = None,
#         stride: int = 5,
#         max_pairs_per_seq: int = 500,
#     ):
#         self.data_root = data_root
#         self.stride    = stride

#         self._pairs: List[Tuple[str, str, np.ndarray, np.ndarray]] = []

#         # splits_dir が未設定の場合は AnyThermal の公式パスを優先する
#         if splits_dir is None:
#             anythermal_splits = os.path.join(
#                 'third_party', 'anythermal',
#                 'custom_datasets', 'tartanRGBT', 'splits')
#             splits_dir = anythermal_splits if os.path.isdir(anythermal_splits) else None

#         seq_dirs = self._find_seq_dirs(data_root, splits_dir)
#         print(f"[SeqTartanRGBT] {len(seq_dirs)} sequences found")

#         for seq_dir in sorted(seq_dirs):
#             thr_dir    = os.path.join(seq_dir, 'thermal_left_rect_8')
#             pose_path  = os.path.join(seq_dir, 'pose_left_rect.txt')
#             ffc_path   = os.path.join(seq_dir, 'thermal_left_ffc', 'data.txt')
#             calib_path = os.path.join(seq_dir, 'calib.yaml')

#             if not os.path.isdir(thr_dir):
#                 continue

#             # FFC フレーム除外
#             ffc_set: set = set()
#             if os.path.isfile(ffc_path):
#                 with open(ffc_path) as f:
#                     for idx_l, line in enumerate(f):
#                         if line.strip() == '1':
#                             ffc_set.add(idx_l)

#             thr_files = sorted(
#                 f for f in os.listdir(thr_dir)
#                 if f.lower().endswith(('.png', '.jpg'))
#             )
#             if len(thr_files) < 2:
#                 continue

#             # 姿勢ファイル
#             poses: Optional[List[np.ndarray]] = None
#             if os.path.isfile(pose_path):
#                 try:
#                     raw = np.loadtxt(pose_path)
#                     if raw.ndim == 2 and raw.shape[1] == 7:
#                         poses = [_pose_vec_to_mat(row) for row in raw]
#                 except Exception:
#                     pass

#             # カメラ行列
#             K = _TARTANRGBT_K_DEFAULT.copy()
#             if os.path.isfile(calib_path):
#                 try:
#                     with open(calib_path) as f:
#                         calib = yaml.safe_load(f)
#                     if isinstance(calib, dict) and 'thermal_left' in calib:
#                         km = calib['thermal_left'].get('K', None)
#                         if km:
#                             K = np.array(km, dtype=np.float64).reshape(3, 3)
#                 except Exception:
#                     pass

#             count = 0
#             for i in range(len(thr_files) - stride):
#                 j = i + stride
#                 if i in ffc_set or j in ffc_set:
#                     continue

#                 tp_t  = os.path.join(thr_dir, thr_files[i])
#                 tp_t1 = os.path.join(thr_dir, thr_files[j])
#                 if not (os.path.isfile(tp_t) and os.path.isfile(tp_t1)):
#                     continue

#                 T_rel = np.eye(4, dtype=np.float64)
#                 if poses and i < len(poses) and j < len(poses):
#                     T_rel = _relative_pose(poses[i], poses[j])
#                     # 並進移動が 0.1m 未満ならペアとして採用しない
#                     if np.linalg.norm(T_rel[:3, 3]) < 0.1:
#                         continue

#                 self._pairs.append((tp_t, tp_t1, T_rel, K))
#                 count += 1
#                 if count >= max_pairs_per_seq:
#                     break

#             if count > 0:
#                 seq_name = os.path.basename(seq_dir)
#                 print(f"  {seq_name}: {count} pairs (pose={'✓' if poses else '×'})")

#     @staticmethod
#     def _find_seq_dirs(data_root: str,
#                        splits_dir: Optional[str]) -> List[str]:
#         """
#         thermal_left_rect_8 ディレクトリを持つシーケンスを収集する。
#         sequence.yaml がある場合はそれを優先する。
#         """
#         # sequence.yaml を探す
#         for sd in [splits_dir, os.path.join(data_root, 'splits')]:
#             if sd and os.path.isfile(os.path.join(sd, 'sequence.yaml')):
#                 try:
#                     with open(os.path.join(sd, 'sequence.yaml')) as f:
#                         seq_map = yaml.safe_load(f)
#                     traj_map = seq_map.get('traj_list', seq_map)
#                     dirs = []
#                     for key, label in traj_map.items():
#                         if not isinstance(label, str):
#                             continue
#                         day_prefix = key.split('/')[0]
#                         seq_dir = os.path.join(data_root, day_prefix, label)
#                         if not os.path.isdir(seq_dir):
#                             seq_dir = os.path.join(data_root, key)
#                         if os.path.isdir(seq_dir):
#                             dirs.append(seq_dir)
#                     if dirs:
#                         return dirs
#                 except Exception:
#                     pass

#         # sequence.yaml がない → directory walk で探す
#         # TartanRGBT の構造:
#         #   data_root/
#         #     {scene_name}/
#         #       thermal_left_rect_8/   ← このディレクトリがあればシーケンス
#         #       pose_left_rect.txt
#         result = []
#         for name in os.listdir(data_root):
#             d = os.path.join(data_root, name)
#             if not os.path.isdir(d):
#                 continue
#             # 直下に thermal_left_rect_8 がある場合
#             if os.path.isdir(os.path.join(d, 'thermal_left_rect_8')):
#                 result.append(d)
#                 continue
#             # 1階層下に thermal_left_rect_8 がある場合
#             for sub in os.listdir(d):
#                 sd = os.path.join(d, sub)
#                 if os.path.isdir(sd) and os.path.isdir(
#                         os.path.join(sd, 'thermal_left_rect_8')):
#                     result.append(sd)
#         return result

#     def __len__(self) -> int:
#         return len(self._pairs)

#     def __getitem__(self, idx: int) -> Dict[str, Tensor]:
#         tp_t, tp_t1, T_rel, K = self._pairs[idx]
#         thr_t  = self._read_thr(tp_t)
#         thr_t1 = self._read_thr(tp_t1)
#         return {
#             'thr_t'  : thr_t,
#             'thr_t1' : thr_t1,
#             'T_rel'  : torch.from_numpy(T_rel).float(),
#             'K'      : torch.from_numpy(K).float(),
#             'valid'  : torch.tensor(True),
#         }

#     @staticmethod
#     def _read_thr(path: str) -> Tensor:
#         img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#         if img is None:
#             raise FileNotFoundError(f"[SeqTartanRGBT] not found: {path}")
#         img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0


# ---------------------------------------------------------------------------
# Freiburg シーケンスデータセット（姿勢なし近似）
# ---------------------------------------------------------------------------

class FreiburgSequentialDataset(Dataset):
    """
    Freiburg の連続フレームペアデータセット。
    厳密な姿勢情報がないため valid=False を返し、
    Stage 2 では基本行列（F行列）によるエピポーラ損失のみを使用する。

    Args:
        data_root:  FREIBURG_ROOT
        splits_dir: frame_list txt が置かれたディレクトリ（None → data_root/splits/frame_list/）
        stride:     フレーム間隔
    """

    def __init__(
        self,
        data_root: str,
        splits_dir: Optional[str] = None,
        stride: int = 1,
        split: str = 'train',
    ):
        self.data_root = data_root
        self.stride    = stride
        thr_dir = os.path.join(data_root, 'thermal8_clahe')
        splits_dir = splits_dir or os.path.join(data_root, 'splits', 'frame_list')

        _VAL_PREFIXES = ('train_seq_01_night', 'train_seq_02_day')

        self._pairs: List[Tuple[str, str]] = []

        if not os.path.isdir(splits_dir):
            return

        for txt_name in sorted(os.listdir(splits_dir)):
            if not txt_name.endswith('.txt'):
                continue
            stem = txt_name[:-4]
            is_val = any(stem.startswith(p) for p in _VAL_PREFIXES)
            if (split == 'val') != is_val:
                continue

            with open(os.path.join(splits_dir, txt_name)) as f:
                frames = [l.strip() for l in f if l.strip()]

            for i in range(len(frames) - stride):
                j = i + stride
                tp_t  = os.path.join(thr_dir, f'fl_ir_aligned_{frames[i]}.png')
                tp_t1 = os.path.join(thr_dir, f'fl_ir_aligned_{frames[j]}.png')
                if os.path.isfile(tp_t) and os.path.isfile(tp_t1):
                    self._pairs.append((tp_t, tp_t1))

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        tp_t, tp_t1 = self._pairs[idx]
        thr_t  = self._read_thr(tp_t)
        thr_t1 = self._read_thr(tp_t1)
        return {
            'thr_t'  : thr_t,
            'thr_t1' : thr_t1,
            'T_rel'  : torch.eye(4).float(),           # 不明 → 単位行列
            'K'      : torch.from_numpy(_FREIBURG_K_THERMAL).float(),
            'valid'  : torch.tensor(False),             # 幾何損失は使わない
        }

    @staticmethod
    def _read_thr(path: str) -> Tensor:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"[SeqFreiburg] not found: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0


# ===========================================================================
# SThErEO シーケンスデータセット
# ===========================================================================

_STHEREO_K_DEFAULT = np.array([
    [4.2943288714549999e+02, 0., 3.1111923634459998e+02],
    [0., 4.2953142750190000e+02, 2.6612817575460002e+02],
    [0., 0., 1.],
], dtype=np.float64)


def _load_sthereo_K(calib_path: str) -> np.ndarray:
    """
    SThErEO の thermal_14bit_left.yaml から camera_matrix を読み込む。
    thermal8_left_clahe は未 rectify 画像のため camera_matrix を使用する。
    """
    if not os.path.isfile(calib_path):
        return _STHEREO_K_DEFAULT.copy()
    try:
        with open(calib_path) as f:
            text = f.read()
        # camera_matrix.data: [ ... ] を抽出
        import re as _re
        m = _re.search(
            r'camera_matrix.*?data:\s*\[(.*?)\]', text, _re.DOTALL)
        if m:
            raw = m.group(1).replace('\n', ' ').replace('  ', ' ')
            vals = [float(x.strip()) for x in raw.split(',') if x.strip()]
            if len(vals) == 9:
                return np.array(vals, dtype=np.float64).reshape(3, 3)
    except Exception:
        pass
    return _STHEREO_K_DEFAULT.copy()


def _load_sthereo_poses(pose_csv: str) -> List[Tuple[int, np.ndarray]]:
    """
    SThErEO global_pose.csv を読み込む。

    フォーマット: timestamp_sec, E_utm, N_utm, alt_m, roll_deg, pitch_deg, yaw_deg

    ローカル ENU フレームに変換（最初の座標を原点にオフセット）。
    回転: ZYX Euler (yaw, pitch, roll) → 回転行列

    Returns: [(timestamp_ns, T_world(4x4)), ...] ソート済み
    """
    if not os.path.isfile(pose_csv):
        return []

    result: List[Tuple[int, np.ndarray]] = []
    ref_E = ref_N = ref_alt = None

    with open(pose_csv) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) < 7:
                continue
            try:
                ts_sec    = float(parts[0])
                E, N, alt = float(parts[1]), float(parts[2]), float(parts[3])
                roll_deg  = float(parts[4])
                pitch_deg = float(parts[5])
                yaw_deg   = float(parts[6])
            except ValueError:
                continue

            if ref_E is None:
                ref_E, ref_N, ref_alt = E, N, alt

            t = np.array([E - ref_E, N - ref_N, alt - ref_alt], dtype=np.float64)

            roll  = np.radians(roll_deg)
            pitch = np.radians(pitch_deg)
            yaw   = np.radians(yaw_deg)

            cr, sr = np.cos(roll),  np.sin(roll)
            cp, sp = np.cos(pitch), np.sin(pitch)
            cy, sy = np.cos(yaw),   np.sin(yaw)

            # ZYX Euler: world_R_body = Rz(yaw) @ Ry(pitch) @ Rx(roll)
            Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
            Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
            Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
            R  = Rz @ Ry @ Rx

            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = R
            T[:3,  3] = t
            result.append((int(ts_sec * 1e9), T))

    return result


def _nearest_pose_idx(ts_ns: int,
                      sorted_ts: List[int]) -> int:
    """ソート済みタイムスタンプリストから最近傍インデックスを返す（二分探索）"""
    lo, hi = 0, len(sorted_ts) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if sorted_ts[mid] < ts_ns:
            lo = mid + 1
        else:
            hi = mid
    if lo > 0 and abs(sorted_ts[lo-1] - ts_ns) < abs(sorted_ts[lo] - ts_ns):
        return lo - 1
    return lo


class SThErEOSequentialDataset(Dataset):
    """
    SThErEO の連続フレームペア + GT 相対姿勢データセット。

    GT: pose/global_pose.csv (UTM ENU + ZYX Euler)
    K:  calibration/thermal_14bit_left.yaml (camera_matrix)
    Images: image/thermal8_left_clahe/{timestamp_ns}.png

    Args:
        data_root:  SThErEO ルート（kaist_morning 等が直下にある）
        stride:     フレーム間隔（推奨: 5 以上。1 は baseline が退化して trivial）
        split:      'train' | 'val' | 'all'
        max_dt_ns:  画像-ポーズ タイムスタンプ許容差 [ns]
    """

    _VAL_SEQS = frozenset(['snu_afternoon', 'kaist_morning', 'valley_afternoon'])

    def __init__(
        self,
        data_root:         str,
        stride:            int = 5,
        split:             str = 'val',
        max_dt_ns:         int = 250_000_000,   # 250ms（ポーズ 2.5Hz = 400ms 間隔の半分強）
        max_pairs_per_seq: int = 2000,
    ):
        self.data_root = data_root
        self.stride    = stride
        self._pairs: List[Tuple[str, str, np.ndarray, np.ndarray]] = []

        for seq_name in sorted(os.listdir(data_root)):
            seq_dir = os.path.join(data_root, seq_name)
            if not os.path.isdir(seq_dir):
                continue

            is_val = seq_name in self._VAL_SEQS
            if split == 'train' and is_val:
                continue
            if split == 'val' and not is_val:
                continue

            # K
            calib_path = os.path.join(
                seq_dir, 'calibration', 'thermal_14bit_left.yaml')
            K = _load_sthereo_K(calib_path)

            # ポーズ
            poses = _load_sthereo_poses(
                os.path.join(seq_dir, 'pose', 'global_pose.csv'))
            if len(poses) < 2:
                print(f"[SThErEOSeq] {seq_name}: no pose data")
                continue
            pose_ts  = [p[0] for p in poses]
            pose_Ts  = [p[1] for p in poses]

            # 画像ディレクトリ
            img_dir = os.path.join(seq_dir, 'image', 'thermal8_left_clahe')
            if not os.path.isdir(img_dir):
                img_dir = os.path.join(seq_dir, 'image', 'thermal8_left')
            if not os.path.isdir(img_dir):
                print(f"[SThErEOSeq] {seq_name}: image dir not found")
                continue

            img_files = sorted(f for f in os.listdir(img_dir)
                               if f.endswith('.png'))
            if len(img_files) < 2:
                continue

            # 画像をポーズに対応付け（タイムスタンプ最近傍）
            matched: List[Tuple[str, np.ndarray]] = []
            for fname in img_files:
                try:
                    img_ts_ns = int(fname.split('.')[0])
                except ValueError:
                    continue
                idx = _nearest_pose_idx(img_ts_ns, pose_ts)
                if abs(pose_ts[idx] - img_ts_ns) < max_dt_ns:
                    matched.append((os.path.join(img_dir, fname), pose_Ts[idx]))

            if len(matched) < 2:
                print(f"[SThErEOSeq] {seq_name}: no matched frames")
                continue

            # stride ペアを構築
            n_added = 0
            # for i in range(0, len(matched) - stride, stride):
            #     j = i + stride
            #     p_t,  T_t  = matched[i]
            #     p_t1, T_t1 = matched[j]
            #     T_rel = np.linalg.inv(T_t) @ T_t1
            #     self._pairs.append((p_t, p_t1, T_rel, K))
            for i in range(0, len(matched) - stride, stride):
                j = i + stride
                p_t,  T_t  = matched[i]
                p_t1, T_t1 = matched[j]
                T_rel = np.linalg.inv(T_t) @ T_t1
                
                # --- ここを 0.01 から 0.5 に変更 ---
                if np.linalg.norm(T_rel[:3, 3]) < 0.5:
                    continue
                # -----------------------------------
                
                self._pairs.append((p_t, p_t1, T_rel, K))
                n_added += 1
                if n_added >= max_pairs_per_seq:
                    break

            print(f"[SThErEOSeq] {seq_name}: {n_added} pairs (stride={stride})")

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: int) -> Dict:
        p_t, p_t1, T_rel, K = self._pairs[idx]
        return {
            'thr_t'  : self._read_thr(p_t),
            'thr_t1' : self._read_thr(p_t1),
            'T_rel'  : torch.from_numpy(T_rel).float(),
            'K'      : torch.from_numpy(K).float(),
            'valid'  : torch.tensor(True),
        }

    @staticmethod
    def _read_thr(path: str) -> Tensor:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(path)
        img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR),
                           cv2.COLOR_BGR2RGB)
        return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0


# ===========================================================================
# VIVID シーケンスデータセット
# ===========================================================================

_VIVID_K_DEFAULT = np.array([
    [463.34, 0.0,   320.85],
    [0.0,   463.34, 254.37],
    [0.0,     0.0,    1.0 ],
], dtype=np.float64)


def _quat_to_rot(qx: float, qy: float,
                 qz: float, qw: float) -> np.ndarray:
    """クォータニオン → 3×3 回転行列"""
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    q /= np.linalg.norm(q) + 1e-12
    x, y, z, w = q
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-z*w),   2*(x*z+y*w)],
        [  2*(x*y+z*w), 1-2*(x*x+z*z),   2*(y*z-x*w)],
        [  2*(x*z-y*w),   2*(y*z+x*w), 1-2*(x*x+y*y)],
    ], dtype=np.float64)


def _load_vivid_indoor_gt(pose_csv: str) -> List[Tuple[float, np.ndarray]]:
    """
    VIVID indoor GT を読み込む（モーションキャプチャ）。
    フォーマット: time_sec, x, y, z, qx, qy, qz, qw
    """
    result = []
    if not os.path.isfile(pose_csv):
        return result
    with open(pose_csv) as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 8:
                continue
            try:
                t = float(parts[0])
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                R = _quat_to_rot(
                    float(parts[4]), float(parts[5]),
                    float(parts[6]), float(parts[7]))
            except ValueError:
                continue
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = R
            T[:3,  3] = [x, y, z]
            result.append((t, T))
    return result


def _load_vivid_loam_poses(
    pose_txt: str,
    times_txt: str,
) -> List[Tuple[float, np.ndarray]]:
    """
    VIVID driving LOAM poses を読み込む。

    対応フォーマット:
      A) 12値 per line = 3×4 [R|t] row-major（旧形式）
      B) g2o グラフ形式（新形式）:
           VERTEX_SE3:QUAT id x y z qx qy qz qw
           EDGE_SE3:QUAT ...（スキップ）
    """
    if not os.path.isfile(pose_txt) or not os.path.isfile(times_txt):
        return []

    poses, times = [], []

    # pose ファイルの形式を自動判定
    with open(pose_txt) as f:
        first_line = f.readline().strip()

    is_g2o = first_line.startswith('VERTEX_SE3:QUAT') or              first_line.startswith('EDGE_SE3:QUAT')

    with open(pose_txt) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if is_g2o:
                # g2o 形式: VERTEX_SE3:QUAT id x y z qx qy qz qw
                tokens = line.split()
                if tokens[0] != 'VERTEX_SE3:QUAT' or len(tokens) < 9:
                    continue
                try:
                    x, y, z = float(tokens[2]), float(tokens[3]), float(tokens[4])
                    qx, qy, qz, qw = (float(tokens[5]), float(tokens[6]),
                                       float(tokens[7]), float(tokens[8]))
                except ValueError:
                    continue
                R = _quat_to_rot(qx, qy, qz, qw)
                T = np.eye(4, dtype=np.float64)
                T[:3, :3] = R
                T[:3,  3] = [x, y, z]
                poses.append(T)
            else:
                # 3×4 row-major 形式
                try:
                    vals = list(map(float, line.split()))
                except ValueError:
                    continue
                if len(vals) != 12:
                    continue
                T = np.eye(4, dtype=np.float64)
                T[:3, :4] = np.array(vals).reshape(3, 4)
                poses.append(T)

    with open(times_txt) as f:
        for line in f:
            s = line.strip()
            if s:
                try:
                    times.append(float(s))
                except ValueError:
                    pass

    n = min(len(poses), len(times))
    return [(times[i], poses[i]) for i in range(n)]


class VividSequentialDataset(Dataset):
    """
    VIVID の連続フレームペア + GT 相対姿勢データセット。

    GT 優先順位:
        1. handheld_indoor/pose/*_gt.csv (モーションキャプチャ)
        2. driving_full/loampose/*.txt   (LiDAR SLAM)
        3. handheld_outdoor/pose/*.csv   (LiDAR SLAM)

    画像: extracted_data/{seq_name}/thermal/ または類似ディレクトリ

    Args:
        data_root: VIVID ルートディレクトリ
        stride:    フレーム間隔
        split:     'train' | 'val' | 'all'
    """

    _VAL_PATTERN = 'campus'   # AnyThermal 準拠: campus = val

    def __init__(
        self,
        data_root:         str,
        stride:            int = 5,
        split:             str = 'val',
        max_pairs_per_seq: int = 2000,
    ):
        self.data_root = data_root
        self.stride    = stride
        self._pairs: List[Tuple[str, str, np.ndarray, np.ndarray]] = []

        extracted = os.path.join(data_root, 'extracted_data')
        if not os.path.isdir(extracted):
            print(f"[VividSeq] extracted_data not found at {extracted}")
            return

        # 実際のデータ構造:
        #   extracted_data/{category}/{seq_name}/img/thermal_raw/*.png
        # category = driving_full | driving_vision | handheld_indoor | ...
        # → 2段階でシーケンスを列挙する

        # カテゴリ一覧を取得（driving_full, driving_vision 等）
        categories = sorted(
            c for c in os.listdir(extracted)
            if os.path.isdir(os.path.join(extracted, c)))

        if not categories:
            print(f"[VividSeq] no categories found in {extracted}")
            return

        for category in categories:
            cat_dir = os.path.join(extracted, category)

            for seq_name in sorted(os.listdir(cat_dir)):
                seq_dir = os.path.join(cat_dir, seq_name)
                if not os.path.isdir(seq_dir):
                    continue

                is_val = self._VAL_PATTERN in seq_name.lower()
                if split == 'train' and is_val:
                    continue
                if split == 'val' and not is_val:
                    continue

                # 熱画像ディレクトリを探す
                # 実際のパス: img/thermal_raw/ または thermal/ 等
                thr_dir = None
                for cand in [
                    os.path.join('img', 'thermal_clahe'),  # VIVID: CLAHE 強調（SThErEO と同手法）
                    os.path.join('img', 'thermal_8'),       # VIVID: 8bit 正規化
                    os.path.join('img', 'thermal_raw'),     # VIVID: 生データ
                    'thermal', 'thermal8', 'thermal8_clahe', 'ir',
                ]:
                    d = os.path.join(seq_dir, cand)
                    if os.path.isdir(d):
                        pngs = [f for f in os.listdir(d) if f.endswith('.png')]
                        if pngs:
                            thr_dir = d
                            break

                if thr_dir is None:
                    # 再帰的にサブディレクトリを探す（最大2段）
                    for sub in os.listdir(seq_dir):
                        sub_d = os.path.join(seq_dir, sub)
                        if not os.path.isdir(sub_d):
                            continue
                        pngs = [f for f in os.listdir(sub_d)
                                if f.endswith('.png')]
                        if pngs:
                            thr_dir = sub_d
                            break

                if thr_dir is None:
                    continue

                img_files = sorted(f for f in os.listdir(thr_dir)
                                   if f.endswith('.png'))
                if len(img_files) < 2:
                    continue

                # GT ポーズを探す（category 情報も渡す）
                poses_t, K = self._get_gt(data_root, seq_name, category)
                if not poses_t:
                    print(f"[VividSeq] {category}/{seq_name}: no GT pose → skip")
                    continue

                # 画像とポーズを対応付け
                matched = self._match(thr_dir, img_files, poses_t)
                if len(matched) < 2:
                    continue

                # stride ペアを構築
                n_added = 0
                # for i in range(0, len(matched) - stride, stride):
                #     j = i + stride
                #     p_t,  T_t  = matched[i]
                #     p_t1, T_t1 = matched[j]
                #     T_rel = np.linalg.inv(T_t) @ T_t1
                #     self._pairs.append((p_t, p_t1, T_rel, K))
                for i in range(0, len(matched) - stride, stride):
                    j = i + stride
                    p_t,  T_t  = matched[i]
                    p_t1, T_t1 = matched[j]
                    T_rel = np.linalg.inv(T_t) @ T_t1

                    # --- ここを 0.01 から 0.5 に変更 ---
                    if np.linalg.norm(T_rel[:3, 3]) < 0.5:
                        continue
                    # -----------------------------------

                    self._pairs.append((p_t, p_t1, T_rel, K))
                    n_added += 1
                    if n_added >= max_pairs_per_seq:
                        break

                print(f"[VividSeq] {category}/{seq_name}: "
                      f"{n_added} pairs (stride={stride})")

    @staticmethod
    def _get_gt(
        data_root: str,
        seq_name:  str,
        category:  str = '',
    ) -> Tuple[List[Tuple[float, np.ndarray]], np.ndarray]:
        K = _VIVID_K_DEFAULT.copy()

        # extracted_data/{category}/{seq_name} 内に loampose があれば優先
        # 例: extracted_data/driving_full/campus_day1/loampose/
        cat_seq_loam = os.path.join(
            data_root, 'extracted_data', category, seq_name, 'loampose')
        if os.path.isdir(cat_seq_loam):
            for fname in sorted(os.listdir(cat_seq_loam)):
                if not fname.endswith('_poses.txt') and not fname.endswith('.txt'):
                    continue
                times_cands = ['times.txt', 'time.txt',
                               fname.replace('_poses.txt', '_times.txt')]
                for tc in times_cands:
                    times_path = os.path.join(cat_seq_loam, tc)
                    if os.path.isfile(times_path):
                        poses = _load_vivid_loam_poses(
                            os.path.join(cat_seq_loam, fname), times_path)
                        if poses:
                            return poses, K

        # 1. handheld_indoor mocap GT
        indoor_dir = os.path.join(data_root, 'handheld_indoor', 'pose')
        if os.path.isdir(indoor_dir):
            for fname in sorted(os.listdir(indoor_dir)):
                stem = fname.replace('_gt.csv', '').replace('.csv', '')
                if stem in seq_name or seq_name in stem:
                    poses = _load_vivid_indoor_gt(
                        os.path.join(indoor_dir, fname))
                    if poses:
                        return poses, K

        # 2. driving LOAM
        loam_dir = os.path.join(data_root, 'driving_full', 'loampose')
        if os.path.isdir(loam_dir):
            for fname in sorted(os.listdir(loam_dir)):
                if not fname.endswith('_poses.txt'):
                    continue
                stem = fname.replace('_optimized_poses.txt', '').replace(
                    '_poses.txt', '')
                if stem in seq_name or seq_name.startswith(stem):
                    for times_fname in [
                        stem + '_times.txt', stem + '_time.txt']:
                        times_path = os.path.join(loam_dir, times_fname)
                        if os.path.isfile(times_path):
                            poses = _load_vivid_loam_poses(
                                os.path.join(loam_dir, fname), times_path)
                            if poses:
                                return poses, K

        # 3. handheld_outdoor
        outdoor_dir = os.path.join(data_root, 'handheld_outdoor', 'pose')
        if os.path.isdir(outdoor_dir):
            for fname in sorted(os.listdir(outdoor_dir)):
                if not fname.endswith('.csv'):
                    continue
                stem = fname.replace('.csv', '').replace('path_', '')
                if stem in seq_name or seq_name in stem:
                    poses = _load_vivid_indoor_gt(
                        os.path.join(outdoor_dir, fname))
                    if poses:
                        return poses, K

        return [], K

    @staticmethod
    def _match(
        thr_dir:    str,
        img_files:  List[str],
        poses_t:    List[Tuple[float, np.ndarray]],
        max_dt:     float = 0.1,
    ) -> List[Tuple[str, np.ndarray]]:
        """画像ファイルをポーズに対応付け（タイムスタンプまたは順序）"""
        p_times = [t for t, _ in poses_t]
        p_Ts    = [T for _, T in poses_t]
        matched = []

        # タイムスタンプの種別を判定
        # ファイル名形式の例:
        #   1621839203.519869.png  ← VIVID: 秒単位 float（小数点あり）
        #   1630118400019787687.png ← SThErEO: ナノ秒整数
        #   000001.png             ← 連番
        stem_first = img_files[0].rsplit('.', 1)[0]  # 拡張子を除いた部分全体

        try:
            first_val = float(stem_first)
            # ナノ秒か秒かを値の大きさで判断（1e12 以上はナノ秒）
            use_ts_ns  = first_val > 1e12
            use_ts_sec = not use_ts_ns
        except ValueError:
            use_ts_ns  = False
            use_ts_sec = False

        if use_ts_ns:
            for fname in img_files:
                stem = fname.rsplit('.', 1)[0]
                try:
                    ts_sec = float(stem) / 1e9
                except ValueError:
                    continue
                diffs = [abs(ts_sec - pt) for pt in p_times]
                idx   = int(np.argmin(diffs))
                if diffs[idx] < max_dt:
                    matched.append((os.path.join(thr_dir, fname), p_Ts[idx]))
        elif use_ts_sec:
            # まずスケール一致を確認する
            # VIVID の問題: 画像が Unix 秒（~1.6e9）だが
            # times.txt が相対秒（~1587）の場合がある
            # → 最初の画像と最近傍 pose の差が 1000s 超ならスケール不一致
            stem0 = img_files[0].rsplit('.', 1)[0]
            try:
                ts0 = float(stem0)
                min_diff_check = min(abs(ts0 - pt) for pt in p_times)
                scale_ok = min_diff_check < 1000.0  # 1000秒以内ならスケール一致
            except Exception:
                scale_ok = False

            if scale_ok:
                # 通常のタイムスタンプマッチング
                for fname in img_files:
                    stem = fname.rsplit('.', 1)[0]
                    try:
                        ts_sec = float(stem)
                    except ValueError:
                        continue
                    diffs = [abs(ts_sec - pt) for pt in p_times]
                    idx   = int(np.argmin(diffs))
                    if diffs[idx] < max_dt:
                        matched.append((os.path.join(thr_dir, fname), p_Ts[idx]))
            else:
                # スケール不一致 → インデックスベースマッチング
                # 画像数 と pose 数 の比率で対応付け
                n     = min(len(img_files), len(p_Ts))
                ratio = len(p_Ts) / max(len(img_files), 1)
                for i in range(n):
                    pidx = min(int(i * ratio), len(p_Ts) - 1)
                    matched.append((
                        os.path.join(thr_dir, img_files[i]), p_Ts[pidx]))

        else:
            # 連番インデックスの場合: 等間隔サンプリング
            n = min(len(img_files), len(p_Ts))
            ratio = len(p_Ts) / max(n, 1)
            for i in range(n):
                pidx = min(int(i * ratio), len(p_Ts) - 1)
                matched.append((
                    os.path.join(thr_dir, img_files[i]), p_Ts[pidx]))

        return matched

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: int) -> Dict:
        p_t, p_t1, T_rel, K = self._pairs[idx]
        return {
            'thr_t'  : self._read_thr(p_t),
            'thr_t1' : self._read_thr(p_t1),
            'T_rel'  : torch.from_numpy(T_rel).float(),
            'K'      : torch.from_numpy(K).float(),
            'valid'  : torch.tensor(True),
        }

    @staticmethod
    def _read_thr(path: str) -> Tensor:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(path)
        img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR),
                           cv2.COLOR_BGR2RGB)
        return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0


# ---------------------------------------------------------------------------
# MS2 Sequential Dataset
# ---------------------------------------------------------------------------

_MS2_K_DEFAULT = np.array([
    [517.87, 0.0,   326.81],
    [0.0,   517.87, 261.02],
    [0.0,     0.0,    1.0 ],
], dtype=np.float64)
# NOTE: MS2 のカメラパラメータはシーケンスごとに異なる可能性がある。
#       odom ディレクトリにキャリブレーション情報がなければデフォルト値を使用。
#       実際の K は datasets/ms2/sync_data/{seq}/calib/ で確認すること。

_MS2_TRAIN_SEQS = [
    '_2021-08-06-10-59-33',
    '_2021-08-06-17-44-55',
    '_2021-08-13-17-06-04',
    '_2021-08-13-21-18-04',
    '_2021-08-13-16-50-57',
    '_2021-08-06-16-59-13',
    '_2021-08-13-16-31-10',
    '_2021-08-13-22-16-02',
    '_2021-08-13-16-08-46',
    '_2021-08-13-21-58-13',
    '_2021-08-13-22-36-41',
]

_MS2_VAL_SEQS = [
    '_2021-08-06-11-23-45',
    '_2021-08-06-16-45-28',
    '_2021-08-13-16-14-48',
    '_2021-08-13-22-03-03',
]


# def _load_ms2_frame_pose(pose_txt: str) -> Optional[np.ndarray]:
#     """
#     MS2 の 1フレーム分の pose ファイルを読み込む。

#     実際のデータ構造:
#         odom/{seq}/thr/{idx:06d}.txt  ← フレームごとに個別ファイル
#         例: 000668.txt → frame 668 の pose

#     形式を自動判別:
#         16値: 4×4 変換行列（row-major, 同次座標）
#         12値: 3×4 変換行列（row-major, KITTI 形式）
#          7値: tx ty tz qx qy qz qw（quaternion 形式）
#     """
#     if not os.path.isfile(pose_txt):
#         return None
#     try:
#         with open(pose_txt) as f:
#             raw = f.read()
#         vals = list(map(float, raw.strip().split()))
#     except (ValueError, OSError):
#         return None

#     T = np.eye(4, dtype=np.float64)
#     if len(vals) == 16:
#         T = np.array(vals, dtype=np.float64).reshape(4, 4)
#     elif len(vals) == 12:
#         T[:3, :4] = np.array(vals, dtype=np.float64).reshape(3, 4)
#     elif len(vals) == 7:
#         tx, ty, tz, qx, qy, qz, qw = vals
#         T[:3, :3] = _quat_to_rot(qx, qy, qz, qw)
#         T[:3,  3] = [tx, ty, tz]
#     else:
#         return None

#     return T

def _load_ms2_frame_pose(pose_txt: str) -> Optional[np.ndarray]:
    """
    MS2 の 1フレーム分の pose ファイルを読み込む。
    """
    if not os.path.isfile(pose_txt):
        return None
    try:
        with open(pose_txt) as f:
            raw = f.read()
        vals = list(map(float, raw.strip().split()))
    except (ValueError, OSError):
        return None

    T = np.eye(4, dtype=np.float64)
    if len(vals) == 16:
        T = np.array(vals, dtype=np.float64).reshape(4, 4)
    elif len(vals) == 12:
        T[:3, :4] = np.array(vals, dtype=np.float64).reshape(3, 4)
    elif len(vals) == 7:
        tx, ty, tz, qx, qy, qz, qw = vals
        T[:3, :3] = _quat_to_rot(qx, qy, qz, qw)
        T[:3,  3] = [tx, ty, tz]
    else:
        return None

    return T

def _load_ms2_K(calib_path: str) -> np.ndarray:
    """
    MS2の calib.npy から Thermal Left カメラの内部パラメータ (K_thrL) を読み込む。
    calib.npy が見つからない場合のみデフォルト値にフォールバックする。
    """
    if os.path.isfile(calib_path):
        try:
            # .npy が辞書として保存されているため allow_pickle=True でロード
            calib_data = np.load(calib_path, allow_pickle=True).item()
            if 'K_thrL' in calib_data:
                K_matrix = np.array(calib_data['K_thrL'], dtype=np.float64).reshape(3, 3)
                return K_matrix
            else:
                print(f"[MS2Seq] Warning: 'K_thrL' が {calib_path} 内に見つかりません。")
        except Exception as e:
            print(f"[MS2Seq] Warning: {calib_path} の読み込みに失敗しました ({e})")
    else:
        print(f"[MS2Seq] Warning: キャリブレーションファイル {calib_path} が見つかりません。")
        
    # 失敗した時のみデフォルトを使用
    return _MS2_K_DEFAULT.copy()

class MS2SequentialDataset(Dataset):
    """
    MS2 データセットの連続フレームペア + GT 相対姿勢。

    GT ポーズ:
        odom/{seq}/thr/*.txt  ← 熱画像カメラのオドメトリ・タイムスタンプ
        実際のファイル名はシーケンスによって異なる場合がある。
        odom_thr.txt または *.txt の最初のファイルを使用する。
    画像:
        sync_data/{seq}/thr/img_left/*.png
        → hist_99 + bilateral 前処理済み

    前処理の注意:
        MS2 は hist_99（99パーセンタイル正規化）+ bilateral フィルタを使用。
        SThErEO/VIVID は CLAHE を使用。
        → MS2 の画像は CLAHE をオプションで追加適用して統一可能。
        → apply_clahe=True（デフォルト）で SThErEO と前処理を統一。

    Args:
        data_root:         MS2 ルートディレクトリ
        stride:            フレーム間隔（推奨: 3-5）
        split:             'train' | 'val' | 'all'
        max_pairs_per_seq: 1シーケンスあたりの最大ペア数
        apply_clahe:       True=CLAHE 追加適用（SThErEO との前処理統一）
        clahe_clip_range:  CLAHE clipLimit のランダム範囲 (min, max)
    """

    def __init__(
        self,
        data_root:         str,
        stride:            int = 3,
        split:             str = 'all',
        max_pairs_per_seq: int = 2000,
        apply_clahe:       bool = True,
        clahe_clip_range:  tuple = (1.5, 3.0),
    ):
        self.data_root       = data_root
        self.stride          = stride
        self.apply_clahe     = apply_clahe
        self.clahe_clip_range = clahe_clip_range
        self._pairs: List[Tuple[str, str, np.ndarray, np.ndarray]] = []

        if split == 'all':
            seqs = _MS2_TRAIN_SEQS + _MS2_VAL_SEQS
        elif split == 'train':
            seqs = _MS2_TRAIN_SEQS
        else:
            seqs = _MS2_VAL_SEQS

        sync_root = os.path.join(data_root, 'sync_data')
        odom_root = os.path.join(data_root, 'odom')

        for seq_name in seqs:
            # 画像: sync_data/{seq}/thr/img_left/{idx:06d}.png
            thr_img_dir  = os.path.join(sync_root, seq_name, 'thr', 'img_left')
            # pose:  odom/{seq}/thr/{idx:06d}.txt （フレームごとに個別ファイル）
            thr_pose_dir = os.path.join(odom_root,  seq_name, 'thr')

            if not os.path.isdir(thr_img_dir):
                continue
            if not os.path.isdir(thr_pose_dir):
                print(f"[MS2Seq] {seq_name}: pose dir なし → skip")
                continue

            # 画像ファイル一覧（インデックス順）
            img_files = sorted(
                f for f in os.listdir(thr_img_dir) if f.endswith('.png'))
            if len(img_files) < 2:
                continue

            # フレームインデックスで画像と pose を直接対応付け
            # 000668.png ↔ 000668.txt（タイムスタンプ照合不要）
            matched = []
            for fname in img_files:
                stem      = fname.rsplit('.', 1)[0]   # '000668'
                pose_path = os.path.join(thr_pose_dir, stem + '.txt')
                T = _load_ms2_frame_pose(pose_path)
                if T is not None:
                    matched.append((
                        os.path.join(thr_img_dir, fname), T))

            if len(matched) < 2:
                print(f"[MS2Seq] {seq_name}: matched={len(matched)} → skip")
                continue

            # stride でペアを構築
            calib_path = os.path.join(sync_root, seq_name, 'calib.npy')
            K = _load_ms2_K(calib_path)

            n_added = 0
            # for i in range(0, len(matched) - stride, stride):
            #     j = i + stride
            #     p_t,  T_t  = matched[i]
            #     p_t1, T_t1 = matched[j]
            #     T_rel = np.linalg.inv(T_t) @ T_t1
            #     if np.linalg.norm(T_rel[:3, 3]) < 0.01:
            #         continue
            #     # 読み込んだK行列をペア情報として保存
            #     self._pairs.append((p_t, p_t1, T_rel, K))
            for i in range(0, len(matched) - stride, stride):
                j = i + stride
                p_t,  T_t  = matched[i]
                p_t1, T_t1 = matched[j]
                T_rel = np.linalg.inv(T_t) @ T_t1
                
                # --- ここを 0.01 から 0.5 に変更 ---
                if np.linalg.norm(T_rel[:3, 3]) < 0.5:
                    continue
                # -----------------------------------
                
                self._pairs.append((p_t, p_t1, T_rel, K))
                n_added += 1
                if n_added >= max_pairs_per_seq:
                    break

            print(f"[MS2Seq] {seq_name}: {n_added} pairs (stride={stride})")

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: int) -> Dict:
        p_t, p_t1, T_rel, K = self._pairs[idx]
        return {
            'thr_t'  : self._read_thr(p_t),
            'thr_t1' : self._read_thr(p_t1),
            'T_rel'  : torch.from_numpy(T_rel).float(),
            'K'      : torch.from_numpy(K).float(),
            'valid'  : torch.tensor(True),
        }

    global _DEBUG_MS2_SAVE_COUNT
    _DEBUG_MS2_SAVE_COUNT = 10

    def _read_thr(self, path: str) -> Tensor:
        global _DEBUG_MS2_SAVE_COUNT
        
        # 1. RAW読み込み (8-bitダウンキャストを防ぎ、16-bit等の生データを保持)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"[MS2Seq] not found: {path}")

        img_float = img.astype(np.float32)
        if img_float.ndim == 3:
            img_float = cv2.cvtColor(img_float, cv2.COLOR_BGR2GRAY)
            
        # ==========================================================
        # 【正しい前処理 1】 hist_99 (1%〜99%パーセンタイル正規化)
        # ==========================================================
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

        # ==========================================================
        # 【正しい前処理 2】 enhance_image (CLAHE + Bilateral Filter)
        # ==========================================================
        # MS2の評価時は、パラメータをランダム化せず AnyThermal 固定値を使用
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_img = clahe.apply(image_out)
        # ノイズを平滑化しつつエッジを保持
        img_final = cv2.bilateralFilter(clahe_img, 5, 20, 15)

        # ==========================================================
        # 【正しい前処理 3】 crop (静的ノイズ領域：自車の枠の切り抜き)
        # ==========================================================
        h, w = img_final.shape[:2]
        crop_top, crop_bottom = 9, 35
        crop_left, crop_right = 28, 34
        img_final = img_final[crop_top:h - crop_bottom, crop_left:w - crop_right]

        # ----------------------------------------------------------
        # デバッグ用：前処理が完了した画像を最初の10枚だけディスクに保存
        # ----------------------------------------------------------
        if _DEBUG_MS2_SAVE_COUNT < 10:
            save_dir = "debug_ms2_preprocessed"
            os.makedirs(save_dir, exist_ok=True)
            
            # 元のファイル名を取得して保存名にする
            original_name = os.path.basename(path)
            save_path = os.path.join(save_dir, f"prep_{_DEBUG_MS2_SAVE_COUNT:02d}_{original_name}")
            
            cv2.imwrite(save_path, img_final)
            print(f"[DEBUG] MS2前処理確認用画像を保存しました: {save_path}")
            _DEBUG_MS2_SAVE_COUNT += 1
        # ----------------------------------------------------------

        # XFeatに入力するため、3チャンネル化してテンソル(0.0~1.0)に変換
        img_final = cv2.cvtColor(img_final, cv2.COLOR_GRAY2RGB)
        return torch.from_numpy(img_final).permute(2, 0, 1).float() / 255.0