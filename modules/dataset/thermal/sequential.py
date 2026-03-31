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
    [320.0,   0.0, 320.0],
    [  0.0, 320.0, 256.0],
    [  0.0,   0.0,   1.0],
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
        stride: int = 1,
        max_pairs_per_seq: int = 500,
    ):
        self.data_root = data_root
        self.stride    = stride
        splits_dir = splits_dir or os.path.join(data_root, 'splits')

        yaml_path = os.path.join(splits_dir, 'sequence.yaml')
        if not os.path.isfile(yaml_path):
            raise RuntimeError(f"[SeqTartanRGBT] sequence.yaml not found: {yaml_path}")
        with open(yaml_path) as f:
            seq_map: dict = yaml.safe_load(f)

        self._pairs: List[Tuple[str, str, np.ndarray, np.ndarray]] = []
        # (thr_path_t, thr_path_t1, T_rel, K)

        for _, dir_name in seq_map.items():
            seq_dir = os.path.join(data_root, dir_name)
            thr_dir = os.path.join(seq_dir, 'thermal_left_rect_8')
            pose_path = os.path.join(seq_dir, 'pose_left_rect.txt')
            ffc_path  = os.path.join(seq_dir, 'thermal_left_ffc', 'data.txt')
            calib_path = os.path.join(seq_dir, 'calib.yaml')

            if not os.path.isdir(thr_dir):
                continue

            # FFC フレーム除外
            ffc_set: set = set()
            if os.path.isfile(ffc_path):
                with open(ffc_path) as f:
                    for i, line in enumerate(f):
                        if line.strip() == '1':
                            ffc_set.add(i)

            # ファイルリスト
            thr_files = sorted(
                f for f in os.listdir(thr_dir)
                if f.lower().endswith(('.png', '.jpg'))
            )

            # 姿勢ファイル読み込み
            poses: Optional[List[np.ndarray]] = None
            if os.path.isfile(pose_path):
                raw = np.loadtxt(pose_path)
                if raw.ndim == 2 and raw.shape[1] == 7:
                    poses = [_pose_vec_to_mat(row) for row in raw]

            # カメラ行列
            K = _TARTANRGBT_K_DEFAULT.copy()
            if os.path.isfile(calib_path):
                with open(calib_path) as f:
                    calib = yaml.safe_load(f)
                if 'thermal_left' in calib:
                    km = calib['thermal_left'].get('K', None)
                    if km:
                        K = np.array(km, dtype=np.float64).reshape(3, 3)

            count = 0
            for i in range(len(thr_files) - stride):
                j = i + stride
                if i in ffc_set or j in ffc_set:
                    continue

                tp_t  = os.path.join(thr_dir, thr_files[i])
                tp_t1 = os.path.join(thr_dir, thr_files[j])
                if not (os.path.isfile(tp_t) and os.path.isfile(tp_t1)):
                    continue

                # 相対姿勢
                T_rel = np.eye(4, dtype=np.float64)
                if poses and i < len(poses) and j < len(poses):
                    T_rel = _relative_pose(poses[i], poses[j])

                self._pairs.append((tp_t, tp_t1, T_rel, K))
                count += 1
                if count >= max_pairs_per_seq:
                    break

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