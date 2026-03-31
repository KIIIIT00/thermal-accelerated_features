"""
modules/dataset/thermal/tartanrgbt.py
TartanRGBT データセット。

AnyThermal tartanrgbt_dataset.py の return_tartanrgbt_split() に完全準拠。

ディレクトリ構造（ダウンロード済みデータ）:
    {data_root}/
    └── day{N}/
        └── {label}/              例: indoor_SQH_office, outdoor_campus_NSH_TO_CUT
            ├── RGB_aligned_with_thermal/
            │   └── {frame:08d}_rgb_in_thermal.png
            ├── thermal_left_rect_8/
            │   └── {frame:08d}.png
            └── thermal_left_ffc/
                └── data.txt      (1=FFC フレーム → 除外)

スプリットファイルの場所:
    AnyThermal 使用時: {anythermal}/custom_datasets/tartanRGBT/splits/sequence.yaml
    スタンドアロン時:  {data_root}/splits/sequence.yaml

sequence.yaml の構造:
    traj_list:
      day1/undistorted_images_all_cameras_20250822_115703: indoor_SQH_office
      ...
    キー = 元の長いディレクトリ名（参照用）
    値   = ラベル名（実際のディレクトリ名）

val/train 分割（AnyThermal return_tartanrgbt_split() に完全準拠）:
    train: 31シーケンス（屋内・屋外・オフロード・公園）
    val:   9シーケンス（mill19・turnpike等）
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Set, Tuple

import cv2
import torch
import yaml
from torch import Tensor

from modules.dataset.thermal.base import ThermalDatasetBase


# ---------------------------------------------------------------------------
# AnyThermal return_tartanrgbt_split() に完全準拠
# ---------------------------------------------------------------------------

_TRAIN_LABELS: Set[str] = {
    'indoor_NSH_third_floor', 'indoor_NSH_fourth_floor',
    'indoor_NSH_first_floor', 'indoor_SQH_office',
    'outdoor_campus_NSH_TO_CUT', 'outdoor_resedential_SQH_block',
    'outdoor_urban_road_campus_to_marget_morrison',
    'indoor_GATES_garage_1', 'indoor_GATES_garage_3',
    'indoor_GATES_seq_1', 'indoor_GATES_seq_2',
    'outdoor_urban_road_mill_19_seq_1',
    'outdoor_urban_mill_19_circle_building_seq_1',
    'outdoor_urban_road_mill_19_seq_2',
    'outdoor_urban_mill_19_exterior',
    'park_frick_seq_1_riverview_trail',
    'park_frick_seq_3_openarea_to_tranquil_trail',
    'park_frick_seq_5_return_falls_ravine_to_riverview',
    'park_frick_seq_6_return_reverview',
    'park_frick_seq_8_nine_mile_run',
    'offroad_figure_eight_morning_1', 'offroad_figure_eight_morning_2',
    'offroad_warehouse_loop_morning', 'offroad_warehouse_fence_morning',
    'offroad_rough_rider_afternoon',
    'offroad_figure_eight_morning_3', 'offroad_figure_eight_morning_4',
    'offroad_figure_eight_rough_rider_start',
    'offroad_warehouse_loop_evening', 'offroad_warehouse_to_garage',
    'offroad_figure_eight_to_fence_to_garage',
}  # 31 シーケンス

_VAL_LABELS: Set[str] = {
    'indoor_outdoor_mill19_building_interior_exterior',
    'indoor_CFA_seq_2',
    'urban_resedential_frick_park',
    'park_frick_seq_4_return_tranquil_trail_start_to_falls_ravine',
    'park_frick_seq_7_deer_creek_trail_and_nine_mile_start',
    'offroad_turnpike_seq_1', 'offroad_turnpike_seq_4',
    'offroad_turnpike_seq_3', 'offroad_turnpike_seq_2',
}  # 9 シーケンス


class TartanRGBTDataset(ThermalDatasetBase):
    """
    Args:
        data_root:  day1/, day2/, ... が置かれたディレクトリ (TARTANRGBT_ROOT)
        splits_dir: sequence.yaml が置かれたディレクトリ
                    None → data_root/splits/ を使う
    """

    def __init__(
        self,
        data_root: str,
        splits_dir: Optional[str] = None,
        split: str = 'train',
        augment: bool = True,
        aug_list: Optional[List[str]] = None,
        p_diurnal_inversion: float = 0.3,
    ):
        self.data_root = data_root
        resolved_splits = splits_dir if splits_dir is not None \
            else os.path.join(data_root, 'splits')
        super().__init__(
            splits_dir=resolved_splits,
            split=split,
            augment=augment,
            aug_list=aug_list,
            p_diurnal_inversion=p_diurnal_inversion,
        )

    def _load_sequence_yaml(self) -> Dict[str, str]:
        """
        sequence.yaml を読み込んで {dir_name: label} の辞書を返す。
        トップレベルキー 'traj_list' を剥がす。
        """
        yaml_path = os.path.join(self.splits_dir, 'sequence.yaml')
        if not os.path.isfile(yaml_path):
            raise RuntimeError(
                f"[TartanRGBT] sequence.yaml not found: {yaml_path}\n"
                f"  splits_dir={self.splits_dir}")
        with open(yaml_path) as f:
            raw = yaml.safe_load(f)
        if isinstance(raw, dict) and 'traj_list' in raw:
            return raw['traj_list']
        return raw

    def _load_ffc_set(self, seq_dir: str) -> Set[int]:
        ffc_path = os.path.join(seq_dir, 'thermal_left_ffc', 'data.txt')
        ffc_set: Set[int] = set()
        if not os.path.isfile(ffc_path):
            return ffc_set
        with open(ffc_path) as f:
            for i, line in enumerate(f):
                if line.strip() == '1':
                    ffc_set.add(i)
        return ffc_set

    def _resolve_seq_dir(self, key: str, label: str) -> str:
        """
        sequence.yaml のキーと値から実際のディレクトリパスを解決する。
        day_prefix/label の形式で存在する場合はそれを使用。
        """
        day_prefix = key.split('/')[0]
        candidate = os.path.join(self.data_root, day_prefix, label)
        if os.path.isdir(candidate):
            return candidate
        return os.path.join(self.data_root, key)

    _RGB_DIR_CANDIDATES = ('RGB_aligned_with_thermal', 'rgb_in_thermal')

    def _find_rgb_dir(self, seq_dir: str) -> Optional[str]:
        for name in self._RGB_DIR_CANDIDATES:
            path = os.path.join(seq_dir, name)
            if os.path.isdir(path):
                return path
        return None

    def _build_pairs(self) -> List[Tuple[str, str]]:
        """
        AnyThermal return_tartanrgbt_split() に準拠した val/train 分離を実装。

        label がどちらのセットに属するかで split を判定する。
        どちらにも属さないラベル（debug, discard 等）はスキップする。
        """
        seq_map = self._load_sequence_yaml()
        target_labels = _TRAIN_LABELS if self.split == 'train' else _VAL_LABELS
        pairs: List[Tuple[str, str]] = []

        for key, label in seq_map.items():
            if not isinstance(label, str):
                continue

            # val/train 判定: ラベルが対象セットに含まれるもののみ処理
            if label not in target_labels:
                continue

            seq_dir  = self._resolve_seq_dir(key, label)
            rgb_dir  = self._find_rgb_dir(seq_dir)
            thr_dir  = os.path.join(seq_dir, 'thermal_left_rect_8')

            if rgb_dir is None or not os.path.isdir(thr_dir):
                continue

            ffc_set = self._load_ffc_set(seq_dir)
            rgb_files = sorted(
                f for f in os.listdir(rgb_dir)
                if 'rgb_in_thermal' in f and f.lower().endswith(('.png', '.jpg'))
            )
            if not rgb_files:
                rgb_files = sorted(
                    f for f in os.listdir(rgb_dir)
                    if f.lower().endswith(('.png', '.jpg'))
                )

            for i, fname in enumerate(rgb_files):
                if i in ffc_set:
                    continue
                rgb_path = os.path.join(rgb_dir, fname)
                stem = os.path.splitext(fname)[0]
                thr_stem = stem.replace('_rgb_in_thermal', '')
                thr_fname = thr_stem + os.path.splitext(fname)[1]
                thr_path = os.path.join(thr_dir, thr_fname)
                if os.path.isfile(thr_path):
                    pairs.append((rgb_path, thr_path))

        if not pairs:
            raise RuntimeError(
                f"[TartanRGBT] No valid pairs found for split='{self.split}'\n"
                f"  data_root: {self.data_root}\n"
                f"  target_labels ({len(target_labels)}): "
                f"{sorted(target_labels)[:3]}...")
        return pairs

    def _load_rgb(self, path: str) -> Tensor:
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"[TartanRGBT] RGB not found: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

    def _load_thr(self, path: str) -> Tensor:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"[TartanRGBT] Thermal not found: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0