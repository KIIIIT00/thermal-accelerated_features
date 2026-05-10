# """
# modules/dataset/thermal/stage1_kd_datasets.py
# Stage 1 (XFeat Knowledge Distillation) 専用のデータローダー群。

# 【設計思想】
# 1. AnyThermal公式の非対称クロップ＆リサイズ（FOVアラインメント）を完全再現。
# 2. K行列や相対ポーズなどの幾何学情報は不要なためロードしない（I/O高速化）。
# 3. Stage1_KDAugmentWrapper を通すことで、バッチ学習のための固定サイズクロップと、
#    config_master.yaml の設定に基づく動的なデータ拡張（Ablation対応）を適用する。
# """

# import os
# import cv2
# import torch
# import numpy as np
# import random
# import torchvision.transforms.functional as torch_F
# from torch.utils.data import Dataset
# from typing import Dict, List, Tuple

# # =====================================================================
# # 1. 個別データセットクラス (FOVアラインメント専任)
# # =====================================================================

# class Stage1_VIVIDDataset(Dataset):
#     """VIVID: 広角なRGB画像を大きくクロップしてThermalの画角に合わせる"""
#     def __init__(self, pairs: List[Tuple[str, str]]):
#         self.pairs = pairs
#         self.thr_res = (512, 640) # (H, W)

#     def __len__(self): return len(self.pairs)

#     def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
#         rgb_path, thr_path = self.pairs[idx]

#         # --- Thermal (基準: クロップなし) ---
#         thr_img = cv2.imread(thr_path, cv2.IMREAD_GRAYSCALE)
#         if thr_img is None: raise FileNotFoundError(f"[VIVID] Thermal not found: {thr_path}")
#         thr_img = cv2.cvtColor(thr_img, cv2.COLOR_GRAY2RGB)

#         # --- RGB (広角補正クロップ＆リサイズ) ---
#         rgb_img = cv2.imread(rgb_path)
#         if rgb_img is None: raise FileNotFoundError(f"[VIVID] RGB not found: {rgb_path}")
#         rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        
#         h, w = rgb_img.shape[:2]
#         rgb_img = rgb_img[122:h-122, 145:w-108]
#         rgb_img = cv2.resize(rgb_img, (self.thr_res[1], self.thr_res[0]), interpolation=cv2.INTER_AREA)

#         return {
#             'image0': torch.from_numpy(rgb_img).permute(2, 0, 1).float() / 255.0,
#             'image1': torch.from_numpy(thr_img).permute(2, 0, 1).float() / 255.0,
#             'dataset_name': 'vivid'
#         }


# class Stage1_MS2Dataset(Dataset):
#     """MS2: Thermalの静的ノイズをクロップし、RGBをそのサイズに縮小する"""
#     def __init__(self, pairs: List[Tuple[str, str]]):
#         self.pairs = pairs
#         self.thr_res_after_crop = (212, 578) # (H, W)

#     def __len__(self): return len(self.pairs)

#     def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
#         rgb_path, thr_path = self.pairs[idx]

#         # --- Thermal (16-bitRaw読込 -> hist_99 -> CLAHE -> Bilateral -> クロップ) ---
#         thr_raw = cv2.imread(thr_path, cv2.IMREAD_UNCHANGED)
#         if thr_raw is None: raise FileNotFoundError(f"[MS2] Thermal not found: {thr_path}")
#         thr_raw = thr_raw.astype(np.float32)

#         v_min, v_max = np.percentile(thr_raw, [1.0, 99.0])
#         thr_8bit = np.clip((thr_raw - v_min) / (v_max - v_min + 1e-6) * 255.0, 0, 255).astype(np.uint8)
        
#         clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#         thr_8bit = clahe.apply(thr_8bit)
#         thr_8bit = cv2.bilateralFilter(thr_8bit, 5, 20, 15)
#         thr_8bit = cv2.cvtColor(thr_8bit, cv2.COLOR_GRAY2RGB)
        
#         h_t, w_t = thr_8bit.shape[:2]
#         thr_cropped = thr_8bit[9:h_t-35, 28:w_t-34].copy()

#         # --- RGB (Thermalの有効画角にリサイズ) ---
#         rgb_img = cv2.imread(rgb_path)
#         if rgb_img is None: raise FileNotFoundError(f"[MS2] RGB not found: {rgb_path}")
#         rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        
#         rgb_img = cv2.resize(rgb_img, (self.thr_res_after_crop[1], self.thr_res_after_crop[0]), interpolation=cv2.INTER_AREA)

#         return {
#             'image0': torch.from_numpy(rgb_img).permute(2, 0, 1).float() / 255.0,
#             'image1': torch.from_numpy(thr_cropped).permute(2, 0, 1).float() / 255.0,
#             'dataset_name': 'ms2'
#         }


# class Stage1_SThErEODataset(Dataset):
#     """SThErEO: Thermalの歪み補正黒枠をクロップし、RGBをそのサイズに縮小する"""
#     def __init__(self, pairs: List[Tuple[str, str]]):
#         self.pairs = pairs
#         self.thr_res = (284, 558) # (H, W)

#     def __len__(self): return len(self.pairs)

#     def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
#         rgb_path, thr_path = self.pairs[idx]

#         # --- Thermal (無効領域クロップ) ---
#         thr_img = cv2.imread(thr_path, cv2.IMREAD_GRAYSCALE)
#         if thr_img is None: raise FileNotFoundError(f"[SThErEO] Thermal not found: {thr_path}")
#         thr_img = cv2.cvtColor(thr_img, cv2.COLOR_GRAY2RGB)
        
#         h_t, w_t = thr_img.shape[:2]
#         thr_cropped = thr_img[121:h_t-107, 52:w_t-30]

#         # --- RGB (Thermalの有効画角にリサイズ) ---
#         rgb_img = cv2.imread(rgb_path)
#         if rgb_img is None: raise FileNotFoundError(f"[SThErEO] RGB not found: {rgb_path}")
#         rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        
#         rgb_img = cv2.resize(rgb_img, (self.thr_res[1], self.thr_res[0]), interpolation=cv2.INTER_AREA)

#         return {
#             'image0': torch.from_numpy(rgb_img).permute(2, 0, 1).float() / 255.0,
#             'image1': torch.from_numpy(thr_cropped).permute(2, 0, 1).float() / 255.0,
#             'dataset_name': 'sthereo'
#         }


# class Stage1_FreiburgDataset(Dataset):
#     """Freiburg: OpenCV(RGB)とTorchvision(Thr)による異種リサイズの再現"""
#     def __init__(self, pairs: List[Tuple[str, str]]):
#         self.pairs = pairs

#     def __len__(self): return len(self.pairs)

#     def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
#         rgb_path, thr_path = self.pairs[idx]

#         # --- RGB ---
#         rgb_img = cv2.imread(rgb_path)
#         if rgb_img is None: raise FileNotFoundError(f"[Freiburg] RGB not found: {rgb_path}")
#         rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        
#         rgb_img = cv2.resize(rgb_img, (960, 320), interpolation=cv2.INTER_AREA)
#         rgb_img = rgb_img[0:320, 148:858]
#         rgb_tensor = torch.from_numpy(rgb_img).permute(2, 0, 1).float() / 255.0

#         # --- Thermal ---
#         thr_img = cv2.imread(thr_path, cv2.IMREAD_GRAYSCALE)
#         if thr_img is None: raise FileNotFoundError(f"[Freiburg] Thermal not found: {thr_path}")
#         thr_img = cv2.cvtColor(thr_img, cv2.COLOR_GRAY2RGB)
#         thr_tensor = torch.from_numpy(thr_img).permute(2, 0, 1).float() / 255.0
        
#         _, h, w = thr_tensor.shape
#         thr_tensor = torch_F.resize(thr_tensor, (int(h/2), int(w/2)), antialias=True, interpolation=torch_F.InterpolationMode.BILINEAR)
#         thr_tensor = torch_F.resize(thr_tensor, (320, thr_tensor.shape[-1]), antialias=True, interpolation=torch_F.InterpolationMode.BILINEAR)

#         return {
#             'image0': rgb_tensor,
#             'image1': thr_tensor,
#             'dataset_name': 'freiburg'
#         }


# class Stage1_CleanDataset(Dataset):
#     """TartanRGBT等: 既にアライメント済みのクリーンデータ用"""
#     def __init__(self, pairs: List[Tuple[str, str]], dataset_name: str = 'tartanrgbt'):
#         self.pairs = pairs
#         self.dataset_name = dataset_name

#     def __len__(self): return len(self.pairs)

#     def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
#         rgb_path, thr_path = self.pairs[idx]

#         rgb_img = cv2.imread(rgb_path)
#         rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

#         thr_img = cv2.imread(thr_path, cv2.IMREAD_GRAYSCALE)
#         thr_img = cv2.cvtColor(thr_img, cv2.COLOR_GRAY2RGB)

#         return {
#             'image0': torch.from_numpy(rgb_img).permute(2, 0, 1).float() / 255.0,
#             'image1': torch.from_numpy(thr_img).permute(2, 0, 1).float() / 255.0,
#             'dataset_name': self.dataset_name
#         }


# # =====================================================================
# # 2. ラッパー: バッチ化 (Random Crop) と データ拡張マネージャー
# # =====================================================================

# class Stage1_KDAugmentWrapper(Dataset):
#     """
#     config_master.yaml の設定 (aug_list) に基づいて、
#     1. DataLoaderでバッチ化するための共通固定サイズへのランダムクロップ
#     2. 動的なデータ拡張 (Ablation対応)
#     を適用するラッパークラス。
#     """
#     def __init__(self, base_dataset: Dataset, crop_size: Tuple[int, int] = (256, 256), is_train: bool = True, aug_list: List[str] = None):
#         self.base_dataset = base_dataset
#         self.crop_size = crop_size
#         self.is_train = is_train
#         # YAMLから渡されたメソッドのリスト（例: ['flip', 'color', 'cutout']）
#         self.aug_list = aug_list or [] 

#     def __len__(self):
#         return len(self.base_dataset)

#     def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
#         data = self.base_dataset[idx]
#         rgb = data['image0'] # (3, H, W)
#         thr = data['image1'] # (3, H, W)

#         # -------------------------------------------------------------
#         # 1. AnyThermal公式: Aspect-Ratio Preserving Square Crop & Resize
#         # -------------------------------------------------------------
#         _, h, w = rgb.shape
#         target_size = self.crop_size[0] # 例: 256
        
#         # 短辺（通常は高さH）に合わせて正方形のクロップサイズを決定
#         crop_size = min(h, w)
#         max_top = h - crop_size
#         max_left = w - crop_size

#         # RGBとThermalで完全に同じ場所をクロップする
#         top = int(torch.randint(0, max_top + 1, (1,)).item()) if max_top > 0 else 0
#         left = int(torch.randint(0, max_left + 1, (1,)).item()) if max_left > 0 else 0

#         rgb_cropped = torch_F.crop(rgb, top, left, crop_size, crop_size)
#         thr_cropped = torch_F.crop(thr, top, left, crop_size, crop_size)

#         # 切り抜いた「正方形」を、モデル入力用の固定サイズ(256x256)にリサイズする
#         rgb = torch_F.resize(rgb_cropped, (target_size, target_size), antialias=True, interpolation=torch_F.InterpolationMode.BILINEAR)
#         thr = torch_F.resize(thr_cropped, (target_size, target_size), antialias=True, interpolation=torch_F.InterpolationMode.BILINEAR)

#         # -------------------------------------------------------------
#         # 2. データ拡張 (config.yaml の aug_list に基づく適用)
#         # -------------------------------------------------------------
#         if self.is_train and len(self.aug_list) > 0:
#             rgb, thr = self._apply_augmentations(rgb, thr)

#         data['image0'] = rgb
#         data['image1'] = thr
#         return data

#     def _apply_augmentations(self, rgb: torch.Tensor, thr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         # [A] 空間・幾何学拡張 (完全同期)
#         if 'flip' in self.aug_list and random.random() > 0.5:
#             rgb = torch_F.hflip(rgb)
#             thr = torch_F.hflip(thr)

#         if 'affine' in self.aug_list and random.random() > 0.5:
#             # アフィン変換 (回転、並進、スケール、シアー)
#             angle = random.uniform(-10, 10)
#             th, tw = self.crop_size
#             translate = [int(random.uniform(-0.02, 0.02) * tw),
#                          int(random.uniform(-0.02, 0.02) * th)]
#             scale = random.uniform(0.95, 1.05)
#             shear = random.uniform(-5, 5)
            
#             rgb = torch_F.affine(rgb, angle, translate, scale, [shear], interpolation=torch_F.InterpolationMode.BILINEAR)
#             thr = torch_F.affine(thr, angle, translate, scale, [shear], interpolation=torch_F.InterpolationMode.BILINEAR)

#         # [B] 色調・輝度拡張 (非同期: 個別の乱数で照明不変性を学習)
#         if 'color' in self.aug_list:
#             if random.random() > 0.5:
#                 rgb = torch_F.adjust_brightness(rgb, random.uniform(0.7, 1.3))
#                 rgb = torch_F.adjust_contrast(rgb, random.uniform(0.8, 1.2))
            
#             if random.random() > 0.5:
#                 thr = torch_F.adjust_brightness(thr, random.uniform(0.7, 1.3))
#                 thr = torch_F.adjust_contrast(thr, random.uniform(0.8, 1.2))

#         # [C] Cutout (完全同期: 同じ場所を黒塗りして局所特徴の欠損へのロバスト性を上げる)
#         if 'cutout' in self.aug_list and random.random() > 0.5:
#             th, tw = self.crop_size
#             cutout_size = int(0.1 * min(th, tw))
#             x0 = random.randint(0, tw - cutout_size)
#             y0 = random.randint(0, th - cutout_size)
#             rgb[:, y0:y0+cutout_size, x0:x0+cutout_size] = 0.0
#             thr[:, y0:y0+cutout_size, x0:x0+cutout_size] = 0.0

#         return rgb, thr

"""
modules/dataset/thermal/stage1_kd_datasets.py
Stage 1 (XFeat Knowledge Distillation) 専用のデータローダー群。

【設計思想】
1. AnyThermal公式の非対称クロップ＆リサイズ（FOVアラインメント）を完全再現。
2. 異種混合バッチ対応: 14/16-bitのRawデータを持つデータセットと持たないデータセットを
   1つのバッチに安全に結合するため、ダミーのRawテンソルと `has_raw` フラグを導入。
3. 正方形クロップ: アスペクト比を破壊せずにFOVを合わせるAnyThermalのバッチ化手法を適用。
"""

import os
import cv2
import torch
import numpy as np
import random
import torchvision.transforms.functional as torch_F
from torch.utils.data import Dataset
from typing import Dict, List, Tuple

# =====================================================================
# 1. 個別データセットクラス (FOVアラインメント & Rawフラグ管理)
# =====================================================================

class Stage1_VIVIDDataset(Dataset):
    """VIVID: 16-bit Rawあり。広角なRGB画像をクロップしてThermalの画角に合わせる"""
    def __init__(self, pairs: List[Tuple[str, str]]):
        self.pairs = pairs

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rgb_path, thr_8bit_path = self.pairs[idx]

        # --- Thermal (8-bit表示用 と 16-bit Raw用) ---
        thr_8bit = cv2.imread(thr_8bit_path, cv2.IMREAD_GRAYSCALE)
        if thr_8bit is None: raise FileNotFoundError(f"[VIVID] Thermal not found: {thr_8bit_path}")
        thr_8bit = cv2.cvtColor(thr_8bit, cv2.COLOR_GRAY2RGB)

        thr_raw_path = thr_8bit_path.replace("thermal_fieldscale_clahe", "thermal_raw")
        thr_raw = cv2.imread(thr_raw_path, cv2.IMREAD_UNCHANGED) # 16-bit Raw
        if thr_raw is None: raise FileNotFoundError(f"[VIVID] Thermal not found: {thr_raw_path}")

        # --- RGB (広角補正ハードクロップ) ---
        rgb_img = cv2.imread(rgb_path)
        if rgb_img is None: raise FileNotFoundError(f"[VIVID] RGB not found: {rgb_path}")
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        
        h, w = rgb_img.shape[:2]
        rgb_img = rgb_img[122:h-122, 145:w-108] # AnyThermal公式のクロップ座標

        # Thermalの解像度に一旦合わせる (リサイズ)
        h_t, w_t = thr_8bit.shape[:2]
        rgb_img = cv2.resize(rgb_img, (w_t, h_t), interpolation=cv2.INTER_AREA)

        return {
            'image0': torch.from_numpy(rgb_img).permute(2, 0, 1).float() / 255.0,
            'image1': torch.from_numpy(thr_8bit).permute(2, 0, 1).float() / 255.0,
            'image_raw': torch.from_numpy(thr_raw.astype(np.float32)).unsqueeze(0), # (1, H, W)
            'has_raw': torch.tensor(True, dtype=torch.bool), # 🎯 Rawあり
            'dataset_name': 'vivid'
        }


class Stage1_MS2Dataset(Dataset):
    """MS2: Rawなし(8-bit)。Thermalの静的ノイズをクロップし、RGBをそのサイズに縮小する"""
    def __init__(self, pairs: List[Tuple[str, str]]):
        self.pairs = pairs

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rgb_path, thr_path = self.pairs[idx]

        # --- Thermal (ノイズ除去と無効領域クロップ) ---
        thr_raw = cv2.imread(thr_path, cv2.IMREAD_UNCHANGED)
        if thr_raw is None: raise FileNotFoundError(f"[MS2] Thermal not found: {thr_path}")
        thr_raw = thr_raw.astype(np.float32)

        # 公式の hist_99 完全再現
        v_min, v_max = np.percentile(thr_raw, [1.0, 99.0])
        thr_8bit = np.clip((thr_raw - v_min) / (v_max - v_min + 1e-6) * 255.0, 0, 255).astype(np.uint8)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        thr_8bit = clahe.apply(thr_8bit)
        thr_8bit = cv2.bilateralFilter(thr_8bit, 5, 20, 15)
        thr_8bit = cv2.cvtColor(thr_8bit, cv2.COLOR_GRAY2RGB)
        
        h_t, w_t = thr_8bit.shape[:2]
        thr_cropped = thr_8bit[9:h_t-35, 28:w_t-34].copy() # AnyThermal公式クロップ
        thr_raw_cropped = thr_raw[9:h_t-35, 28:w_t-34].copy()

        # --- RGB (Thermalの有効画角にリサイズ) ---
        rgb_img = cv2.imread(rgb_path)
        if rgb_img is None: raise FileNotFoundError(f"[MS2] RGB not found: {rgb_path}")
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        
        h_c, w_c = thr_cropped.shape[:2]
        rgb_img = cv2.resize(rgb_img, (w_c, h_c), interpolation=cv2.INTER_AREA)

        thr_tensor = torch.from_numpy(thr_cropped).permute(2, 0, 1).float() / 255.0

        return {
            'image0': torch.from_numpy(rgb_img).permute(2, 0, 1).float() / 255.0,
            'image1': thr_tensor,
            'image_raw': torch.from_numpy(thr_raw_cropped).unsqueeze(0),
            'has_raw': torch.tensor(True, dtype=torch.bool), # 🎯 Rawあり
            'dataset_name': 'ms2'
        }


class Stage1_SThErEODataset(Dataset):
    """SThErEO: 14-bit Rawあり。Thermalの歪み補正黒枠をクロップし、RGBをそのサイズに縮小する"""
    def __init__(self, pairs: List[Tuple[str, str]]):
        self.pairs = pairs

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rgb_path, thr_8bit_path = self.pairs[idx]

        # --- Thermal (無効領域クロップ) ---
        thr_8bit = cv2.imread(thr_8bit_path, cv2.IMREAD_GRAYSCALE)
        if thr_8bit is None: raise FileNotFoundError(f"[SThErEO] Thermal not found: {thr_8bit_path}")
        thr_8bit = cv2.cvtColor(thr_8bit, cv2.COLOR_GRAY2RGB)

        thr_raw_path = thr_8bit_path.replace("thermal8_left_clahe", "stereo_thermal_14_left")
        
        thr_raw = cv2.imread(thr_raw_path, cv2.IMREAD_UNCHANGED) # 14-bit Raw
        if thr_raw is None: raise FileNotFoundError(f"[SThErEO] 14-bit Raw not found: {thr_raw_path}")
        
        h_t, w_t = thr_8bit.shape[:2]
        thr_cropped = thr_8bit[121:h_t-107, 52:w_t-30]
        raw_cropped = thr_raw[121:h_t-107, 52:w_t-30]

        # --- RGB (Thermalの有効画角にリサイズ) ---
        rgb_img = cv2.imread(rgb_path)
        if rgb_img is None: raise FileNotFoundError(f"[SThErEO] RGB not found: {rgb_path}")
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        
        h_c, w_c = thr_cropped.shape[:2]
        rgb_img = cv2.resize(rgb_img, (w_c, h_c), interpolation=cv2.INTER_AREA)

        return {
            'image0': torch.from_numpy(rgb_img).permute(2, 0, 1).float() / 255.0,
            'image1': torch.from_numpy(thr_cropped).permute(2, 0, 1).float() / 255.0,
            'image_raw': torch.from_numpy(raw_cropped.astype(np.float32)).unsqueeze(0),
            'has_raw': torch.tensor(True, dtype=torch.bool), # 🎯 Rawあり
            'dataset_name': 'sthereo'
        }


class Stage1_FreiburgDataset(Dataset):
    """Freiburg: Rawなし(8-bit)。OpenCV(RGB)とTorchvision(Thr)による異種リサイズの再現"""
    def __init__(self, pairs: List[Tuple[str, str]]):
        self.pairs = pairs

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rgb_path, thr_path = self.pairs[idx]

        # --- RGB ---
        rgb_img = cv2.imread(rgb_path)
        if rgb_img is None: raise FileNotFoundError(f"[Freiburg] RGB not found: {rgb_path}")
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        
        rgb_img = cv2.resize(rgb_img, (960, 320), interpolation=cv2.INTER_AREA)
        rgb_img = rgb_img[0:320, 148:858]
        rgb_tensor = torch.from_numpy(rgb_img).permute(2, 0, 1).float() / 255.0

        # --- Thermal ---
        thr_img = cv2.imread(thr_path, cv2.IMREAD_GRAYSCALE)
        if thr_img is None: raise FileNotFoundError(f"[Freiburg] Thermal not found: {thr_path}")
        thr_img = cv2.cvtColor(thr_img, cv2.COLOR_GRAY2RGB)
        thr_tensor = torch.from_numpy(thr_img).permute(2, 0, 1).float() / 255.0
        
        _, h, w = thr_tensor.shape
        thr_tensor = torch_F.resize(thr_tensor, (int(h/2), int(w/2)), antialias=True, interpolation=torch_F.InterpolationMode.BILINEAR)
        thr_tensor = torch_F.resize(thr_tensor, (320, thr_tensor.shape[-1]), antialias=True, interpolation=torch_F.InterpolationMode.BILINEAR)

        # RGBをThermalの幅に合わせる (Freiburgの独特な仕様を吸収)
        rgb_tensor = torch_F.resize(rgb_tensor, (thr_tensor.shape[1], thr_tensor.shape[2]), antialias=True)

        return {
            'image0': rgb_tensor,
            'image1': thr_tensor,
            'image_raw': thr_tensor.clone()[:1, :, :], # 🎯 ダミー
            'has_raw': torch.tensor(False, dtype=torch.bool), # 🎯 Rawなし
            'dataset_name': 'freiburg'
        }


class Stage1_CleanDataset(Dataset):
    """TartanRGBT等: 16-bit Rawあり。既にアライメント済みのクリーンデータ用"""
    def __init__(self, pairs: List[Tuple[str, str]], dataset_name: str = 'tartanrgbt'):
        self.pairs = pairs
        self.dataset_name = dataset_name

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rgb_path, thr_8bit_path = self.pairs[idx]

        # # if self.dataset_name == 'tartanrgbt':
        #     # 🎯 1. RGBパスを『thermalにアラインされた専用フォルダ』へ置換
        #     # zed_left_rect などの標準RGBフォルダから RGB_aligned_with_thermal へ
        # rgb_path = rgb_path.replace("zed_left_rect", "RGB_aligned_with_thermal")
        
        # # 🎯 2. Thermalは 8bit (Left) をそのまま使用
        # # 16bit(Right)は視差があるため、ここでは「Rawなし」として扱う
        # has_raw_flag = False
        # thr_raw_path = thr_8bit_path 
        # # else:
        # #     # 他のクリーンデータセット（あれば）のデフォルト処理
        # #     has_raw_flag = True
        # #     thr_raw_path = thr_8bit_path.replace("_8", "_16")

        # # --- 画像のロード ---
        # rgb_img = cv2.imread(rgb_path)
        # if rgb_img is None: 
        #     # フェイルセーフ：置換に失敗した場合は元のパスを試す
        #     rgb_img = cv2.imread(self.pairs[idx][0])
        #     rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        # else:
        #     rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

        # thr_8bit = cv2.imread(thr_8bit_path, cv2.IMREAD_GRAYSCALE)
        # if thr_8bit is None: raise FileNotFoundError(f"Thermal not found: {thr_8bit_path}")
        # thr_8bit = cv2.cvtColor(thr_8bit, cv2.COLOR_GRAY2RGB)
        
        # # Rawデータの処理（TartanRGBTの場合は 8bit をダミーとして送るが flag で無視される）
        # thr_raw = cv2.imread(thr_raw_path, cv2.IMREAD_UNCHANGED) if has_raw_flag else None
        # if thr_raw is None:
        #     thr_raw = cv2.imread(thr_8bit_path, cv2.IMREAD_UNCHANGED) # ダミーとして8bitをロード

        # return {
        #     'image0': torch.from_numpy(rgb_img).permute(2, 0, 1).float() / 255.0,
        #     'image1': torch.from_numpy(thr_8bit).permute(2, 0, 1).float() / 255.0,
        #     'image_raw': torch.from_numpy(thr_raw.astype(np.float32)).unsqueeze(0),
        #     'has_raw': torch.tensor(has_raw_flag, dtype=torch.bool), # 🎯 TartanRGBTはFalseになる
        #     'dataset_name': self.dataset_name
        # }

        rgb_img = cv2.imread(rgb_path)
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

        thr_8bit = cv2.imread(thr_8bit_path, cv2.IMREAD_GRAYSCALE)
        thr_8bit = cv2.cvtColor(thr_8bit, cv2.COLOR_GRAY2RGB)
        
        # Rawデータの処理（TartanRGBTの場合は 8bit をダミーとして送るが flag で無視される）
        has_raw_flag = False
        thr_raw = cv2.imread(thr_raw_path, cv2.IMREAD_UNCHANGED) if has_raw_flag else None
        if thr_raw is None:
            thr_raw = cv2.imread(thr_8bit_path, cv2.IMREAD_UNCHANGED) # ダミーとして8bitをロード
        
        return {
            'image0': torch.from_numpy(rgb_img).permute(2, 0, 1).float() / 255.0,
            'image1': torch.from_numpy(thr_8bit).permute(2, 0, 1).float() / 255.0,
            'image_raw': torch.from_numpy(thr_raw.astype(np.float32)).unsqueeze(0),
            'has_raw': torch.tensor(has_raw_flag, dtype=torch.bool), # 🎯 TartanRGBTはFalseになる
            'dataset_name': self.dataset_name
        }


# =====================================================================
# 2. ラッパー: アスペクト比保存クロップ と 安全なデータ拡張
# =====================================================================

class Stage1_KDAugmentWrapper(Dataset):
    def __init__(self, base_dataset: Dataset, crop_size: Tuple[int, int] = (256, 256), is_train: bool = True, aug_list: List[str] = None):
        self.base_dataset = base_dataset
        self.crop_size = crop_size
        self.is_train = is_train
        self.aug_list = aug_list or [] 

    def __len__(self): return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = self.base_dataset[idx]
        rgb = data['image0']
        thr = data['image1']
        raw = data['image_raw']
        
        _, h, w = rgb.shape
        target_size = self.crop_size[0]
        
        # 歪みを防ぐ正方形クロップ
        crop_s = min(h, w)
        max_top = h - crop_s
        max_left = w - crop_s

        top = int(torch.randint(0, max_top + 1, (1,)).item()) if max_top > 0 else 0
        left = int(torch.randint(0, max_left + 1, (1,)).item()) if max_left > 0 else 0

        # RGB, 8-bit, Raw のすべてを【完全に同じ座標】でクロップ
        rgb_cropped = torch_F.crop(rgb, top, left, crop_s, crop_s)
        thr_cropped = torch_F.crop(thr, top, left, crop_s, crop_s)
        raw_cropped = torch_F.crop(raw, top, left, crop_s, crop_s)

        # モデル入力用の固定サイズ(256x256)にリサイズ
        rgb = torch_F.resize(rgb_cropped, (target_size, target_size), antialias=True, interpolation=torch_F.InterpolationMode.BILINEAR)
        thr = torch_F.resize(thr_cropped, (target_size, target_size), antialias=True, interpolation=torch_F.InterpolationMode.BILINEAR)
        raw = torch_F.resize(raw_cropped, (target_size, target_size), antialias=True, interpolation=torch_F.InterpolationMode.BILINEAR)

        # データ拡張 (Rawデータ保護仕様)
        if self.is_train and len(self.aug_list) > 0:
            # [A] 空間・幾何学拡張 (RGB, 8-bit, Raw すべて同期)
            if 'flip' in self.aug_list and random.random() > 0.5:
                rgb, thr, raw = torch_F.hflip(rgb), torch_F.hflip(thr), torch_F.hflip(raw)

            if 'affine' in self.aug_list and random.random() > 0.5:
                angle = random.uniform(-10, 10)
                th, tw = target_size, target_size
                translate = [int(random.uniform(-0.02, 0.02) * tw), int(random.uniform(-0.02, 0.02) * th)]
                scale = random.uniform(0.95, 1.05)
                shear = random.uniform(-5, 5)
                rgb = torch_F.affine(rgb, angle, translate, scale, [shear], interpolation=torch_F.InterpolationMode.BILINEAR)
                thr = torch_F.affine(thr, angle, translate, scale, [shear], interpolation=torch_F.InterpolationMode.BILINEAR)
                raw = torch_F.affine(raw, angle, translate, scale, [shear], interpolation=torch_F.InterpolationMode.BILINEAR)

            # [B] 色調拡張 (⚠️ Rawデータは絶対に変更しない)
            if 'color' in self.aug_list:
                if random.random() > 0.5:
                    rgb = torch_F.adjust_brightness(rgb, random.uniform(0.7, 1.3))
                    rgb = torch_F.adjust_contrast(rgb, random.uniform(0.8, 1.2))
                if random.random() > 0.5:
                    thr = torch_F.adjust_brightness(thr, random.uniform(0.7, 1.3))
                    thr = torch_F.adjust_contrast(thr, random.uniform(0.8, 1.2))

            # [C] Cutout (RGB, 8-bit, Raw すべて同期)
            if 'cutout' in self.aug_list and random.random() > 0.5:
                cutout_size = int(0.1 * target_size)
                x0, y0 = random.randint(0, target_size - cutout_size), random.randint(0, target_size - cutout_size)
                rgb[:, y0:y0+cutout_size, x0:x0+cutout_size] = 0.0
                thr[:, y0:y0+cutout_size, x0:x0+cutout_size] = 0.0
                raw[:, y0:y0+cutout_size, x0:x0+cutout_size] = 0.0

        data['image0'] = rgb
        data['image1'] = thr
        data['image_raw'] = raw
        return data