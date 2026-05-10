# """
# modules/dataset/thermal/stage23_geometry_datasets.py
# Stage 2 & 3 (LightGlue FT / Joint FT) 専用データローダー群。

# 【設計思想】
# 1. VIVID, TartanRGBT: 幾何学的に純粋なデータ。クロップを一切行わず、
#    32の倍数パディングのみを適用し、パディング前の元サイズをマスク用に返す。
# 2. SThErEO: 歪み補正前（Distorted）データのため、ロード時に cv2.undistort() で
#    ピンホールモデル化。その後クロップを適用し、K行列の主点を数学的にシフトする。
# """

# import os
# import cv2
# import yaml
# import torch
# import numpy as np
# import torch.nn.functional as F
# import bisect
# from torch.utils.data import Dataset
# from typing import Dict, Tuple, List, Optional

# # =====================================================================
# # 1. 姿勢・数学ユーティリティ (ポーズ抽出の心臓部)
# # =====================================================================

# def _quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
#     """クォータニオン(x,y,z,w)を 3x3 回転行列に変換"""
#     q = np.array([qx, qy, qz, qw], dtype=np.float64)
#     q /= np.linalg.norm(q) + 1e-12
#     x, y, z, w = q
#     return np.array([
#         [1-2*(y*y+z*z),   2*(x*y-z*w),   2*(x*z+y*w)],
#         [  2*(x*y+z*w), 1-2*(x*x+z*z),   2*(y*z-x*w)],
#         [  2*(x*z-y*w),   2*(y*z+x*w), 1-2*(x*x+y*y)],
#     ], dtype=np.float64)

# def _euler_to_rot(roll: float, pitch: float, yaw: float) -> np.ndarray:
#     """ZYXオイラー角(rad)を回転行列に変換 (SThErEO用)"""
#     rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
#     ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
#     rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
#     return rz @ ry @ rx

# def _nearest_idx(target: int, sorted_list: List[int]) -> int:
#     """タイムスタンプ同期用の近傍検索"""
#     idx = bisect.bisect_left(sorted_list, target)
#     if idx == 0: return 0
#     if idx == len(sorted_list): return len(sorted_list) - 1
#     before = sorted_list[idx - 1]
#     after = sorted_list[idx]
#     return idx if (after - target) < (target - before) else idx - 1

# # =====================================================================
# # 汎用ユーティリティ & 幾何学安全な画像ロード
# # =====================================================================

# def _read_thr_geometric(path: str) -> Tuple[torch.Tensor, torch.Tensor]:
#     """
#     [VIVID / TartanRGBT用]
#     クロップを一切行わず、32の倍数に右下パディングしてロードする。
#     """
#     img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#     if img is None:
#         raise FileNotFoundError(f"File not found: {path}")
        
#     orig_h, orig_w = img.shape[:2]
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
#     tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    
#     pad_h = (32 - (orig_h % 32)) % 32
#     pad_w = (32 - (orig_w % 32)) % 32
#     if pad_h > 0 or pad_w > 0:
#         tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode='replicate')
        
#     return tensor, torch.tensor([orig_w, orig_h])

# def _read_sthereo_geometric(path: str, K_orig: np.ndarray, D: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
#     """
#     [SThErEO用]
#     歪み補正(Undistort) -> クロップ -> K行列補正 -> パディング を実行する。
#     """
#     img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#     if img is None: raise FileNotFoundError(f"File not found: {path}")
        
#     # 1. 歪み補正 (ピンホールカメラモデルへ)
#     if D is not None and np.any(D != 0):
#         img = cv2.undistort(img, K_orig, D)
        
#     # 2. クロップの適用 (Stage 1と同じ領域を切り出す)
#     h, w = img.shape[:2]
#     top, bottom, left, right = 121, 107, 52, 30
#     img = img[top : h - bottom, left : w - right]
    
#     # 3. K行列の数学的補正 (主点のシフト)
#     K_corrected = K_orig.copy()
#     K_corrected[0, 2] -= left  # cx
#     K_corrected[1, 2] -= top   # cy
    
#     orig_h, orig_w = img.shape[:2]
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
#     tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    
#     # 4. パディング
#     pad_h = (32 - (orig_h % 32)) % 32
#     pad_w = (32 - (orig_w % 32)) % 32
#     if pad_h > 0 or pad_w > 0:
#         tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode='replicate')
        
#     return tensor, torch.tensor([orig_w, orig_h]), K_corrected

# def _quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
#     q = np.array([qx, qy, qz, qw], dtype=np.float64)
#     q /= np.linalg.norm(q) + 1e-12
#     x, y, z, w = q
#     return np.array([
#         [1-2*(y*y+z*z),   2*(x*y-z*w),   2*(x*z+y*w)],
#         [  2*(x*y+z*w), 1-2*(x*x+z*z),   2*(y*z-x*w)],
#         [  2*(x*z-y*w),   2*(y*z+x*w), 1-2*(x*x+y*y)],
#     ], dtype=np.float64)


# # =====================================================================
# # 1. TartanRGBT (完全幾何学シミュレーションデータ)
# # =====================================================================

# class Stage23_TartanRGBTDataset(Dataset):
#     def __init__(self, data_root: str, stride: int = 5, max_pairs: int = 500):
#         self.data_root = data_root
#         self._pairs = []
        
#         # デフォルトK行列
#         self.K_default = np.array([
#             [421.23, 0.0, 317.55],
#             [0.0, 420.80, 255.54],
#             [0.0, 0.0, 1.0]
#         ], dtype=np.float64)

#         # anythermal splits を検索 (省略版の実装: 実際の検索ロジックはsequential.pyに準ずる)
#         seq_dirs = self._find_seqs(data_root)
        
#         for seq_dir in seq_dirs:
#             thr_dir = os.path.join(seq_dir, 'thermal_left_rect_8')
#             odom_path = os.path.join(seq_dir, 'stereo_depth', 'poses.npy')
            
#             if not os.path.isdir(thr_dir) or not os.path.isfile(odom_path): continue
            
#             thr_files = sorted([f for f in os.listdir(thr_dir) if f.endswith('.png')])
#             try:
#                 raw_poses = np.load(odom_path)
#                 poses = []
#                 for row in raw_poses:
#                     pose_vec = row if len(row) == 7 else row[1:8]
#                     T = np.eye(4, dtype=np.float64)
#                     T[:3, :3] = _quat_to_rot(pose_vec[3], pose_vec[4], pose_vec[5], pose_vec[6])
#                     T[:3, 3] = pose_vec[:3]
#                     poses.append(T)
#             except Exception: continue

#             count = 0
#             for i in range(len(thr_files) - stride):
#                 j = i + stride
#                 tp_t = os.path.join(thr_dir, thr_files[i])
#                 tp_t1 = os.path.join(thr_dir, thr_files[j])
                
#                 if i < len(poses) and j < len(poses):
#                     T_rel = np.linalg.inv(poses[i]) @ poses[j]
#                     if np.linalg.norm(T_rel[:3, 3]) >= 0.1: # 0.1m以上動いたもののみ
#                         self._pairs.append((tp_t, tp_t1, T_rel, self.K_default))
#                         count += 1
#                         if count >= max_pairs: break

#     def _find_seqs(self, root):
#         res = []
#         for d in os.listdir(root):
#             p = os.path.join(root, d)
#             if os.path.isdir(p) and 'day' in d:
#                 for sub in os.listdir(p):
#                     sub_p = os.path.join(p, sub)
#                     if os.path.isdir(os.path.join(sub_p, 'thermal_left_rect_8')):
#                         res.append(sub_p)
#         return res

#     def __len__(self): return len(self._pairs)

#     def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
#         p_t, p_t1, T_rel, K = self._pairs[idx]
#         thr_t, size_t = _read_thr_geometric(p_t)
#         thr_t1, size_t1 = _read_thr_geometric(p_t1)
        
#         return {
#             'image0': thr_t, 'image1': thr_t1,
#             'orig_size0': size_t, 'orig_size1': size_t1,
#             'T_rel': torch.from_numpy(T_rel).float(),
#             'K': torch.from_numpy(K).float(),
#             'dataset_name': 'tartanrgbt'
#         }


# # =====================================================================
# # 2. VIVID (実世界の高品質LOAM/MoCapポーズ)
# # =====================================================================
# # ※VividDatasetのポーズ取得ロジックは非常に長いため、
# # sequential.pyのVividSequentialDatasetの初期化部をそのまま流用する前提のラッパー。

# from modules.dataset.thermal.sequential import VividSequentialDataset

# class Stage23_VIVIDDataset(VividSequentialDataset):
#     def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
#         p_t, p_t1, T_rel, K = self._pairs[idx]
        
#         # 🎯 元の単純な読み込みを _read_thr_geometric に差し替える
#         thr_t, size_t = _read_thr_geometric(p_t)
#         thr_t1, size_t1 = _read_thr_geometric(p_t1)
        
#         return {
#             'image0': thr_t, 'image1': thr_t1,
#             'orig_size0': size_t, 'orig_size1': size_t1,
#             'T_rel': torch.from_numpy(T_rel).float(),
#             'K': torch.from_numpy(K).float(),
#             'dataset_name': 'vivid'
#         }


# # =====================================================================
# # 3. SThErEO (歪み補正＆K行列シフト必須データセット)
# # =====================================================================

# class Stage23_SThErEODataset(Dataset):
#     def __init__(self, data_root: str, stride: int = 5, split: str = 'val', max_pairs: int = 2000):
#         self.data_root = data_root
#         self.stride = stride
#         self._pairs = []
        
#         _VAL_SEQS = frozenset(['snu_afternoon', 'kaist_morning', 'valley_afternoon'])
        
#         for seq_name in sorted(os.listdir(data_root)):
#             seq_dir = os.path.join(data_root, seq_name)
#             if not os.path.isdir(seq_dir): continue
            
#             is_val = seq_name in _VAL_SEQS
#             if (split == 'train' and is_val) or (split == 'val' and not is_val): continue
            
#             # K行列と歪み係数(D)の読み込み
#             calib_path = os.path.join(seq_dir, 'calibration', 'thermal_14bit_left.yaml')
#             K_orig, D = self._load_sthereo_calib(calib_path)
            
#             # ポーズの読み込み (sequential.py の _load_sthereo_poses を利用と仮定)
#             from modules.dataset.thermal.sequential import _load_sthereo_poses, _nearest_pose_idx
#             poses = _load_sthereo_poses(os.path.join(seq_dir, 'pose', 'global_pose.csv'))
#             if len(poses) < 2: continue
            
#             pose_ts, pose_Ts = [p[0] for p in poses], [p[1] for p in poses]
#             img_dir = os.path.join(seq_dir, 'image', 'thermal8_left_clahe')
#             if not os.path.isdir(img_dir): continue
            
#             img_files = sorted(f for f in os.listdir(img_dir) if f.endswith('.png'))
#             matched = []
#             for fname in img_files:
#                 try: ts_ns = int(fname.split('.')[0])
#                 except: continue
#                 idx = _nearest_pose_idx(ts_ns, pose_ts)
#                 if abs(pose_ts[idx] - ts_ns) < 250_000_000:
#                     matched.append((os.path.join(img_dir, fname), pose_Ts[idx]))
                    
#             count = 0
#             for i in range(0, len(matched) - stride, stride):
#                 j = i + stride
#                 p_t, T_t = matched[i]
#                 p_t1, T_t1 = matched[j]
#                 T_rel = np.linalg.inv(T_t) @ T_t1
#                 if np.linalg.norm(T_rel[:3, 3]) < 0.5: continue
                
#                 # K_orig と D もペア情報として保存
#                 self._pairs.append((p_t, p_t1, T_rel, K_orig, D))
#                 count += 1
#                 if count >= max_pairs: break

#     def _load_sthereo_calib(self, calib_path: str) -> Tuple[np.ndarray, np.ndarray]:
#         """YAMLから K と 歪み係数 D を抽出する"""
#         K = np.array([[429.4, 0, 311.1], [0, 429.5, 266.1], [0, 0, 1]]) # 失敗時デフォルト
#         D = np.zeros(5, dtype=np.float64)
#         if not os.path.isfile(calib_path): return K, D
        
#         try:
#             with open(calib_path) as f: text = f.read()
#             import re
#             m_K = re.search(r'camera_matrix.*?data:\s*\[(.*?)\]', text, re.DOTALL)
#             if m_K:
#                 vals = [float(x.strip()) for x in m_K.group(1).replace('\n', ' ').split(',') if x.strip()]
#                 if len(vals) == 9: K = np.array(vals).reshape(3, 3)
                
#             m_D = re.search(r'distortion_coefficients.*?data:\s*\[(.*?)\]', text, re.DOTALL)
#             if m_D:
#                 vals = [float(x.strip()) for x in m_D.group(1).replace('\n', ' ').split(',') if x.strip()]
#                 if len(vals) >= 4: D = np.array(vals)
#         except Exception: pass
#         return K, D

#     def __len__(self): return len(self._pairs)

#     def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
#         p_t, p_t1, T_rel, K_orig, D = self._pairs[idx]
        
#         # 🎯 SThErEO専用: Undistort -> クロップ -> K行列シフト -> パディング
#         thr_t, size_t, K_corr = _read_sthereo_geometric(p_t, K_orig, D)
#         thr_t1, size_t1, _ = _read_sthereo_geometric(p_t1, K_orig, D)
        
#         return {
#             'image0': thr_t, 'image1': thr_t1,
#             'orig_size0': size_t, 'orig_size1': size_t1,
#             'T_rel': torch.from_numpy(T_rel).float(),
#             'K': torch.from_numpy(K_corr).float(), # 🎯 クロップに合わせてシフト済みのKを返す
#             'dataset_name': 'sthereo'
#         }


"""
modules/dataset/thermal/stage23_geometry_datasets.py
Stage 2 & 3 (LightGlue FT / Joint FT) 専用データローダー群。

【完全自己完結版】
古い `sequential.py` への依存を完全に排除し、各データセットの
GTポーズ(相対姿勢)の抽出から幾何学的な画像ロードまでをこのファイル単独で完結させています。
"""

import os
import cv2
import bisect
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import Dict, Tuple, List

# =====================================================================
# 汎用ユーティリティ (画像ロード & 幾何学変換)
# =====================================================================

# def _read_thr_geometric(path: str) -> Tuple[torch.Tensor, torch.Tensor]:
#     """[VIVID / TartanRGBT] クロップなし、32の倍数右下パディング"""
#     img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#     if img is None: raise FileNotFoundError(f"File not found: {path}")
        
#     orig_h, orig_w = img.shape[:2]
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
#     tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    
#     pad_h = (32 - (orig_h % 32)) % 32
#     pad_w = (32 - (orig_w % 32)) % 32
#     if pad_h > 0 or pad_w > 0:
#         tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode='replicate')
        
#     return tensor, torch.tensor([orig_w, orig_h])

def _read_thr_geometric(path: str):
    """
    [VIVID / TartanRGBT Stage 2/3用]
    パディング処理を廃止し、生の解像度(512x640等)をそのまま出力する。
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"File not found: {path}")
        
    # 1. 画像サイズの取得
    orig_h, orig_w = img.shape[:2]
    
    # 2. テンソル化
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    
    # ✂️ 【削除】32の倍数にするための計算と F.pad(tensor, ...) を完全削除

    return tensor, torch.tensor([orig_w, orig_h])

def _read_sthereo_geometric(path: str, K_orig: np.ndarray, D: np.ndarray):
    """
    [SThErEO Stage 2/3用]
    RGBアラインメント用のクロップを廃止し、元のFOV(512x640)を維持する。
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"File not found: {path}")
        
    # 1. 歪み補正 (魚眼/RadTanモデルの適用)
    if D is not None and np.any(D != 0):
        img = cv2.undistort(img, K_orig, D)
        
    # ✂️ 【削除】クロップ処理 (img = img[top:...]) を完全削除
    
    # 2. 画像サイズの取得 (512, 640 になる)
    orig_h, orig_w = img.shape[:2]
    
    # 3. テンソル化 (LightGlue入力用のRGB化)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    
    # ✂️ 【削除】K_corrected のシフト補正 (K_orig[0, 2] -= left) を完全削除
    
    # 4. 生のK行列をそのまま返す
    return tensor, torch.tensor([orig_w, orig_h]), K_orig.copy()

def _quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    q /= np.linalg.norm(q) + 1e-12
    x, y, z, w = q
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-z*w),   2*(x*z+y*w)],
         [  2*(x*y+z*w), 1-2*(x*x+z*z),   2*(y*z-x*w)],
         [  2*(x*z-y*w),   2*(y*z+x*w), 1-2*(x*x+y*y)],
    ], dtype=np.float64)

# =====================================================================
# SThErEO 専用ポーズ抽出ユーティリティ
# =====================================================================

def _nearest_pose_idx(ts_ns: int, pose_ts: List[int]) -> int:
    """タイムスタンプ(ナノ秒)に最も近いGTポーズのインデックスを二分探索"""
    idx = bisect.bisect_left(pose_ts, ts_ns)
    if idx == 0: return 0
    if idx == len(pose_ts): return len(pose_ts) - 1
    if abs(pose_ts[idx] - ts_ns) < abs(pose_ts[idx - 1] - ts_ns):
        return idx
    return idx - 1

# def _load_sthereo_poses(csv_path: str) -> List[Tuple[int, np.ndarray]]:
#     """global_pose.csv から UTM座標とオイラー角を読み取り、4x4変換行列を構築"""
#     poses = []
#     if not os.path.isfile(csv_path): return poses
#     with open(csv_path, 'r') as f: lines = f.readlines()
        
#     for line in lines[1:]: # ヘッダーをスキップ
#         parts = line.strip().split(',')
#         if len(parts) < 8: continue
#         try:
#             ts_ns = int(parts[0])
#             E, N, alt = float(parts[1]), float(parts[2]), float(parts[3])
#             roll, pitch, yaw = float(parts[5]), float(parts[6]), float(parts[7])
            
#             Rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
#             Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
#             Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
#             R = Rz @ Ry @ Rx
            
#             T = np.eye(4, dtype=np.float64)
#             T[:3, :3] = R
#             T[:3, 3] = [E, N, alt]
#             poses.append((ts_ns, T))
#         except Exception: continue
            
#     poses.sort(key=lambda x: x[0])
#     return poses

def _load_sthereo_poses(csv_path: str) -> List[Tuple[int, np.ndarray]]:
    """global_pose.csv (7列, ヘッダーなし, [秒, X, Y, Z, Roll, Pitch, Yaw(度)]) をパース"""
    poses = []
    if not os.path.isfile(csv_path): return poses
    
    with open(csv_path, 'r') as f: 
        lines = f.readlines()
        
    for line in lines: # 🌟 修正1: ヘッダーがないため [1:] を削除し全行読み込む
        parts = line.strip().split(',')
        if len(parts) < 7: continue # 🌟 修正2: 7列以上に変更
        
        try:
            # 🌟 修正3: 小数点の「秒」を、画像名に合わせて「ナノ秒(19桁)」に変換
            ts_ns = int(float(parts[0]) * 1e9)
            
            # 並進 (Translation)
            E, N, alt = float(parts[1]), float(parts[2]), float(parts[3])
            
            # 🌟 修正4: オイラー角のインデックスを 4, 5, 6 に変更し、Degree から Radian に変換！
            roll = np.deg2rad(float(parts[4]))
            pitch = np.deg2rad(float(parts[5]))
            yaw = np.deg2rad(float(parts[6]))
            
            Rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
            Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
            Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
            R = Rz @ Ry @ Rx
            
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = R
            T[:3, 3] = [E, N, alt]
            poses.append((ts_ns, T))
        except Exception: 
            continue
            
    poses.sort(key=lambda x: x[0])
    return poses

# =====================================================================
# 1. TartanRGBT
# =====================================================================

class Stage23_TartanRGBTDataset(Dataset):
    def __init__(self, data_root: str, stride: int = 5, max_pairs: int = 500):
        self.data_root = data_root
        self._pairs = []
        self.K_default = np.array([[421.23237248, 0.0, 317.55165969], [0.0, 420.80872096, 255.54588954], [0.0, 0.0, 1.0]], dtype=np.float64)

        seq_dirs = self._find_seqs(data_root)
        for seq_dir in seq_dirs:
            thr_dir = os.path.join(seq_dir, 'thermal_left_rect_8')
            odom_path = os.path.join(seq_dir, 'stereo_depth', 'poses.npy')
            if not os.path.isdir(thr_dir) or not os.path.isfile(odom_path): continue
            
            thr_files = sorted([f for f in os.listdir(thr_dir) if f.endswith('.png')])
            try:
                raw_poses = np.load(odom_path)
                poses = []
                for row in raw_poses:
                    pose_vec = row if len(row) == 7 else row[1:8]
                    T = np.eye(4, dtype=np.float64)
                    T[:3, :3] = _quat_to_rot(pose_vec[3], pose_vec[4], pose_vec[5], pose_vec[6])
                    T[:3, 3] = pose_vec[:3]
                    poses.append(T)
            except Exception: continue

            count = 0
            for i in range(len(thr_files) - stride):
                j = i + stride
                if i < len(poses) and j < len(poses):
                    T_rel = np.linalg.inv(poses[i]) @ poses[j]
                    if np.linalg.norm(T_rel[:3, 3]) >= 0.1:
                        self._pairs.append((os.path.join(thr_dir, thr_files[i]), os.path.join(thr_dir, thr_files[j]), T_rel, self.K_default))
                        count += 1
                        if count >= max_pairs: break

    def _find_seqs(self, root):
        res = []
        for d in os.listdir(root):
            p = os.path.join(root, d)
            if os.path.isdir(p) and 'day' in d:
                for sub in os.listdir(p):
                    sub_p = os.path.join(p, sub)
                    if os.path.isdir(os.path.join(sub_p, 'thermal_left_rect_8')):
                        res.append(sub_p)
        return res

    def __len__(self): 
        return len(self._pairs)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        p_t, p_t1, T_rel, K = self._pairs[idx]
        thr_t, size_t = _read_thr_geometric(p_t)
        thr_t1, size_t1 = _read_thr_geometric(p_t1)
        return {'image0': thr_t, 'image1': thr_t1, 'orig_size0': size_t, 'orig_size1': size_t1, 'T_rel': torch.from_numpy(T_rel).float(), 'K': torch.from_numpy(K).float(), 'dataset_name': 'tartanrgbt'}

# =====================================================================
# 2. VIVID (完全自己完結版)
# =====================================================================

# class Stage23_VIVIDDataset(Dataset):
#     def __init__(self, data_root: str, stride: int = 5, max_pairs: int = 2000):
#         self.data_root = data_root
#         self._pairs = []
#         # 代表的なVIVIDのK行列
#         self.K_default = np.array([[419.8, 0.0, 319.5], [0.0, 419.8, 239.5], [0.0, 0.0, 1.0]], dtype=np.float64)

#         for seq_name in os.listdir(data_root):
#             seq_dir = os.path.join(data_root, seq_name)
#             if not os.path.isdir(seq_dir): continue
            
#             # 1. 8-bitまたは16-bitのサーマル画像フォルダを探す
#             thr_dir = next((os.path.join(seq_dir, d) for d in os.listdir(seq_dir) if "thermal" in d.lower() and os.path.isdir(os.path.join(seq_dir, d))), None)
#             if not thr_dir: continue
            
#             img_files = sorted([f for f in os.listdir(thr_dir) if f.endswith('.png')])
#             if len(img_files) < 10: continue
            
#             # 2. GTポーズの自動探索とロード
#             poses = self._get_gt(seq_dir)
#             if len(poses) < 10: continue
            
#             # 3. タイムスタンプのスケール不一致を吸収するマッチング
#             matched_poses = self._match(img_files, poses)
#             if len(matched_poses) != len(img_files): continue

#             # 4. ペアの構築
#             count = 0
#             for i in range(0, len(img_files) - stride, stride):
#                 j = i + stride
#                 T_t = matched_poses[i]
#                 T_t1 = matched_poses[j]
                
#                 T_rel = np.linalg.inv(T_t) @ T_t1
#                 if np.linalg.norm(T_rel[:3, 3]) >= 0.1: # 移動量が0.1m以上
#                     p_t = os.path.join(thr_dir, img_files[i])
#                     p_t1 = os.path.join(thr_dir, img_files[j])
#                     self._pairs.append((p_t, p_t1, T_rel, self.K_default))
#                     count += 1
#                     if count >= max_pairs: break

#     def _get_gt(self, seq_dir: str) -> List[Tuple[float, np.ndarray]]:
#         """Mocap, LOAM などからGTポーズを抽出"""
#         poses = []
#         # MoCap (Vicon等)
#         mocap_file = next((f for f in os.listdir(seq_dir) if f.endswith('_gt.csv')), None)
#         if mocap_file:
#             with open(os.path.join(seq_dir, mocap_file)) as f:
#                 for line in f.readlines()[1:]:
#                     parts = line.strip().split(',')
#                     if len(parts) >= 8:
#                         T = np.eye(4, dtype=np.float64)
#                         T[:3, :3] = _quat_to_rot(float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7]))
#                         T[:3, 3] = [float(parts[1]), float(parts[2]), float(parts[3])]
#                         poses.append((float(parts[0]), T))
#             return poses
            
#         # LOAM (poses.txt)
#         loam_file = next((f for f in os.listdir(seq_dir) if f.endswith('poses.txt')), None)
#         if loam_file:
#             with open(os.path.join(seq_dir, loam_file)) as f:
#                 for idx, line in enumerate(f.readlines()):
#                     parts = [float(x) for x in line.strip().split()]
#                     if len(parts) == 12:
#                         T = np.eye(4, dtype=np.float64)
#                         T[:3, :] = np.array(parts).reshape(3, 4)
#                         poses.append((float(idx), T)) # ダミータイムスタンプ
#             return poses
            
#         return poses

#     def _match(self, img_files: List[str], poses: List[Tuple[float, np.ndarray]]) -> List[np.ndarray]:
#         """画像とポーズ数が合わない場合、インデックスの比率で強引に補間・マッチさせる"""
#         matched = []
#         ratio = len(poses) / len(img_files)
#         for i in range(len(img_files)):
#             pose_idx = min(int(i * ratio), len(poses) - 1)
#             matched.append(poses[pose_idx][1])
#         return matched

#     def __len__(self):
#         return len(self._pairs)

#     def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
#         p_t, p_t1, T_rel, K = self._pairs[idx]
#         thr_t, size_t = _read_thr_geometric(p_t)
#         thr_t1, size_t1 = _read_thr_geometric(p_t1)
#         return {'image0': thr_t, 'image1': thr_t1, 'orig_size0': size_t, 'orig_size1': size_t1, 'T_rel': torch.from_numpy(T_rel).float(), 'K': torch.from_numpy(K).float(), 'dataset_name': 'vivid'}

# # =====================================================================
# # 3. SThErEO (完全自己完結版)
# # =====================================================================

# class Stage23_SThErEODataset(Dataset):
#     def __init__(self, data_root: str, stride: int = 5, split: str = 'val', max_pairs: int = 2000):
#         self.data_root = data_root
#         self._pairs = []
#         _VAL_SEQS = frozenset(['snu_afternoon', 'kaist_morning', 'valley_afternoon'])
        
#         for seq_name in sorted(os.listdir(data_root)):
#             seq_dir = os.path.join(data_root, seq_name)
#             if not os.path.isdir(seq_dir): continue
            
#             is_val = seq_name in _VAL_SEQS
#             if (split == 'train' and is_val) or (split == 'val' and not is_val): continue
            
#             calib_path = os.path.join(seq_dir, 'calibration', 'thermal_14bit_left.yaml')
#             K_orig, D = self._load_sthereo_calib(calib_path)
            
#             # 🎯 クラス外に配置した自己完結関数を使用
#             poses = _load_sthereo_poses(os.path.join(seq_dir, 'pose', 'global_pose.csv'))
#             if len(poses) < 2: continue
            
#             pose_ts, pose_Ts = [p[0] for p in poses], [p[1] for p in poses]
#             img_dir = os.path.join(seq_dir, 'image', 'thermal8_left_clahe')
#             if not os.path.isdir(img_dir): continue
            
#             img_files = sorted(f for f in os.listdir(img_dir) if f.endswith('.png'))
#             matched = []
#             for fname in img_files:
#                 try: ts_ns = int(fname.split('.')[0])
#                 except: continue
#                 # 🎯 自己完結関数を使用
#                 idx = _nearest_pose_idx(ts_ns, pose_ts)
#                 if abs(pose_ts[idx] - ts_ns) < 250_000_000:
#                     matched.append((os.path.join(img_dir, fname), pose_Ts[idx]))
                    
#             count = 0
#             for i in range(0, len(matched) - stride, stride):
#                 j = i + stride
#                 p_t, T_t = matched[i]
#                 p_t1, T_t1 = matched[j]
#                 T_rel = np.linalg.inv(T_t) @ T_t1
#                 if np.linalg.norm(T_rel[:3, 3]) < 0.5: continue
                
#                 self._pairs.append((p_t, p_t1, T_rel, K_orig, D))
#                 count += 1
#                 if count >= max_pairs: break

#     def _load_sthereo_calib(self, calib_path: str) -> Tuple[np.ndarray, np.ndarray]:
#         K = np.array([[429.4, 0, 311.1], [0, 429.5, 266.1], [0, 0, 1]])
#         D = np.zeros(5, dtype=np.float64)
#         if not os.path.isfile(calib_path): return K, D
#         try:
#             with open(calib_path) as f: text = f.read()
#             import re
#             m_K = re.search(r'camera_matrix.*?data:\s*\[(.*?)\]', text, re.DOTALL)
#             if m_K:
#                 vals = [float(x.strip()) for x in m_K.group(1).replace('\n', ' ').split(',') if x.strip()]
#                 if len(vals) == 9: K = np.array(vals).reshape(3, 3)
#             m_D = re.search(r'distortion_coefficients.*?data:\s*\[(.*?)\]', text, re.DOTALL)
#             if m_D:
#                 vals = [float(x.strip()) for x in m_D.group(1).replace('\n', ' ').split(',') if x.strip()]
#                 if len(vals) >= 4: D = np.array(vals)
#         except Exception: pass
#         return K, D

#     def __len__(self):
#         return len(self._pairs)

#     def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
#         p_t, p_t1, T_rel, K_orig, D = self._pairs[idx]
#         thr_t, size_t, K_corr = _read_sthereo_geometric(p_t, K_orig, D)
#         thr_t1, size_t1, _ = _read_sthereo_geometric(p_t1, K_orig, D)
#         return {'image0': thr_t, 'image1': thr_t1, 'orig_size0': size_t, 'orig_size1': size_t1, 'T_rel': torch.from_numpy(T_rel).float(), 'K': torch.from_numpy(K_corr).float(), 'dataset_name': 'sthereo'}

# =====================================================================
# 2. VIVID (デバッグ版)
# =====================================================================
# class Stage23_VIVIDDataset(Dataset):
#     def __init__(self, data_root: str, stride: int = 5, max_pairs: int = 2000):
#         self.data_root = data_root
#         self._pairs = []
#         self.K_default = np.array([[419.8, 0.0, 319.5], [0.0, 419.8, 239.5], [0.0, 0.0, 1.0]], dtype=np.float64)

#         print(f"\n=======================================================")
#         print(f"🔍 [DEBUG] Starting VIVID Dataset Initialization")
#         print(f"🔍 Data Root: {data_root}")
#         print(f"=======================================================")
        
#         for seq_name in os.listdir(data_root):
#             seq_dir = os.path.join(data_root, seq_name)
#             if not os.path.isdir(seq_dir): continue
            
#             print(f"  [VIVID] 📁 Checking sequence: {seq_name}")
            
#             thr_dir = next((os.path.join(seq_dir, d) for d in os.listdir(seq_dir) if "thermal" in d.lower() and os.path.isdir(os.path.join(seq_dir, d))), None)
#             if not thr_dir:
#                 print(f"    ❌ Skipped: Could not find any directory containing 'thermal'.")
#                 continue
            
#             img_files = sorted([f for f in os.listdir(thr_dir) if f.endswith('.png') or f.endswith('.jpg')])
#             print(f"    ✅ Found {len(img_files)} thermal images in: {os.path.basename(thr_dir)}")
#             if len(img_files) < 10: 
#                 print(f"    ❌ Skipped: Less than 10 images.")
#                 continue
            
#             poses = self._get_gt(seq_dir)
#             if len(poses) == 0:
#                 print(f"    ❌ Skipped: No Ground Truth pose files found (checked for _gt.csv and poses.txt).")
#                 continue
#             print(f"    ✅ Found {len(poses)} Ground Truth poses.")
            
#             matched_poses = self._match(img_files, poses)
#             if len(matched_poses) != len(img_files):
#                 print(f"    ❌ Skipped: Pose match length mismatch (Images: {len(img_files)}, Poses: {len(matched_poses)}).")
#                 continue

#             count = 0
#             for i in range(0, len(img_files) - stride, stride):
#                 j = i + stride
#                 T_rel = np.linalg.inv(matched_poses[i]) @ matched_poses[j]
                
#                 # 🌟 [DEBUG] 移動量の制限をなくす (-1.0 以上)
#                 if np.linalg.norm(T_rel[:3, 3]) >= 0.1: 
#                     p_t = os.path.join(thr_dir, img_files[i])
#                     p_t1 = os.path.join(thr_dir, img_files[j])
#                     self._pairs.append((p_t, p_t1, T_rel, self.K_default))
#                     count += 1
#                     if count >= max_pairs: break
                    
#             print(f"    🎯 Successfully extracted {count} pairs.")

#         print(f"✅ [VIVID DEBUG COMPLETE] Total valid pairs loaded: {len(self._pairs)}\n")

#     def _get_gt(self, seq_dir: str) -> List[Tuple[float, np.ndarray]]:
#         poses = []
#         mocap_file = next((f for f in os.listdir(seq_dir) if f.endswith('_gt.csv')), None)
#         if mocap_file:
#             with open(os.path.join(seq_dir, mocap_file)) as f:
#                 for line in f.readlines()[1:]:
#                     parts = line.strip().split(',')
#                     if len(parts) >= 8:
#                         T = np.eye(4, dtype=np.float64)
#                         T[:3, :3] = _quat_to_rot(float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7]))
#                         T[:3, 3] = [float(parts[1]), float(parts[2]), float(parts[3])]
#                         poses.append((float(parts[0]), T))
#             return poses
#         loam_file = next((f for f in os.listdir(seq_dir) if f.endswith('poses.txt')), None)
#         if loam_file:
#             with open(os.path.join(seq_dir, loam_file)) as f:
#                 for idx, line in enumerate(f.readlines()):
#                     parts = [float(x) for x in line.strip().split()]
#                     if len(parts) == 12:
#                         T = np.eye(4, dtype=np.float64)
#                         T[:3, :] = np.array(parts).reshape(3, 4)
#                         poses.append((float(idx), T))
#             return poses
#         return poses

#     def _match(self, img_files: List[str], poses: List[Tuple[float, np.ndarray]]) -> List[np.ndarray]:
#         matched = []
#         ratio = len(poses) / len(img_files)
#         for i in range(len(img_files)):
#             pose_idx = min(int(i * ratio), len(poses) - 1)
#             matched.append(poses[pose_idx][1])
#         return matched
#     def __len__(self): return len(self._pairs)
#     def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
#         p_t, p_t1, T_rel, K = self._pairs[idx]
#         thr_t, size_t = _read_thr_geometric(p_t)
#         thr_t1, size_t1 = _read_thr_geometric(p_t1)
#         return {'image0': thr_t, 'image1': thr_t1, 'orig_size0': size_t, 'orig_size1': size_t1, 'T_rel': torch.from_numpy(T_rel).float(), 'K': torch.from_numpy(K).float(), 'dataset_name': 'vivid'}

# =====================================================================
# 2. VIVID (完全制覇版: img階層 ＆ 集約ポーズディレクトリ対応)
# =====================================================================
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Dict

class Stage23_VIVIDDataset(Dataset):
    def __init__(self, data_root: str, stride: int = 5, max_pairs: int = 2000):
        self.data_root = data_root
        self._pairs = []
        # self.K_default = np.array([[419.8, 0.0, 319.5], [0.0, 419.8, 239.5], [0.0, 0.0, 1.0]], dtype=np.float64)
        self.K_default = np.array([[445.34173838383924, 0., 310.74708274781557], [0., 446.40695195454197, 249.54892754326676], [0., 0., 1.]])

        print(f"\n=======================================================")
        print(f"🔍 [DEBUG] Starting VIVID Dataset Initialization (v5: Final)")
        print(f"=======================================================")

        extracted_dir = os.path.join(data_root, 'extracted_data')
        if not os.path.isdir(extracted_dir): return

        categories = [d for d in os.listdir(extracted_dir) if os.path.isdir(os.path.join(extracted_dir, d))]
        
        for category in categories:
            cat_dir = os.path.join(extracted_dir, category)
            sequences = [d for d in os.listdir(cat_dir) if os.path.isdir(os.path.join(cat_dir, d))]

            for seq_name in sequences:
                seq_dir_extracted = os.path.join(cat_dir, seq_name)
                print(f"  [VIVID] 🎞️ Checking sequence: {category} / {seq_name}")

                # -------------------------------------------------------------
                # 1. 🖼️ 画像の検索 (extracted_data/カテゴリ/シーケンス/img/ の中を探す)
                # -------------------------------------------------------------
                img_base_dir = os.path.join(seq_dir_extracted, 'img')
                if not os.path.isdir(img_base_dir): continue
                    
                subdirs = [d for d in os.listdir(img_base_dir) if os.path.isdir(os.path.join(img_base_dir, d))]
                priority_list = ['thermal_fieldscale_clahe', 'thermal_clahe', 'thermal_fieldscale', 'thermal_8']
                thr_dir_name = next((pref for pref in priority_list if pref in subdirs), None)
                
                if not thr_dir_name:
                    fallback = [d for d in subdirs if "thermal" in d.lower()]
                    if fallback: thr_dir_name = fallback[0]
                        
                if not thr_dir_name: continue
                    
                thr_dir = os.path.join(img_base_dir, thr_dir_name)
                img_files = sorted([f for f in os.listdir(thr_dir) if f.endswith(('.png', '.jpg'))])
                
                if len(img_files) < 10: continue
                
                # -------------------------------------------------------------
                # 2. 📍 ポーズの検索 (🌟修正: 集約フォルダを直接狙い撃ち)
                # -------------------------------------------------------------
                poses = self._get_gt(data_root, category, seq_name)
                
                if len(poses) == 0: 
                    print(f"    ❌ Skipped: No GT poses found.")
                    continue
                print(f"    ✅ Found {len(poses)} images & {len(poses)} GT poses.")
                
                # -------------------------------------------------------------
                # 3. マッチングとペア構築
                # -------------------------------------------------------------
                matched_poses = self._match(img_files, poses)
                if len(matched_poses) != len(img_files): continue

                count = 0
                for i in range(0, len(img_files) - stride, stride):
                    j = i + stride
                    T_t = matched_poses[i]
                    T_t1 = matched_poses[j]
                    
                    T_rel = np.linalg.inv(T_t) @ T_t1
                    # 🌟 正常に動作することを確認するため、一時的に 0.0m 以上の制約にしています
                    if np.linalg.norm(T_rel[:3, 3]) >= 0.0: 
                        p_t = os.path.join(thr_dir, img_files[i])
                        p_t1 = os.path.join(thr_dir, img_files[j])
                        self._pairs.append((p_t, p_t1, T_rel, self.K_default))
                        count += 1
                        if count >= max_pairs: break
                        
                print(f"    🎯 Successfully extracted {count} pairs!")

        print(f"\n✅ [VIVID INITIALIZATION COMPLETE] Total valid pairs loaded: {len(self._pairs)}\n")

    def _get_gt(self, data_root: str, category: str, seq_name: str) -> List[Tuple[float, np.ndarray]]:
        """集約フォルダと分散フォルダの両方から、複数のフォーマットのポーズを探し出す"""
        poses = []
        candidates = [
            os.path.join(data_root, category, 'loampose', f'{seq_name}_optimized_poses.txt'),
            os.path.join(data_root, category, 'pose', f'{seq_name}_gt.csv'),
            os.path.join(data_root, category, seq_name, 'loampose', 'optimized_poses.txt')
        ]
        
        for file_path in candidates:
            if not os.path.isfile(file_path): continue
            
            # CSV形式 (MoCap)
            if file_path.endswith('.csv'):
                with open(file_path) as file:
                    for line in file.readlines()[1:]:
                        parts = line.strip().split(',')
                        if len(parts) >= 8:
                            try:
                                T = np.eye(4, dtype=np.float64)
                                T[:3, :3] = _quat_to_rot(float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7]))
                                T[:3, 3] = [float(parts[1]), float(parts[2]), float(parts[3])]
                                poses.append((float(parts[0]), T))
                            except ValueError: continue
                if poses: return poses
            
            # TXT形式 (LOAM/KITTI 混在対応)
            elif file_path.endswith('.txt'):
                with open(file_path) as file:
                    for idx, line in enumerate(file.readlines()):
                        parts = line.strip().split()
                        if not parts: continue
                        
                        # パターン1: g2oフォーマット (VERTEX_SE3:QUAT id x y z qx qy qz qw)
                        if parts[0] == 'VERTEX_SE3:QUAT':
                            if len(parts) >= 9:
                                try:
                                    T = np.eye(4, dtype=np.float64)
                                    # g2oのクォータニオン順序は x, y, z, w
                                    T[:3, :3] = _quat_to_rot(float(parts[5]), float(parts[6]), float(parts[7]), float(parts[8]))
                                    T[:3, 3] = [float(parts[2]), float(parts[3]), float(parts[4])]
                                    poses.append((float(parts[1]), T)) # IDをタイムスタンプ代わりに
                                except ValueError: continue
                                
                        # パターン2: KITTIフォーマット等 (12個の数字) またはその他の数値データ
                        else:
                            try:
                                num_parts = [float(x) for x in parts]
                                if len(num_parts) == 12:
                                    T = np.eye(4, dtype=np.float64)
                                    T[:3, :] = np.array(num_parts).reshape(3, 4)
                                    poses.append((float(idx), T))
                            except ValueError:
                                # 'VERTEX_SE3:QUAT' 以外の予期せぬ文字列(ヘッダー等)があった場合はスキップ
                                continue
                                
                if poses: return poses
                
        return poses

    def _match(self, img_files: List[str], poses: List[Tuple[float, np.ndarray]]) -> List[np.ndarray]:
        matched = []
        ratio = len(poses) / len(img_files)
        for i in range(len(img_files)):
            pose_idx = min(int(i * ratio), len(poses) - 1)
            matched.append(poses[pose_idx][1])
        return matched

    def __len__(self): return len(self._pairs)
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        p_t, p_t1, T_rel, K = self._pairs[idx]
        thr_t, size_t = _read_thr_geometric(p_t)
        thr_t1, size_t1 = _read_thr_geometric(p_t1)
        return {'image0': thr_t, 'image1': thr_t1, 'orig_size0': size_t, 'orig_size1': size_t1, 'T_rel': torch.from_numpy(T_rel).float(), 'K': torch.from_numpy(K).float(), 'dataset_name': 'vivid'}

# =====================================================================
# 3. SThErEO (デバッグ版)
# =====================================================================
class Stage23_SThErEODataset(Dataset):
    def __init__(self, data_root: str, stride: int = 5, split: str = 'val', max_pairs: int = 2000):
        self.data_root = data_root
        self._pairs = []
        
        print(f"\n=======================================================")
        print(f"🔍 [DEBUG] Starting SThErEO Dataset Initialization")
        print(f"🔍 Target Split: '{split}' | Data Root: {data_root}")
        print(f"=======================================================")
        
        # 🌟 [DEBUG] SThErEO の Split チェックを一時的に無効化し、すべてのシーケンスを読み込むように変更
        for seq_name in sorted(os.listdir(data_root)):
            seq_dir = os.path.join(data_root, seq_name)
            if not os.path.isdir(seq_dir): continue
            
            print(f"  [SThErEO] 📁 Checking sequence: {seq_name}")
            
            calib_path = os.path.join(seq_dir, 'calibration', 'thermal_14bit_left.yaml')
            K_orig, D = self._load_sthereo_calib(calib_path)
            
            pose_path = os.path.join(seq_dir, 'pose', 'global_pose.csv')
            poses = _load_sthereo_poses(pose_path)
            
            if len(poses) < 2: 
                print(f"    ❌ Skipped: Not enough valid poses extracted from CSV.")
                continue
            
            pose_ts, pose_Ts = [p[0] for p in poses], [p[1] for p in poses]
            
            img_dir = os.path.join(seq_dir, 'image', 'thermal8_left_clahe')
            if not os.path.isdir(img_dir): 
                # 別のフォルダ名で保存されている可能性を探る
                img_dir_alt = os.path.join(seq_dir, 'image')
                if os.path.isdir(img_dir_alt):
                    print(f"    ⚠️ Warning: 'thermal8_left_clahe' not found. Searching in '{img_dir_alt}' directly.")
                    img_dir = img_dir_alt
                else:
                    print(f"    ❌ Skipped: Image directory not found ({img_dir})")
                    continue
            
            img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.png') or f.endswith('.jpg')])
            print(f"    ✅ Found {len(img_files)} thermal images.")
            
            matched = []
            for fname in img_files:
                try: 
                    # 拡張子を除いたファイル名を数値(タイムスタンプ)として解釈
                    ts_ns = int(float(fname.split('.')[0]))
                except: continue
                idx = _nearest_pose_idx(ts_ns, pose_ts)
                
                # 🌟 [DEBUG] タイムスタンプのマッチング許容誤差を大幅に緩和 (250ms -> 10秒)
                # ファイル名が秒単位(sec)かナノ秒(ns)かでスケールが違うバグを防ぐため
                time_diff = abs(pose_ts[idx] - ts_ns)
                if time_diff < 10_000_000_000: # 10秒以内なら許容
                    matched.append((os.path.join(img_dir, fname), pose_Ts[idx]))
                    
            print(f"    ✅ Successfully time-matched {len(matched)} / {len(img_files)} images to poses.")
            if len(matched) < 10:
                print(f"    ❌ Skipped: Not enough time-matched pairs. (Timestamp format mismatch?)")
                continue
                    
            count = 0
            for i in range(0, len(matched) - stride, stride):
                j = i + stride
                p_t, T_t = matched[i]
                p_t1, T_t1 = matched[j]
                T_rel = np.linalg.inv(T_t) @ T_t1
                
                # 🌟 [DEBUG] 移動量の制限をなくす (-1.0 以上)
                if np.linalg.norm(T_rel[:3, 3]) >= 0.5: 
                    self._pairs.append((p_t, p_t1, T_rel, K_orig, D))
                    count += 1
                    if count >= max_pairs: break
                    
            print(f"    🎯 Successfully extracted {count} valid geometric pairs.")

        print(f"✅ [SThErEO DEBUG COMPLETE] Total valid pairs loaded: {len(self._pairs)}\n")

    def _load_sthereo_calib(self, calib_path: str) -> Tuple[np.ndarray, np.ndarray]:
        K = np.array([[429.4, 0, 311.1], [0, 429.5, 266.1], [0, 0, 1]])
        D = np.zeros(5, dtype=np.float64)
        if not os.path.isfile(calib_path): return K, D
        try:
            with open(calib_path) as f: text = f.read()
            import re
            m_K = re.search(r'camera_matrix.*?data:\s*\[(.*?)\]', text, re.DOTALL)
            if m_K:
                vals = [float(x.strip()) for x in m_K.group(1).replace('\n', ' ').split(',') if x.strip()]
                if len(vals) == 9: K = np.array(vals).reshape(3, 3)
            m_D = re.search(r'distortion_coefficients.*?data:\s*\[(.*?)\]', text, re.DOTALL)
            if m_D:
                vals = [float(x.strip()) for x in m_D.group(1).replace('\n', ' ').split(',') if x.strip()]
                if len(vals) >= 4: D = np.array(vals)
        except Exception: pass
        return K, D
    def __len__(self): return len(self._pairs)
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        p_t, p_t1, T_rel, K_orig, D = self._pairs[idx]
        thr_t, size_t, K_corr = _read_sthereo_geometric(p_t, K_orig, D)
        thr_t1, size_t1, _ = _read_sthereo_geometric(p_t1, K_orig, D)
        return {'image0': thr_t, 'image1': thr_t1, 'orig_size0': size_t, 'orig_size1': size_t1, 'T_rel': torch.from_numpy(T_rel).float(), 'K': torch.from_numpy(K_corr).float(), 'dataset_name': 'sthereo'}