"""
modules/dataset/thermal/preprocessing.py
熱画像の前処理を一元管理するモジュール。
学習(train)と評価(eval)で完全に同一の前処理を保証する。
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F

def pad_to_32_multiple(tensor: torch.Tensor):
    """
    テンソルの H と W を 32 の倍数になるように、右側と下側にパディングします。
    BORDER_REPLICATE 相当の処理を行い、エッジのピクセルをコピーすることで、
    人工的な境界（黒枠）による物理勾配損失への悪影響を防ぎます。
    """
    _, h, w = tensor.shape
    pad_h = (32 - (h % 32)) % 32
    pad_w = (32 - (w % 32)) % 32
    if pad_h > 0 or pad_w > 0:
        # F.pad の引数は (左, 右, 上, 下) の順
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode='replicate')
    return tensor


def read_and_preprocess_thermal(path: str, is_ms2: bool = False, return_dual: bool = False):
    """
    熱画像を読み込みます。MS2 の場合は hist_99 正規化を適用します。
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None: return None
    img_float = img.astype(np.float32)
    
    if is_ms2 or 'ms2' in path.lower():
        # hist_99: 1% - 99% のパーセンタイルでスケーリングしノイズを排除
        v_min, v_max = np.percentile(img_float, [1.0, 99.0])
        img_8bit = np.clip((img_float - v_min) / (v_max - v_min + 1e-6) * 255, 0, 255).astype(np.uint8)
        # 固有のクロップ (上:9, 下:35, 左:28, 右:34)
        # img_8bit = img_8bit[9:img_8bit.shape[0]-35, 28:img_8bit.shape[1]-34]
        # img_raw = img_float[9:img_float.shape[0]-35, 28:img_float.shape[1]-34]
    else:
        # その他のデータセットは 8-bit 変換のみ
        img_8bit = (img / 256).astype(np.uint8) if img.dtype == np.uint16 else img.astype(np.uint8)
        img_raw = img_float

    if return_dual:
        t_8bit = torch.from_numpy(cv2.cvtColor(img_8bit, cv2.COLOR_GRAY2RGB)).permute(2, 0, 1).float() / 255.0
        t_raw = torch.from_numpy(img_raw).unsqueeze(0).float()
        return {'8bit': t_8bit, 'raw': t_raw}
    return img_8bit

# def read_and_preprocess_thermal(path: str, is_ms2: bool = False, return_dual: bool = False):
#     """
#     熱画像を読み込み、前処理を適用する共通関数。
    
#     Args:
#         path: 画像のファイルパス
#         is_ms2: MS2データセット特有の前処理を適用するかどうか
#         return_dual: True なら学習用の {'8bit': Tensor, 'raw': Tensor} を返す。
#                      False なら評価用の 8-bit NumPy配列 を返す。
#     """
#     # 1. 16-bit等の生データ情報を維持して読み込む
#     img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
#     if img is None:
#         return None

#     img_float = img.astype(np.float32)
#     if img_float.ndim == 3:
#         img_float = cv2.cvtColor(img_float, cv2.COLOR_BGR2GRAY)

#     # MS2特有の動的前処理 (AnyThermal準拠)
#     if is_ms2 or 'ms2' in path.lower() or 'sync_data' in path.lower():
#         # a. hist_99 正規化
#         im_srt = np.sort(img_float.reshape(-1))
#         upper_bound = im_srt[round(len(im_srt) * 0.99) - 1]
#         lower_bound = im_srt[round(len(im_srt) * 0.01)]

#         img_norm = img_float.copy()
#         img_norm[img_norm < lower_bound] = lower_bound
#         img_norm[img_norm > upper_bound] = upper_bound
        
#         if upper_bound - lower_bound > 1e-5:
#             image_out = ((img_norm - lower_bound) / (upper_bound - lower_bound)) * 255.0
#         else:
#             image_out = img_norm * 0
#         image_out = image_out.astype(np.uint8)

#         # b. CLAHE + Bilateral Filter
#         clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#         clahe_img = clahe.apply(image_out)
#         img_final = cv2.bilateralFilter(clahe_img, 5, 20, 15)
        
#         # c. Static Crop (8-bitとRawで空間を完全に同期)
#         h, w = img_final.shape[:2]
#         crop_top, crop_bottom = 9, 35
#         crop_left, crop_right = 28, 34

#         img_8bit = img_final[crop_top:h - crop_bottom, crop_left:w - crop_right]
#         img_raw = img_float[crop_top:h - crop_bottom, crop_left:w - crop_right]

#         curr_h, curr_w = img_8bit.shape[:2]
#         pad_h = (32 - (curr_h % 32)) % 32
#         pad_w = (32 - (curr_w % 32)) % 32
        
#         # img_8bit = img_final[crop_top:h - crop_bottom, crop_left:w - crop_right]
#         # img_raw = img_float[crop_top:h - crop_bottom, crop_left:w - crop_right]
#         if pad_h > 0 or pad_w > 0:
#             # 下(Bottom)と右(Right)にのみパディング。
#             # BORDER_REPLICATE を使い、端のピクセルを自然に引き伸ばす。
#             # 原点(左上)は動かないため、カメラ行列 K への悪影響はゼロ。
#             img_8bit = cv2.copyMakeBorder(img_8bit, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)
#             img_raw = cv2.copyMakeBorder(img_raw, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)

#     else:
#         # SThErEOやVIVID等（既に前処理済みのデータセット用）
#         if img.dtype == np.uint16:
#             img_8bit = (img / 256).astype(np.uint8)
#         else:
#             img_8bit = img.astype(np.uint8)
#         img_raw = img_float

#     # --- 戻り値の形式を選択 ---
#     if return_dual:
#         # 学習用: PyTorchテンソル化 (8-bitは3ch化、Rawは1chのまま)
#         t_8bit = torch.from_numpy(cv2.cvtColor(img_8bit, cv2.COLOR_GRAY2RGB)).permute(2, 0, 1).float() / 255.0
#         t_raw = torch.from_numpy(img_raw).unsqueeze(0).float()
#         return {'8bit': t_8bit, 'raw': t_raw}
#     else:
#         # 評価用: 8-bitグレースケールのNumPy配列
#         return img_8bit