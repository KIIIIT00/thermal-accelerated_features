"""
modules/dataset/thermal/sequential_hybrid.py
知識蒸留（Stage 1）専用の拡張データセットクラス。
各データセット固有のロード処理を継承し、色バイアスの除去と
ネットワーク制約（32の倍数パディング）を適用してテンソルを出力します。
"""

import torch
import torch.nn.functional as F

# 🚨 重要：'sequential.py' ではなく、クロップ/リサイズが実装された個別のファイルからインポート
from modules.dataset.thermal.sthereo import SthEreoDataset as OrigSThErEO
from modules.dataset.thermal.ms2 import MS2Dataset as OrigMS2
from modules.dataset.thermal.vivid import VividDataset as OrigVivid
from modules.dataset.thermal.tartanrgbt import TartanRGBTDataset as OrigTartan
from modules.dataset.thermal.freiburg import FreiburgDataset as OrigFreiburg

from modules.dataset.thermal.preprocessing import read_and_preprocess_thermal

# =============================================================================
# ユーティリティ関数（内部処理用）
# =============================================================================

def pad_to_32_multiple(tensor: torch.Tensor) -> torch.Tensor:
    """
    XFeatのダウンサンプリング/アップサンプリング時の端数エラーを防ぐための右下パディング。
    'replicate' モードを使用することで、端のピクセルを引き伸ばし、
    物理勾配損失において人工的なエッジ（黒枠）が検出されるのを防ぎます。
    原点(0,0)を固定するため、カメラ内部パラメータ K との整合性も維持されます。
    """
    _, h, w = tensor.shape
    pad_h = (32 - (h % 32)) % 32
    pad_w = (32 - (w % 32)) % 32
    if pad_h > 0 or pad_w > 0:
        # F.pad は (左, 右, 上, 下) の順で指定
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode='replicate')
    return tensor

def rgb_to_gray_3ch(rgb_tensor: torch.Tensor) -> torch.Tensor:
    """
    RGBテンソル(C,H,W)をグレースケール化し、再度3chに複製します。
    これにより、教師モデル（RGB用重み）から「色情報」への依存を剥奪し、
    熱画像でも模倣可能な「幾何構造とエッジ」の知識のみを抽出させます。
    """
    # 輝度計算式: Y = 0.2989*R + 0.5870*G + 0.1140*B
    r, g, b = rgb_tensor[0], rgb_tensor[1], rgb_tensor[2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    # 3チャンネルに複製して (3, H, W) に戻す
    return gray.unsqueeze(0).repeat(3, 1, 1)

# =============================================================================
# 知識蒸留用 ベースクラス
# =============================================================================

class BaseHybridKDDataset:
    """
    すべてのデータセットに共通する知識蒸留用の getitem ロジック。
    多重継承を通じて、各データセットのパス解決能力（_pairs）と結合します。
    """
    def __init__(self, *args, **kwargs):
        # 知識蒸留（RGBと熱画像の同期ペア）において連続フレームの stride は不要なため、
        # 引数として渡されてきてもここで安全に抜き取って破棄する。
        kwargs.pop('stride', None)
        
        # 残りの正しい引数（data_root, split など）だけを親クラスに渡して初期化する
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx: int) -> dict:
        rgb_path, thr_path = self._pairs[idx]
        
        # 1. 教師用 RGB のロード と 色の除去
        rgb_t = self._load_rgb(rgb_path)
        rgb_t = rgb_to_gray_3ch(rgb_t)
        
        # 2. 生徒用 Thermal のロード
        is_ms2 = "ms2" in str(type(self)).lower()
        dict_thr = read_and_preprocess_thermal(thr_path, is_ms2=is_ms2, return_dual=True)
        
        # ====================================================================
        # 🎯 【超重要追加】 RGBの空間解像度を、Thermalのネイティブ解像度に強制同期させる
        # ====================================================================
        _, h_thr, w_thr = dict_thr['8bit'].shape
        _, h_rgb, w_rgb = rgb_t.shape
        
        if h_thr != h_rgb or w_thr != w_rgb:
            # bilinear補間でRGBをThermalのサイズにリサイズ
            rgb_t = F.interpolate(
                rgb_t.unsqueeze(0), size=(h_thr, w_thr), mode='bilinear', align_corners=False
            ).squeeze(0)

        # ====================================================================
        # 3. サイズが完全に揃った両者を、XFeatクラッシュ防止のため32倍数へパディング
        # ====================================================================
        rgb_t    = pad_to_32_multiple(rgb_t)
        thr_8bit = pad_to_32_multiple(dict_thr['8bit'])
        thr_raw  = pad_to_32_multiple(dict_thr['raw'])
        
        return {
            'rgb_t':      rgb_t,
            'thr_t_8bit': thr_8bit,
            'thr_t_raw':  thr_raw,
            'rgb_path':   rgb_path,
            'thr_path':   thr_path
        }

# =============================================================================
# 公開クラス定義
# =============================================================================

class SThErEOSequentialDataset(BaseHybridKDDataset, OrigSThErEO):
    pass

class MS2SequentialDataset(BaseHybridKDDataset, OrigMS2):
    pass

class VividSequentialDataset(BaseHybridKDDataset, OrigVivid):
    pass

class TartanRGBTSequentialDataset(BaseHybridKDDataset, OrigTartan):
    pass

class FreiburgSequentialDataset(BaseHybridKDDataset, OrigFreiburg):
    pass

# # ==============================================================================
# # SThErEO (デュアルテンソル対応版)
# # ==============================================================================
# class SThErEOSequentialDataset(OrigSThErEO):
#     def __getitem__(self, idx: int) -> dict:
#         p_t, p_t1, T_rel, K = self._pairs[idx]
        
#         # 共通モジュールを使って 8-bit と Raw の辞書を取得
#         dict_t  = read_and_preprocess_thermal(p_t, return_dual=True)
#         dict_t1 = read_and_preprocess_thermal(p_t1, return_dual=True)
        
#         return {
#             'thr_t_8bit':  dict_t['8bit'],
#             'thr_t_raw':   dict_t['raw'],
#             'thr_t1_8bit': dict_t1['8bit'],
#             'thr_t1_raw':  dict_t1['raw'],
#             'T_rel': torch.from_numpy(T_rel).float(),
#             'K':     torch.from_numpy(K).float(),
#             'valid': torch.tensor(True),
#         }

# # ==============================================================================
# # MS2 (デュアルテンソル ＆ 動的前処理対応版)
# # ==============================================================================
# class MS2SequentialDataset(OrigMS2):
#     def __getitem__(self, idx: int) -> dict:
#         p_t, p_t1, T_rel, K = self._pairs[idx]
        
#         # is_ms2=True を渡すことで、hist_99 や Crop などの黄金前処理を適用
#         dict_t  = read_and_preprocess_thermal(p_t, is_ms2=True, return_dual=True)
#         dict_t1 = read_and_preprocess_thermal(p_t1, is_ms2=True, return_dual=True)
        
#         return {
#             'thr_t_8bit':  dict_t['8bit'],
#             'thr_t_raw':   dict_t['raw'],
#             'thr_t1_8bit': dict_t1['8bit'],
#             'thr_t1_raw':  dict_t1['raw'],
#             'T_rel': torch.from_numpy(T_rel).float(),
#             'K':     torch.from_numpy(K).float(),
#             'valid': torch.tensor(True),
#         }

# # ==============================================================================
# # VIVID (デュアルテンソル対応版)
# # ==============================================================================
# class VividSequentialDataset(OrigVivid):
#     def __getitem__(self, idx: int) -> dict:
#         p_t, p_t1, T_rel, K = self._pairs[idx]
#         dict_t  = read_and_preprocess_thermal(p_t, return_dual=True)
#         dict_t1 = read_and_preprocess_thermal(p_t1, return_dual=True)
#         return {
#             'thr_t_8bit':  dict_t['8bit'],
#             'thr_t_raw':   dict_t['raw'],
#             'thr_t1_8bit': dict_t1['8bit'],
#             'thr_t1_raw':  dict_t1['raw'],
#             'T_rel': torch.from_numpy(T_rel).float(),
#             'K':     torch.from_numpy(K).float(),
#             'valid': torch.tensor(True),
#         }

# # ==============================================================================
# # TartanRGBT (デュアルテンソル対応版)
# # ==============================================================================
# class TartanRGBTSequentialDataset(OrigTartan):
#     def __getitem__(self, idx: int) -> dict:
#         p_t, p_t1, T_rel, K = self._pairs[idx]
#         dict_t  = read_and_preprocess_thermal(p_t, return_dual=True)
#         dict_t1 = read_and_preprocess_thermal(p_t1, return_dual=True)
#         return {
#             'thr_t_8bit':  dict_t['8bit'],
#             'thr_t_raw':   dict_t['raw'],
#             'thr_t1_8bit': dict_t1['8bit'],
#             'thr_t1_raw':  dict_t1['raw'],
#             'T_rel': torch.from_numpy(T_rel).float(),
#             'K':     torch.from_numpy(K).float(),
#             'valid': torch.tensor(True),
#         }

# # ==============================================================================
# # Freiburg (デュアルテンソル対応版・ポーズなし近似)
# # ==============================================================================
# class FreiburgSequentialDataset(OrigFreiburg):
#     def __getitem__(self, idx: int) -> dict:
#         p_t, p_t1 = self._pairs[idx]
#         dict_t  = read_and_preprocess_thermal(p_t, return_dual=True)
#         dict_t1 = read_and_preprocess_thermal(p_t1, return_dual=True)
#         return {
#             'thr_t_8bit':  dict_t['8bit'],
#             'thr_t_raw':   dict_t['raw'],
#             'thr_t1_8bit': dict_t1['8bit'],
#             'thr_t1_raw':  dict_t1['raw'],
#             'T_rel': torch.eye(4).float(),
#             'K':     torch.from_numpy(self._FREIBURG_K_THERMAL).float() if hasattr(self, '_FREIBURG_K_THERMAL') else torch.eye(3).float(),
#             'valid': torch.tensor(False),
#         }