# modules/slam/thermal_frontend.py
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple, Optional

from modules.model import XFeatModel
from modules.dataset.thermal.preprocessing import read_and_preprocess_thermal


class ThermalFrontend:
    """
    ORB-SLAM3 互換の熱画像特徴抽出フロントエンド。

    設計方針:
        - letterbox resize で K 行列を数学的に補正
        - K 行列の不整合を排除し RANSAC の精度を保証
        - MS2 / SThErEO / VIVID の前処理を統一
    """

    def __init__(
        self,
        weights_path: str,
        max_kp: int = 1024,
        device: torch.device = None,
    ):
        self.device  = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_kp  = max_kp

        self.model = XFeatModel().to(self.device).eval()
        state = torch.load(weights_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        for p in self.model.parameters():
            p.requires_grad_(False)

    # ------------------------------------------------------------------
    # 画像前処理
    # ------------------------------------------------------------------

    def preprocess(
        self, img_gray: np.ndarray, target_hw: Tuple[int, int] = (480, 640)
    ) -> Tuple[torch.Tensor, np.ndarray, Tuple[float, float, int, int]]:
        """
        グレースケール熱画像をネットワーク入力テンソルへ変換。

        Returns
        -------
        tensor   : (1, 3, H, W) float32 [0,1]
        img_vis  : (H, W, 3) uint8  可視化用
        meta     : (scale_x, scale_y, pad_x, pad_y)
                   座標逆変換に使用
        """
        h_orig, w_orig = img_gray.shape[:2]
        th, tw = target_hw
        scale  = min(tw / w_orig, th / h_orig)
        nw, nh = int(w_orig * scale), int(h_orig * scale)

        img_resized = cv2.resize(img_gray, (nw, nh), interpolation=cv2.INTER_LINEAR)
        pad_w = tw - nw
        pad_h = th - nh
        top   = pad_h // 2
        left  = pad_w // 2

        img_padded = cv2.copyMakeBorder(
            img_resized, top, pad_h - top, left, pad_w - left,
            cv2.BORDER_CONSTANT, value=0
        )

        img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_GRAY2RGB)
        tensor  = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        tensor  = tensor.unsqueeze(0).to(self.device)

        # meta: (元画像 / リサイズ後のスケール, パディング量)
        meta = (w_orig / nw, h_orig / nh, left, top)
        return tensor, img_padded, meta

    def adjust_K(
        self, K: np.ndarray, meta: Tuple[float, float, int, int]
    ) -> np.ndarray:
        """
        letterbox 変換後の K 行列を数学的に補正する。

        letterbox: スケール → パディング の順なので
            fx_new = fx / sx,  cx_new = cx / sx + pad_x
        """
        sx, sy, pad_x, pad_y = meta
        K_new = K.copy().astype(np.float64)
        K_new[0, 0] /= sx          # fx
        K_new[1, 1] /= sy          # fy
        K_new[0, 2] = K[0, 2] / sx + pad_x  # cx
        K_new[1, 2] = K[1, 2] / sy + pad_y  # cy
        return K_new

    # ------------------------------------------------------------------
    # 特徴検出
    # ------------------------------------------------------------------

    @torch.no_grad()
    def detect(
        self, tensor: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        ThermalXFeat でキーポイントを検出する。

        Returns
        -------
        kpts   : (N, 2)  画素座標 (x, y) in padded image space
        descs  : (N, 64) L2 正規化済み記述子
        scores : (N,)    検出スコア
        """
        feats, kp_logits, _ = self.model(tensor)
        feats = F.normalize(feats, dim=1)
        B, C, Hf, Wf = feats.shape
        H, W = tensor.shape[2], tensor.shape[3]

        probs    = F.softmax(kp_logits, dim=1)
        kp_score = probs[:, :64].sum(dim=1)  # P(keypoint) = 1 - P(dustbin)

        scores_flat = kp_score[0].flatten()
        feats_flat  = feats[0].reshape(C, -1).T

        k       = min(self.max_kp, scores_flat.shape[0])
        top_idx = scores_flat.topk(k).indices

        iy = (top_idx // Wf).float() * (H / Hf)
        ix = (top_idx %  Wf).float() * (W / Wf)

        kpts   = torch.stack([ix, iy], dim=1).cpu().numpy()
        descs  = feats_flat[top_idx].cpu().numpy()
        scores = scores_flat[top_idx].cpu().numpy()

        return kpts, descs, scores

    # ------------------------------------------------------------------
    # 座標逆変換（padded space → original image space）
    # ------------------------------------------------------------------

    @staticmethod
    def unpad_kpts(
        kpts: np.ndarray, meta: Tuple[float, float, int, int]
    ) -> np.ndarray:
        """
        letterbox 後の画素座標を元画像座標に戻す。
        RANSAC / E 行列推定は元の K 行列空間で行う場合に使用。
        """
        sx, sy, pad_x, pad_y = meta
        kpts_out = kpts.copy().astype(np.float32)
        kpts_out[:, 0] = (kpts[:, 0] - pad_x) * sx
        kpts_out[:, 1] = (kpts[:, 1] - pad_y) * sy
        return kpts_out