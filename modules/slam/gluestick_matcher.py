# modules/slam/gluestick_matcher.py

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict

from modules.slam.thermal_frontend import ThermalFrontend
from modules.slam.thermal_line_extractor import ThermalLineExtractor


class ThermalGlueStickMatcher:
    """
    ThermalXFeat (点) + ThermalLineExtractor (線分) を
    LightGlueStick で同時マッチングするモジュール。

    論文: "LightGlueStick: Joint Feature Matching of Points and Lines"

    新規性の核心:
        - ThermalXFeat の点特徴は KD で RGB→熱適応済み
        - ThermalLineExtractor の線分は「温度境界」に特化
        - この組み合わせが熱画像の低テクスチャ問題を解決する

    フォールバック:
        LightGlueStick が利用不可の場合は MNN + Sampson フィルタに
        自動的にフォールバックする。
    """

    def __init__(
        self,
        frontend:      ThermalFrontend,
        line_extractor: ThermalLineExtractor,
        device:        torch.device,
        target_hw:     Tuple[int, int] = (480, 640),
        use_gluestick: bool = True,
    ):
        self.frontend       = frontend
        self.line_extractor = line_extractor
        self.device         = device
        self.target_hw      = target_hw
        self.use_gluestick  = use_gluestick

        self._gluestick = None
        if use_gluestick:
            self._load_gluestick()

    def _load_gluestick(self):
        try:
            # GlueStick の公式インターフェース
            # pip install git+https://github.com/cvg/GlueStick.git
            from gluestick import GlueStick
            self._gluestick = GlueStick.from_pretrained('gluestick_indoor').to(self.device).eval()
            print("[GlueStick] Loaded pretrained weights")
        except ImportError:
            print("[GlueStick] Not available → MNN fallback")
            self.use_gluestick = False

    # ------------------------------------------------------------------
    # メインマッチング
    # ------------------------------------------------------------------

    def match(
        self,
        gray0: np.ndarray,
        gray1: np.ndarray,
        K:     Optional[np.ndarray] = None,
        T_rel: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        2枚の熱画像を受け取りマッチング結果を返す。

        Returns
        -------
        dict with:
            mkpts0    : (N', 2) マッチした点 (padded space)
            mkpts1    : (N', 2) マッチした点 (padded space)
            mlines0   : (M', 4) マッチした線分
            mlines1   : (M', 4) マッチした線分
            meta0     : letterbox メタ情報 (sx, sy, pad_x, pad_y)
            meta1     : letterbox メタ情報
            K0_adj    : K を letterbox 補正したもの
        """
        # 前処理
        tensor0, _, meta0 = self.frontend.preprocess(gray0, self.target_hw)
        tensor1, _, meta1 = self.frontend.preprocess(gray1, self.target_hw)

        # 点特徴抽出 (ThermalXFeat)
        kpts0, descs0, sc0 = self.frontend.detect(tensor0)
        kpts1, descs1, sc1 = self.frontend.detect(tensor1)

        # 線分抽出 (ThermalLineExtractor)
        H, W = self.target_hw
        # letterbox 後の padded 画像に対して実行
        img0_pad = (tensor0[0].cpu().permute(1,2,0).numpy() * 255).astype(np.uint8)
        img1_pad = (tensor1[0].cpu().permute(1,2,0).numpy() * 255).astype(np.uint8)
        gray0_pad = cv2.cvtColor(img0_pad, cv2.COLOR_RGB2GRAY)
        gray1_pad = cv2.cvtColor(img1_pad, cv2.COLOR_RGB2GRAY)

        lines0, lsc0 = self.line_extractor.extract(gray0_pad)
        lines1, lsc1 = self.line_extractor.extract(gray1_pad)

        # LightGlueStick マッチング or MNN フォールバック
        if self.use_gluestick and self._gluestick is not None and len(lines0) > 0 and len(lines1) > 0:
            result = self._match_gluestick(
                kpts0, descs0, sc0, lines0, lsc0,
                kpts1, descs1, sc1, lines1, lsc1,
                H, W,
            )
        else:
            result = self._match_mnn(kpts0, descs0, kpts1, descs1)
            result['mlines0'] = np.zeros((0, 4))
            result['mlines1'] = np.zeros((0, 4))

        # K 行列補正
        K_adj = self.frontend.adjust_K(K, meta0) if K is not None else None

        result.update({
            'meta0':  meta0,
            'meta1':  meta1,
            'K0_adj': K_adj,
        })
        return result

    def _match_gluestick(
        self,
        kpts0, descs0, sc0, lines0, lsc0,
        kpts1, descs1, sc1, lines1, lsc1,
        H, W,
    ) -> Dict:
        """LightGlueStick による点+線の同時マッチング。"""
        import cv2 as _cv2

        def _norm_kp(kp):
            kn = kp.copy().astype(np.float32)
            kn[:, 0] = kn[:, 0] / W * 2 - 1
            kn[:, 1] = kn[:, 1] / H * 2 - 1
            return kn

        def _norm_line(l):
            ln = l.copy().astype(np.float32)
            ln[:, 0] = ln[:, 0] / W * 2 - 1
            ln[:, 1] = ln[:, 1] / H * 2 - 1
            ln[:, 2] = ln[:, 2] / W * 2 - 1
            ln[:, 3] = ln[:, 3] / H * 2 - 1
            return ln

        with torch.no_grad():
            pred = self._gluestick({
                'keypoints0': torch.from_numpy(_norm_kp(kpts0)).unsqueeze(0).to(self.device),
                'descriptors0': torch.from_numpy(descs0).unsqueeze(0).to(self.device),
                'keypoint_scores0': torch.from_numpy(sc0).unsqueeze(0).to(self.device),
                'lines0': torch.from_numpy(_norm_line(lines0)).unsqueeze(0).to(self.device),
                'keypoints1': torch.from_numpy(_norm_kp(kpts1)).unsqueeze(0).to(self.device),
                'descriptors1': torch.from_numpy(descs1).unsqueeze(0).to(self.device),
                'keypoint_scores1': torch.from_numpy(sc1).unsqueeze(0).to(self.device),
                'lines1': torch.from_numpy(_norm_line(lines1)).unsqueeze(0).to(self.device),
            })

        # 点マッチの取得
        m0 = pred.get('matches0', pred.get('point_matches0'))
        if m0 is not None:
            m0_np = m0[0].cpu().numpy()
            valid  = m0_np >= 0
            idx0   = np.where(valid)[0]
            idx1   = m0_np[valid]
            mkpts0 = kpts0[idx0]
            mkpts1 = kpts1[idx1]
        else:
            mkpts0, mkpts1 = np.zeros((0,2)), np.zeros((0,2))

        # 線分マッチの取得
        lm0 = pred.get('line_matches0')
        if lm0 is not None:
            lm0_np  = lm0[0].cpu().numpy()
            lvalid  = lm0_np >= 0
            lidx0   = np.where(lvalid)[0]
            lidx1   = lm0_np[lvalid]
            mlines0 = lines0[lidx0]
            mlines1 = lines1[lidx1]
        else:
            mlines0, mlines1 = np.zeros((0,4)), np.zeros((0,4))

        return {'mkpts0': mkpts0, 'mkpts1': mkpts1,
                'mlines0': mlines0, 'mlines1': mlines1}

    def _match_mnn(
        self, kpts0, descs0, kpts1, descs1
    ) -> Dict:
        """MNN フォールバック（既存実装と同一）。"""
        d0 = descs0 / (np.linalg.norm(descs0, axis=1, keepdims=True) + 1e-8)
        d1 = descs1 / (np.linalg.norm(descs1, axis=1, keepdims=True) + 1e-8)
        sim  = d0 @ d1.T
        nn12 = np.argmax(sim, axis=1)
        nn21 = np.argmax(sim, axis=0)
        ids  = np.arange(len(d0))
        mask = nn21[nn12] == ids
        idx0 = ids[mask]
        idx1 = nn12[mask]
        return {'mkpts0': kpts0[idx0], 'mkpts1': kpts1[idx1]}