# modules/slam/thermal_line_extractor.py

import cv2
import numpy as np
import torch
from typing import List, Tuple, Optional


class ThermalLineExtractor:
    """
    熱画像から温度境界線分を抽出する。

    物理的根拠:
        熱画像の温度勾配が大きい領域（物体輪郭・材質境界）は
        RGB のテクスチャが失われた夜間・逆光条件でも安定して存在する。
        LSD (Line Segment Detector) は照明不変性があり熱画像に適合する。

    LightGlueStick との接続:
        抽出した線分 (M, 4) を LightGlueStick の line_segments 入力として渡す。
    """

    def __init__(
        self,
        method:            str   = 'lsd',       # 'lsd' | 'canny_hough' | 'deeplsd'
        gradient_threshold: float = 15.0,       # 温度勾配フィルタ閾値 (0〜255)
        min_line_length:   int   = 20,          # 最小線分長 (px)
        max_lines:         int   = 200,         # 最大線分数
    ):
        self.method             = method
        self.gradient_threshold = gradient_threshold
        self.min_line_length    = min_line_length
        self.max_lines          = max_lines

        if method == 'lsd':
            self._lsd = cv2.createLineSegmentDetector(
                cv2.LSD_REFINE_ADV,
                scale      = 0.8,    # ガウスピラミッドのスケール
                sigma_scale= 0.6,
                quant      = 2.0,
                ang_th     = 22.5,   # 角度許容幅 (deg)
                log_eps    = 0.0,
                density_th = 0.7,
                n_bins     = 1024,
            )

    # ------------------------------------------------------------------
    # 前処理: CLAHE + 温度勾配マスク
    # ------------------------------------------------------------------

    def _preprocess(self, gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        CLAHE でコントラスト強調し、温度勾配マスクを計算する。

        Returns
        -------
        enhanced  : CLAHE 適用済み 8-bit 画像
        grad_mask : 温度勾配強度マスク（閾値以上の領域）
        """
        clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Sobel 勾配（losses_kd.py と同一の計算）
        gx = cv2.Sobel(enhanced.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(enhanced.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        grad = np.sqrt(gx**2 + gy**2)

        grad_mask = (grad > self.gradient_threshold).astype(np.uint8)
        return enhanced, grad_mask

    # ------------------------------------------------------------------
    # メイン抽出
    # ------------------------------------------------------------------

    def extract(
        self, gray: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        熱画像から温度境界線分を抽出する。

        Parameters
        ----------
        gray : (H, W) uint8

        Returns
        -------
        lines       : (M, 4)  [(x1, y1, x2, y2), ...] in pixel
        line_scores : (M,)    NFA ベースのスコア（高いほど信頼性高）
        """
        enhanced, grad_mask = self._preprocess(gray)

        # 温度勾配が低い領域を白飛ばし（LSD が検出しないよう）
        # → エッジが温度境界と一致している線分のみを残す
        masked = enhanced.copy()
        masked[grad_mask == 0] = np.median(enhanced[grad_mask == 1]) \
            if grad_mask.any() else 128

        if self.method == 'lsd':
            return self._detect_lsd(masked)
        elif self.method == 'canny_hough':
            return self._detect_canny_hough(masked, grad_mask)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _detect_lsd(
        self, img: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        lines, widths, precs, nfa = self._lsd.detect(img)

        if lines is None or len(lines) == 0:
            return np.zeros((0, 4), dtype=np.float32), np.zeros(0)

        lines = lines.reshape(-1, 4).astype(np.float32)   # (M, 4)
        nfa   = nfa.flatten()

        # 最小長フィルタ
        lengths = np.sqrt(
            (lines[:, 2] - lines[:, 0])**2 +
            (lines[:, 3] - lines[:, 1])**2
        )
        valid = lengths >= self.min_line_length
        lines, nfa = lines[valid], nfa[valid]

        # スコア降順でソートして上位 max_lines を返す
        order = np.argsort(-nfa)[:self.max_lines]
        return lines[order], nfa[order]

    def _detect_canny_hough(
        self, img: np.ndarray, grad_mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        edges = cv2.Canny(img, 50, 150, apertureSize=3)
        edges = cv2.bitwise_and(edges, edges, mask=grad_mask * 255)

        linesP = cv2.HoughLinesP(
            edges,
            rho             = 1,
            theta           = np.pi / 180,
            threshold       = 30,
            minLineLength   = self.min_line_length,
            maxLineGap      = 10,
        )
        if linesP is None:
            return np.zeros((0, 4), dtype=np.float32), np.zeros(0)

        lines = linesP.reshape(-1, 4).astype(np.float32)
        lengths = np.sqrt(
            (lines[:, 2] - lines[:, 0])**2 +
            (lines[:, 3] - lines[:, 1])**2
        )
        order = np.argsort(-lengths)[:self.max_lines]
        return lines[order], lengths[order]