"""
gf_modules/models/thermal_xfeat.py
ThermalXFeat を gluefactory の BaseModel として定義。

gluefactory の TwoViewPipeline が呼ぶインターフェース:
    extractor({'image': tensor(B,3,H,W)})
    → {'keypoints': List[Tensor(N,2)],
       'keypoint_scores': List[Tensor(N,)],
       'descriptors': List[Tensor(N,64)]}

設計方針:
    - ThermalXFeat は完全 frozen（重みを変更しない）
    - P(keypoint) = 1 - P(dustbin) で 65ch softmax を正しく使用
    - ボーダー除去で画像端のノイジーなキーポイントを除外
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _ensure_repo_in_path() -> None:
    """
    プロジェクトルートを sys.path に追加する。

    このファイルは以下のどちらに置かれるかによって深さが変わる:
      A) gf_modules/models/thermal_xfeat.py          → 2階層上がルート
      B) third_party/glue-factory/gluefactory/
         models/extractors/thermal_xfeat.py           → 5階層上がルート

    'modules/model.py' が存在するディレクトリをルートとして探索する。
    """
    _THIS = os.path.dirname(os.path.abspath(__file__))
    candidate = _THIS
    for _ in range(8):
        candidate = os.path.dirname(candidate)
        if os.path.isfile(os.path.join(candidate, 'modules', 'model.py')):
            if candidate not in sys.path:
                sys.path.insert(0, candidate)
            return
    # 見つからなかった場合は 2階層上をフォールバックとして追加
    fallback = os.path.dirname(os.path.dirname(_THIS))
    if fallback not in sys.path:
        sys.path.insert(0, fallback)


try:
    from gluefactory.models.base_model import BaseModel
    _HAS_GLUEFACTORY = True
except ImportError:
    # gluefactory が未インストールの場合のフォールバック
    BaseModel = nn.Module
    _HAS_GLUEFACTORY = False


class ThermalXFeat(BaseModel):
    """
    ThermalXFeat を gluefactory の特徴抽出器として登録。

    gluefactory の TwoViewPipeline から呼ばれる。
    ThermalXFeat の重みは frozen（fine-tuning しない）。
    LightGlue だけが学習対象になる。
    """

    # gluefactory BaseModel が必要とするデフォルト設定
    default_conf = {
        'weights':           None,    # ThermalXFeat の重みファイルパス
        'max_num_keypoints': 512,     # 最大キーポイント数
        'remove_borders':    4,       # 画像端から除外するピクセル数
        'detection_threshold': 0.0,  # スコア閾値（0 = 全点使用）
    }
    required_data_keys = ['image']

    # gluefactory BaseModel のサブクラスは _init で初期化する
    def _init(self, conf) -> None:
        _ensure_repo_in_path()
        from modules.model import XFeatModel

        self.net = XFeatModel()

        w = getattr(conf, 'weights', None)
        if w and os.path.isfile(str(w)):
            state = torch.load(str(w), map_location='cpu', weights_only=True)
            self.net.load_state_dict(state)
            print(f'[ThermalXFeat] Loaded: {w}')
        else:
            print('[ThermalXFeat] WARNING: weights not found → random init')

        # 完全 frozen
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.net.eval()

    def _forward(self, data: dict) -> dict:
        """
        特徴抽出。gluefactory の LightGlue が要求する
        stacked Tensor (B, N, *) 形式で返す。

        LightGlue の forward は:
            b, m, _ = kpts0.shape  # (B, N, 2) の Tensor を期待
        しているため、List[Tensor] ではなく stacked Tensor が必須。
        バッチ内で N を揃えるため max_num_keypoints にゼロパディングする。

        Args:
            data['image']: (B, 3, H, W) または (3, H, W)

        Returns:
            {
                'keypoints':       Tensor(B, N, 2)   画素座標 (x, y)
                'keypoint_scores': Tensor(B, N)
                'descriptors':     Tensor(B, N, 64)  L2正規化済み
            }
        """
        img = data['image']
        if img.dim() == 3:
            img = img.unsqueeze(0)

        B, _, H, W = img.shape
        max_kp = self.conf.max_num_keypoints
        border = self.conf.remove_borders
        thr    = self.conf.detection_threshold
        device = img.device

        with torch.no_grad():
            feats, kp_logits, _ = self.net(img)

        feats = F.normalize(feats, dim=1)
        _, C, Hf, Wf = feats.shape

        # P(keypoint) = 1 - P(dustbin)
        probs    = F.softmax(kp_logits, dim=1)
        kp_score = probs[:, :64].sum(dim=1)            # (B, Hf, Wf)

        kpts_out   = []
        scores_out = []
        descs_out  = []
        valid_masks = []

        for b in range(B):
            s_flat = kp_score[b].flatten()
            f_flat = feats[b].reshape(C, -1).T         # (Hf*Wf, C)

            if thr > 0:
                s_flat = s_flat.clone()
                s_flat[s_flat < thr] = 0.0

            k   = min(max_kp, s_flat.shape[0])
            top = s_flat.topk(k).indices

            iy  = (top // Wf).float() * (H / Hf)
            ix  = (top %  Wf).float() * (W / Wf)
            kp  = torch.stack([ix, iy], dim=1)         # (k, 2)
            sc  = s_flat[top]
            dc  = f_flat[top]                          # (k, C)

            # ボーダー除去
            mask = (
                (kp[:, 0] >= border) & (kp[:, 0] < W - border) &
                (kp[:, 1] >= border) & (kp[:, 1] < H - border)
            )
            kp, sc, dc = kp[mask], sc[mask], dc[mask]

            # max_num_keypoints にゼロパディング（全バッチで N を揃える）
            n = kp.shape[0]
            if n < max_kp:
                pad = max_kp - n
                kp = torch.cat([kp, kp.new_zeros(pad, 2)], dim=0)
                sc = torch.cat([sc, sc.new_zeros(pad)],    dim=0)
                dc = torch.cat([dc, dc.new_zeros(pad, C)], dim=0)
            else:
                kp, sc, dc = kp[:max_kp], sc[:max_kp], dc[:max_kp]

            # 有効なキーポイント数を記録（パディング位置は False）
            valid = torch.zeros(max_kp, dtype=torch.bool, device=device)
            valid[:n] = True
            valid_masks.append(valid)

            kpts_out.append(kp)
            scores_out.append(sc)
            descs_out.append(dc)

        return {
            'keypoints':       torch.stack(kpts_out,   dim=0),  # (B, N, 2)
            'keypoint_scores': torch.stack(scores_out, dim=0),  # (B, N)
            'descriptors':     torch.stack(descs_out,  dim=0),  # (B, N, C)
            # valid_mask: パディングされたゼロ点を homography_matcher に伝える
            # True = 有効なキーポイント / False = ゼロパディング
            'valid_mask':      torch.stack(valid_masks, dim=0),  # (B, N)
        }

    def loss(self, pred: dict, data: dict):
        """特徴抽出器は損失を持たない。"""
        raise NotImplementedError(
            'ThermalXFeat does not have a loss. '
            'Use TwoViewPipeline.loss() instead.'
        )


# gluefactory が未インストールの場合の standalone 使用
if not _HAS_GLUEFACTORY:

    class ThermalXFeatStandalone(nn.Module):
        """gluefactory なしでも動作する Standalone 版。"""

        def __init__(self, weights: Optional[str] = None,
                     max_num_keypoints: int = 512,
                     remove_borders: int = 4):
            super().__init__()
            _ensure_repo_in_path()
            from modules.model import XFeatModel
            self.net = XFeatModel()
            if weights and os.path.isfile(weights):
                self.net.load_state_dict(
                    torch.load(weights, map_location='cpu', weights_only=True))
            for p in self.net.parameters():
                p.requires_grad_(False)
            self.net.eval()
            self.max_num_keypoints = max_num_keypoints
            self.remove_borders    = remove_borders

        @torch.no_grad()
        def forward(self, image: Tensor):
            if image.dim() == 3:
                image = image.unsqueeze(0)
            B, _, H, W = image.shape
            feats, kp_logits, _ = self.net(image)
            feats = F.normalize(feats, dim=1)
            _, C, Hf, Wf = feats.shape
            probs    = F.softmax(kp_logits, dim=1)
            kp_score = probs[:, :64].sum(dim=1)
            kpts_l, scores_l, descs_l = [], [], []
            for b in range(B):
                s = kp_score[b].flatten()
                f = feats[b].reshape(C, -1).T
                k   = min(self.max_num_keypoints, s.shape[0])
                top = s.topk(k).indices
                iy  = (top // Wf).float() * (H / Hf)
                ix  = (top %  Wf).float() * (W / Wf)
                kp  = torch.stack([ix, iy], dim=1)
                bd  = self.remove_borders
                mask = ((kp[:,0]>=bd)&(kp[:,0]<W-bd)&
                        (kp[:,1]>=bd)&(kp[:,1]<H-bd))
                kpts_l.append(kp[mask])
                scores_l.append(s[top[mask]])
                descs_l.append(f[top[mask]])
            return {'keypoints': kpts_l,
                    'keypoint_scores': scores_l,
                    'descriptors': descs_l}