"""
train_joint.py
XFeat + LightGlue 同時 fine-tune（Stage 4）。

目的:
    Stage2 で XFeat を単独更新すると記述子空間が移動し、
    Stage3 で LG を再適応しても Recall が完全には回復しない。
    XFeat と LG を同一 backward path で同時更新することで
    記述子変化に LG が追従し、Recall > 100% の維持を目指す。

根拠:
    proposed(init) Recall=135.9% は XFeat+LG が整合していた証拠。
    XFeat 単独更新後は Recall=65.7%（記述子ドリフト）。

損失:
    L = L_kd(XFeat) + λ_match × L_match(LG, GT)
    L_kd    : MSE(normalize(student_feat), normalize(teacher_feat))
    L_match : NegativeLogAssignment（LG 公式損失）

使用方法:
    python train_joint.py --config configs/joint_config_stage4.yaml
"""

from __future__ import annotations

import argparse, os, sys, time, yaml
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# 設定ロード
# ---------------------------------------------------------------------------

def load_cfg(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--config',     required=True)
    p.add_argument('--device_num', default='0')
    return p.parse_args()


# ---------------------------------------------------------------------------
# KD 損失（XFeat 側）
# ---------------------------------------------------------------------------

def kd_loss(student_out, teacher_out) -> Tensor:
    """正規化 L2 蒸留損失。"""
    def _feat(out):
        if isinstance(out, (tuple, list)): return out[0]
        if isinstance(out, dict):
            k = 'feats' if 'feats' in out else list(out.keys())[0]
            return out[k]
        return out
    s = F.normalize(_feat(student_out), dim=1)
    t = F.normalize(_feat(teacher_out), dim=1)
    return F.mse_loss(s, t.detach())


# ---------------------------------------------------------------------------
# GT 対応点の計算（SThErEO 連続フレームペア）
# ---------------------------------------------------------------------------

def compute_gt_matches(
    kpts0: Tensor,   # (N, 2)
    kpts1: Tensor,   # (M, 2)
    T_rel: Tensor,   # (4, 4)
    K:     Tensor,   # (3, 3)
    th_pos: float = 3.0,
    th_neg: float = 5.0,
) -> Tuple[Tensor, Tensor]:
    """
    GT 相対姿勢から GT マッチングラベルを生成する。

    E行列からエピポーラ線を計算し、kpts1 上の最近傍点を GT マッチとする。
    th_pos px 以内 → matchable, th_neg px 以上 → unmatchable。
    """
    N, M = kpts0.shape[0], kpts1.shape[0]
    gt0 = torch.full((N,), -1, dtype=torch.long, device=kpts0.device)
    gt1 = torch.full((M,), -1, dtype=torch.long, device=kpts0.device)

    if N == 0 or M == 0:
        return gt0, gt1

    # F行列の計算
    K_np    = K.cpu().numpy().astype(np.float64)
    T_np    = T_rel.cpu().numpy().astype(np.float64)
    R, t    = T_np[:3, :3], T_np[:3, 3]
    t_cross = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    E       = t_cross @ R
    Ki      = np.linalg.inv(K_np)
    F_mat   = (Ki.T @ E @ Ki).astype(np.float32)

    k0 = kpts0.cpu().numpy()   # (N, 2)
    k1 = kpts1.cpu().numpy()   # (M, 2)

    ones0 = np.ones((N, 1), dtype=np.float32)
    ones1 = np.ones((M, 1), dtype=np.float32)
    p0h   = np.hstack([k0, ones0])   # (N, 3)
    p1h   = np.hstack([k1, ones1])   # (M, 3)

    # エピポーラ線 (N, 3)
    lines = (F_mat @ p0h.T).T   # l = F p0

    # k1 から各エピポーラ線への距離 (N, M)
    num  = np.abs(p1h @ lines.T)                          # (M, N)
    denom = np.sqrt(lines[:, 0]**2 + lines[:, 1]**2)[None] + 1e-8  # (1, N)
    dist  = (num / denom).T   # (N, M)

    # 各 kp0 に対して最近傍 kp1 を探す
    min_idx  = dist.argmin(axis=1)   # (N,)
    min_dist = dist[np.arange(N), min_idx]   # (N,)

    gt0_np = np.full(N, -1, dtype=np.int64)
    gt1_np = np.full(M, -1, dtype=np.int64)

    matchable = min_dist < th_pos
    gt0_np[matchable] = min_idx[matchable]

    # gt1: gt0 の逆引き（最初に一致したもののみ）
    for i in range(N):
        if gt0_np[i] >= 0:
            j = gt0_np[i]
            if gt1_np[j] < 0:
                gt1_np[j] = i

    # 遠い点は -1 のまま（unmatchable はデフォルト -1）
    return (torch.from_numpy(gt0_np).to(kpts0.device),
            torch.from_numpy(gt1_np).to(kpts0.device))


# ---------------------------------------------------------------------------
# メイントレーナー
# ---------------------------------------------------------------------------

class JointTrainer:
    """XFeat + LG 同時 fine-tune トレーナー。"""

    def __init__(self, cfg: Dict, device: torch.device):
        from modules.model import XFeatModel
        self.cfg    = cfg
        self.device = device

        # ── XFeat（学習可能）────────────────────────────────────────────
        self.xfeat = XFeatModel().to(device).train()
        xw = cfg.get('xfeat_weights')
        if xw and os.path.isfile(xw):
            self.xfeat.load_state_dict(
                torch.load(xw, map_location=device, weights_only=True))
            print(f"[Joint] XFeat loaded: {xw}")
        else:
            print("[Joint] XFeat: default weights")

        # ── Teacher（frozen）─────────────────────────────────────────────
        self.teacher = XFeatModel().to(device).eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

        # ── LightGlue（学習可能）─────────────────────────────────────────
        from train_lightglue_ft import build_lg_model
        lg_weights = cfg.get('lg_weights')
        self.lg = build_lg_model(cfg, device)
        if lg_weights and os.path.isfile(lg_weights):
            self.lg.load_state_dict(
                torch.load(lg_weights, map_location=device, weights_only=True))
            print(f"[Joint] LG loaded: {lg_weights}")

        self.lambda_match = cfg.get('lambda_match', 0.5)
        self.th_pos       = cfg.get('th_pos', 3.0)
        self.th_neg       = cfg.get('th_neg', 5.0)
        self.max_kp       = cfg.get('max_keypoints', 512)

        # ── Optimizer（XFeat + LG 同時）──────────────────────────────────
        all_params = (list(self.xfeat.parameters()) +
                      list(self.lg.parameters()))
        lr = cfg.get('lr', 1e-5)
        self.opt = optim.Adam(all_params, lr=lr)
        self.ckpt_path = cfg.get('ckpt_save_path',
                                 'checkpoints/pipeline/stage4_joint')
        os.makedirs(self.ckpt_path, exist_ok=True)

        # wandb
        self.use_wandb = self._init_wandb(cfg)

    def _init_wandb(self, cfg):
        if cfg.get('no_wandb', False):
            return False
        try:
            import wandb
            wandb.init(
                project=cfg.get('wandb_project', 'thermal-xfeat-kd'),
                name=cfg.get('wandb_run_name', 'stage4_joint'),
                config=cfg,
            )
            return True
        except Exception as e:
            print(f"[wandb] {e}")
            return False

    def _extract_kp(self, img: Tensor) -> Dict:
        """XFeat で KP と記述子を抽出する。"""
        feats, kp_logits, _ = self.xfeat(img)
        feats = F.normalize(feats, dim=1)
        B, C, Hf, Wf = feats.shape
        _, H, W = img.shape[0], img.shape[-2], img.shape[-1]
        probs    = F.softmax(kp_logits, dim=1)
        kp_score = probs[:, :64].sum(dim=1)

        kpts_l, scores_l, descs_l = [], [], []
        for b in range(B):
            s   = kp_score[b].flatten()
            f   = feats[b].reshape(C, -1).T
            k   = min(self.max_kp, s.shape[0])
            top = s.topk(k).indices
            iy  = (top // Wf).float() * (H / Hf)
            ix  = (top %  Wf).float() * (W / Wf)
            kp  = torch.stack([ix, iy], dim=1)
            kpts_l.append(kp)
            scores_l.append(s[top])
            descs_l.append(f[top])
        return {'keypoints': kpts_l, 'descriptors': descs_l,
                'keypoint_scores': scores_l}

    def _pad(self, lst: List[Tensor]) -> Tensor:
        max_n = max(t.shape[0] for t in lst)
        D     = lst[0].shape[1] if lst[0].dim() > 1 else None
        B     = len(lst)
        out   = lst[0].new_zeros(B, max_n, D) if D else lst[0].new_zeros(B, max_n)
        for b, t in enumerate(lst):
            n = t.shape[0]
            if D:
                out[b, :n] = t
            else:
                out[b, :n] = t
        return out

    def train(self, loader: DataLoader, n_steps: int):
        print(f"\n{'='*60}")
        print(f"  Joint Fine-tune  steps={n_steps}  λ_match={self.lambda_match}")
        print(f"  XFeat + LG 同時学習（記述子ドリフト根本解決）")
        print(f"{'='*60}")

        it   = iter(loader)
        step = 0
        while step < n_steps:
            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader)
                batch = next(it)

            img0  = batch['thr_t'].to(self.device)
            img1  = batch['thr_t1'].to(self.device)
            T_rel = batch['T_rel'].to(self.device)
            K     = batch['K'].to(self.device)
            B     = img0.shape[0]

            # ── XFeat KD 損失 ────────────────────────────────────────────
            with torch.no_grad():
                t_out = self.teacher(img0)
            s_out     = self.xfeat(img0)
            l_kd      = kd_loss(s_out, t_out)

            # ── LG マッチング損失 ─────────────────────────────────────────
            p0 = self._extract_kp(img0)
            p1 = self._extract_kp(img1)

            sz = torch.tensor([[img0.shape[-2], img0.shape[-1]]] * B,
                               dtype=torch.float32, device=self.device)

            # GT マッチの生成（バッチ内で1ペアずつ）
            gt0_list, gt1_list = [], []
            for b in range(B):
                g0, g1 = compute_gt_matches(
                    p0['keypoints'][b], p1['keypoints'][b],
                    T_rel[b], K[b],
                    self.th_pos, self.th_neg)
                gt0_list.append(g0)
                gt1_list.append(g1)

            gt_m0 = self._pad([g.float() for g in gt0_list])   # (B, N)
            gt_m1 = self._pad([g.float() for g in gt1_list])   # (B, M)

            lg_in = {
                'image0': {
                    'keypoints':       self._pad(p0['keypoints']),
                    'descriptors':     self._pad(p0['descriptors']),
                    'keypoint_scores': self._pad(p0['keypoint_scores']),
                    'image_size':      sz,
                },
                'image1': {
                    'keypoints':       self._pad(p1['keypoints']),
                    'descriptors':     self._pad(p1['descriptors']),
                    'keypoint_scores': self._pad(p1['keypoint_scores']),
                    'image_size':      sz,
                },
                'gt_matches0': gt_m0.long(),
                'gt_matches1': gt_m1.long(),
            }

            try:
                pred      = self.lg(lg_in)
                loss_dict = self.lg.loss(pred, lg_in)
                l_match   = (loss_dict.get('total', next(iter(loss_dict.values())))
                             if isinstance(loss_dict, dict) else loss_dict)
            except Exception as e:
                print(f"[step {step}] LG error: {e}")
                step += 1
                continue

            # ── 統合損失 ─────────────────────────────────────────────────
            l_total = l_kd + self.lambda_match * l_match

            self.opt.zero_grad()
            l_total.backward()
            nn.utils.clip_grad_norm_(
                list(self.xfeat.parameters()) + list(self.lg.parameters()), 1.0)
            self.opt.step()

            if step % 100 == 0:
                msg = (f"[{step:05d}/{n_steps}] "
                       f"total={l_total.item():.4f}  "
                       f"kd={l_kd.item():.4f}  "
                       f"match={l_match.item():.4f}")
                print(msg)
                if self.use_wandb:
                    try:
                        import wandb
                        wandb.log({'joint/total': l_total.item(),
                                   'joint/kd':    l_kd.item(),
                                   'joint/match': l_match.item()},
                                  step=step)
                    except Exception:
                        pass

            step += 1

        # ── 保存 ─────────────────────────────────────────────────────────
        xfeat_out = os.path.join(self.ckpt_path, 'xfeat_joint.pth')
        lg_out    = os.path.join(self.ckpt_path, 'lg_joint.pth')
        torch.save(self.xfeat.state_dict(), xfeat_out)
        torch.save(self.lg.state_dict(),    lg_out)
        print(f"\n[Joint] saved: {xfeat_out}")
        print(f"[Joint] saved: {lg_out}")

        if self.use_wandb:
            try:
                import wandb
                wandb.finish()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# データローダー（SThErEO Sequential を使用）
# ---------------------------------------------------------------------------

def build_loader(cfg: Dict) -> DataLoader:
    from modules.dataset.thermal.sequential import SThErEOSequentialDataset

    sthereo_root = cfg['data_roots']['sthereo']
    stride       = cfg.get('stride', 3)

    ds = SThErEOSequentialDataset(
        data_root=sthereo_root,
        stride=stride,
        split='train',
    )
    print(f"[Joint] SThErEO train: {len(ds._pairs)} pairs")

    return DataLoader(
        ds,
        batch_size  = cfg.get('batch_size', 16),
        shuffle     = True,
        num_workers = 2,
        drop_last   = True,
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_num
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg    = load_cfg(args.config)

    print(f"[Joint] config: {args.config}")
    print(f"[Joint] device: {device}")

    loader  = build_loader(cfg)
    trainer = JointTrainer(cfg, device)
    n_steps = cfg.get('n_steps', 5000)
    trainer.train(loader, n_steps)


if __name__ == '__main__':
    main()