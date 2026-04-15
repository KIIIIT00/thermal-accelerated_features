"""
modules/training/train_lightglue_ft.py
LightGlue Fine-tuning トレーナー。

設計方針:
    - ThermalXFeat は完全 frozen（重みを変更しない）
    - LightGlue のみを Thermal ペアで fine-tuning
    - 損失: L_match（NLL）+ lambda_epi * L_epi（Sampson）
    - 学習データ: Freiburg（train） + TartanRGBT（train）
    - 評価データ: SThErEO, VIVID（fine-tuning に使用しない）

損失の意味:
    L_match:
        GT ポーズから F 行列を計算し、Sampson 距離でインライアを判定。
        インライア対応の log P(match) を最大化、
        アウトライア対応の log P(unmatch) を最大化する NLL 損失。

    L_epi:
        LightGlue が出力したマッチ点のエピポーラ整合性を直接測定。
        GT ポーズがある場合のみ有効（TartanRGBT）。
        Freiburg は valid=False のため L_epi はスキップ。
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from modules.model import XFeatModel


# ---------------------------------------------------------------------------
# ユーティリティ
# ---------------------------------------------------------------------------

def _init_wandb(args: Any) -> bool:
    if getattr(args, 'no_wandb', False):
        return False
    try:
        import wandb
        wandb.init(
            project  = getattr(args, 'wandb_project',  'thermal-xfeat-lightglue-ft'),
            name     = getattr(args, 'wandb_run_name',  None),
            group    = getattr(args, 'wandb_group',     'lightglue_ft'),
            tags     = getattr(args, 'wandb_tags',      []),
            config   = vars(args),
            dir      = getattr(args, 'ckpt_save_path',  'checkpoints/lightglue_ft'),
        )
        return True
    except Exception as e:
        print(f"[LG-FT] wandb init failed: {e}")
        return False


def total_grad_norm(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return total ** 0.5


# ---------------------------------------------------------------------------
# 損失関数
# ---------------------------------------------------------------------------

def _compute_F_matrix(K: Tensor, T_rel: Tensor) -> Tensor:
    """
    GT ポーズから基本行列 F を計算する。
    K_inv は CPU で計算して GPU に転送（cuSOLVER エラー回避）。
    """
    R = T_rel[:3, :3]
    t = T_rel[:3, 3]
    t_s = torch.zeros(3, 3, device=t.device, dtype=t.dtype)
    t_s[0, 1] = -t[2]; t_s[0, 2] =  t[1]
    t_s[1, 0] =  t[2]; t_s[1, 2] = -t[0]
    t_s[2, 0] = -t[1]; t_s[2, 1] =  t[0]
    E     = t_s @ R
    K_inv = torch.inverse(K.cpu().double()).float().to(K.device)
    return K_inv.T @ E @ K_inv   # (3, 3)


def _sampson_dist(pts1: Tensor, pts2: Tensor, F_mat: Tensor) -> Tensor:
    """Sampson 距離 (N,)"""
    N    = pts1.shape[0]
    ones = pts1.new_ones(N, 1)
    p1h  = torch.cat([pts1, ones], dim=1)  # (N, 3)
    p2h  = torch.cat([pts2, ones], dim=1)
    Fp1  = (F_mat @ p1h.T).T               # (N, 3)
    Ftp2 = (F_mat.T @ p2h.T).T
    numer = (p2h * Fp1).sum(1) ** 2
    denom = Fp1[:, 0]**2 + Fp1[:, 1]**2 + Ftp2[:, 0]**2 + Ftp2[:, 1]**2
    return numer / denom.clamp(min=1e-8)


def lightglue_matching_loss(
    scores:    Tensor,
    pts1:      Tensor,
    pts2:      Tensor,
    K:         Tensor,
    T_rel:     Tensor,
    use_gt:    bool,
    inlier_thr: float = 3.0,
) -> Tensor:
    """
    LightGlue のマッチングスコアに対する NLL 損失。

    GT ポーズ T_rel と K から基本行列 F を計算し、
    Sampson 距離でインライア/アウトライアを判定する。

    インライア対応:   log P(matched)    を最大化 → loss を下げる
    アウトライア対応: log P(unmatched)  を最大化 → loss を下げる

    Args:
        scores   : (N1+1, N2+1) LightGlue のダスト付きスコア行列
                   最終行・列はダストビン（unmatch）スコア
        pts1     : (N1, 2) 画像1のキーポイント（画素座標）
        pts2     : (N2, 2) 画像2のキーポイント（画素座標）
        K        : (3, 3)  カメラ内部行列
        T_rel    : (4, 4)  相対姿勢
        use_gt   : True なら GT ポーズ使用（TartanRGBT）
                   False なら 8 点法で F を推定（Freiburg）
        inlier_thr: Sampson 距離でのインライア判定閾値（画素単位）

    Returns:
        loss: scalar
    """
    N1, N2 = pts1.shape[0], pts2.shape[0]
    if N1 < 8 or N2 < 8:
        return scores.new_zeros(1).squeeze()

    # ── 基本行列の計算 ─────────────────────────────────────────────────────
    with torch.no_grad():
        if use_gt:
            F_mat = _compute_F_matrix(K, T_rel)
        else:
            # 8 点法（Freiburg: GT ポーズなし）
            import cv2
            p1_np = pts1.cpu().numpy().astype(np.float32)
            p2_np = pts2.cpu().numpy().astype(np.float32)
            F_np, mask = cv2.findFundamentalMat(
                p1_np, p2_np, cv2.FM_8POINT)
            if F_np is None:
                return scores.new_zeros(1).squeeze()
            F_mat = torch.from_numpy(F_np).float().to(pts1.device)

    # ── MNN で仮対応を生成（GT ラベル作成用）──────────────────────────────
    # scores は (N1+1, N2+1): dustbin を除いた (N1, N2) 部分を使う
    sim     = scores[:N1, :N2]                        # (N1, N2)
    nn12    = sim.argmax(dim=1)                        # (N1,)
    nn21    = sim.argmax(dim=0)                        # (N2,)
    ids     = torch.arange(N1, device=sim.device)
    mutual  = nn21[nn12] == ids                        # (N1,) 相互最近傍マスク

    with torch.no_grad():
        # Sampson 距離でインライア判定
        if mutual.sum() >= 4:
            pts1_m = pts1[mutual]
            pts2_m = pts2[nn12[mutual]]
            sd     = _sampson_dist(pts1_m, pts2_m, F_mat)
            inlier = sd < (inlier_thr ** 2)
        else:
            inlier = torch.zeros(mutual.sum(), dtype=torch.bool,
                                 device=pts1.device)

    # ── NLL 損失 ──────────────────────────────────────────────────────────
    # LightGlue の scores は log-softmax 済み想定
    # インライア: log P(pts2[j] | pts1[i]) を最大化
    # アウトライア/unmatch: log P(dustbin | pts1[i]) を最大化

    log_probs = F.log_softmax(scores, dim=1)           # (N1+1, N2+1)
    total_loss = scores.new_zeros(1)
    n_terms    = 0

    mutual_idx = ids[mutual]
    for k, i in enumerate(mutual_idx):
        j = nn12[i]
        if inlier[k]:
            # インライア → マッチを促進
            total_loss = total_loss - log_probs[i, j]
        else:
            # アウトライア → ダストビンを促進
            total_loss = total_loss - log_probs[i, N2]   # pts1[i] → dustbin
            total_loss = total_loss - log_probs[N1, j]   # pts2[j] → dustbin
        n_terms += 1

    # マッチなし点のダストビン損失
    unmatched1 = ~mutual
    if unmatched1.any():
        total_loss = total_loss - log_probs[ids[unmatched1], N2].sum()
        n_terms   += unmatched1.sum().item()

    if n_terms == 0:
        return scores.new_zeros(1).squeeze()
    return total_loss / n_terms


def lightglue_epi_loss(
    pts1:      Tensor,
    pts2:      Tensor,
    scores:    Tensor,
    K:         Tensor,
    T_rel:     Tensor,
    threshold: float = 2.0,
) -> Tensor:
    """
    LightGlue 出力マッチのエピポーラ整合性損失。

    LightGlue が出力したマッチ（argmax で取得）の Sampson 距離を最小化する。
    これにより「LightGlue がエピポーラ幾何に整合したマッチを学習する」。

    Args:
        pts1, pts2 : キーポイント座標
        scores     : (N1+1, N2+1) LightGlue スコア行列
        K, T_rel   : GT カメラパラメータ
        threshold  : ソフトインライア重みのスケール
    """
    N1, N2 = pts1.shape[0], pts2.shape[0]
    if N1 < 8 or N2 < 8:
        return scores.new_zeros(1).squeeze()

    with torch.no_grad():
        F_mat  = _compute_F_matrix(K, T_rel)
        sim    = scores[:N1, :N2]
        nn12   = sim.argmax(dim=1)
        nn21   = sim.argmax(dim=0)
        ids    = torch.arange(N1, device=sim.device)
        mutual = nn21[nn12] == ids

    if mutual.sum() < 4:
        return scores.new_zeros(1).squeeze()

    pts1_m = pts1[mutual]
    pts2_m = pts2[nn12[mutual]]

    sd      = _sampson_dist(pts1_m, pts2_m, F_mat)     # (M,)
    weights = torch.exp(-sd.detach() / (threshold ** 2))
    loss    = (sd * weights).mean()
    return loss


# ---------------------------------------------------------------------------
# LightGlue Fine-tuning トレーナー
# ---------------------------------------------------------------------------

class LightGlueFTTrainer:

    def __init__(self, args: Any):
        self.args = args
        self.dev  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[LG-FT] Device: {self.dev}")

        # ── ThermalXFeat（完全 frozen）─────────────────────────────────────
        self.thermal_feat = XFeatModel().to(self.dev).eval()
        w = getattr(args, 'thermal_weights', None)
        if w and os.path.isfile(w):
            self.thermal_feat.load_state_dict(
                torch.load(w, map_location=self.dev, weights_only=True))
            print(f"[LG-FT] ThermalXFeat loaded: {w}")
        else:
            print("[LG-FT] WARNING: thermal_weights not found → random weights")
        for p in self.thermal_feat.parameters():
            p.requires_grad_(False)

        # ── LightGlue（fine-tuning 対象）──────────────────────────────────
        self.lightglue = self._load_lightglue(args)

        # ── チェックポイント・ログ ─────────────────────────────────────────
        self.ckpt_path = getattr(args, 'ckpt_save_path',
                                  'checkpoints/lightglue_ft/default')
        os.makedirs(self.ckpt_path, exist_ok=True)
        logdir = os.path.join(self.ckpt_path, 'logdir',
                              'lg_ft_' + time.strftime('%Y_%m_%d-%H_%M_%S'))
        os.makedirs(logdir, exist_ok=True)
        self.writer    = SummaryWriter(log_dir=logdir)
        self.use_wandb = _init_wandb(args)
        if self.use_wandb:
            try:
                import wandb
                if wandb.run is not None:
                    print(f"[LG-FT] wandb run dir: {wandb.run.dir}")
            except Exception:
                pass

    def _load_lightglue(self, args: Any) -> nn.Module:
        """LightGlue をロードして GPU に転送する。"""
        try:
            from lightglue import LightGlue
        except ImportError:
            raise ImportError(
                "LightGlue が見つかりません。\n"
                "pip install git+https://github.com/cvg/LightGlue.git"
            )
        features        = getattr(args, 'lg_features',          'xfeat')
        depth_conf      = getattr(args, 'lg_depth_confidence',  -1)
        width_conf      = getattr(args, 'lg_width_confidence',  -1)

        lg = LightGlue(
            features         = features,
            depth_confidence = depth_conf,
            width_confidence = width_conf,
        ).to(self.dev).train()

        print(f"[LG-FT] LightGlue loaded (features={features})")
        n = sum(p.numel() for p in lg.parameters() if p.requires_grad)
        print(f"[LG-FT] Trainable params: {n:,}")
        return lg

    @torch.no_grad()
    def _extract_features(
        self,
        img: Tensor,
        max_kp: int,
    ):
        """
        ThermalXFeat で特徴抽出（frozen）。

        Returns:
            kpts  : (N, 2)  画素座標 (x, y)
            descs : (N, 64) L2 正規化済み記述子
            scores: (N,)    キーポイントスコア
        """
        feats, kp_logits, hmap = self.thermal_feat(img)
        feats = F.normalize(feats, dim=1)
        B, C, Hf, Wf = feats.shape
        H, W = img.shape[2], img.shape[3]

        # P(keypoint) = 1 - P(dustbin)
        probs    = F.softmax(kp_logits, dim=1)
        kp_score = probs[:, :64].sum(dim=1)   # (B, Hf, Wf)

        kpts_list, descs_list, scores_list = [], [], []
        for b in range(B):
            scores_flat = kp_score[b].flatten()
            feats_flat  = feats[b].reshape(C, -1).T   # (Hf*Wf, C)
            k           = min(max_kp, scores_flat.shape[0])
            top_idx     = scores_flat.topk(k).indices
            iy = (top_idx // Wf).float() * (H / Hf)
            ix = (top_idx %  Wf).float() * (W / Wf)
            kpts_list.append(torch.stack([ix, iy], dim=1))    # (k, 2)
            descs_list.append(feats_flat[top_idx])             # (k, C)
            scores_list.append(scores_flat[top_idx])           # (k,)

        return kpts_list, descs_list, scores_list

    def _to_lg_input(
        self,
        kpts:   Tensor,
        descs:  Tensor,
        scores: Tensor,
        H: int,
        W: int,
    ) -> dict:
        """LightGlue の入力形式に変換する。"""
        # 画素座標 → [-1, 1] に正規化
        kpts_norm = kpts.clone()
        kpts_norm[:, 0] = kpts[:, 0] / W * 2.0 - 1.0
        kpts_norm[:, 1] = kpts[:, 1] / H * 2.0 - 1.0
        return {
            'keypoints':   kpts_norm.unsqueeze(0),  # (1, N, 2)
            'descriptors': descs.unsqueeze(0),       # (1, N, 64)
            'keypoint_scores': scores.unsqueeze(0),  # (1, N)
        }

    def _log(self, metrics: dict, step: int) -> None:
        for k, v in metrics.items():
            self.writer.add_scalar(k, v, step)
        if self.use_wandb:
            try:
                import wandb
                wandb.log(metrics, step=step)
            except Exception:
                pass

    def _save(self, tag: str) -> None:
        filename = f'lightglue_ft_{tag}.pth'
        local    = os.path.join(self.ckpt_path, filename)
        torch.save(self.lightglue.state_dict(), local)
        print(f"[LG-FT] Saved (local): {local}")
        if self.use_wandb:
            try:
                import wandb
                if wandb.run is not None:
                    wp = os.path.join(wandb.run.dir, filename)
                    torch.save(self.lightglue.state_dict(), wp)
                    wandb.save(wp, base_path=wandb.run.dir)
                    print(f"[LG-FT] Saved (wandb): {wp}")
            except Exception as e:
                print(f"[LG-FT] WARNING: wandb save failed: {e}")

    def run(self, loader: DataLoader) -> None:
        print("\n" + "=" * 60)
        print("  LightGlue Fine-tuning")
        print(f"  Train datasets : Freiburg + TartanRGBT")
        print(f"  Eval datasets  : SThErEO, VIVID（未使用）")
        print("=" * 60)

        lr         = getattr(self.args, 'lr',             1e-5)
        n_steps    = getattr(self.args, 'n_steps',        20000)
        grad_clip  = getattr(self.args, 'grad_clip',      1.0)
        log_every  = getattr(self.args, 'log_every',      100)
        save_every = getattr(self.args, 'save_ckpt_every', 2000)
        lambda_epi = getattr(self.args, 'lambda_epi',     0.1)
        inlier_thr = getattr(self.args, 'inlier_threshold', 3.0)
        epi_thr    = getattr(self.args, 'epi_threshold',  2.0)
        max_kp     = getattr(self.args, 'max_keypoints',  512)

        opt = optim.Adam(
            filter(lambda p: p.requires_grad, self.lightglue.parameters()),
            lr=lr,
        )
        scheduler = optim.lr_scheduler.StepLR(
            opt,
            step_size = getattr(self.args, 'lr_step', 10000),
            gamma     = getattr(self.args, 'lr_gamma', 0.5),
        )

        data_iter = iter(loader)
        step      = 0

        while step < n_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch     = next(data_iter)

            thr_t  = batch['thr_t'].to(self.dev)    # (B, 3, H, W)
            thr_t1 = batch['thr_t1'].to(self.dev)
            T_rel  = batch['T_rel'].to(self.dev)    # (B, 4, 4)
            K      = batch['K'].to(self.dev)        # (B, 3, 3)
            valid  = batch['valid']                  # (B,)

            B, _, H, W = thr_t.shape

            # ── 特徴抽出（frozen）─────────────────────────────────────────
            kpts_list1, descs_list1, scores_list1 = \
                self._extract_features(thr_t,  max_kp)
            kpts_list2, descs_list2, scores_list2 = \
                self._extract_features(thr_t1, max_kp)

            losses_match = []
            losses_epi   = []

            for b in range(B):
                kpts1  = kpts_list1[b]
                descs1 = descs_list1[b]
                sc1    = scores_list1[b]
                kpts2  = kpts_list2[b]
                descs2 = descs_list2[b]
                sc2    = scores_list2[b]

                if kpts1.shape[0] < 8 or kpts2.shape[0] < 8:
                    continue

                # ── LightGlue フォワード ─────────────────────────────────
                inp0 = self._to_lg_input(kpts1, descs1, sc1, H, W)
                inp1 = self._to_lg_input(kpts2, descs2, sc2, H, W)
                try:
                    pred = self.lightglue({'image0': inp0, 'image1': inp1})
                except Exception as e:
                    print(f"[LG-FT] LightGlue forward error: {e}")
                    continue

                # LightGlue の出力からスコア行列を取得
                # matches0: (1, N1) 各キーポイントのマッチ先インデックス(-1=unmatch)
                # scores は log_assignment: (1, N1+1, N2+1) または matching_scores
                if 'log_assignment' in pred:
                    scores_mat = pred['log_assignment'][0]  # (N1+1, N2+1)
                elif 'matching_scores0' in pred:
                    # older API: スコアから再構築
                    scores_mat = pred['matching_scores0'][0].unsqueeze(-1)
                    scores_mat = scores_mat.expand(-1, kpts2.shape[0] + 1)
                else:
                    continue

                # ── 損失計算 ─────────────────────────────────────────────
                use_gt = valid[b].item()

                l_match = lightglue_matching_loss(
                    scores     = scores_mat,
                    pts1       = kpts1,
                    pts2       = kpts2,
                    K          = K[b],
                    T_rel      = T_rel[b],
                    use_gt     = use_gt,
                    inlier_thr = inlier_thr,
                )
                losses_match.append(l_match)

                # L_epi は GT ポーズがある場合のみ
                if use_gt:
                    l_epi = lightglue_epi_loss(
                        pts1      = kpts1,
                        pts2      = kpts2,
                        scores    = scores_mat,
                        K         = K[b],
                        T_rel     = T_rel[b],
                        threshold = epi_thr,
                    )
                    losses_epi.append(l_epi)

            if not losses_match:
                step += 1
                continue

            l_match_mean = torch.stack(losses_match).mean()
            l_epi_mean   = (torch.stack(losses_epi).mean()
                            if losses_epi else l_match_mean.new_zeros(1).squeeze())
            loss = l_match_mean + lambda_epi * l_epi_mean

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.lightglue.parameters(), grad_clip)
            opt.step()
            scheduler.step()

            if step % log_every == 0:
                self._log({
                    'lg_ft/loss_total' : loss.item(),
                    'lg_ft/loss_match' : l_match_mean.item(),
                    'lg_ft/loss_epi'   : l_epi_mean.item(),
                    'lg_ft/lr'         : opt.param_groups[0]['lr'],
                    'lg_ft/grad_norm'  : total_grad_norm(self.lightglue),
                    'lg_ft/n_batch'    : len(losses_match),
                }, step)
                print(
                    f"[LG-FT {step:06d}] "
                    f"total={loss.item():.4f}  "
                    f"match={l_match_mean.item():.4f}  "
                    f"epi={l_epi_mean.item():.4f}  "
                    f"lr={opt.param_groups[0]['lr']:.2e}"
                )

            if (step + 1) % save_every == 0:
                self._save(f'step{step + 1}')

            step += 1

        self._save('final')
        print("[LG-FT] Fine-tuning done.")

    def finalize(self) -> None:
        self.writer.close()
        if self.use_wandb:
            try:
                import wandb
                wandb.finish()
            except Exception:
                pass