"""
modules/training/train_post_kd.py
Post-KD トレーナークラス。

3段階の訓練を管理する:
  Stage 1: Thermal Homographic Adaptation
           キーポイントブランチ・fine_matcherを自己教師あり適応
           バックボーン（block1〜block_fusion）とheatmap_headはKD済みで凍結

  Stage 2: 幾何整合ファインチューニング
    Step 2a: バックボーン凍結維持 → キーポイント+fine_matcherを幾何損失で再最適化
    Step 2b: 後段バックボーク（block4/5/fusion）を極小LRで解凍

  Stage 3: Stub（Future Work）

設計原則:
  - KD で確立した特徴マップ品質を破壊しない（段階的解凍）
  - 損失関数は losses_post_kd.py に分離（alike依存なし）
  - wandb + TensorBoard の二重ロギングを継続（train_kd.py と同じ方針）
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from modules.model import XFeatModel
from modules.training.train_kd import total_grad_norm, _init_wandb
from modules.training.losses_post_kd import (
    repeatability_loss,
    fine_matching_loss,
    epipolar_loss,
    reprojection_loss,
)


# ---------------------------------------------------------------------------
# バックボーン凍結ユーティリティ
# ---------------------------------------------------------------------------

# Stage 1 / Stage 2a: 凍結するモジュール名（KD済み部分を保護）
_FREEZE_STAGE1 = [
    'norm', 'skip1',
    'block1', 'block2', 'block3', 'block4', 'block5',
    'block_fusion', 'heatmap_head',
]

# Stage 2b: さらに後段だけ解凍するモジュール名（極小LR）
_UNFREEZE_STAGE2B = ['block4', 'block5', 'block_fusion']


def _freeze_modules(model: nn.Module, names: list[str]) -> None:
    for name, param in model.named_parameters():
        if any(name.startswith(n) for n in names):
            param.requires_grad_(False)


def _unfreeze_modules(model: nn.Module, names: list[str]) -> None:
    for name, param in model.named_parameters():
        if any(name.startswith(n) for n in names):
            param.requires_grad_(True)


def _trainable_params(model: nn.Module) -> list:
    return [p for p in model.parameters() if p.requires_grad]


# ---------------------------------------------------------------------------
# PostKDTrainer
# ---------------------------------------------------------------------------

class PostKDTrainer:
    """
    Post-KD 訓練の全ステージを管理するトレーナー。

    使い方:
        trainer = PostKDTrainer(args)
        trainer.run_stage1(homo_loader)
        trainer.run_stage2(seq_loader)
    """

    def __init__(self, args: Any):
        self.dev  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.args = args

        # ── モデルロード ───────────────────────────────────────────────────
        self.model = XFeatModel().to(self.dev)
        kd_weights = getattr(args, 'kd_weights', None)  # KD 済みの生徒重み
        if kd_weights and os.path.isfile(kd_weights):
            self.model.load_state_dict(
                torch.load(kd_weights, map_location=self.dev, weights_only=True))
            print(f"[PostKD] Loaded KD weights from: {kd_weights}")
        else:
            print("[PostKD] WARNING: kd_weights not specified. Using random weights.")

        # ── 保存ディレクトリ ───────────────────────────────────────────────
        self.ckpt_path = getattr(args, 'ckpt_save_path', 'checkpoints/post_kd')
        os.makedirs(self.ckpt_path, exist_ok=True)

        # ── ロギング ──────────────────────────────────────────────────────
        logdir = os.path.join(
            self.ckpt_path, 'logdir',
            'post_kd_' + time.strftime('%Y_%m_%d-%H_%M_%S'))
        os.makedirs(logdir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=logdir)
        self.use_wandb = _init_wandb(args)

    # ── ロギングヘルパー ──────────────────────────────────────────────────

    def _log(self, metrics: Dict[str, float], step: int) -> None:
        for k, v in metrics.items():
            self.writer.add_scalar(k, v, step)
        if self.use_wandb:
            try:
                import wandb
                wandb.log(metrics, step=step)
            except Exception:
                pass

    def _save(self, tag: str) -> None:
        path = os.path.join(self.ckpt_path, f'post_kd_{tag}.pth')
        torch.save(self.model.state_dict(), path)
        print(f"[PostKD] Saved: {path}")
        if self.use_wandb:
            try:
                import wandb
                wandb.save(path)
            except Exception:
                pass

    # ======================================================================
    # Stage 1: Thermal Homographic Adaptation
    # ======================================================================

    def run_stage1(self, homo_loader: DataLoader) -> None:
        """
        Stage 1: キーポイントブランチ + fine_matcher を自己教師あり適応。

        凍結: norm, skip1, block1~block5, block_fusion, heatmap_head
        学習: keypoint_head, fine_matcher
        """
        print("\n" + "=" * 60)
        print("  Post-KD Stage 1: Thermal Homographic Adaptation")
        print("=" * 60)

        # バックボーン凍結
        self.model.train()
        _freeze_modules(self.model, _FREEZE_STAGE1)

        trainable = _trainable_params(self.model)
        n_trainable = sum(p.numel() for p in trainable)
        print(f"[Stage1] Trainable parameters: {n_trainable:,}")
        print(f"[Stage1] Training: keypoint_head, fine_matcher")

        lr         = getattr(self.args, 's1_lr',             1e-4)
        n_steps    = getattr(self.args, 's1_n_steps',        30_000)
        grad_clip  = getattr(self.args, 'grad_clip',         1.0)
        log_every  = getattr(self.args, 'log_every',         100)
        save_every = getattr(self.args, 'save_ckpt_every',   2_000)
        lambda_fine= getattr(self.args, 's1_lambda_fine',    0.5)

        opt = optim.Adam(trainable, lr=lr)
        scheduler = optim.lr_scheduler.StepLR(
            opt,
            step_size=getattr(self.args, 's1_lr_step', 10_000),
            gamma=getattr(self.args, 'lr_gamma', 0.5),
        )

        data_iter = iter(homo_loader)
        step = 0

        while step < n_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(homo_loader)
                batch = next(data_iter)

            thr   = batch['thr'].to(self.dev)      # (B, 3, H, W)
            thr_w = batch['thr_w'].to(self.dev)    # (B, 3, H, W)
            H_mat = batch['H_mat'].to(self.dev)    # (B, 3, 3)

            # ── frozen バックボーンで feats / hmap を取得 ─────────────────
            with torch.no_grad():
                feats_f,  kp_logits,   hmap_f  = self.model(thr)
                feats_wf, kp_logits_w, hmap_wf = self.model(thr_w)
                feats_f  = F.normalize(feats_f,  dim=1)
                feats_wf = F.normalize(feats_wf, dim=1)

            # ── 学習可能な keypoint_head を別途フォワード ─────────────────
            # NOTE: XFeatModel.forward() は内部で keypoint_head も呼ぶが、
            #       frozen バックボーンの中間表現を取り出せないため
            #       forward() をそのまま使い、その出力のうち kp_logits だけ使う。
            #       backbone は no_grad ブロックで保護されているため
            #       forward() 内の norm/block1-5/fusion には勾配が流れない。
            _, kp_logits_train,  _ = self.model(thr)
            _, kp_logits_w_train,_ = self.model(thr_w)

            # ── 損失計算 ─────────────────────────────────────────────────
            l_repeat = repeatability_loss(
                kp_logits_train, kp_logits_w_train,
                H_mat, hmap_f, hmap_wf,
            )
            l_fine = fine_matching_loss(
                feats_f, feats_wf,
                kp_logits_train, kp_logits_w_train,
                self.model.fine_matcher,
                H_mat,
                n_pts=getattr(self.args, 's1_n_pts', 256),
            )
            loss = l_repeat + lambda_fine * l_fine

            # ── バックワード ───────────────────────────────────────────────
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
            opt.step()
            scheduler.step()

            if step % log_every == 0:
                gnorm = total_grad_norm(self.model)
                self._log({
                    's1/loss_total'   : loss.item(),
                    's1/loss_repeat'  : l_repeat.item(),
                    's1/loss_fine'    : l_fine.item(),
                    's1/lr'           : opt.param_groups[0]['lr'],
                    's1/grad_norm'    : gnorm,
                }, step)
                print(
                    f"[S1 {step:06d}] total={loss.item():.4f}  "
                    f"repeat={l_repeat.item():.4f}  "
                    f"fine={l_fine.item():.4f}"
                )

            if (step + 1) % save_every == 0:
                self._save(f's1_step{step + 1}')

            step += 1

        self._save('s1_final')
        print("[Stage1] Done.")

    # ======================================================================
    # Stage 2: 幾何整合ファインチューニング
    # ======================================================================

    def run_stage2(self, seq_loader: DataLoader) -> None:
        """
        Stage 2: 再投影誤差・エピポーラ拘束でキーポイントを幾何的に最適化。

        Step 2a: バックボーン凍結維持 → KD済み特徴空間を破壊しない
        Step 2b: block4/5/fusion のみ極小 LR で解凍
        """
        print("\n" + "=" * 60)
        print("  Post-KD Stage 2: Geometric Consistency Fine-tuning")
        print("=" * 60)

        n_steps_2a = getattr(self.args, 's2a_n_steps', 10_000)
        n_steps_2b = getattr(self.args, 's2b_n_steps',  5_000)

        self._run_stage2_step(
            seq_loader,
            unfreeze_extra=[],
            lr=getattr(self.args, 's2a_lr', 5e-5),
            n_steps=n_steps_2a,
            step_offset=0,
            tag='2a',
        )
        self._run_stage2_step(
            seq_loader,
            unfreeze_extra=_UNFREEZE_STAGE2B,
            lr=getattr(self.args, 's2b_lr', 1e-5),
            n_steps=n_steps_2b,
            step_offset=n_steps_2a,
            tag='2b',
        )

        self._save('s2_final')
        print("[Stage2] Done.")

    def _run_stage2_step(
        self,
        seq_loader: DataLoader,
        unfreeze_extra: list[str],
        lr: float,
        n_steps: int,
        step_offset: int,
        tag: str,
    ) -> None:
        """Stage 2 の各ステップ共通ループ。"""
        print(f"\n[Stage2-{tag}] lr={lr}  n_steps={n_steps}")

        self.model.train()
        # Stage 1 と同じ凍結状態から開始
        _freeze_modules(self.model, _FREEZE_STAGE1)

        # 追加解凍（Step 2b のみ）
        if unfreeze_extra:
            _unfreeze_modules(self.model, unfreeze_extra)
            print(f"[Stage2-{tag}] Unfrozen: {unfreeze_extra}")

        trainable = _trainable_params(self.model)
        n_tp = sum(p.numel() for p in trainable)
        print(f"[Stage2-{tag}] Trainable: {n_tp:,}")

        opt = optim.Adam(trainable, lr=lr)
        grad_clip  = getattr(self.args, 'grad_clip',    1.0)
        log_every  = getattr(self.args, 'log_every',    100)
        save_every = getattr(self.args, 'save_ckpt_every', 2_000)
        lambda_epi = getattr(self.args, 's2_lambda_epi', 0.5)
        thr_epi    = getattr(self.args, 's2_epi_threshold', 2.0)

        data_iter = iter(seq_loader)
        step = 0

        while step < n_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(seq_loader)
                batch = next(data_iter)

            thr_t  = batch['thr_t'].to(self.dev)   # (B, 3, H, W)
            thr_t1 = batch['thr_t1'].to(self.dev)  # (B, 3, H, W)
            T_rel  = batch['T_rel'].to(self.dev)    # (B, 4, 4)
            K      = batch['K'].to(self.dev)        # (B, 3, 3)
            valid  = batch['valid']                 # (B,) bool

            # ── フォワード ────────────────────────────────────────────────
            feats_t,  kp_t,  hmap_t  = self.model(thr_t)
            feats_t1, kp_t1, hmap_t1 = self.model(thr_t1)
            feats_t  = F.normalize(feats_t,  dim=1)
            feats_t1 = F.normalize(feats_t1, dim=1)

            # ── キーポイント座標の抽出（信頼性マップ上位 N 点）──────────
            B = thr_t.shape[0]
            total_loss = feats_t.new_zeros(1)
            n_valid_batches = 0

            for b in range(B):
                # 信頼性マップから上位 N 点を選択
                hmap_b = hmap_t[b, 0]                          # (H/8, W/8)
                topk_val, topk_idx = torch.topk(
                    hmap_b.reshape(-1),
                    k=min(256, hmap_b.numel()),
                )
                iy = topk_idx // hmap_b.shape[1]
                ix = topk_idx % hmap_b.shape[1]
                # 8x8 ブロック中心座標（元解像度）
                pts1_b = torch.stack([
                    ix.float() * 8 + 4,
                    iy.float() * 8 + 4,
                ], dim=1)  # (N, 2)

                # 対応する t+1 フレームの特徴点を最近傍マッチング
                # feats をサンプリングして MNN で pts2 を推定
                with torch.no_grad():
                    f1 = feats_t[b, :, iy, ix].permute(1, 0)   # (N, 64)
                    Hf, Wf = feats_t1.shape[-2:]
                    f2_all = feats_t1[b].reshape(64, -1).T       # (HW, 64)
                    sim = f1 @ f2_all.T                           # (N, HW)
                    nn_idx = sim.argmax(dim=1)                    # (N,)
                    iy2 = nn_idx // Wf
                    ix2 = nn_idx % Wf
                    pts2_b = torch.stack([
                        ix2.float() * 8 + 4,
                        iy2.float() * 8 + 4,
                    ], dim=1)  # (N, 2)

                K_b     = K[b]      # (3, 3)
                T_rel_b = T_rel[b]  # (4, 4)
                use_gt  = valid[b].item()

                # エピポーラ損失
                l_epi = epipolar_loss(
                    pts1_b, pts2_b, K_b, T_rel_b,
                    use_gt_pose=use_gt,
                    inlier_threshold=thr_epi,
                )
                # 再投影損失（姿勢既知の場合のみ）
                l_reproj = reprojection_loss(
                    pts1_b, pts2_b, K_b, T_rel_b,
                    inlier_threshold=thr_epi,
                ) if use_gt else feats_t.new_zeros(1).squeeze()

                loss_b = l_reproj + lambda_epi * l_epi
                total_loss = total_loss + loss_b
                n_valid_batches += 1

            loss = total_loss / max(n_valid_batches, 1)

            # ── バックワード ───────────────────────────────────────────────
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
            opt.step()

            global_step = step + step_offset
            if step % log_every == 0:
                gnorm = total_grad_norm(self.model)
                self._log({
                    f's{tag}/loss_total': loss.item(),
                    f's{tag}/lr'        : opt.param_groups[0]['lr'],
                    f's{tag}/grad_norm' : gnorm,
                }, global_step)
                print(
                    f"[S{tag} {step:06d}] loss={loss.item():.4f}  "
                    f"lr={opt.param_groups[0]['lr']:.2e}"
                )

            if (step + 1) % save_every == 0:
                self._save(f's{tag}_step{step + 1}')

            step += 1

    # ======================================================================
    # Stage 3: Stub（Future Work）
    # ======================================================================

    def run_stage3_stub(self) -> None:
        """
        Stage 3: Tightly-coupled SLAM End-to-End 微調整（Future Work）。

        本実装ではスタブのみ。将来的に以下を実装する:
          - 微分可能なSLAMフロントエンドを構築
          - 再投影誤差 + IMUプリインテグレーション誤差を
            特徴抽出器まで逆伝播
          - 極小LR (1e-6) でモデル全体を微調整
        """
        print("\n[Stage3] Future Work: End-to-End SLAM fine-tuning.")
        print("  → To implement: differentiable SLAM front-end")
        print("    + reprojection error + IMU preintegration loss")
        print("    + full model fine-tuning with lr=1e-6")

    # ======================================================================
    # 終了処理
    # ======================================================================

    def finalize(self) -> None:
        self.writer.close()
        if self.use_wandb:
            try:
                import wandb
                wandb.finish()
            except Exception:
                pass