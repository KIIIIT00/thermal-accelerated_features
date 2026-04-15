"""
modules/training/train_post_kd.py
Post-KD トレーナークラス（改善版）

変更点:
  - Stage 1: フォワード4回→2回に削減
  - Stage 2: hmap を no_grad で取得（一貫性）
  - コメントを詳細化
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from modules.model import XFeatModel
from modules.training.train_kd import total_grad_norm, _init_wandb
from modules.training.losses_post_kd import (
    repeatability_loss,
    fine_matching_loss,
    geometric_feature_consistency_loss,
)


# ---------------------------------------------------------------------------
# バックボーン凍結ユーティリティ
# ---------------------------------------------------------------------------

_FREEZE_STAGE1    = ['norm', 'skip1', 'block1', 'block2', 'block3',
                      'block4', 'block5', 'block_fusion', 'heatmap_head']
# Stage2a: block_fusion を解放 → feats_t に grad_fn が付く
_UNFREEZE_STAGE2A = ['block_fusion']
# Stage2b: 中間層まで追加解放
_UNFREEZE_STAGE2B = ['block4', 'block5', 'block_fusion']


def _freeze_modules(model: nn.Module, names: List[str]) -> None:
    for name, param in model.named_parameters():
        if any(name.startswith(n) for n in names):
            param.requires_grad_(False)


def _unfreeze_modules(model: nn.Module, names: List[str]) -> None:
    for name, param in model.named_parameters():
        if any(name.startswith(n) for n in names):
            param.requires_grad_(True)


def _trainable_params(model: nn.Module) -> List[nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


# ---------------------------------------------------------------------------
# PostKDTrainer
# ---------------------------------------------------------------------------

class PostKDTrainer:

    def __init__(self, args: Any):
        self.dev  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.args = args

        self.model = XFeatModel().to(self.dev)
        kd_weights = getattr(args, 'kd_weights', None)
        if kd_weights and os.path.isfile(kd_weights):
            self.model.load_state_dict(
                torch.load(kd_weights, map_location=self.dev, weights_only=True))
            print(f"[PostKD] Loaded KD weights from: {kd_weights}")
        else:
            print("[PostKD] WARNING: kd_weights not specified. Using random weights.")

        self.ckpt_path = getattr(args, 'ckpt_save_path', 'checkpoints/post_kd')
        os.makedirs(self.ckpt_path, exist_ok=True)

        logdir = os.path.join(self.ckpt_path, 'logdir',
                              'post_kd_' + time.strftime('%Y_%m_%d-%H_%M_%S'))
        os.makedirs(logdir, exist_ok=True)
        self.writer   = SummaryWriter(log_dir=logdir)
        self.use_wandb = _init_wandb(args)

        # wandb が有効な場合、run.dir を表示して保存先を明示する
        if self.use_wandb:
            try:
                import wandb
                if wandb.run is not None:
                    print(f"[PostKD] wandb run dir: {wandb.run.dir}")
                    print(f"[PostKD] Checkpoints will be saved to:")
                    print(f"         Local : {self.ckpt_path}/")
                    print(f"         wandb : {wandb.run.dir}/")
            except Exception:
                pass

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
        """
        チェックポイントを保存する。

        保存先:
            1. self.ckpt_path/{tag}.pth         ← 常に保存（ローカル）
            2. wandb.run.dir/post_kd_{tag}.pth  ← wandb 有効時（run 配下）
               → wandb.save() で wandb クラウドにもアップロードされる
        """
        filename = f'post_kd_{tag}.pth'

        # ── 1. ローカル保存 ───────────────────────────────────────────────
        local_path = os.path.join(self.ckpt_path, filename)
        torch.save(self.model.state_dict(), local_path)
        print(f"[PostKD] Saved (local): {local_path}")

        # ── 2. wandb run ディレクトリへの保存 ────────────────────────────
        if self.use_wandb:
            try:
                import wandb
                if wandb.run is not None:
                    wandb_path = os.path.join(wandb.run.dir, filename)
                    torch.save(self.model.state_dict(), wandb_path)
                    # wandb.save() でクラウドにアップロード
                    wandb.save(wandb_path, base_path=wandb.run.dir)
                    print(f"[PostKD] Saved (wandb): {wandb_path}")
            except Exception as e:
                print(f"[PostKD] WARNING: wandb save failed: {e}")

    # ======================================================================
    # Stage 1: Thermal Homographic Adaptation
    # ======================================================================

    def run_stage1(self, homo_loader: DataLoader) -> None:
        print("\n" + "=" * 60)
        print("  Post-KD Stage 1: Thermal Homographic Adaptation")
        print("=" * 60)

        self.model.train()
        _freeze_modules(self.model, _FREEZE_STAGE1)

        trainable  = _trainable_params(self.model)
        n_trainable = sum(p.numel() for p in trainable)
        print(f"[Stage1] Trainable: {n_trainable:,}  (keypoint_head + fine_matcher)")

        lr          = getattr(self.args, 's1_lr',           1e-4)
        n_steps     = getattr(self.args, 's1_n_steps',      30_000)
        grad_clip   = getattr(self.args, 'grad_clip',        1.0)
        log_every   = getattr(self.args, 'log_every',        100)
        save_every  = getattr(self.args, 'save_ckpt_every',  2_000)
        lambda_fine = getattr(self.args, 's1_lambda_fine',   0.5)

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

            thr   = batch['thr'].to(self.dev)
            thr_w = batch['thr_w'].to(self.dev)
            H_mat = batch['H_mat'].to(self.dev)

            # ── 改善: フォワード4回→2回に削減 ────────────────────────────
            #
            # 旧実装:
            #   no_grad: model(thr)   → feats_f, kp, hmap_f     (1回目)
            #   no_grad: model(thr_w) → feats_wf, kp, hmap_wf   (2回目)
            #   勾配あり: model(thr)  → kp_logits_train          (3回目)
            #   勾配あり: model(thr_w)→ kp_logits_w_train        (4回目)
            #
            # 新実装:
            #   凍結済みパラメータは requires_grad=False なので
            #   勾配ありフォワードでも凍結層への勾配は発生しない。
            #   したがって1回のフォワードで feats/hmap/kp_logits を全て取得できる。
            #
            #   勾配あり: model(thr)  → feats_f, kp_logits_train, hmap_f    (1回目)
            #   勾配あり: model(thr_w)→ feats_wf, kp_logits_w_train, hmap_wf (2回目)
            #
            # ただし feats と hmap は損失計算で stop_gradient が必要なため detach()。
            # kp_logits_train のみ勾配を保持する。

            feats_f,  kp_logits_train,   hmap_f  = self.model(thr)
            feats_wf, kp_logits_w_train, hmap_wf = self.model(thr_w)

            # feats と hmap は目標値として使うため stop_gradient
            feats_f   = F.normalize(feats_f.detach(),  dim=1)
            feats_wf  = F.normalize(feats_wf.detach(), dim=1)
            hmap_f    = hmap_f.detach()
            hmap_wf   = hmap_wf.detach()

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

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
            opt.step()
            scheduler.step()

            if step % log_every == 0:
                self._log({
                    's1/loss_total' : loss.item(),
                    's1/loss_repeat': l_repeat.item(),
                    's1/loss_fine'  : l_fine.item(),
                    's1/lr'         : opt.param_groups[0]['lr'],
                    's1/grad_norm'  : total_grad_norm(self.model),
                }, step)
                print(f"[S1 {step:06d}] total={loss.item():.4f}  "
                      f"repeat={l_repeat.item():.4f}  fine={l_fine.item():.4f}")

            if (step + 1) % save_every == 0:
                self._save(f's1_step{step + 1}')

            step += 1

        self._save('s1_final')
        print("[Stage1] Done.")

    # ======================================================================
    # Stage 2: 幾何整合ファインチューニング
    # ======================================================================

    def run_stage2(self, seq_loader: DataLoader) -> None:
        print("\n" + "=" * 60)
        print("  Post-KD Stage 2: Geometric Consistency Fine-tuning")
        print("=" * 60)
        print("  Stage2a: keypoint_head + fine_matcher + block_fusion")
        print("  Stage2b: + block4 + block5（中間層まで追加解放）")
        print("=" * 60)

        self._run_stage2_step(
            seq_loader, unfreeze_extra=_UNFREEZE_STAGE2A,
            lr=getattr(self.args, 's2a_lr', 5e-5),
            n_steps=getattr(self.args, 's2a_n_steps', 10_000),
            step_offset=0, tag='2a',
        )
        self._run_stage2_step(
            seq_loader, unfreeze_extra=_UNFREEZE_STAGE2B,
            lr=getattr(self.args, 's2b_lr', 1e-5),
            n_steps=getattr(self.args, 's2b_n_steps', 5_000),
            step_offset=getattr(self.args, 's2a_n_steps', 10_000),
            tag='2b',
        )

        self._save('s2_final')
        print("[Stage2] Done.")

    def _run_stage2_step(
        self,
        seq_loader: DataLoader,
        unfreeze_extra: List[str],
        lr: float,
        n_steps: int,
        step_offset: int,
        tag: str,
    ) -> None:
        print(f"\n[Stage2-{tag}] lr={lr}  n_steps={n_steps}")

        self.model.train()
        _freeze_modules(self.model, _FREEZE_STAGE1)
        if unfreeze_extra:
            _unfreeze_modules(self.model, unfreeze_extra)
            print(f"[Stage2-{tag}] Unfrozen: {unfreeze_extra}")

        trainable = _trainable_params(self.model)
        print(f"[Stage2-{tag}] Trainable: {sum(p.numel() for p in trainable):,}")

        opt        = optim.Adam(trainable, lr=lr)
        grad_clip  = getattr(self.args, 'grad_clip',          1.0)
        log_every  = getattr(self.args, 'log_every',          100)
        save_every = getattr(self.args, 'save_ckpt_every',    2_000)
        n_pts      = getattr(self.args, 's2_n_pts',           256)
        epi_scale  = getattr(self.args, 's2_epi_threshold',   2.0)

        data_iter = iter(seq_loader)
        step = 0

        while step < n_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(seq_loader)
                batch = next(data_iter)

            thr_t  = batch['thr_t'].to(self.dev)    # (B, 3, H, W)
            thr_t1 = batch['thr_t1'].to(self.dev)   # (B, 3, H, W)
            T_rel  = batch['T_rel'].to(self.dev)    # (B, 4, 4)
            K      = batch['K'].to(self.dev)        # (B, 3, 3)
            valid  = batch['valid']                  # (B,) bool

            B, _, H, W = thr_t.shape

            # ── hmap を no_grad で取得（キーポイント選択に使うだけ） ───────
            with torch.no_grad():
                _, _, hmap_t  = self.model(thr_t)   # (B, 1, Hf, Wf)
                _, _, hmap_t1 = self.model(thr_t1)

            # ── 学習側フォワード（feats に勾配を流す） ───────────────────
            feats_t,  _, _ = self.model(thr_t)      # (B, C, Hf, Wf)
            feats_t1, _, _ = self.model(thr_t1)
            feats_t  = F.normalize(feats_t,  dim=1)
            feats_t1 = F.normalize(feats_t1, dim=1)

            # ── バッチ内各サンプルで損失を計算 ───────────────────────────
            losses = []   # grad_fn を持つ損失だけを蓄積

            for b in range(B):
                # GTポーズが信頼できないサンプルはスキップ
                if not valid[b].item():
                    continue

                loss_b = geometric_feature_consistency_loss(
                    feats_t=feats_t[b],          # (C, Hf, Wf) 勾配あり
                    feats_t1=feats_t1[b],         # (C, Hf, Wf) 勾配あり
                    hmap_t=hmap_t[b],             # (1, Hf, Wf) no_grad
                    K=K[b],                       # (3, 3)
                    T_rel=T_rel[b],               # (4, 4)
                    H=H, W=W,
                    n_pts=n_pts,
                    epi_weight_scale=epi_scale,
                )
                # None = 有効投影点が少なすぎてスキップ
                if loss_b is None:
                    continue
                losses.append(loss_b)

            # 有効な損失が 1 つもない場合はバックワードをスキップ
            if len(losses) == 0:
                step += 1
                continue

            # torch.stack で平均（各 loss_b は grad_fn を持つ）
            loss = torch.stack(losses).mean()

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
            opt.step()

            global_step = step + step_offset
            if step % log_every == 0:
                self._log({
                    f's{tag}/loss'              : loss.item(),
                    f's{tag}/lr'               : opt.param_groups[0]['lr'],
                    f's{tag}/grad'             : total_grad_norm(self.model),
                    f's{tag}/n_valid_per_batch': len(losses),
                }, global_step)
                print(f"[S{tag} {step:06d}] loss={loss.item():.4f}  "
                      f"n_valid={len(losses)}/{B}  "
                      f"lr={opt.param_groups[0]['lr']:.2e}")

            if (step + 1) % save_every == 0:
                self._save(f's{tag}_step{step + 1}')

            step += 1

    # ======================================================================
    # Stage 3: Future Work
    # ======================================================================

    def run_stage3_stub(self) -> None:
        print("\n[Stage3] Future Work: End-to-End SLAM fine-tuning.")

    def finalize(self) -> None:
        self.writer.close()
        if self.use_wandb:
            try:
                import wandb
                wandb.finish()
            except Exception:
                pass