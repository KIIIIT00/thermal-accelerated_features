"""
modules/training/train_kd.py
Thermal XFeat KD トレーナークラス。

NOTE:
  - modules/xfeat.py の XFeat クラスは torch.compile 済み推論ラッパーのため
    訓練には使用しない。XFeatModel を直接インスタンス化する。
  - modules/training/losses.py は alike_wrapper をトップレベル import するため
    import しない。losses_kd.py 内の実装のみを使う。
"""

from __future__ import annotations

import os
import time
from typing import Any, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from modules.model import XFeatModel
from modules.training.losses_kd import (
    kd_feature_loss,
    kd_reliability_loss,
    fpn_invariance_loss,
    fpn_invariance_loss_fast,
    make_fpn_noise,
    relational_kd_loss,
)


# ---------------------------------------------------------------------------
# ユーティリティ
# ---------------------------------------------------------------------------

def total_grad_norm(model: nn.Module) -> float:
    """全パラメータの勾配ノルムを計算する (backward 後に呼ぶ)。"""
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return total ** 0.5


def layer_grad_norms(model: nn.Module) -> dict:
    """
    レイヤーグループごとの勾配ノルムを返す (backward 後に呼ぶ)。

    XFeatModel の named_parameters() の先頭トークン（最初の '.' の前）で
    グループ化する。例: 'norm.weight' → 'norm', 'block1.0.conv' → 'block1'

    Returns:
        {'grad_norm/block1': float, 'grad_norm/block2': float, ...}
    """
    group_sq: dict = {}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        group = name.split('.')[0]
        group_sq[group] = group_sq.get(group, 0.0) + \
                          p.grad.data.norm(2).item() ** 2
    return {f'grad_norm/{g}': v ** 0.5 for g, v in group_sq.items()}


def compute_feature_metrics(
    feats_s: torch.Tensor,
    feats_t: torch.Tensor,
    hmap_s: torch.Tensor,
    hmap_t: torch.Tensor,
    n_pts: int = 512,
) -> dict:
    """
    特徴空間の品質指標を計算する。eval 時に呼ぶ。

    Args:
        feats_s: 生徒特徴マップ (B, C, Hf, Wf)  L2 正規化済み
        feats_t: 教師特徴マップ (B, C, Hf, Wf)  L2 正規化済み
        hmap_s:  生徒信頼性マップ (B, 1, Hf, Wf)
        hmap_t:  教師信頼性マップ (B, 1, Hf, Wf)
        n_pts:   サブサンプリング数

    Returns:
        dict:
            feature/cosine_sim      高いほど特徴空間が一致（目標: 収束とともに増加）
            feature/collapse_score  低いほど良い（1=完全崩壊、0=多様）
            feature/active_dim_ratio 高いほど良い（有効な特徴次元の割合）
            hmap/reliability_corr   高いほど良い（信頼性マップの相関）
    """
    B, C, Hf, Wf = feats_s.shape
    n = min(Hf * Wf, n_pts)
    device = feats_s.device

    # ランダムに空間位置をサブサンプリング
    idx = torch.randperm(Hf * Wf, device=device)[:n]
    # (B, C, Hf*Wf) → (B*n, C) に変形
    s = feats_s.reshape(B, C, -1)[:, :, idx].permute(0, 2, 1).reshape(-1, C)
    t = feats_t.reshape(B, C, -1)[:, :, idx].permute(0, 2, 1).reshape(-1, C)

    # ① 教師-生徒間コサイン類似度（高いほど特徴空間が一致）
    cosine_sim = F.cosine_similarity(s, t, dim=1).mean().item()

    # ② 崩壊スコア（生徒特徴のバッチ内ペア間類似度の平均）
    # 1 に近いほど全ベクトルが同方向に潰れている（表現崩壊）
    s_norm = F.normalize(s, dim=1)
    # メモリ節約のため最大256点に制限
    n_sub = min(len(s_norm), 256)
    s_sub = s_norm[:n_sub]
    sim_mat = s_sub @ s_sub.T                                    # (n_sub, n_sub)
    mask = ~torch.eye(n_sub, dtype=torch.bool, device=device)
    collapse_score = sim_mat[mask].mean().item()

    # ③ 有効次元比率（標準偏差が閾値を超える次元の割合）
    std_per_dim = s.std(dim=0)                                   # (C,)
    active_dim_ratio = (std_per_dim > 1e-3).float().mean().item()

    # ④ 信頼性マップのピアソン相関（収束とともに増加するはず）
    hs = hmap_s.reshape(B, -1).float()
    ht = hmap_t.reshape(B, -1).float()
    hs_c = hs - hs.mean(dim=1, keepdim=True)
    ht_c = ht - ht.mean(dim=1, keepdim=True)
    corr = (hs_c * ht_c).sum(dim=1) / (
        hs_c.norm(dim=1) * ht_c.norm(dim=1) + 1e-8)
    reliability_corr = corr.mean().item()

    return {
        'feature/cosine_sim':       cosine_sim,
        'feature/collapse_score':   collapse_score,
        'feature/active_dim_ratio': active_dim_ratio,
        'hmap/reliability_corr':    reliability_corr,
    }


# ---------------------------------------------------------------------------
# wandb 初期化ヘルパー
# ---------------------------------------------------------------------------

def _args_to_wandb_config(args: Any) -> dict:
    """argparse.Namespace からハイパーパラメータ辞書を生成する。"""
    return {
        # Sweep で探索する変数
        'lambda_kd_rel':       getattr(args, 'lambda_kd_rel',       0.1),
        'lambda_fpn':          getattr(args, 'lambda_fpn',          0.05),
        'lambda_relkd':        getattr(args, 'lambda_relkd',        0.5),
        'fpn_sigma_min':       getattr(args, 'fpn_sigma_min',       2.0),
        'fpn_sigma_max':       getattr(args, 'fpn_sigma_max',       8.0),
        'p_diurnal_inversion': getattr(args, 'p_diurnal_inversion', 0.3),
        'infonce_temp':        getattr(args, 'infonce_temp',         0.2),
        'n_kd_samples':        getattr(args, 'n_kd_samples',        1024),
        'n_relkd_samples':     getattr(args, 'n_relkd_samples',     512),
        # 訓練設定
        'n_steps':             getattr(args, 'n_steps',             100_000),
        'lr':                  getattr(args, 'lr',                  3e-4),
        'lr_step':             getattr(args, 'lr_step',             30_000),
        'lr_gamma':            getattr(args, 'lr_gamma',            0.5),
        'batch_size':          getattr(args, 'batch_size',          8),
        'grad_clip':           getattr(args, 'grad_clip',           1.0),
        # データ
        'dataset':             getattr(args, 'dataset',             []),
        'eval_dataset':        getattr(args, 'eval_dataset',        []),
        'aug_list':            getattr(args, 'aug_list',            []),
        # モデル
        'teacher_weights':     getattr(args, 'teacher_weights',     ''),
        # 環境（再現性のため）
        'env/torch_version':   torch.__version__,
        'env/gpu_name':        (torch.cuda.get_device_name(0)
                                if torch.cuda.is_available() else 'cpu'),
    }


def _init_wandb(args: Any) -> bool:
    """
    wandb を初期化する。失敗した場合 False を返して TensorBoard のみで継続する。
    --no_wandb フラグが立っている場合は即時 False を返す。
    """
    try:
        import wandb  # noqa: F401
    except ImportError:
        print("[wandb] wandb is not installed. Falling back to TensorBoard only.")
        return False

    if getattr(args, 'no_wandb', False):
        print("[wandb] Disabled by --no_wandb flag")
        return False

    try:
        import wandb
        wandb.init(
            project=getattr(args, 'wandb_project',  'thermal-xfeat-kd'),
            name=getattr(args, 'wandb_run_name',    None),
            group=getattr(args, 'wandb_group',      'grid_search'),
            tags=getattr(args, 'wandb_tags',        []),
            config=_args_to_wandb_config(args),
            dir=getattr(args, 'ckpt_save_path',     '.'),
            resume='allow',
        )
        print(f"[wandb] Run: {wandb.run.url}")
        return True
    except Exception as e:
        print(f"[wandb] Init failed ({e}). Falling back to TensorBoard only.")
        return False


# ---------------------------------------------------------------------------
# メイントレーナークラス
# ---------------------------------------------------------------------------

class ThermalXFeatKDTrainer:
    """
    RGB→Thermal Knowledge Distillation トレーナー。

    教師 (frozen XFeatModel) が RGB を、
    生徒 (学習中 XFeatModel) が熱画像を受け取る。

    損失:
        L_total = L_KD
                + λ_rel × L_KD_rel
                + λ_fpn × L_FPN
    """

    def __init__(self, args: Any):
        # ── デバイス ──────────────────────────────────────────────────────
        self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # ── ハイパーパラメータ ────────────────────────────────────────────
        self.n_steps      = getattr(args, 'n_steps',             100_000)
        self.lr           = getattr(args, 'lr',                  3e-4)
        self.lr_step      = getattr(args, 'lr_step',             30_000)
        self.lr_gamma     = getattr(args, 'lr_gamma',            0.5)
        self.grad_clip    = getattr(args, 'grad_clip',           1.0)
        self.lambda_rel   = getattr(args, 'lambda_kd_rel',       0.1)
        self.lambda_fpn   = getattr(args, 'lambda_fpn',          0.05)
        self.lambda_relkd = getattr(args, 'lambda_relkd',        0.5)
        self.sigma_min    = getattr(args, 'fpn_sigma_min',       2.0)
        self.sigma_max    = getattr(args, 'fpn_sigma_max',       8.0)
        self.n_samples    = getattr(args, 'n_kd_samples',        1024)
        self.n_relkd      = getattr(args, 'n_relkd_samples',     512)
        self.temp         = getattr(args, 'infonce_temp',         0.2)
        self.save_every   = getattr(args, 'save_ckpt_every',     2_000)
        self.log_every    = getattr(args, 'log_every',           100)
        self.eval_every   = getattr(args, 'eval_every',           2_000)
        self.image_log_every = getattr(args, 'image_log_every',  500)
        self.ckpt_path    = getattr(args, 'ckpt_save_path',      'checkpoints')

        os.makedirs(self.ckpt_path, exist_ok=True)

        # ── 教師モデル（frozen）────────────────────────────────────────────
        self.teacher = XFeatModel().to(self.dev).eval()
        teacher_weights = getattr(args, 'teacher_weights', None)
        if teacher_weights and os.path.isfile(teacher_weights):
            self.teacher.load_state_dict(
                torch.load(teacher_weights, map_location=self.dev,
                           weights_only=True))
            print(f"[Trainer] Teacher loaded from: {teacher_weights}")
        else:
            print("[Trainer] WARNING: teacher_weights not found. "
                  "Using random weights.")
        for p in self.teacher.parameters():
            p.requires_grad_(False)

        # ── 生徒モデル（学習）────────────────────────────────────────────
        self.student = XFeatModel().to(self.dev)
        student_init = getattr(args, 'student_init', None)
        if student_init and os.path.isfile(student_init):
            self.student.load_state_dict(
                torch.load(student_init, map_location=self.dev,
                           weights_only=True))
            print(f"[Trainer] Student initialized from: {student_init}")
        elif teacher_weights and os.path.isfile(teacher_weights):
            # デフォルト: teacher_weights で初期化
            self.student.load_state_dict(
                torch.load(teacher_weights, map_location=self.dev,
                           weights_only=True))
            print("[Trainer] Student initialized from teacher weights")

        self.student.train()

        # ── オプティマイザ・スケジューラ ──────────────────────────────────
        self.opt = optim.Adam(
            filter(lambda p: p.requires_grad, self.student.parameters()),
            lr=self.lr,
        )
        self.scheduler = optim.lr_scheduler.StepLR(
            self.opt, step_size=self.lr_step, gamma=self.lr_gamma)

        # ── ロギング ──────────────────────────────────────────────────────
        logdir = os.path.join(
            self.ckpt_path, 'logdir',
            'thermal_kd_' + time.strftime('%Y_%m_%d-%H_%M_%S'))
        os.makedirs(logdir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=logdir)

        self.use_wandb = _init_wandb(args)

        # ── AMP（自動混合精度）────────────────────────────────────────────
        self.use_amp = getattr(args, 'use_amp', False) and torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        if self.use_amp:
            print("[Trainer] AMP enabled (float16 mixed precision)")

    # ── ロギングヘルパー ─────────────────────────────────────────────────

    def _log(self, metrics: dict, step: int) -> None:
        """TensorBoard と wandb に同時記録する。"""
        for k, v in metrics.items():
            self.writer.add_scalar(k, v, step)
        if self.use_wandb:
            try:
                import wandb
                wandb.log(metrics, step=step)
            except Exception:
                pass

    def _log_sample_images(
        self, rgb: torch.Tensor, thr: torch.Tensor, step: int
    ) -> None:
        """
        バッチ内の最初の 4 枚を記録する。
        昼夜反転拡張 (diurnal_inversion) が適用されたかを目視確認する。
        """
        B = min(4, rgb.shape[0])
        with torch.no_grad():
            rgb_grid = make_grid(rgb[:B].cpu(), normalize=True)
            thr_grid = make_grid(thr[:B].cpu(), normalize=True)

        self.writer.add_image('sample/rgb', rgb_grid, step)
        self.writer.add_image('sample/thr', thr_grid, step)

        if self.use_wandb:
            try:
                import wandb
                wandb.log({
                    'sample/rgb': wandb.Image(
                        rgb_grid.permute(1, 2, 0).numpy(),
                        caption=f'RGB input (step {step})'
                    ),
                    'sample/thr': wandb.Image(
                        thr_grid.permute(1, 2, 0).numpy(),
                        caption=f'Thermal input w/ diurnal aug (step {step})'
                    ),
                }, step=step)
            except Exception:
                pass

    @torch.no_grad()
    def _log_heatmaps(
        self,
        hmap_s: torch.Tensor,
        hmap_t: torch.Tensor,
        step: int,
    ) -> None:
        """
        eval 時に生徒・教師の信頼性マップを wandb に画像としてログする。

        信頼性マップが収束につれて類似していくことを目視確認できる。
        hmap_s, hmap_t: (B, 1, Hf, Wf) — sigmoid 済みを想定。
        """
        if not self.use_wandb:
            return
        try:
            import wandb
            B = min(4, hmap_s.shape[0])
            # (B, 1, Hf, Wf) → グリッド画像
            s_grid = make_grid(hmap_s[:B].cpu(), normalize=True)
            t_grid = make_grid(hmap_t[:B].cpu(), normalize=True)
            wandb.log({
                'sample/heatmap_student': wandb.Image(
                    s_grid.permute(1, 2, 0).numpy(),
                    caption=f'Student reliability map (step {step})'
                ),
                'sample/heatmap_teacher': wandb.Image(
                    t_grid.permute(1, 2, 0).numpy(),
                    caption=f'Teacher reliability map (step {step})'
                ),
            }, step=step)
        except Exception:
            pass

    # ── 訓練ループ ────────────────────────────────────────────────────────

    def train(self, train_loader, eval_loader=None) -> None:
        """
        訓練ループ。DataLoader からバッチを無限に引き出して学習する。

        Args:
            train_loader: build_thermal_loader() が返す DataLoader
            eval_loader:  build_thermal_eval_loader() が返す DataLoader（None可）
        """
        self.student.train()
        data_iter = iter(train_loader)
        step = 0
        best_val_loss = float('inf')

        # スループット計測用
        step_start_time = time.time()
        # diurnal_inversion 適用率計測用
        diurnal_count   = 0
        diurnal_total   = 0

        print(f"[Trainer] Starting KD training for {self.n_steps} steps ...")

        while step < self.n_steps:
            # ── バッチ取得 ────────────────────────────────────────────────
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            # AnyThermal 互換: batch["item"][0] = {"rgb": ..., "thr": ...}
            item_dict = batch['item'][0]
            rgb = item_dict['rgb'].to(self.dev)   # 教師入力 (B, 3, H, W)
            thr = item_dict['thr'].to(self.dev)   # 生徒入力 (B, 3, H, W)

            # ── diurnal_inversion 適用率の計測 ────────────────────────────
            # 熱画像の平均輝度が 0.5 をまたいで反転していれば適用されたと判定
            thr_raw = item_dict.get('thr_raw', None)
            if thr_raw is not None:
                thr_raw_dev = thr_raw.to(self.dev)
                inv_mask = (
                    (thr_raw_dev.mean(dim=[1, 2, 3]) > 0.5) !=
                    (thr.mean(dim=[1, 2, 3]) > 0.5)
                )
                diurnal_count += inv_mask.sum().item()
            diurnal_total += thr.shape[0]

            # ── 教師フォワード（勾配なし）─────────────────────────────────
            with torch.no_grad():
                feats_t, _, hmap_t = self.teacher(rgb)
                feats_t = F.normalize(feats_t, dim=1)

            # ── FPN ノイズ画像を生成 ──────────────────────────────────────
            # 旧実装: clean フォワード(1) + fpn_invariance_loss 内で clean(1) + noisy(1) = 3回
            # 新実装: no_grad clean(1) + 学習 noisy(1) = 2回（約33%削減）
            img_fpn, fpn_sigma = make_fpn_noise(
                thr,
                sigma_min=self.sigma_min,
                sigma_max=self.sigma_max,
            )

            # ── 生徒フォワード（クリーン・stop_gradient）────────────────
            amp_ctx = torch.cuda.amp.autocast() if self.use_amp else torch.no_grad()
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    feats_clean, _, hmap_s = self.student(thr)
                    feats_clean = F.normalize(feats_clean, dim=1)

            # ── 生徒フォワード（ノイズ付き・学習側）────────────────────
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                feats_s, _, _ = self.student(img_fpn)
                feats_s = F.normalize(feats_s, dim=1)

            # ── 損失計算 ──────────────────────────────────────────────────
            # L_KD: ノイズ付き特徴 vs 教師（FPN 不変性も同時に学習）
            l_kd = kd_feature_loss(
                feats_s, feats_t,
                n_samples=self.n_samples, temp=self.temp)

            # L_KD_rel: clean 信頼性マップ vs 教師信頼性マップ
            l_rel = kd_reliability_loss(hmap_s, hmap_t.detach(), thr)

            # L_FPN: noisy 特徴 → clean 特徴（stop_gradient）に近づける
            l_fpn = fpn_invariance_loss_fast(feats_s, feats_clean)

            # L_relkd: intra-modal 構造転移
            l_relkd = relational_kd_loss(
                feats_s, feats_t,
                n_samples=self.n_relkd)

            loss = (l_kd
                    + self.lambda_rel   * l_rel
                    + self.lambda_fpn   * l_fpn
                    + self.lambda_relkd * l_relkd)

            # ── バックワード ──────────────────────────────────────────────
            self.opt.zero_grad()
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_(
                    self.student.parameters(), self.grad_clip)
                self.scaler.step(self.opt)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.student.parameters(), self.grad_clip)
                self.opt.step()
            self.scheduler.step()

            # ── ロギング（スカラー）───────────────────────────────────────
            if step % self.log_every == 0:
                gnorm = total_grad_norm(self.student)
                elapsed = time.time() - step_start_time
                elapsed = max(elapsed, 1e-6)

                metrics = {
                    # ── 損失（既存）──────────────────────────────────────
                    'loss/total':      loss.item(),
                    'loss/kd':         l_kd.item(),
                    'loss/kd_rel':     l_rel.item(),
                    'loss/fpn':        l_fpn.item(),
                    # Relational KD 損失（新規追加）
                    'loss/relkd':      l_relkd.item(),
                    # KD 損失が total に占める割合（FPN・rel の寄与確認）
                    'loss/kd_ratio':   l_kd.item() / (loss.item() + 1e-8),
                    # Relational KD が total に占める割合
                    # → cross-modal と intra-modal の学習バランスを確認
                    'loss/relkd_ratio': (
                        self.lambda_relkd * l_relkd.item() / (loss.item() + 1e-8)),
                    # ── 最適化（既存 + 拡張）────────────────────────────
                    'lr':              self.opt.param_groups[0]['lr'],
                    'grad_norm/total': gnorm,
                    # ── データ拡張確認（新規性③の直接証拠）────────────────
                    # diurnal_inversion が p=0.3 付近で機能しているか確認
                    'aug/diurnal_inv_rate': (
                        diurnal_count / diurnal_total
                        if diurnal_total > 0 else 0.0),
                    # FPN 列ノイズの実際の強度（sigma_min〜sigma_max の範囲を確認）
                    'aug/fpn_sigma_mean': fpn_sigma,
                    # ── スループット ─────────────────────────────────────
                    'throughput/steps_per_sec': self.log_every / elapsed,
                    'throughput/pairs_per_sec': (
                        self.log_every * thr.shape[0] / elapsed),
                }

                # レイヤーごと勾配ノルム（特定レイヤーへの勾配集中を検出）
                metrics.update(layer_grad_norms(self.student))

                self._log(metrics, step)
                print(
                    f"[step {step:06d}] "
                    f"total={loss.item():.4f}  "
                    f"kd={l_kd.item():.4f}  "
                    f"rel={l_rel.item():.4f}  "
                    f"fpn={l_fpn.item():.4f}  "
                    f"relkd={l_relkd.item():.4f}  "
                    f"lr={self.opt.param_groups[0]['lr']:.2e}  "
                    f"diurnal={diurnal_count}/{diurnal_total}"
                )

                # カウンタリセット
                step_start_time = time.time()
                diurnal_count   = 0
                diurnal_total   = 0

            # ── サンプル画像ロギング ───────────────────────────────────────
            if step % self.image_log_every == 0:
                self._log_sample_images(rgb, thr, step)

            # ── 評価ループ ────────────────────────────────────────────────
            if eval_loader is not None and (step + 1) % self.eval_every == 0:
                val_metrics = self._evaluate(eval_loader, current_step=step)
                self._log(
                    {f'val/{k}': v for k, v in val_metrics.items()}, step)
                val_loss = val_metrics.get('loss_total', float('inf'))
                print(
                    f"[eval  {step:06d}] "
                    + "  ".join(f"{k}={v:.4f}" for k, v in val_metrics.items())
                )
                # best モデルを保存
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_ckpt = os.path.join(self.ckpt_path,
                                             'thermal_kd_student_best.pth')
                    torch.save(self.student.state_dict(), best_ckpt)
                    print(f"[Trainer] Best checkpoint updated: {best_ckpt}")

            # ── チェックポイント保存 ──────────────────────────────────────
            if (step + 1) % self.save_every == 0:
                ckpt_file = os.path.join(
                    self.ckpt_path, f'thermal_kd_student_{step + 1}.pth')
                torch.save(self.student.state_dict(), ckpt_file)
                print(f"[Trainer] Checkpoint saved: {ckpt_file}")
                if self.use_wandb:
                    try:
                        import wandb
                        wandb.save(ckpt_file)
                    except Exception:
                        pass

            step += 1

        # 最終保存
        final_ckpt = os.path.join(self.ckpt_path, 'thermal_kd_student_final.pth')
        torch.save(self.student.state_dict(), final_ckpt)
        print(f"[Trainer] Training done. Final checkpoint: {final_ckpt}")
        self._finalize()

    @torch.no_grad()
    def _evaluate(self, eval_loader, current_step: int = 0) -> dict:
        """
        val split 全体で KD 損失と特徴空間品質を計算して返す。

        Args:
            eval_loader:  評価用 DataLoader
            current_step: 呼び出し時の訓練ステップ数（wandb の step に使用）

        Returns:
            {
                "loss_total": float, "loss_kd": float, "loss_kd_rel": float,
                "feature/cosine_sim": float, "feature/collapse_score": float,
                "feature/active_dim_ratio": float,
                "hmap/reliability_corr": float,
            }
        """
        self.student.eval()
        totals = {
            'loss_total':               0.0,
            'loss_kd':                  0.0,
            'loss_kd_rel':              0.0,
            'loss_relkd':               0.0,
            'feature/cosine_sim':       0.0,
            'feature/collapse_score':   0.0,
            'feature/active_dim_ratio': 0.0,
            'hmap/reliability_corr':    0.0,
        }
        n_batches = 0
        heatmap_logged = False   # 最初のバッチのみ heatmap 画像をログ

        for batch in eval_loader:
            item_dict = batch['item'][0]
            rgb = item_dict['rgb'].to(self.dev)
            thr = item_dict['thr'].to(self.dev)

            feats_t, _, hmap_t = self.teacher(rgb)
            feats_t = F.normalize(feats_t, dim=1)

            feats_s, _, hmap_s = self.student(thr)
            feats_s = F.normalize(feats_s, dim=1)

            l_kd    = kd_feature_loss(
                feats_s, feats_t,
                n_samples=self.n_samples, temp=self.temp)
            l_rel   = kd_reliability_loss(hmap_s, hmap_t, thr)
            l_relkd = relational_kd_loss(feats_s, feats_t,
                                         n_samples=self.n_relkd)
            loss    = (l_kd
                       + self.lambda_rel   * l_rel
                       + self.lambda_relkd * l_relkd)

            totals['loss_total']  += loss.item()
            totals['loss_kd']     += l_kd.item()
            totals['loss_kd_rel'] += l_rel.item()
            totals['loss_relkd']  += l_relkd.item()

            # 特徴空間品質指標（バッチごとに累積して最後に平均）
            feat_metrics = compute_feature_metrics(
                feats_s, feats_t, hmap_s, hmap_t)
            for k, v in feat_metrics.items():
                totals[k] += v

            # 最初のバッチのみ信頼性マップを画像としてログ
            # current_step を渡すことで wandb の step が単調増加を保つ
            if not heatmap_logged and self.use_wandb:
                self._log_heatmaps(hmap_s, hmap_t, current_step)
                heatmap_logged = True

            n_batches += 1

        self.student.train()

        if n_batches == 0:
            return totals
        return {k: v / n_batches for k, v in totals.items()}

    def _finalize(self) -> None:
        self.writer.close()
        if self.use_wandb:
            try:
                import wandb
                wandb.finish()
            except Exception:
                pass