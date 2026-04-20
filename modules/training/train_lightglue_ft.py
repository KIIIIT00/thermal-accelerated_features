# """
# modules/training/train_lightglue_ft.py
# LightGlue Fine-tuning トレーナー。

# 設計方針:
#     - ThermalXFeat は完全 frozen（重みを変更しない）
#     - LightGlue のみを Thermal ペアで fine-tuning
#     - 損失: L_match（NLL）+ lambda_epi * L_epi（Sampson）
#     - 学習データ: Freiburg（train） + TartanRGBT（train）
#     - 評価データ: SThErEO, VIVID（fine-tuning に使用しない）

# 損失の意味:
#     L_match:
#         GT ポーズから F 行列を計算し、Sampson 距離でインライアを判定。
#         インライア対応の log P(match) を最大化、
#         アウトライア対応の log P(unmatch) を最大化する NLL 損失。

#     L_epi:
#         LightGlue が出力したマッチ点のエピポーラ整合性を直接測定。
#         GT ポーズがある場合のみ有効（TartanRGBT）。
#         Freiburg は valid=False のため L_epi はスキップ。
# """

# from __future__ import annotations

# import os
# import time
# from typing import Any, Dict, List, Optional

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch import Tensor
# from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter

# from modules.model import XFeatModel


# # ---------------------------------------------------------------------------
# # ユーティリティ
# # ---------------------------------------------------------------------------

# def _init_wandb(args: Any) -> bool:
#     if getattr(args, 'no_wandb', False):
#         return False
#     try:
#         import wandb
#         wandb.init(
#             project  = getattr(args, 'wandb_project',  'thermal-xfeat-lightglue-ft'),
#             name     = getattr(args, 'wandb_run_name',  None),
#             group    = getattr(args, 'wandb_group',     'lightglue_ft'),
#             tags     = getattr(args, 'wandb_tags',      []),
#             config   = vars(args),
#             dir      = getattr(args, 'ckpt_save_path',  'checkpoints/lightglue_ft'),
#         )
#         return True
#     except Exception as e:
#         print(f"[LG-FT] wandb init failed: {e}")
#         return False


# def total_grad_norm(model: nn.Module) -> float:
#     total = 0.0
#     for p in model.parameters():
#         if p.grad is not None:
#             total += p.grad.data.norm(2).item() ** 2
#     return total ** 0.5


# # ---------------------------------------------------------------------------
# # 損失関数
# # ---------------------------------------------------------------------------

# def _compute_F_matrix(K: Tensor, T_rel: Tensor) -> Tensor:
#     """
#     GT ポーズから基本行列 F を計算する。
#     K_inv は CPU で計算して GPU に転送（cuSOLVER エラー回避）。
#     """
#     R = T_rel[:3, :3]
#     t = T_rel[:3, 3]
#     t_s = torch.zeros(3, 3, device=t.device, dtype=t.dtype)
#     t_s[0, 1] = -t[2]; t_s[0, 2] =  t[1]
#     t_s[1, 0] =  t[2]; t_s[1, 2] = -t[0]
#     t_s[2, 0] = -t[1]; t_s[2, 1] =  t[0]
#     E     = t_s @ R
#     K_inv = torch.inverse(K.cpu().double()).float().to(K.device)
#     return K_inv.T @ E @ K_inv   # (3, 3)


# def _sampson_dist(pts1: Tensor, pts2: Tensor, F_mat: Tensor) -> Tensor:
#     """Sampson 距離 (N,)"""
#     N    = pts1.shape[0]
#     ones = pts1.new_ones(N, 1)
#     p1h  = torch.cat([pts1, ones], dim=1)  # (N, 3)
#     p2h  = torch.cat([pts2, ones], dim=1)
#     Fp1  = (F_mat @ p1h.T).T               # (N, 3)
#     Ftp2 = (F_mat.T @ p2h.T).T
#     numer = (p2h * Fp1).sum(1) ** 2
#     denom = Fp1[:, 0]**2 + Fp1[:, 1]**2 + Ftp2[:, 0]**2 + Ftp2[:, 1]**2
#     return numer / denom.clamp(min=1e-8)


# def lightglue_matching_loss(
#     scores:     Tensor,
#     pts1:       Tensor,
#     pts2:       Tensor,
#     K:          Tensor,
#     T_rel:      Tensor,
#     use_gt:     bool,
#     inlier_thr: float = 3.0,
# ) -> Tensor:
#     """
#     LightGlue 公式の NegativeLogAssignment 損失（MNN ベース GT ラベル版）。

#     LightGlue (Lindenberger 2023) の式:
#       L = -(1/|M+|)  Σ_{(i,j)∈M+} log σ(i→j)      GT 対応点
#           -(1/|U1|)  Σ_{i∈U1}     log σ(i→dustbin) frame1 非対応点
#           -(1/|U2|)  Σ_{j∈U2}     log σ(dustbin→j) frame2 非対応点

#     GT ラベル: エピポーラ幾何 + 相互最近傍 (MNN)
#       修正前（最近傍存在性）: P(偶然インライア)≈1.0 で 99% インライア（数学的に証明済み）
#       修正後（MNN）: i→j かつ j→i が成立する場合のみ真の対応（期待 30-60%）
#     """
#     N1 = scores.shape[0] - 1
#     N2 = scores.shape[1] - 1
#     pts1 = pts1[:N1]
#     pts2 = pts2[:N2]

#     if N1 < 8 or N2 < 8:
#         return scores.new_zeros(1).squeeze()

#     # ── GT 基本行列の計算 ──────────────────────────────────────────────
#     with torch.no_grad():
#         if use_gt:
#             F_mat = _compute_F_matrix(K, T_rel)
#         else:
#             import cv2
#             p1_np = pts1.cpu().numpy().astype(np.float32)
#             p2_np = pts2.cpu().numpy().astype(np.float32)
#             F_np, _ = cv2.findFundamentalMat(p1_np, p2_np, cv2.FM_8POINT)
#             if F_np is None:
#                 return scores.new_zeros(1).squeeze()
#             F_mat = torch.from_numpy(F_np).float().to(pts1.device)

#     # ── GT ラベル生成（MNN + Sampson）─────────────────────────────────
#     with torch.no_grad():
#         sd_map = _compute_all_sampson_dist(pts1, pts2, F_mat)   # (N1, N2)
#         min_sd1, nn1 = sd_map.min(dim=1)
#         min_sd2, nn2 = sd_map.min(dim=0)
#         ids1   = torch.arange(N1, device=sd_map.device)
#         ids2   = torch.arange(N2, device=sd_map.device)
#         mnn1   = (nn2[nn1] == ids1) & (min_sd1 < inlier_thr ** 2)
#         mnn2   = (nn1[nn2] == ids2) & (min_sd2 < inlier_thr ** 2)
#         n_pos  = int(mnn1.sum())
#         if n_pos < 4:
#             return scores.new_zeros(1).squeeze()

#     if torch.rand(1) < 0.1:
#         print(f"[GT-Loss] MNN Pairs: {n_pos}/{N1} ({100*n_pos/N1:.0f}%)")

#     # ── NegativeLogAssignment 損失 ─────────────────────────────────────
#     log_p_row = F.log_softmax(scores, dim=1)
#     log_p_col = F.log_softmax(scores, dim=0)

#     n_pos_t  = max(n_pos, 1)
#     n_neg1_t = max(int((~mnn1).sum()), 1)
#     n_neg2_t = max(int((~mnn2).sum()), 1)

#     loss_pos = scores.new_zeros(1)
#     for i in range(N1):
#         if mnn1[i]:
#             loss_pos = loss_pos - log_p_row[i, int(nn1[i])]
#     loss_pos = loss_pos / n_pos_t

#     loss_neg1 = scores.new_zeros(1)
#     for i in range(N1):
#         if not mnn1[i]:
#             loss_neg1 = loss_neg1 - log_p_row[i, N2]
#     loss_neg1 = loss_neg1 / n_neg1_t

#     loss_neg2 = scores.new_zeros(1)
#     for j in range(N2):
#         if not mnn2[j]:
#             loss_neg2 = loss_neg2 - log_p_col[N1, j]
#     loss_neg2 = loss_neg2 / n_neg2_t

#     return (loss_pos + loss_neg1 + loss_neg2) / 3.0


# def _compute_all_sampson_dist(pts1, pts2, F):
#     """(N1, 2) と (N2, 2) の全ペア間の Sampson 距離を計算するヘルパー。"""
#     N1, N2 = pts1.shape[0], pts2.shape[0]
#     ones1 = torch.ones((N1, 1), device=pts1.device)
#     ones2 = torch.ones((N2, 1), device=pts2.device)
#     p1 = torch.cat([pts1, ones1], dim=1) # (N1, 3)
#     p2 = torch.cat([pts2, ones2], dim=1) # (N2, 3)

#     # エピポーラ線 L1 = F * p1, L2 = F^T * p2
#     L1 = p1 @ F.t()  # (N1, 3)
#     L2 = p2 @ F      # (N2, 3)

#     # 代数的距離 p2^T * F * p1
#     p2_F_p1 = (p2 @ F @ p1.t()).t() # (N1, N2)

#     # Sampson 距離の分母計算
#     denom = L1[:, 0:1]**2 + L1[:, 1:2]**2 + L2[None, :, 0]**2 + L2[None, :, 1]**2
#     return (p2_F_p1**2) / (denom + 1e-9)

# def lightglue_epi_loss(
#     pts1:      Tensor,
#     pts2:      Tensor,
#     scores:    Tensor,
#     K:         Tensor,
#     T_rel:     Tensor,
#     threshold: float = 2.0,
# ) -> Tensor:
#     """
#     LightGlue 出力マッチのエピポーラ整合性損失。

#     LightGlue が出力したマッチ（argmax で取得）の Sampson 距離を最小化する。
#     これにより「LightGlue がエピポーラ幾何に整合したマッチを学習する」。

#     Args:
#         pts1, pts2 : キーポイント座標
#         scores     : (N1+1, N2+1) LightGlue スコア行列
#         K, T_rel   : GT カメラパラメータ
#         threshold  : ソフトインライア重みのスケール
#     """
#     N1, N2 = pts1.shape[0], pts2.shape[0]
#     if N1 < 8 or N2 < 8:
#         return scores.new_zeros(1).squeeze()

#     with torch.no_grad():
#         F_mat  = _compute_F_matrix(K, T_rel)
#         sim    = scores[:N1, :N2]
#         nn12   = sim.argmax(dim=1)
#         nn21   = sim.argmax(dim=0)
#         ids    = torch.arange(N1, device=sim.device)
#         mutual = nn21[nn12] == ids

#     if mutual.sum() < 4:
#         return scores.new_zeros(1).squeeze()

#     pts1_m = pts1[mutual]
#     pts2_m = pts2[nn12[mutual]]

#     sd      = _sampson_dist(pts1_m, pts2_m, F_mat)     # (M,)
#     weights = torch.exp(-sd.detach() / (threshold ** 2))
#     loss    = (sd * weights).mean()
#     return loss


# # ---------------------------------------------------------------------------
# # LightGlue Fine-tuning トレーナー
# # ---------------------------------------------------------------------------

# class LightGlueFTTrainer:

#     def __init__(self, args: Any):
#         self.args = args
#         self.dev  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         print(f"[LG-FT] Device: {self.dev}")

#         # ── ThermalXFeat（完全 frozen）─────────────────────────────────────
#         self.thermal_feat = XFeatModel().to(self.dev).eval()
#         w = getattr(args, 'thermal_weights', None)
#         if w and os.path.isfile(w):
#             self.thermal_feat.load_state_dict(
#                 torch.load(w, map_location=self.dev, weights_only=True))
#             print(f"[LG-FT] ThermalXFeat loaded: {w}")
#         else:
#             print("[LG-FT] WARNING: thermal_weights not found → random weights")
#         for p in self.thermal_feat.parameters():
#             p.requires_grad_(False)

#         # ── LightGlue（fine-tuning 対象）──────────────────────────────────
#         self.lightglue = self._load_lightglue(args)

#         # LightGlue の confidence 設定（属性がある場合のみ）
#         # depth_confidence / width_confidence = -1 で早期停止を無効化
#         # これにより全レイヤーが実行され log_assignment が生成される
#         for attr, val in [('filter_threshold', None),
#                           ('depth_confidence', -1),
#                           ('width_confidence', -1)]:
#             if hasattr(self.lightglue, attr):
#                 setattr(self.lightglue, attr, val)
#         # log_assignment を出力するよう強制（バージョン依存）
#         if hasattr(self.lightglue, 'conf') and hasattr(self.lightglue.conf, 'log_assignment'):
#             self.lightglue.conf.log_assignment = True
#         # self.lightglue.training = True は .train() 呼び出しで保証

#         # ── チェックポイント・ログ ─────────────────────────────────────────
#         self.ckpt_path = getattr(args, 'ckpt_save_path',
#                                   'checkpoints/lightglue_ft/default')
#         os.makedirs(self.ckpt_path, exist_ok=True)
#         logdir = os.path.join(self.ckpt_path, 'logdir',
#                               'lg_ft_' + time.strftime('%Y_%m_%d-%H_%M_%S'))
#         os.makedirs(logdir, exist_ok=True)
#         self.writer    = SummaryWriter(log_dir=logdir)
#         self.use_wandb = _init_wandb(args)
#         if self.use_wandb:
#             try:
#                 import wandb
#                 if wandb.run is not None:
#                     print(f"[LG-FT] wandb run dir: {wandb.run.dir}")
#             except Exception:
#                 pass

#     def _load_lightglue(self, args: Any) -> nn.Module:
#         """LightGlue をロードして GPU に転送する。"""
#         try:
#             from lightglue import LightGlue
#         except ImportError:
#             raise ImportError(
#                 "LightGlue が見つかりません。\n"
#                 "pip install git+https://github.com/cvg/LightGlue.git"
#             )

#         input_dim = getattr(args, 'input_dim', 64)

#         # 全 confidence 機能を無効化（学習時は全レイヤーを通す）
#         # depth_confidence=-1, width_confidence=-1 で早期停止・枝刈りを無効化
#         lg = LightGlue(
#             features         = None,
#             input_dim        = input_dim,
#             filter_threshold = -1.0,
#             depth_confidence = -1.0,
#             width_confidence = -1.0,
#             flash            = False,
#         ).to(self.dev)

#         n = sum(p.numel() for p in lg.parameters() if p.requires_grad)
#         print(f"[LG-FT] LightGlue loaded (input_dim={input_dim}, trainable={n:,})")

#         # ── forward を wrap して log_assignment を強制取得 ────────────────
#         # 根拠: cvg/LightGlue は training=True でも log_assignment を返さない
#         #       バージョンがある。wrap して内部の log_assignment を取得する。
#         original_forward = lg.forward.__func__

#         def patched_forward(self_lg, data: dict) -> dict:
#             # 通常の forward を実行
#             pred = original_forward(self_lg, data)

#             # log_assignment が存在しない場合は内部から取得を試みる
#             if 'log_assignment' not in pred and self_lg.training:
#                 # LightGlue の内部変数 log_assignment を取得する
#                 # gluefactory 版: _get_log_assignment()
#                 # cvg 版: log_assignment は token の最終状態から計算
#                 # 代替: matching_scores0 から soft assignment を再構成
#                 if 'matching_scores0' in pred and 'matching_scores1' in pred:
#                     ms0 = pred['matching_scores0']   # (B, N1)
#                     ms1 = pred['matching_scores1']   # (B, N2)
#                     B, N1 = ms0.shape
#                     N2 = ms1.shape[1]
#                     # (B, N1+1, N2+1) の擬似 log-assignment を構築
#                     # ms0[b,i] = LG が "KP i はマッチする" と判断した確率
#                     # ここから soft assignment を再構成する
#                     # 対角方向に ms0, dustbin に 1-ms0 を配置
#                     la = ms0.new_zeros(B, N1 + 1, N2 + 1)
#                     n  = min(N1, N2)
#                     la[:, torch.arange(n), torch.arange(n)] = ms0[:, :n]
#                     la[:, :N1, N2] = 1.0 - ms0
#                     pred['log_assignment'] = la
#             return pred

#         import types
#         lg.forward = types.MethodType(patched_forward, lg)
#         return lg

#     @torch.no_grad()
#     def _extract_features(
#         self,
#         img: Tensor,
#         max_kp: int,
#     ):
#         """
#         ThermalXFeat で特徴抽出（frozen）。

#         Returns:
#             kpts  : (N, 2)  画素座標 (x, y)
#             descs : (N, 64) L2 正規化済み記述子
#             scores: (N,)    キーポイントスコア
#         """
#         feats, kp_logits, hmap = self.thermal_feat(img)
#         feats = F.normalize(feats, dim=1)
#         B, C, Hf, Wf = feats.shape
#         H, W = img.shape[2], img.shape[3]

#         # P(keypoint) = 1 - P(dustbin)
#         probs    = F.softmax(kp_logits, dim=1)
#         kp_score = probs[:, :64].sum(dim=1)   # (B, Hf, Wf)

#         kpts_list, descs_list, scores_list = [], [], []
#         for b in range(B):
#             scores_flat = kp_score[b].flatten()
#             feats_flat  = feats[b].reshape(C, -1).T   # (Hf*Wf, C)
#             k           = min(max_kp, scores_flat.shape[0])
#             top_idx     = scores_flat.topk(k).indices
#             iy = (top_idx // Wf).float() * (H / Hf)
#             ix = (top_idx %  Wf).float() * (W / Wf)
#             kpts_list.append(torch.stack([ix, iy], dim=1))    # (k, 2)
#             descs_list.append(feats_flat[top_idx])             # (k, C)
#             scores_list.append(scores_flat[top_idx])           # (k,)

#         return kpts_list, descs_list, scores_list

#     def _to_lg_input(
#         self,
#         kpts:   Tensor,
#         descs:  Tensor,
#         scores: Tensor,
#         H: int,
#         W: int,
#     ) -> dict:
#         """LightGlue の入力形式に変換する。"""
#         scores = scores + 1e-6
#         descs = F.normalize(descs, p=2, dim=-1)
#         # DEBUG ログは不要なため削除

#         # 画素座標 → [-1, 1] に正規化
#         kpts_norm = kpts.clone()
#         kpts_norm[:, 0] = kpts[:, 0] / W * 2.0 - 1.0
#         kpts_norm[:, 1] = kpts[:, 1] / H * 2.0 - 1.0
#         return {
#             'keypoints':   kpts_norm.unsqueeze(0),  # (1, N, 2)
#             'descriptors': descs.unsqueeze(0),       # (1, N, 64)
#             'keypoint_scores': scores.unsqueeze(0),  # (1, N)
#         }

#     def _log(self, metrics: dict, step: int) -> None:
#         for k, v in metrics.items():
#             self.writer.add_scalar(k, v, step)
#         if self.use_wandb:
#             try:
#                 import wandb
#                 wandb.log(metrics, step=step)
#             except Exception:
#                 pass

#     def _save(self, tag: str) -> None:
#         filename = f'lightglue_ft_{tag}.pth'
#         local    = os.path.join(self.ckpt_path, filename)
#         torch.save(self.lightglue.state_dict(), local)
#         print(f"[LG-FT] Saved (local): {local}")
#         if self.use_wandb:
#             try:
#                 import wandb
#                 if wandb.run is not None:
#                     wp = os.path.join(wandb.run.dir, filename)
#                     torch.save(self.lightglue.state_dict(), wp)
#                     wandb.save(wp, base_path=wandb.run.dir)
#                     print(f"[LG-FT] Saved (wandb): {wp}")
#             except Exception as e:
#                 print(f"[LG-FT] WARNING: wandb save failed: {e}")

#     # def run(self, loader: DataLoader) -> None:
#     #     print("\n" + "=" * 60)
#     #     print("  LightGlue Fine-tuning")
#     #     print(f"  Train datasets : Freiburg + TartanRGBT")
#     #     print(f"  Eval datasets  : SThErEO, VIVID（未使用）")
#     #     print("=" * 60)

#     #     lr         = getattr(self.args, 'lr',             1e-5)
#     #     n_steps    = getattr(self.args, 'n_steps',        20000)
#     #     grad_clip  = getattr(self.args, 'grad_clip',      1.0)
#     #     log_every  = getattr(self.args, 'log_every',      100)
#     #     save_every = getattr(self.args, 'save_ckpt_every', 2000)
#     #     lambda_epi = getattr(self.args, 'lambda_epi',     0.1)
#     #     inlier_thr = getattr(self.args, 'inlier_threshold', 3.0)
#     #     epi_thr    = getattr(self.args, 'epi_threshold',  2.0)
#     #     max_kp     = getattr(self.args, 'max_keypoints',  512)

#     #     opt = optim.Adam(
#     #         filter(lambda p: p.requires_grad, self.lightglue.parameters()),
#     #         lr=lr,
#     #     )
#     #     scheduler = optim.lr_scheduler.StepLR(
#     #         opt,
#     #         step_size = getattr(self.args, 'lr_step', 10000),
#     #         gamma     = getattr(self.args, 'lr_gamma', 0.5),
#     #     )

#     #     data_iter = iter(loader)
#     #     step      = 0

#     #     while step < n_steps:
#     #         try:
#     #             batch = next(data_iter)
#     #         except StopIteration:
#     #             data_iter = iter(loader)
#     #             batch     = next(data_iter)

#     #         thr_t  = batch['thr_t'].to(self.dev)    # (B, 3, H, W)
#     #         thr_t1 = batch['thr_t1'].to(self.dev)
#     #         T_rel  = batch['T_rel'].to(self.dev)    # (B, 4, 4)
#     #         K      = batch['K'].to(self.dev)        # (B, 3, 3)
#     #         valid  = batch['valid']                  # (B,)

#     #         B, _, H, W = thr_t.shape

#     #         # ── 特徴抽出（frozen）─────────────────────────────────────────
#     #         kpts_list1, descs_list1, scores_list1 = \
#     #             self._extract_features(thr_t,  max_kp)
#     #         kpts_list2, descs_list2, scores_list2 = \
#     #             self._extract_features(thr_t1, max_kp)

#     #         losses_match = []
#     #         losses_epi   = []

#     #         for b in range(B):
#     #             kpts1  = kpts_list1[b]
#     #             descs1 = descs_list1[b]
#     #             sc1    = scores_list1[b]
#     #             kpts2  = kpts_list2[b]
#     #             descs2 = descs_list2[b]
#     #             sc2    = scores_list2[b]

#     #             if kpts1.shape[0] < 8 or kpts2.shape[0] < 8:
#     #                 continue

#     #             # ── LightGlue フォワード ─────────────────────────────────
#     #             inp0 = self._to_lg_input(kpts1, descs1, sc1, H, W)
#     #             inp1 = self._to_lg_input(kpts2, descs2, sc2, H, W)
#     #             try:
#     #                 pred = self.lightglue({'image0': inp0, 'image1': inp1})
#     #             except Exception as e:
#     #                 print(f"[LG-FT] LightGlue forward error: {e}")
#     #                 import traceback
#     #                 traceback.print_exc()
#     #                 continue

#     #             # LightGlue の出力からスコア行列を取得
#     #             # matches0: (1, N1) 各キーポイントのマッチ先インデックス(-1=unmatch)
#     #             # scores は log_assignment: (1, N1+1, N2+1) または matching_scores
#     #             if 'log_assignment' in pred:
#     #                 scores_mat = pred['log_assignment'][0]  # (N1+1, N2+1)
#     #             elif 'matching_scores0' in pred:
#     #                 # older API: スコアから再構築
#     #                 scores_mat = pred['matching_scores0'][0].unsqueeze(-1)
#     #                 scores_mat = scores_mat.expand(-1, kpts2.shape[0] + 1)
#     #             else:
#     #                 continue

#     #             # ── 損失計算 ─────────────────────────────────────────────
#     #             use_gt = valid[b].item()

#     #             l_match = lightglue_matching_loss(
#     #                 scores     = scores_mat,
#     #                 pts1       = kpts1,
#     #                 pts2       = kpts2,
#     #                 K          = K[b],
#     #                 T_rel      = T_rel[b],
#     #                 use_gt     = use_gt,
#     #                 inlier_thr = inlier_thr,
#     #             )
#     #             losses_match.append(l_match)

#     #             # L_epi は GT ポーズがある場合のみ
#     #             if use_gt:
#     #                 l_epi = lightglue_epi_loss(
#     #                     pts1      = kpts1,
#     #                     pts2      = kpts2,
#     #                     scores    = scores_mat,
#     #                     K         = K[b],
#     #                     T_rel     = T_rel[b],
#     #                     threshold = epi_thr,
#     #                 )
#     #                 losses_epi.append(l_epi)

#     #         if not losses_match:
#     #             step += 1
#     #             continue

#     #         l_match_mean = torch.stack(losses_match).mean()
#     #         l_epi_mean   = (torch.stack(losses_epi).mean()
#     #                         if losses_epi else l_match_mean.new_zeros(1).squeeze())
#     #         loss = l_match_mean + lambda_epi * l_epi_mean

#     #         opt.zero_grad()
#     #         loss.backward()
#     #         torch.nn.utils.clip_grad_norm_(
#     #             self.lightglue.parameters(), grad_clip)
#     #         opt.step()
#     #         scheduler.step()

#     #         pred = self.lightglue({'image0': inp0, 'image1': inp1})

#     #         if step % 10 == 0:
#     #             # ネットワークの最初のパラメータ（重み）を取得
#     #             # 学習が進んでいれば、この値の合計(sum)がステップごとに変化します
#     #             param = next(self.lightglue.parameters())
#     #             weight_sum = param.data.sum().item()
                
#     #             print(f"Pred Keys: {pred.keys()}")
#     #             if 'log_assignment' in pred:
#     #                 # スコアの統計値を詳細に確認
#     #                 s = pred['log_assignment']
#     #                 print(f"Raw Logits: mean={s.mean().item():.6f}, std={s.std().item():.6f}")
#     #             # 非常に小さな変化も見逃さないように小数第10位まで表示
#     #             print(f"[DEBUG {step:06d}] Weight Sum: {weight_sum:.10f} | Loss: {loss.item():.6f}")
#     #             print(f"      Score Range: min={scores_mat.min().item():.4f}, max={scores_mat.max().item():.4f}")

#     #         if step % log_every == 0:
#     #             self._log({
#     #                 'lg_ft/loss_total' : loss.item(),
#     #                 'lg_ft/loss_match' : l_match_mean.item(),
#     #                 'lg_ft/loss_epi'   : l_epi_mean.item(),
#     #                 'lg_ft/lr'         : opt.param_groups[0]['lr'],
#     #                 'lg_ft/grad_norm'  : total_grad_norm(self.lightglue),
#     #                 'lg_ft/n_batch'    : len(losses_match),
#     #             }, step)
#     #             print(
#     #                 f"[LG-FT {step:06d}] "
#     #                 f"total={loss.item():.4f}  "
#     #                 f"match={l_match_mean.item():.4f}  "
#     #                 f"epi={l_epi_mean.item():.4f}  "
#     #                 f"lr={opt.param_groups[0]['lr']:.2e}"
#     #             )

#     #         if (step + 1) % save_every == 0:
#     #             self._save(f'step{step + 1}')

#     #         step += 1

#     #     self._save('final')
#     #     print("[LG-FT] Fine-tuning done.")
#     # def run(self, loader: DataLoader) -> None:
#     #     print("\n" + "=" * 60)
#     #     print("  LightGlue Fine-tuning")
#     #     print(f"  Train datasets : Freiburg + TartanRGBT")
#     #     print(f"  Eval datasets  : SThErEO, VIVID（未使用）")
#     #     print("=" * 60)

#     #     lr         = getattr(self.args, 'lr',             1e-4)
#     #     n_steps    = getattr(self.args, 'n_steps',        10000)
#     #     grad_clip  = getattr(self.args, 'grad_clip',      1.0)
#     #     log_every  = getattr(self.args, 'log_every',      100)
#     #     save_every = getattr(self.args, 'save_ckpt_every', 2000)
#     #     lambda_epi = getattr(self.args, 'lambda_epi',      0.1)
#     #     inlier_thr = getattr(self.args, 'inlier_threshold', 5.0) # 閾値を少し厳しく(5.0)設定
#     #     epi_thr    = getattr(self.args, 'epi_threshold',  2.0)
#     #     max_kp     = getattr(self.args, 'max_keypoints',  512)

#     #     # 勾配が必要なパラメータのみを抽出
#     #     opt = optim.Adam(
#     #         filter(lambda p: p.requires_grad, self.lightglue.parameters()),
#     #         lr=lr,
#     #     )
#     #     scheduler = optim.lr_scheduler.StepLR(
#     #         opt,
#     #         step_size = getattr(self.args, 'lr_step', 10000),
#     #         gamma     = getattr(self.args, 'lr_gamma', 0.5),
#     #     )

#     #     data_iter = iter(loader)
#     #     step      = 0

#     #     self.lightglue.train() # ループ前に明示的に train モードに
#     #     while step < n_steps:
#     #         try:
#     #             batch = next(data_iter)
#     #         except StopIteration:
#     #             data_iter = iter(loader)
#     #             batch     = next(data_iter)

#     #         thr_t  = batch['thr_t'].to(self.dev)
#     #         thr_t1 = batch['thr_t1'].to(self.dev)
#     #         T_rel  = batch['T_rel'].to(self.dev)
#     #         K      = batch['K'].to(self.dev)
#     #         valid  = batch['valid']

#     #         B, _, H, W = thr_t.shape

#     #         # 特徴抽出
#     #         kpts_list1, descs_list1, scores_list1 = self._extract_features(thr_t,  max_kp)
#     #         kpts_list2, descs_list2, scores_list2 = self._extract_features(thr_t1, max_kp)

#     #         losses_match = []
#     #         losses_epi   = []

#     #         for b in range(B):
#     #             kpts1, descs1, sc1 = kpts_list1[b], descs_list1[b], scores_list1[b]
#     #             kpts2, descs2, sc2 = kpts_list2[b], descs_list2[b], scores_list2[b]

#     #             if kpts1.shape[0] < 8 or kpts2.shape[0] < 8:
#     #                 continue

#     #             inp0 = self._to_lg_input(kpts1, descs1, sc1, H, W)
#     #             inp1 = self._to_lg_input(kpts2, descs2, sc2, H, W)

#     #             try:
#     #                 # モデルを train モードで実行し、割り当て行列を生成させる
#     #                 self.lightglue.train()
#     #                 pred = self.lightglue({'image0': inp0, 'image1': inp1})
                    
#     #                 # ── [重要] 正しい割当行列の抽出ロジック ──
#     #                 # if 'log_assignment' in pred:
#     #                 #     # 最後の層(最精緻)の割当行列を取得
#     #                 #     res = pred['log_assignment']
#     #                 #     scores_mat = res[-1][0] if isinstance(res, list) else res[0]
#     #                 # elif 'scores' in pred and pred['scores'].numel() > 0:
#     #                 #     # 'scores' に行列が入っている場合
#     #                 #     res = pred['scores']
#     #                 #     scores_mat = res[-1][0] if res.ndim > 2 else res
#     #                 # else:
#     #                 #     # 学習に必要なデータが取れない場合はスキップ
#     #                 #     continue
#     #                 # ── [修正：リストとテンソルの両方に対応する抽出ロジック] ───────────────────
#     #                 if 'log_assignment' in pred:
#     #                     res = pred['log_assignment']
#     #                     # リストなら最後の要素（最終層）を取り、さらにバッチの0番目を取る
#     #                     if isinstance(res, list):
#     #                         scores_mat = res[-1][0] 
#     #                     else:
#     #                         scores_mat = res[0]
                            
#     #                 elif 'scores' in pred:
#     #                     res = pred['scores']
#     #                     # リストの場合
#     #                     if isinstance(res, list):
#     #                         if len(res) > 0:
#     #                             scores_mat = res[-1]
#     #                             # もし取り出したものがまだ (Batch, N1+1, N2+1) なら [0] で次元を落とす
#     #                             if scores_mat.ndim > 2:
#     #                                 scores_mat = scores_mat[0]
#     #                         else:
#     #                             continue
#     #                     # テンソルの場合
#     #                     else:
#     #                         scores_mat = res[0] if res.ndim > 2 else res
#     #                 else:
#     #                     continue

#     #                 # ─────────────────────────────────────────

#     #                 # 損失計算 (温度パラメータ τ=0.1 を適用して尖らせる)
#     #                 tau = 0.1
#     #                 l_match = lightglue_matching_loss(
#     #                     scores     = scores_mat / tau, 
#     #                     pts1       = kpts1,
#     #                     pts2       = kpts2,
#     #                     K          = K[b],
#     #                     T_rel      = T_rel[b],
#     #                     use_gt     = valid[b].item(),
#     #                     inlier_thr = inlier_thr,
#     #                 )
#     #                 losses_match.append(l_match)

#     #                 if valid[b].item():
#     #                     l_epi = lightglue_epi_loss(kpts1, kpts2, scores_mat, K[b], T_rel[b], epi_thr)
#     #                     losses_epi.append(l_epi)

#     #             except Exception as e:
#     #                 print(f"[LG-FT] Skip batch due to error: {e}")
#     #                 continue

#     #         if not losses_match:
#     #             step += 1
#     #             continue

#     #         # バックプロパゲーション
#     #         l_match_mean = torch.stack(losses_match).mean()
#     #         l_epi_mean   = torch.stack(losses_epi).mean() if losses_epi else l_match_mean.new_zeros(1).squeeze()
#     #         loss = l_match_mean + lambda_epi * l_epi_mean

#     #         opt.zero_grad()
#     #         loss.backward()
#     #         torch.nn.utils.clip_grad_norm_(self.lightglue.parameters(), grad_clip)
#     #         opt.step()
#     #         scheduler.step()

#     #         # デバッグ表示
#     #         if step % 10 == 0:
#     #             param = next(self.lightglue.parameters())
#     #             weight_sum = param.data.sum().item()
#     #             # スコアが 0 以外に動いているか監視
#     #             s_min, s_max = scores_mat.min().item(), scores_mat.max().item()
#     #             print(f"[DEBUG {step:06d}] WeightSum: {weight_sum:.8f} | Loss: {loss.item():.4f} | ScoreRange: [{s_min:.2f}, {s_max:.2f}]")

#     #         if step % log_every == 0:
#     #             print(f"[LG-FT {step:06d}] total={loss.item():.4f}  match={l_match_mean.item():.4f}  lr={opt.param_groups[0]['lr']:.2e}")

#     #         if (step + 1) % save_every == 0:
#     #             self._save(f'step{step + 1}')
#     #         step += 1

#     #     self._save('final')
#     #     print("[LG-FT] Fine-tuning done.")
#     def run(self, loader: DataLoader) -> None:
#         print("\n" + "=" * 60)
#         print("  LightGlue Fine-tuning (Final Debug & Run)")
#         print("=" * 60)

#         # パラメータ設定
#         lr = getattr(self.args, 'lr', 1e-4)
#         n_steps = getattr(self.args, 'n_steps', 10000)
#         inlier_thr = 5.0  # 精度のために引き締め
#         grad_clip = getattr(self.args, 'grad_clip', 1.0)
#         log_every = getattr(self.args, 'log_every', 100)
#         save_every = getattr(self.args, 'save_ckpt_every', 2000)
#         lambda_epi = getattr(self.args, 'lambda_epi', 0.1)
#         epi_thr = getattr(self.args, 'epi_threshold', 2.0)

#         # オプティマイザ設定
#         opt = optim.Adam(
#             filter(lambda p: p.requires_grad, self.lightglue.parameters()),
#             lr=lr,
#         )
#         scheduler = optim.lr_scheduler.StepLR(
#             opt,
#             step_size=getattr(self.args, 'lr_step', 10000),
#             gamma=getattr(self.args, 'lr_gamma', 0.5),
#         )

#         data_iter = iter(loader)
#         step = 0

#         while step < n_steps:
#             try:
#                 # バッチ取得
#                 try:
#                     batch = next(data_iter)
#                 except StopIteration:
#                     data_iter = iter(loader)
#                     batch = next(data_iter)

#                 # デバイス転送
#                 thr_t, thr_t1 = batch['thr_t'].to(self.dev), batch['thr_t1'].to(self.dev)
#                 T_rel, K = batch['T_rel'].to(self.dev), batch['K'].to(self.dev)
#                 valid = batch['valid']
#                 B, _, H, W = thr_t.shape

#                 # 特徴抽出 (XFeat: Frozen)
#                 kpts_list1, descs_list1, scores_list1 = self._extract_features(thr_t, 512)
#                 kpts_list2, descs_list2, scores_list2 = self._extract_features(thr_t1, 512)

#                 losses_match = []
#                 losses_epi = []

#                 for b in range(B):
#                     kpts1, descs1, sc1 = kpts_list1[b], descs_list1[b], scores_list1[b]
#                     kpts2, descs2, sc2 = kpts_list2[b], descs_list2[b], scores_list2[b]

#                     if kpts1.shape[0] < 8 or kpts2.shape[0] < 8:
#                         continue

#                     inp0 = self._to_lg_input(kpts1, descs1, sc1, H, W)
#                     inp1 = self._to_lg_input(kpts2, descs2, sc2, H, W)

#                     # ── LightGlue フォワード ──
#                     self.lightglue.train() 
#                     # 明示的に training=True を渡すことで、log_assignment を強制
#                     pred = self.lightglue({'image0': inp0, 'image1': inp1})

#                     # 【診断ログ】最初の1回だけ、モデルが何を返したか全表示する
#                     if step == 0 and b == 0:
#                         print(f"\n[DIAGNOSTIC] Step 0 Keys: {list(pred.keys())}")
#                         for k, v in pred.items():
#                             if isinstance(v, (torch.Tensor, list)):
#                                 shape = v[0].shape if isinstance(v, list) and len(v) > 0 else (v.shape if isinstance(v, torch.Tensor) else "empty")
#                                 print(f"  - {k:18s} | {type(v).__name__:8s} | Shape: {shape}")

#                     # ── 割当行列の取得（cvg/LightGlue API 対応版）──────────
#                     # cvg/LightGlue の training=True 時の出力:
#                     #   'log_assignment': (B, N1+1, N2+1) ← これが理想
#                     # cvg/LightGlue の testing 時の出力:
#                     #   'matching_scores0': (B, N1)  各KPの最大マッチスコア
#                     #   'scores': list of scalar     ← shape(2,) で2D未満 → 使えない
#                     # 対策: 常に 'log_assignment' を取得するよう LG を強制
#                     scores_mat = None

#                     if 'log_assignment' in pred:
#                         # 理想ケース: training=True で完全なスコア行列
#                         res = pred['log_assignment']
#                         res = res[-1] if isinstance(res, list) else res
#                         scores_mat = res[0] if res.ndim == 3 else res

#                     elif 'matching_scores0' in pred:
#                         # フォールバック: (1,N1) → 対角的な (N1+1, N2+1) を構築
#                         # matching_scores0[b,i] = max_j P(match i→j)
#                         ms0 = pred['matching_scores0'][0]   # (N1,)
#                         ms1 = pred['matching_scores1'][0]   # (N2,) あれば
#                         N1e = kpts1.shape[0]
#                         N2e = kpts2.shape[0]
#                         # (N1+1, N2+1) のゼロ行列を作り対角に ms0 を入れる
#                         sm = kpts1.new_zeros(N1e + 1, N2e + 1)
#                         n  = min(N1e, N2e, ms0.shape[0])
#                         sm[torch.arange(n), torch.arange(n)] = ms0[:n]
#                         # ダストビン行・列はマッチスコアの補数
#                         sm[:N1e, N2e] = 1.0 - ms0.clamp(0, 1)
#                         scores_mat = sm

#                     if scores_mat is None or scores_mat.ndim < 2:
#                         if step == 0:
#                             print(f"[LG-FT] scores_mat 取得失敗. pred keys={list(pred.keys())}")
#                         continue

#                     # ── 損失計算 ──
#                     # 温度パラメータ 0.1 で勾配を尖らせる
#                     l_match = lightglue_matching_loss(
#                         scores=scores_mat / 0.1, 
#                         pts1=kpts1, pts2=kpts2, K=K[b], T_rel=T_rel[b],
#                         use_gt=valid[b].item(), inlier_thr=inlier_thr
#                     )
#                     losses_match.append(l_match)

#                     # エピポーラ損失
#                     if valid[b].item():
#                         l_epi = lightglue_epi_loss(kpts1, kpts2, scores_mat, K[b], T_rel[b], epi_thr)
#                         losses_epi.append(l_epi)

#                 # ── パラメータ更新 ──
#                 if not losses_match:
#                     if step % 10 == 0:
#                         print(f"[DEBUG] Step {step}: Skipping (losses_match empty). Check DIAGNOSTIC log.")
#                     step += 1
#                     continue

#                 l_match_mean = torch.stack(losses_match).mean()
#                 l_epi_mean = torch.stack(losses_epi).mean() if losses_epi else l_match_mean.new_zeros(1).squeeze()
#                 loss = l_match_mean + lambda_epi * l_epi_mean

#                 opt.zero_grad()
#                 loss.backward()
#                 torch.nn.utils.clip_grad_norm_(self.lightglue.parameters(), grad_clip)
#                 opt.step()
#                 scheduler.step()

#                 # ── 進捗表示 ──
#                 if step % 10 == 0:
#                     param = next(self.lightglue.parameters())
#                     s_min, s_max = scores_mat.min().item(), scores_mat.max().item()
#                     print(f"[STEP {step:05d}] Loss: {loss.item():.4f} | Range: [{s_min:.2f}, {s_max:.2f}] | WeightSum: {param.data.sum().item():.6f}")

#                 # ログ
#                 if step % log_every == 0:
#                     self._log({'lg_ft/loss_total': loss.item(), 'lg_ft/lr': opt.param_groups[0]['lr']}, step)

#                 if (step + 1) % save_every == 0:
#                     self._save(f'step{step + 1}')

#                 step += 1

#             except Exception as e:
#                 print(f"[ERROR] Step {step}: {e}")
#                 import traceback
#                 traceback.print_exc()
#                 step += 1
#                 if step > 50: break # 連続エラー回避

#         self._save('final')
#         print("[LG-FT] Fine-tuning done.")

#     def finalize(self) -> None:
#         self.writer.close()
#         if self.use_wandb:
#             try:
#                 import wandb
#                 wandb.finish()
#             except Exception:
#                 pass

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


def build_gt_labels(
    pts1: Tensor,
    pts2: Tensor,
    K:    Tensor,
    T_rel: Tensor,
    use_gt: bool,
    match_thr: float = 3.0,
    outlier_thr: float = 5.0,
):
    """
    SThErEO 用の GT ラベル生成（LightGlue Appendix C.4 の SThErEO 版）。

    論文 Appendix C.4（MegaDepth）の GT ラベル生成:
      Match:   再投影誤差 < 3px かつ相互最近傍   → GT M
      Outlier: 再投影誤差 > 5px               → Ā または B̄
      Ignore:  3px ≤ 誤差 ≤ 5px             → 損失に含めない

    SThErEO 版（深度なし、GPS/IMU GT pose）:
      F 行列から Sampson 距離を計算（深度の代替）
      Match:   MNN + Sampson < match_thr²       → GT M
      Outlier: 全 B 点と min_Sampson > outlier_thr²  → Ā
               全 A 点と min_Sampson > outlier_thr²  → B̄
      Ignore:  match_thr² ≤ min_Sampson ≤ outlier_thr²  → 損失に含めない

    Returns:
        match_ids:   (n_match, 2) tensor of (i, j) GT match indices
        unmatch_A:   (n_uA,)  bool mask for unmatchable points in A
        unmatch_B:   (n_uB,)  bool mask for unmatchable points in B
    """
    N1, N2 = pts1.shape[0], pts2.shape[0]
    device = pts1.device

    # F 行列の計算
    with torch.no_grad():
        if use_gt:
            F_mat = _compute_F_matrix(K, T_rel)
        else:
            import cv2
            p1_np = pts1.cpu().numpy().astype(np.float32)
            p2_np = pts2.cpu().numpy().astype(np.float32)
            F_np, _ = cv2.findFundamentalMat(p1_np, p2_np, cv2.FM_8POINT)
            if F_np is None:
                empty = torch.zeros(0, 2, dtype=torch.long, device=device)
                return empty, torch.zeros(N1, dtype=torch.bool, device=device),                               torch.zeros(N2, dtype=torch.bool, device=device)
            F_mat = torch.from_numpy(F_np).float().to(device)

    with torch.no_grad():
        sd = _compute_all_sampson_dist(pts1, pts2, F_mat)   # (N1, N2)

        min_sd1, nn1 = sd.min(dim=1)   # (N1,) frame2 での最近傍
        min_sd2, nn2 = sd.min(dim=0)   # (N2,) frame1 での最近傍

        # GT Match: MNN + Sampson < match_thr²
        ids1  = torch.arange(N1, device=device)
        ids2  = torch.arange(N2, device=device)
        mnn1  = (nn2[nn1] == ids1) & (min_sd1 < match_thr ** 2)   # (N1,)
        mnn2  = (nn1[nn2] == ids2) & (min_sd2 < match_thr ** 2)   # (N2,)

        match_i = ids1[mnn1]
        match_j = nn1[mnn1]
        match_ids = torch.stack([match_i, match_j], dim=1)   # (n_match, 2)

        # Outlier: 全相手点との最小 Sampson > outlier_thr²（曖昧領域を除外）
        # Ignore: match_thr² ≤ min_Sampson ≤ outlier_thr² （何もしない）
        unmatch_A = min_sd1 > outlier_thr ** 2   # (N1,)
        unmatch_B = min_sd2 > outlier_thr ** 2   # (N2,)

    return match_ids, unmatch_A, unmatch_B


# def nll_one_layer(
#     log_assign: Tensor,
#     match_ids:  Tensor,
#     unmatch_A:  Tensor,
#     unmatch_B:  Tensor,
# ) -> Tensor:
#     """
#     LightGlue 論文 Eq. 11 の一層分の損失。

#     log_assign: (N1+1, N2+1) log-assignment 行列（dustbin を含む）
#     match_ids:  (n_match, 2) GT 対応点インデックス
#     unmatch_A:  (N1,) bool  非対応点マスク（A 側）
#     unmatch_B:  (N2,) bool  非対応点マスク（B 側）

#     損失 (Eq. 11 の一層分):
#       L_ℓ = -(1/|M|)  Σ_{(i,j)∈M}  log P_ij
#             -(1/2|Ā|) Σ_{i∈Ā}       log P_{i,dustbin}   ≈ log(1-σ_i)
#             -(1/2|B̄|) Σ_{j∈B̄}       log P_{dustbin,j}   ≈ log(1-σ_j)
#     """
#     N1 = log_assign.shape[0] - 1
#     N2 = log_assign.shape[1] - 1

#     loss = log_assign.new_zeros(1)
#     n_terms = 0

#     # (A) GT 対応点: log P_{i,j} を最大化
#     n_match = match_ids.shape[0]
#     if n_match > 0:
#         ii = match_ids[:, 0]
#         jj = match_ids[:, 1]
#         loss_pos = -log_assign[ii, jj].sum() / n_match
#         loss = loss + loss_pos
#         n_terms += 1

#     # (B) 非対応点 A: log P_{i, dustbin} を最大化（ = log(1 - σ_i) と等価）
#     n_uA = int(unmatch_A.sum())
#     if n_uA > 0:
#         uA_idx = torch.where(unmatch_A)[0]
#         loss_negA = -log_assign[uA_idx, N2].sum() / (2 * n_uA)
#         loss = loss + loss_negA
#         n_terms += 1

#     # (C) 非対応点 B: log P_{dustbin, j} を最大化（ = log(1 - σ_j) と等価）
#     n_uB = int(unmatch_B.sum())
#     if n_uB > 0:
#         uB_idx = torch.where(unmatch_B)[0]
#         loss_negB = -log_assign[N1, uB_idx].sum() / (2 * n_uB)
#         loss = loss + loss_negB
#         n_terms += 1

#     if n_terms == 0:
#         return log_assign.new_zeros(1).squeeze()

#     return loss.squeeze()
def nll_one_layer(
    log_assign: Tensor,
    match_ids:  Tensor,
    unmatch_A:  Tensor,
    unmatch_B:  Tensor,
) -> Tensor:
    """
    LightGlue 論文 Eq. 11 の一層分の損失を計算する。
    
    論理的修正：
    入力 log_assign が確率空間 [0, 1] にある場合（現状のバグ）、
    自動的に対数空間 [-inf, 0] に変換し、負の損失の発生を防ぐ。
    """
    N1 = log_assign.shape[0] - 1
    N2 = log_assign.shape[1] - 1

    # ── 【論理的ガード】確率から対数尤度への強制変換 ──
    # Range [0.00, 1.00] のログが出ている場合、この処理が必須となる。
    # 0 の対数計算による NaN を防ぐため、1e-8 でクランプする。
    # if log_assign.min() >= 0.0 and log_assign.max() <= 1.05:
    #     log_assign = torch.log(log_assign.clamp(min=1e-8))
    if log_assign.min() >= 0.0:
        log_assign = torch.log(log_assign.clamp(min=1e-8))

    loss = log_assign.new_zeros(1)
    n_terms = 0

    # (A) GT 対応点: -log P_{i,j}
    n_match = match_ids.shape[0]
    if n_match > 0:
        ii = match_ids[:, 0]
        jj = match_ids[:, 1]
        # 対数尤度は負なので、マイナスを掛けることで Loss は正になる
        loss_pos = -log_assign[ii, jj].sum() / n_match
        loss = loss + loss_pos
        n_terms += 1

    # (B) 非対応点 A: -0.5 * log(1 - σ_i) ≈ -0.5 * log P_{i, dustbin}
    n_uA = int(unmatch_A.sum())
    if n_uA > 0:
        uA_idx = torch.where(unmatch_A)[0]
        # 論文 Eq. 11 の係数 1/2|Ā| に準拠
        loss_negA = -log_assign[uA_idx, N2].sum() / (2 * n_uA)
        loss = loss + loss_negA
        n_terms += 1

    # (C) 非対応点 B: -0.5 * log(1 - σ_j) ≈ -0.5 * log P_{dustbin, j}
    n_uB = int(unmatch_B.sum())
    if n_uB > 0:
        uB_idx = torch.where(unmatch_B)[0]
        loss_negB = -log_assign[N1, uB_idx].sum() / (2 * n_uB)
        loss = loss + loss_negB
        n_terms += 1

    # どの項目も計算できなかった場合は 0 を返す
    if n_terms == 0:
        return log_assign.new_zeros(1).squeeze()

    # 論文通りに各項の平均を加算（スケーリングの一貫性を維持）
    return loss.squeeze()


def lightglue_matching_loss(
    scores:      Tensor,
    pts1:        Tensor,
    pts2:        Tensor,
    K:           Tensor,
    T_rel:       Tensor,
    use_gt:      bool,
    inlier_thr:  float = 3.0,
    outlier_thr: float = 5.0,
) -> Tensor:
    """
    LightGlue 論文 Eq. 11 に準拠した NegativeLogAssignment 損失（SThErEO 版）。

    論文との対応:
      GT M:  MNN + Sampson < inlier_thr px  ← MegaDepth の再投影 < 3px に対応
      Ā,B̄:  min_Sampson > outlier_thr px   ← MegaDepth の再投影 > 5px に対応
      Ignore: inlier ≤ Sampson ≤ outlier    ← 曖昧領域（損失なし）
      Deep Supervision: scores がリストなら全層を平均（単一テンソルなら最終層のみ）

    Args:
        scores: (N1+1, N2+1) または [(N1+1,N2+1), ...] の log-assignment
    """
    N1 = pts1.shape[0]
    N2 = pts2.shape[0]

    if N1 < 8 or N2 < 8:
        return pts1.new_zeros(1).squeeze()

    # GT ラベルの生成
    match_ids, unmatch_A, unmatch_B = build_gt_labels(
        pts1, pts2, K, T_rel, use_gt, inlier_thr, outlier_thr)

    n_pos = match_ids.shape[0]
    if n_pos < 4:
        return pts1.new_zeros(1).squeeze()

    # GT ラベルのログ（5%確率）
    if torch.rand(1) < 0.05:
        print(f"[GT] M={n_pos}/{N1}({100*n_pos/N1:.0f}%) "
              f"Ā={int(unmatch_A.sum())} B̄={int(unmatch_B.sum())} "
              f"Ignore={N1 - n_pos - int(unmatch_A.sum())}")

    # Deep Supervision: scores がリスト（各層）または単一テンソル（最終層）
    if isinstance(scores, (list, tuple)):
        # 各層の損失を平均（論文の -(1/L) Σ_ℓ）
        layer_losses = []
        for la in scores:
            la_2d = la[0] if la.ndim == 3 else la   # (N1+1, N2+1)
            l = nll_one_layer(la_2d, match_ids, unmatch_A, unmatch_B)
            if l.requires_grad or l.item() != 0.0:
                layer_losses.append(l)
        if not layer_losses:
            return pts1.new_zeros(1).squeeze()
        return torch.stack(layer_losses).mean()
    else:
        # 最終層のみ
        la_2d = scores[0] if scores.ndim == 3 else scores
        return nll_one_layer(la_2d, match_ids, unmatch_A, unmatch_B)


def _compute_all_sampson_dist(pts1, pts2, F):
    """(N1, 2) と (N2, 2) の全ペア間の Sampson 距離を計算するヘルパー。"""
    N1, N2 = pts1.shape[0], pts2.shape[0]
    ones1 = torch.ones((N1, 1), device=pts1.device)
    ones2 = torch.ones((N2, 1), device=pts2.device)
    p1 = torch.cat([pts1, ones1], dim=1) # (N1, 3)
    p2 = torch.cat([pts2, ones2], dim=1) # (N2, 3)

    # エピポーラ線 L1 = F * p1, L2 = F^T * p2
    L1 = p1 @ F.t()  # (N1, 3)
    L2 = p2 @ F      # (N2, 3)

    # 代数的距離 p2^T * F * p1
    p2_F_p1 = (p2 @ F @ p1.t()).t() # (N1, N2)

    # Sampson 距離の分母計算
    denom = L1[:, 0:1]**2 + L1[:, 1:2]**2 + L2[None, :, 0]**2 + L2[None, :, 1]**2
    return (p2_F_p1**2) / (denom + 1e-9)

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

        # LightGlue の confidence 設定（属性がある場合のみ）
        # depth_confidence / width_confidence = -1 で早期停止を無効化
        # これにより全レイヤーが実行され log_assignment が生成される
        for attr, val in [('filter_threshold', -1.0),
                          ('depth_confidence', -1.0),
                          ('width_confidence', -1.0)]:
            if hasattr(self.lightglue, attr):
                setattr(self.lightglue, attr, val)
                
            if hasattr(self.lightglue, 'conf'):
                setattr(self.lightglue.conf, attr, val)
        # log_assignment を出力するよう強制（バージョン依存）
        if hasattr(self.lightglue, 'conf') and hasattr(self.lightglue.conf, 'log_assignment'):
            self.lightglue.conf.log_assignment = True
        # self.lightglue.training = True は .train() 呼び出しで保証

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

        input_dim = getattr(args, 'input_dim', 64)

        # 全 confidence 機能を無効化（学習時は全レイヤーを通す）
        # depth_confidence=-1, width_confidence=-1 で早期停止・枝刈りを無効化
        lg = LightGlue(
            features         = None,
            input_dim        = input_dim,
            filter_threshold = None,
            depth_confidence = -1.0,
            width_confidence = -1.0,
            flash            = False,
        ).to(self.dev)

        if hasattr(lg, 'conf'):
            lg.conf.filter_threshold = None
            lg.conf.depth_confidence = -1
            lg.conf.width_confidence = -1
            # これを True にすることで、各レイヤーの微分可能な log_assignment がリストで返ります
            lg.conf.log_assignment = True
        
        # n = sum(p.numel() for p in lg.parameters() if p.requires_grad)
        # print(f"[LG-FT] LightGlue loaded (input_dim={input_dim}, trainable={n:,})")

        # ── forward を wrap して log_assignment を強制取得 ────────────────
        # 根拠: cvg/LightGlue は training=True でも log_assignment を返さない
        #       バージョンがある。wrap して内部の log_assignment を取得する。
        original_forward = lg.forward.__func__

        def patched_forward(self_lg, data: dict) -> dict:
            # 通常の forward を実行
            pred = original_forward(self_lg, data)

            # log_assignment が存在しない場合は内部から取得を試みる
            # if 'log_assignment' not in pred and self_lg.training:
            #     # LightGlue の内部変数 log_assignment を取得する
            #     # gluefactory 版: _get_log_assignment()
            #     # cvg 版: log_assignment は token の最終状態から計算
            #     # 代替: matching_scores0 から soft assignment を再構成
            #     if 'matching_scores0' in pred and 'matching_scores1' in pred:
            #         ms0 = pred['matching_scores0']   # (B, N1)
            #         ms1 = pred['matching_scores1']   # (B, N2)
            #         B, N1 = ms0.shape
            #         N2 = ms1.shape[1]
            #         # (B, N1+1, N2+1) の擬似 log-assignment を構築
            #         # ms0[b,i] = LG が "KP i はマッチする" と判断した確率
            #         # ここから soft assignment を再構成する
            #         # 対角方向に ms0, dustbin に 1-ms0 を配置
            #         # la = ms0.new_zeros(B, N1 + 1, N2 + 1)
            #         la = ms0.new_full((B, N1 + 1, N2 + 1), -10.0)
            #         n  = min(N1, N2)
            #         # la[:, torch.arange(n), torch.arange(n)] = ms0[:, :n]
            #         la[:, torch.arange(n), torch.arange(n)] = ms0[:, :n].clamp(min=1e-8).log()
            #         # la[:, :N1, N2] = 1.0 - ms0
            #         pred['log_assignment'] = la
            # return pred
            if 'log_assignment' in pred:
                return pred
            
            if self_lg.training and 'matching_scores0' in pred:
                ms0 = pred['matching_scores0']
                B, N1 = ms0.shape
                # デフォルトを負の大きな値（log(0)に近い値）で埋める
                la = ms0.new_full((B, N1 + 1, N1 + 1), -10.0)
                n = min(N1, N1) # 正方形行列を想定
                # 確率 ms0 を対数化して代入
                la[:, torch.arange(n), torch.arange(n)] = ms0[:, :n].clamp(min=1e-8).log()
                pred['log_assignment'] = la
                
            return pred

        import types
        lg.forward = types.MethodType(patched_forward, lg)
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
        scores = scores + 1e-6
        descs = F.normalize(descs, p=2, dim=-1)
        # DEBUG ログは不要なため削除

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

    # def run(self, loader: DataLoader) -> None:
    #     print("\n" + "=" * 60)
    #     print("  LightGlue Fine-tuning (Final Debug & Run)")
    #     print("=" * 60)

    #     # パラメータ設定
    #     lr = getattr(self.args, 'lr', 1e-4)
    #     n_steps = getattr(self.args, 'n_steps', 10000)
    #     inlier_thr = 5.0  # 精度のために引き締め
    #     grad_clip = getattr(self.args, 'grad_clip', 1.0)
    #     log_every = getattr(self.args, 'log_every', 100)
    #     save_every = getattr(self.args, 'save_ckpt_every', 2000)
    #     lambda_epi = getattr(self.args, 'lambda_epi', 0.1)
    #     epi_thr = getattr(self.args, 'epi_threshold', 2.0)

    #     # オプティマイザ設定
    #     opt = optim.Adam(
    #         filter(lambda p: p.requires_grad, self.lightglue.parameters()),
    #         lr=lr,
    #     )
    #     # lr スケジューラ: 論文 Appendix C.4 準拠
    #     # 「exponentially decay by 0.95 each epoch after 10 epochs」
    #     # n_steps / epoch_steps ≈ 10 epoch → lr_step = epoch_steps
    #     epoch_steps = getattr(self.args, 'epoch_steps', 2619)  # SThErEO 10476/4
    #     lr_decay_start = getattr(self.args, 'lr_decay_start_epoch', 10)
    #     lr_step  = epoch_steps                  # 1 epoch ごとに decay
    #     lr_gamma = getattr(self.args, 'lr_gamma', 0.95)   # 論文の 0.95

    #     # StepLR: lr_decay_start epoch 後から適用
    #     scheduler = optim.lr_scheduler.SequentialLR(
    #         opt,
    #         schedulers=[
    #             optim.lr_scheduler.ConstantLR(opt, factor=1.0,
    #                                           total_iters=lr_decay_start * epoch_steps),
    #             optim.lr_scheduler.ExponentialLR(opt, gamma=lr_gamma ** (1/epoch_steps)),
    #         ],
    #         milestones=[lr_decay_start * epoch_steps],
    #     )

    #     data_iter = iter(loader)
    #     step = 0

    #     while step < n_steps:
    #         try:
    #             # バッチ取得
    #             try:
    #                 batch = next(data_iter)
    #             except StopIteration:
    #                 data_iter = iter(loader)
    #                 batch = next(data_iter)

    #             # デバイス転送
    #             thr_t, thr_t1 = batch['thr_t'].to(self.dev), batch['thr_t1'].to(self.dev)
    #             T_rel, K = batch['T_rel'].to(self.dev), batch['K'].to(self.dev)
    #             valid = batch['valid']
    #             B, _, H, W = thr_t.shape

    #             # 特徴抽出 (XFeat: Frozen)
    #             kpts_list1, descs_list1, scores_list1 = self._extract_features(thr_t, 512)
    #             kpts_list2, descs_list2, scores_list2 = self._extract_features(thr_t1, 512)

    #             losses_match = []
    #             losses_epi = []

    #             for b in range(B):
    #                 kpts1, descs1, sc1 = kpts_list1[b], descs_list1[b], scores_list1[b]
    #                 kpts2, descs2, sc2 = kpts_list2[b], descs_list2[b], scores_list2[b]

    #                 if kpts1.shape[0] < 8 or kpts2.shape[0] < 8:
    #                     continue

    #                 inp0 = self._to_lg_input(kpts1, descs1, sc1, H, W)
    #                 inp1 = self._to_lg_input(kpts2, descs2, sc2, H, W)

    #                 # ── LightGlue フォワード ──
    #                 self.lightglue.train() 
    #                 # 明示的に training=True を渡すことで、log_assignment を強制
    #                 pred = self.lightglue({'image0': inp0, 'image1': inp1})

    #                 # 【診断ログ】最初の1回だけ、モデルが何を返したか全表示する
    #                 if step == 0 and b == 0:
    #                     print(f"\n[DIAGNOSTIC] Step 0 Keys: {list(pred.keys())}")
    #                     for k, v in pred.items():
    #                         if isinstance(v, (torch.Tensor, list)):
    #                             shape = v[0].shape if isinstance(v, list) and len(v) > 0 else (v.shape if isinstance(v, torch.Tensor) else "empty")
    #                             print(f"  - {k:18s} | {type(v).__name__:8s} | Shape: {shape}")

    #                 # ── 割当行列の取得（cvg/LightGlue API 対応版）──────────
    #                 # cvg/LightGlue の training=True 時の出力:
    #                 #   'log_assignment': (B, N1+1, N2+1) ← これが理想
    #                 # cvg/LightGlue の testing 時の出力:
    #                 #   'matching_scores0': (B, N1)  各KPの最大マッチスコア
    #                 #   'scores': list of scalar     ← shape(2,) で2D未満 → 使えない
    #                 # 対策: 常に 'log_assignment' を取得するよう LG を強制
    #                 # ── log_assignment の取得（Deep Supervision 対応）──────
    #                 # LightGlue 論文: 各層 ℓ の log_assignment を平均（Eq. 11）
    #                 # cvg/LightGlue の training=True 時:
    #                 #   list → [layer1(N1+1,N2+1), ..., layerL(N1+1,N2+1)]（Deep Sup）
    #                 #   tensor → (N1+1, N2+1) または (B, N1+1, N2+1)（最終層のみ）
    #                 scores_mat = None

    #                 if 'log_assignment' in pred:
    #                     res = pred['log_assignment']
    #                     if isinstance(res, (list, tuple)):
    #                         # Deep Supervision: リスト → そのまま渡す
    #                         scores_mat = res   # list of (N1+1, N2+1) tensors
    #                     elif isinstance(res, torch.Tensor):
    #                         # 最終層のみ: テンソル形式
    #                         scores_mat = res[0] if res.ndim == 3 else res

    #                 if scores_mat is None:
    #                     if step == 0:
    #                         print(f"[LG-FT] log_assignment なし. keys={list(pred.keys())}")
    #                         print("[LG-FT] → matching_scores0 からフォールバックを使用")
    #                     # フォールバック: matching_scores0 から擬似 log_assignment
    #                     if 'matching_scores0' in pred:
    #                         ms0 = pred['matching_scores0'][0]   # (N1,)
    #                         N1e, N2e = kpts1.shape[0], kpts2.shape[0]
    #                         sm = kpts1.new_full((N1e + 1, N2e + 1), -10.0)
    #                         n  = min(N1e, N2e, ms0.shape[0])
    #                         sm[torch.arange(n), torch.arange(n)] = ms0[:n].log().clamp(-10)
    #                         sm[:N1e, N2e] = (1.0 - ms0.clamp(1e-6, 1-1e-6)).log()
    #                         scores_mat = sm
    #                     else:
    #                         continue

    #                 # ── 損失計算（LightGlue 論文 Eq. 11 準拠）──────────────
    #                 # scores_mat: リスト（Deep Sup）または単一テンソル（最終層）
    #                 # 温度パラメータは log_assignment には不要（既に log-softmax 済み）
    #                 l_match = lightglue_matching_loss(
    #                     scores=scores_mat,
    #                     pts1=kpts1, pts2=kpts2, K=K[b], T_rel=T_rel[b],
    #                     use_gt=valid[b].item(),
    #                     inlier_thr=inlier_thr,
    #                     outlier_thr=5.0,   # 論文の 5px に準拠
    #                 )
    #                 losses_match.append(l_match)

    #                 # エピポーラ損失
    #                 if valid[b].item():
    #                     l_epi = lightglue_epi_loss(kpts1, kpts2, scores_mat, K[b], T_rel[b], epi_thr)
    #                     losses_epi.append(l_epi)

    #             # ── パラメータ更新 ──
    #             if not losses_match:
    #                 if step % 10 == 0:
    #                     print(f"[DEBUG] Step {step}: Skipping (losses_match empty). Check DIAGNOSTIC log.")
    #                 step += 1
    #                 continue

    #             l_match_mean = torch.stack(losses_match).mean()
    #             l_epi_mean = torch.stack(losses_epi).mean() if losses_epi else l_match_mean.new_zeros(1).squeeze()
    #             loss = l_match_mean + lambda_epi * l_epi_mean

    #             opt.zero_grad()
    #             loss.backward()
    #             torch.nn.utils.clip_grad_norm_(self.lightglue.parameters(), grad_clip)
    #             opt.step()
    #             scheduler.step()

    #             # ── 進捗表示 ──
    #             if step % 10 == 0:
    #                 param = next(self.lightglue.parameters())
    #                 s_min, s_max = scores_mat.min().item(), scores_mat.max().item()
    #                 print(f"[STEP {step:05d}] Loss: {loss.item():.4f} | Range: [{s_min:.2f}, {s_max:.2f}] | WeightSum: {param.data.sum().item():.6f}")

    #             # ログ
    #             if step % log_every == 0:
    #                 self._log({'lg_ft/loss_total': loss.item(), 'lg_ft/lr': opt.param_groups[0]['lr']}, step)

    #             if (step + 1) % save_every == 0:
    #                 self._save(f'step{step + 1}')

    #             step += 1

    #         except Exception as e:
    #             print(f"[ERROR] Step {step}: {e}")
    #             import traceback
    #             traceback.print_exc()
    #             step += 1
    #             if step > 50: break # 連続エラー回避

    #     self._save('final')
    #     print("[LG-FT] Fine-tuning done.")

    # def run(self, loader: DataLoader) -> None:
    #     print("\n" + "=" * 60)
    #     print("  LightGlue Fine-tuning (Final Debug & Run)")
    #     print("=" * 60)

    #     # ── パラメータ設定 ──
    #     lr = getattr(self.args, 'lr', 1e-4)
    #     n_steps = getattr(self.args, 'n_steps', 10000)
    #     inlier_thr = 5.0  # 精度のために引き締め
    #     grad_clip = getattr(self.args, 'grad_clip', 1.0)
    #     log_every = getattr(self.args, 'log_every', 100)
    #     save_every = getattr(self.args, 'save_ckpt_every', 2000)
    #     lambda_epi = getattr(self.args, 'lambda_epi', 0.1)
    #     epi_thr = getattr(self.args, 'epi_threshold', 2.0)

    #     # ── オプティマイザ設定 ──
    #     opt = optim.Adam(
    #         filter(lambda p: p.requires_grad, self.lightglue.parameters()),
    #         lr=lr,
    #     )

    #     # ── lr スケジューラ: 論文 Appendix C.4 準拠 ──
    #     # 「10 epoch 後から 0.95 で指数減衰させる」
    #     epoch_steps = getattr(self.args, 'epoch_steps', 2619)  # SThErEO 10476/4
    #     lr_decay_start = getattr(self.args, 'lr_decay_start_epoch', 10)
    #     lr_gamma = getattr(self.args, 'lr_gamma', 0.95)

    #     scheduler = optim.lr_scheduler.SequentialLR(
    #         opt,
    #         schedulers=[
    #             optim.lr_scheduler.ConstantLR(opt, factor=1.0,
    #                                          total_iters=lr_decay_start * epoch_steps),
    #             optim.lr_scheduler.ExponentialLR(opt, gamma=lr_gamma ** (1/epoch_steps)),
    #         ],
    #         milestones=[lr_decay_start * epoch_steps],
    #     )

    #     data_iter = iter(loader)
    #     step = 0

    #     while step < n_steps:
    #         try:
    #             # バッチ取得
    #             try:
    #                 batch = next(data_iter)
    #             except StopIteration:
    #                 data_iter = iter(loader)
    #                 batch = next(data_iter)

    #             # デバイス転送
    #             thr_t, thr_t1 = batch['thr_t'].to(self.dev), batch['thr_t1'].to(self.dev)
    #             T_rel, K = batch['T_rel'].to(self.dev), batch['K'].to(self.dev)
    #             valid = batch['valid']
    #             B, _, H, W = thr_t.shape

    #             # ── 特徴抽出 (XFeat: Frozen) ──
    #             kpts_list1, descs_list1, scores_list1 = self._extract_features(thr_t, 512)
    #             kpts_list2, descs_list2, scores_list2 = self._extract_features(thr_t1, 512)

    #             losses_match = []
    #             losses_epi = []

    #             for b in range(B):
    #                 kpts1, descs1, sc1 = kpts_list1[b], descs_list1[b], scores_list1[b]
    #                 kpts2, descs2, sc2 = kpts_list2[b], descs_list2[b], scores_list2[b]

    #                 if kpts1.shape[0] < 8 or kpts2.shape[0] < 8:
    #                     continue

    #                 inp0 = self._to_lg_input(kpts1, descs1, sc1, H, W)
    #                 inp1 = self._to_lg_input(kpts2, descs2, sc2, H, W)

    #                 # ── LightGlue フォワード ──
    #                 self.lightglue.train() 
    #                 # 明示的に training=True を渡すことで log_assignment を強制
    #                 pred = self.lightglue({'image0': inp0, 'image1': inp1})

    #                 # 【診断ログ】最初の1回だけ、モデルが何を返したか全表示する
    #                 if step == 0 and b == 0:
    #                     print(f"\n[DIAGNOSTIC] Step 0 Keys: {list(pred.keys())}")
    #                     for k, v in pred.items():
    #                         if isinstance(v, (torch.Tensor, list)):
    #                             shape = v[0].shape if isinstance(v, list) and len(v) > 0 else (v.shape if isinstance(v, torch.Tensor) else "empty")
    #                             print(f"  - {k:18s} | {type(v).__name__:8s} | Shape: {shape}")

    #                 # ── 割当行列の取得（Deep Supervision 対応）──
    #                 scores_mat = None

    #                 if 'log_assignment' in pred:
    #                     res = pred['log_assignment']
    #                     if isinstance(res, (list, tuple)):
    #                         # Deep Supervision: 全レイヤーのリストをそのまま渡す
    #                         scores_mat = res 
    #                     elif isinstance(res, torch.Tensor):
    #                         # 単一レイヤーの場合
    #                         scores_mat = res[0] if res.ndim == 3 else res

    #                 # ── フォールバック処理 ──
    #                 if scores_mat is None:
    #                     if step == 0:
    #                         print(f"[LG-FT] log_assignment なし. matching_scores0 から構築を試みます")
    #                     if 'matching_scores0' in pred:
    #                         ms0 = pred['matching_scores0'][0]
    #                         N1e, N2e = kpts1.shape[0], kpts2.shape[0]
    #                         # 確率 [0,1] を対数空間 [-10, 0] へ疑似変換
    #                         sm = kpts1.new_full((N1e + 1, N2e + 1), -10.0)
    #                         n  = min(N1e, N2e, ms0.shape[0])
    #                         sm[torch.arange(n), torch.arange(n)] = ms0[:n].log().clamp(-10)
    #                         sm[:N1e, N2e] = (1.0 - ms0.clamp(1e-6, 1-1e-6)).log()
    #                         scores_mat = sm
    #                     else:
    #                         continue

    #                 # ── 損失計算 ──
    #                 l_match = lightglue_matching_loss(
    #                     scores=scores_mat,
    #                     pts1=kpts1, pts2=kpts2, K=K[b], T_rel=T_rel[b],
    #                     use_gt=valid[b].item(),
    #                     inlier_thr=inlier_thr,
    #                     outlier_thr=5.0, # 論文準拠
    #                 )
    #                 losses_match.append(l_match)

    #                 # ── エピポーラ損失 ──
    #                 if valid[b].item():
    #                     l_epi = lightglue_epi_loss(kpts1, kpts2, scores_mat, K[b], T_rel[b], epi_thr)
    #                     losses_epi.append(l_epi)

    #             # ── パラメータ更新 ──
    #             if not losses_match:
    #                 step += 1
    #                 continue

    #             l_match_mean = torch.stack(losses_match).mean()
    #             l_epi_mean = torch.stack(losses_epi).mean() if losses_epi else l_match_mean.new_zeros(1).squeeze()
    #             loss = l_match_mean + lambda_epi * l_epi_mean

    #             opt.zero_grad()
    #             loss.backward()
    #             torch.nn.utils.clip_grad_norm_(self.lightglue.parameters(), grad_clip)
    #             opt.step()
    #             scheduler.step()

    #             # ── 進捗表示 ──
    #             if step % 10 == 0:
    #                 param = next(self.lightglue.parameters())
    #                 # scores_mat がリスト（Deep Sup）の場合は最終層を監視
    #                 s_ref = scores_mat[-1] if isinstance(scores_mat, list) else scores_mat
    #                 s_min, s_max = s_ref.min().item(), s_ref.max().item()
    #                 print(f"[STEP {step:05d}] Loss: {loss.item():.4f} | Range: [{s_min:.2f}, {s_max:.2f}] | WeightSum: {param.data.sum().item():.6f}")

    #             # ── ログ ──
    #             if step % log_every == 0:
    #                 self._log({'lg_ft/loss_total': loss.item(), 'lg_ft/lr': opt.param_groups[0]['lr']}, step)

    #             if (step + 1) % save_every == 0:
    #                 self._save(f'step{step + 1}')

    #             step += 1

    #         except Exception as e:
    #             print(f"[ERROR] Step {step}: {e}")
    #             import traceback
    #             traceback.print_exc()
    #             step += 1
    #             if step > 50: break 

    #     self._save('final')
    #     print("[LG-FT] Fine-tuning done.")
    def run(self, loader: DataLoader) -> None:
        print("\n" + "=" * 60)
        print("  LightGlue Fine-tuning (Final Debug & Run)")
        print("=" * 60)

        # ── パラメータ設定 ──
        lr = getattr(self.args, 'lr', 1e-4)
        n_steps = getattr(self.args, 'n_steps', 10000)
        inlier_thr = 5.0  # 精度のために引き締め
        grad_clip = getattr(self.args, 'grad_clip', 1.0)
        log_every = getattr(self.args, 'log_every', 100)
        save_every = getattr(self.args, 'save_ckpt_every', 2000)
        lambda_epi = getattr(self.args, 'lambda_epi', 0.1)
        epi_thr = getattr(self.args, 'epi_threshold', 2.0)

        # ── オプティマイザ設定 ──
        opt = optim.Adam(
            filter(lambda p: p.requires_grad, self.lightglue.parameters()),
            lr=lr,
        )

        # ── lr スケジューラ ──
        epoch_steps = getattr(self.args, 'epoch_steps', 2619)
        lr_decay_start = getattr(self.args, 'lr_decay_start_epoch', 10)
        lr_gamma = getattr(self.args, 'lr_gamma', 0.95)

        scheduler = optim.lr_scheduler.SequentialLR(
            opt,
            schedulers=[
                optim.lr_scheduler.ConstantLR(opt, factor=1.0,
                                             total_iters=lr_decay_start * epoch_steps),
                optim.lr_scheduler.ExponentialLR(opt, gamma=lr_gamma ** (1/epoch_steps)),
            ],
            milestones=[lr_decay_start * epoch_steps],
        )

        data_iter = iter(loader)
        step = 0

        while step < n_steps:
            try:
                # バッチ取得
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(loader)
                    batch = next(data_iter)

                # デバイス転送
                thr_t, thr_t1 = batch['thr_t'].to(self.dev), batch['thr_t1'].to(self.dev)
                T_rel, K = batch['T_rel'].to(self.dev), batch['K'].to(self.dev)
                valid = batch['valid']
                B, _, H, W = thr_t.shape

                # ── 特徴抽出 ──
                kpts_list1, descs_list1, scores_list1 = self._extract_features(thr_t, 512)
                kpts_list2, descs_list2, scores_list2 = self._extract_features(thr_t1, 512)

                losses_match = []
                losses_epi = []

                for b in range(B):
                    kpts1, descs1, sc1 = kpts_list1[b], descs_list1[b], scores_list1[b]
                    kpts2, descs2, sc2 = kpts_list2[b], descs_list2[b], scores_list2[b]

                    if kpts1.shape[0] < 8 or kpts2.shape[0] < 8:
                        continue

                    inp0 = self._to_lg_input(kpts1, descs1, sc1, H, W)
                    inp1 = self._to_lg_input(kpts2, descs2, sc2, H, W)

                    # ── LightGlue フォワード ──
                    self.lightglue.train() 
                    pred = self.lightglue({'image0': inp0, 'image1': inp1})

                    # 【診断ログ】
                    if step == 0 and b == 0:
                        print(f"\n[DIAGNOSTIC] Step 0 Keys: {list(pred.keys())}")

                    # ── 割当行列の取得と「対数空間への強制変換」 ──
                    scores_mat = None

                    if 'log_assignment' in pred:
                        res = pred['log_assignment']
                        
                        # ヘルパー: 確率 [0,1] を 対数 [-inf, 0] に変換する
                        def ensure_log_space(m):
                            if m.min() >= 0.0 and m.max() <= 1.05:
                                return torch.log(m.clamp(min=1e-8))
                            return m

                        if isinstance(res, (list, tuple)):
                            # Deep Supervision: 全レイヤーを log 化
                            scores_mat = [ensure_log_space(l[0] if l.ndim == 3 else l) for l in res]
                        elif isinstance(res, torch.Tensor):
                            # 単一レイヤーを log 化
                            scores_mat = ensure_log_space(res[0] if res.ndim == 3 else res)

                    # ── フォールバック ──
                    if scores_mat is None:
                        if 'matching_scores0' in pred:
                            ms0 = pred['matching_scores0'][0]
                            N1e, N2e = kpts1.shape[0], kpts2.shape[0]
                            sm = kpts1.new_full((N1e + 1, N2e + 1), -10.0)
                            n = min(N1e, N2e, ms0.shape[0])
                            sm[torch.arange(n), torch.arange(n)] = ms0[:n].clamp(min=1e-8).log()
                            sm[:N1e, N2e] = (1.0 - ms0.clamp(1e-6, 1-1e-6)).log()
                            scores_mat = sm
                        else:
                            continue

                    # ── 損失計算 ──
                    l_match = lightglue_matching_loss(
                        scores=scores_mat,
                        pts1=kpts1, pts2=kpts2, K=K[b], T_rel=T_rel[b],
                        use_gt=valid[b].item(),
                        inlier_thr=inlier_thr,
                        outlier_thr=5.0,
                    )
                    losses_match.append(l_match)

                    # ── エピポーラ損失 ──
                    if valid[b].item():
                        l_epi = lightglue_epi_loss(kpts1, kpts2, scores_mat, K[b], T_rel[b], epi_thr)
                        losses_epi.append(l_epi)

                # ── パラメータ更新 ──
                if not losses_match:
                    step += 1
                    continue

                l_match_mean = torch.stack(losses_match).mean()
                l_epi_mean = torch.stack(losses_epi).mean() if losses_epi else l_match_mean.new_zeros(1).squeeze()
                loss = l_match_mean + lambda_epi * l_epi_mean

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.lightglue.parameters(), grad_clip)
                opt.step()
                scheduler.step()

                # ── 進捗表示の修正 ──
                if step % 10 == 0:
                    param = next(self.lightglue.parameters())
                    s_ref = scores_mat[-1] if isinstance(scores_mat, list) else scores_mat
                    s_min, s_max = s_ref.min().item(), s_ref.max().item()
                    # これで Range が負の対数領域（例: [-18.42, -0.01]）になれば成功です
                    print(f"[STEP {step:05d}] Loss: {loss.item():.4f} | Range: [{s_min:.2f}, {s_max:.2f}] | WeightSum: {param.data.sum().item():.6f}")

                if step % log_every == 0:
                    self._log({'lg_ft/loss_total': loss.item(), 'lg_ft/lr': opt.param_groups[0]['lr']}, step)

                if (step + 1) % save_every == 0:
                    self._save(f'step{step + 1}')

                step += 1

            except Exception as e:
                print(f"[ERROR] Step {step}: {e}")
                import traceback
                traceback.print_exc()
                step += 1
                if step > 50: break 

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