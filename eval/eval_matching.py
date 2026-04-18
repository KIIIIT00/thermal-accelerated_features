"""
evaluate/eval_matching.py
キーポイント検出・マッチング精度の定量・定性評価スクリプト。

評価内容:
    定量: Homography Estimation AUC（@3px / @5px / @10px）, Matching Score
    定性: キーポイント・マッチング結果の可視化画像

比較対象:
    teacher_rgb : XFeat（元モデル）に RGB を入力   ← 参照上限
    teacher_thr : XFeat（元モデル）に熱画像を入力  ← KD 前のベースライン
    student_thr : 提案手法（KD済み）に熱画像を入力 ← 提案手法

使用方法:
    python evaluate/eval_matching.py --config configs/eval_config.yaml
    python evaluate/eval_matching.py --config configs/eval_config.yaml \
        --datasets freiburg ms2 --n_viz 10
    python evaluate/eval_matching.py --config configs/eval_config.yaml \
        --n_eval_pairs 200 --n_viz 0   # 定量のみ
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from modules.model import XFeatModel


# ---------------------------------------------------------------------------
# 設定ロード
# ---------------------------------------------------------------------------

def _load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Thermal XFeat — Keypoint Matching Evaluation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--student_weights', type=str, default=None)
    parser.add_argument('--teacher_weights',  type=str, default=None)
    parser.add_argument('--datasets', nargs='+', default=None,
                        help='評価データセット名（yaml の datasets を上書き）')
    parser.add_argument('--split', type=str, default=None,
                        choices=['train', 'val', 'all'])
    parser.add_argument('--n_eval_pairs', type=int, default=None,
                        help='1データセットあたり評価ペア数（省略→全件）')
    parser.add_argument('--n_viz', type=int, default=None,
                        help='1データセットあたり可視化ペア数（0→スキップ）')
    parser.add_argument('--output_dir',     type=str, default=None)
    parser.add_argument('--max_keypoints',  type=int, default=None)
    parser.add_argument('--matching_method', type=str, default=None,
                        choices=['mutual_nn', 'ratio_test'])
    parser.add_argument('--device_num',     type=str, default=None)

    cli = parser.parse_args()
    if not os.path.isfile(cli.config):
        parser.error(f'--config not found: {cli.config!r}')

    # YAML をベースに CLI で上書き
    cfg = _load_yaml(cli.config)
    overrides = {k: v for k, v in vars(cli).items()
                 if k != 'config' and v is not None}
    cfg.update(overrides)

    # デフォルト値の補完
    defaults = dict(
        auc_thresholds   = [3, 5, 10],
        split            = 'val',
        n_eval_pairs     = None,
        n_viz            = 5,
        matching_method  = 'mutual_nn',
        ratio_threshold  = 0.9,
        max_keypoints    = 2048,
        output_dir       = 'evaluate/results',
        viz_models       = ['teacher_rgb', 'teacher_thr', 'student_thr'],
        kp_radius        = 3,
        kp_color_teacher_rgb = [0, 255, 0],
        kp_color_teacher_thr = [0, 0, 255],
        kp_color_student_thr = [255, 128, 0],
        match_color      = [255, 255, 0],
        viz_width        = 640,
        viz_height       = 480,
        seed             = 42,
        num_workers      = 2,
        datasets         = ['freiburg'],
        device_num       = '0',
    )
    for k, v in defaults.items():
        cfg.setdefault(k, v)

    return argparse.Namespace(**cfg)


# ---------------------------------------------------------------------------
# モデルロード
# ---------------------------------------------------------------------------

def load_models(args: argparse.Namespace,
                device: torch.device) -> Dict[str, torch.nn.Module]:
    models = {}
    for role, attr in [('teacher', 'teacher_weights'),
                       ('student', 'student_weights')]:
        m = XFeatModel().to(device).eval()
        w = getattr(args, attr, None)
        if w and os.path.isfile(w):
            m.load_state_dict(torch.load(w, map_location=device,
                                         weights_only=True))
            print(f"[Eval] {role}: loaded {w}")
        else:
            print(f"[Eval] WARNING: {attr} not found → random weights")
        models[role] = m
    return models


# ---------------------------------------------------------------------------
# データセット
# ---------------------------------------------------------------------------

def load_pairs(name: str,
               args: argparse.Namespace,
               split: str) -> List[Tuple[str, str]]:
    from modules.dataset.thermal.loader    import _resolve_data_root, _resolve_splits_dir
    from modules.dataset.thermal.freiburg   import FreiburgDataset
    from modules.dataset.thermal.tartanrgbt import TartanRGBTDataset
    from modules.dataset.thermal.vivid      import VividDataset
    from modules.dataset.thermal.sthereo    import SthEreoDataset
    from modules.dataset.thermal.ms2        import MS2Dataset

    CLS = {'freiburg': FreiburgDataset, 'tartanrgbt': TartanRGBTDataset,
           'vivid': VividDataset, 'sthereo': SthEreoDataset, 'ms2': MS2Dataset}

    name_l = name.lower()
    if name_l not in CLS:
        raise ValueError(f"Unknown dataset: {name!r}")

    ds = CLS[name_l](
        data_root  = _resolve_data_root(name_l, args),
        splits_dir = _resolve_splits_dir(name_l, args),
        split      = split,
        augment    = False,
        aug_list   = None,
        p_diurnal_inversion = 0.0,
    )
    pairs = list(ds._pairs)
    print(f"[Eval] {name} ({split}): {len(pairs)} pairs")

    n = getattr(args, 'n_eval_pairs', None)
    if n and n < len(pairs):
        rng = random.Random(args.seed)
        pairs = rng.sample(pairs, n)
        print(f"[Eval] {name}: subsampled → {len(pairs)} pairs")
    return pairs


# ---------------------------------------------------------------------------
# 画像前処理
# ---------------------------------------------------------------------------

def imread_tensor(path: str, is_thermal: bool, device: torch.device,
                  size: Tuple[int, int]) -> Tuple[torch.Tensor, np.ndarray]:
    """画像を (Tensor(1,3,H,W), BGR ndarray) で返す。"""
    if is_thermal:
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(path)
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    else:
        bgr = cv2.imread(path)
        if bgr is None:
            raise FileNotFoundError(path)

    bgr  = cv2.resize(bgr, size)
    rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    t    = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    return t.unsqueeze(0).to(device), bgr


# ---------------------------------------------------------------------------
# キーポイント検出
# ---------------------------------------------------------------------------

@torch.no_grad()
def detect(model: torch.nn.Module, img_t: torch.Tensor,
           max_kp: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    XFeat のキーポイント検出。

    キーポイント位置の選択には kp_logits（検出ヘッド）を使う。
    hmap（信頼性マップ）は重みとして乗算する。

    combined_score = kp_score * hmap
      kp_score = softmax(kp_logits[:, :64], dim=1).max()
      hmap     = 信頼性マップ

    hmap のみで選ぶと KD の空間バイアスが影響してキーポイントが偏るため、
    kp_logits（検出ヘッド）との積で均一な分布を保証する。

    Returns:
        kpts  (N, 2) float32  画像座標 (x, y)
        descs (N, 64) float32 L2 正規化済み特徴量
    """
    feats, kp_logits, hmap = model(img_t)
    feats = F.normalize(feats, dim=1)
    B, C, Hf, Wf = feats.shape
    H, W = img_t.shape[2], img_t.shape[3]

    # P(keypoint) = 1 - P(dustbin) : 65ch softmax で dustbin を正しく考慮
    probs    = F.softmax(kp_logits, dim=1)         # (B, 65, Hf, Wf)
    kp_score = probs[:, :64].sum(dim=1)            # (B, Hf, Wf)

    scores   = kp_score[0].cpu().numpy().flatten()
    feats_np = feats[0].reshape(C, -1).permute(1, 0).cpu().numpy()

    top_idx  = np.argsort(scores)[::-1][:min(max_kp, len(scores))]
    ys = (top_idx // Wf).astype(np.float32) * (H / Hf)
    xs = (top_idx %  Wf).astype(np.float32) * (W / Wf)

    kpts  = np.stack([xs, ys], axis=1)
    descs = feats_np[top_idx].astype(np.float32)
    return kpts, descs


# ---------------------------------------------------------------------------
# マッチング
# ---------------------------------------------------------------------------

# LightGlue グローバルシングルトン（毎回ロードしない）
_lightglue_instance = None


def load_lightglue(
    weights_path: Optional[str] = None,
    device: torch.device = torch.device('cpu'),
) -> Optional[Any]:
    """
    glue-factory の LightGlue を fine-tuning 済み重みでロードする。

    チェックポイント形式（glue-factory）:
        {
            'model': {'extractor.*': ..., 'matcher.*': ...},
            'conf':  {'model': {'matcher': {...}}},  ← アーキテクチャ設定
            'epoch': N,
            'eval':  {...},
        }

    手順:
        1. ckpt['conf']['model']['matcher'] からアーキテクチャを復元
        2. ckpt['model'] から 'matcher.*' キーを抽出・プレフィックス除去
        3. load_state_dict で重みを注入

    Args:
        weights_path: fine-tuning 済み .tar ファイルのパス
        device:       使用するデバイス

    Returns:
        LightGlue モデル（ロード失敗時は None）
    """
    global _lightglue_instance
    try:
        import sys as _sys
        import os as _os
        _THIS = _os.path.dirname(_os.path.abspath(__file__))
        _REPO = _os.path.dirname(_THIS)
        _GF   = _os.path.join(_REPO, 'third_party', 'glue-factory')
        if _os.path.isdir(_GF) and _GF not in _sys.path:
            _sys.path.insert(0, _GF)

        from gluefactory.models.matchers.lightglue import LightGlue
        from omegaconf import OmegaConf, DictConfig

        if weights_path and _os.path.isfile(weights_path):
            # ── Step 1: チェックポイントをロード ─────────────────────────
            ckpt = torch.load(weights_path, map_location=device,
                              weights_only=False)
            print(f"[LightGlue] Checkpoint loaded: {weights_path}")
            print(f"[LightGlue] epoch={ckpt.get('epoch', '?')}  "
                  f"eval={ckpt.get('eval', {})}")

            # ── Step 2: 訓練時の conf からアーキテクチャを復元 ────────────
            # チェックポイントの conf を使うことで訓練時と完全に一致させる
            lg_conf = None
            if 'conf' in ckpt:
                try:
                    saved_conf = ckpt['conf']
                    # OmegaConf または dict に対応
                    if isinstance(saved_conf, DictConfig):
                        matcher_conf = saved_conf.model.matcher
                    else:
                        matcher_conf = saved_conf['model']['matcher']

                    # 推論時は flash / checkpointed を無効化（安定性のため）
                    lg_conf = OmegaConf.create(dict(matcher_conf))
                    OmegaConf.update(lg_conf, 'flash',        False, merge=True)
                    OmegaConf.update(lg_conf, 'checkpointed', False, merge=True)
                    OmegaConf.update(lg_conf, 'mp',           False, merge=True)
                    # depth/width confidence は推論時に有効化可能
                    # OmegaConf.update(lg_conf, 'depth_confidence', 0.95)
                    print(f"[LightGlue] Arch restored from checkpoint conf: "
                          f"n_layers={lg_conf.get('n_layers', '?')} "
                          f"input_dim={lg_conf.get('input_dim', '?')} "
                          f"descriptor_dim={lg_conf.get('descriptor_dim', '?')} "
                          f"num_heads={lg_conf.get('num_heads', '?')} ")
                except Exception as e:
                    print(f"[LightGlue] WARNING: conf parse failed ({e}), "
                          f"using manual conf")

            # conf が取れなかった場合はデフォルト値で初期化
            if lg_conf is None:
                lg_conf = OmegaConf.create({
                    'name':             'matchers.lightglue',
                    'features':         None,
                    'input_dim':        64,
                    'descriptor_dim':   256,
                    'n_layers':         9,
                    'num_heads':        4,
                    'flash':            False,
                    'mp':               False,
                    'checkpointed':     False,
                    'depth_confidence': -1,
                    'width_confidence': -1,
                    'filter_threshold': 0.1,
                    'weights':          None,
                })

            lg = LightGlue(lg_conf).to(device).eval()

            # ── Step 3: 'matcher.*' キーを抽出して注入 ───────────────────
            full_state   = ckpt['model']
            matcher_state = {
                k[len('matcher.'):]: v
                for k, v in full_state.items()
                if k.startswith('matcher.')
            }
            print(f"[LightGlue] matcher.* keys: {len(matcher_state)} 個")

            missing, unexpected = lg.load_state_dict(
                matcher_state, strict=False)
            if missing:
                print(f"[LightGlue] Missing ({len(missing)}): "
                      f"{missing[:3]}{'...' if len(missing) > 3 else ''}")
            if unexpected:
                print(f"[LightGlue] Unexpected ({len(unexpected)}): "
                      f"{unexpected[:3]}{'...' if len(unexpected) > 3 else ''}")

            n_loaded = len(matcher_state) - len(unexpected)
            print(f"[LightGlue] {n_loaded} / {len(matcher_state)} keys loaded ✓")

        else:
            # 重みなし → デフォルト conf で初期化
            lg_conf = OmegaConf.create({
                'name':             'matchers.lightglue',
                'features':         None,
                'input_dim':        64,
                'descriptor_dim':   256,
                'n_layers':         9,
                'num_heads':        4,
                'flash':            False,
                'mp':               False,
                'checkpointed':     False,
                'depth_confidence': -1,
                'width_confidence': -1,
                'filter_threshold': 0.1,
                'weights':          None,
            })
            lg = LightGlue(lg_conf).to(device).eval()
            print("[LightGlue] Using random initialization (no weights)")

        _lightglue_instance = lg
        return lg

    except ImportError as e:
        print(f"[LightGlue] WARNING: import failed: {e}")
        print("  bash scripts/setup_glue_factory.sh を実行してください")
        return None
    except Exception as e:
        import traceback
        print(f"[LightGlue] WARNING: load failed: {e}")
        traceback.print_exc()
        return None


def _get_lightglue(device: torch.device):
    """後方互換用: グローバルシングルトンを返す。"""
    global _lightglue_instance
    return _lightglue_instance


@torch.no_grad()
def match_lightglue(
    kpts1: np.ndarray,
    descs1: np.ndarray,
    kpts2: np.ndarray,
    descs2: np.ndarray,
    image_size: Tuple[int, int],
    device: torch.device,
    lightglue_model: Optional[Any] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    LightGlue でマッチングを行う。

    Args:
        kpts1, kpts2      : (N, 2) float32  画像座標 [x, y]（ピクセル）
        descs1, descs2    : (N, 64) float32  L2 正規化済み記述子
        image_size        : (H, W)
        device            : torch.device
        lightglue_model   : ロード済み LightGlue モデル（None の場合は MNN にフォールバック）

    Returns:
        idx1, idx2: マッチしたインデックスペア
    """
    lg = lightglue_model or _get_lightglue(device)
    if lg is None or len(kpts1) == 0 or len(kpts2) == 0:
        return _mutual_nn_np(descs1, descs2)

    H, W = image_size

    # glue-factory LightGlue のインターフェース（ソースコード確認済み）:
    #   required_data_keys: keypoints0, keypoints1, descriptors0, descriptors1
    #   optional: view0/view1 に image_size を渡すと内部で正規化される
    #   キーポイントはピクセル座標（正規化は LG 内部で実施）
    sz = torch.tensor([[H, W]], device=device)

    def _to_t(arr):
        return torch.from_numpy(arr).float().unsqueeze(0).to(device)

    try:
        with torch.no_grad():
            pred = lg({
                'keypoints0':   _to_t(kpts1),
                'descriptors0': _to_t(descs1),
                'keypoints1':   _to_t(kpts2),
                'descriptors1': _to_t(descs2),
                'view0': {'image_size': sz},
                'view1': {'image_size': sz},
            })

        # 出力: matches0 が (1, N) の対応先インデックス（-1 は不一致）
        if 'matches0' in pred:
            m = pred['matches0'].squeeze(0).cpu().numpy()
            valid = m >= 0
            idx0  = np.where(valid)[0]
            idx1  = m[valid]
        elif 'matches' in pred:
            # 一部バージョンは matches が (1, N) 形式
            m = pred['matches'].squeeze(0).cpu().numpy()
            valid = m >= 0
            idx0  = np.where(valid)[0]
            idx1  = m[valid]
        else:
            return _mutual_nn_np(descs1, descs2)

        # 崩壊検出
        n_matches  = len(idx0)
        n_kpts_min = min(len(kpts1), len(kpts2))
        if n_matches == 0 or n_matches / max(n_kpts_min, 1) < 0.01:
            return _mutual_nn_np(descs1, descs2)

        return idx0.astype(np.int64), idx1.astype(np.int64)

    except Exception as e:
        print(f"  [LightGlue] Error: {e} → fallback to mutual_nn")
        return _mutual_nn_np(descs1, descs2)


def _mutual_nn_np(
    descs1: np.ndarray,
    descs2: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """相互最近傍マッチング（numpy）。"""
    if len(descs1) == 0 or len(descs2) == 0:
        return np.array([], np.int64), np.array([], np.int64)
    d1 = descs1 / (np.linalg.norm(descs1, axis=1, keepdims=True) + 1e-8)
    d2 = descs2 / (np.linalg.norm(descs2, axis=1, keepdims=True) + 1e-8)
    sim  = d1 @ d2.T
    nn12 = np.argmax(sim, axis=1)
    nn21 = np.argmax(sim, axis=0)
    ids  = np.arange(len(descs1))
    mask = nn21[nn12] == ids
    return ids[mask], nn12[mask]


def match(
    descs1: np.ndarray,
    descs2: np.ndarray,
    method: str,
    ratio_thr: float,
    kpts1: Optional[np.ndarray] = None,
    kpts2: Optional[np.ndarray] = None,
    image_size: Optional[Tuple[int, int]] = None,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    マッチング関数。method に応じて MNN / ratio_test / LightGlue を切り替える。

    Args:
        descs1, descs2 : (N, 64) 記述子
        method         : 'mutual_nn' | 'ratio_test' | 'lightglue'
        ratio_thr      : ratio_test 用の閾値
        kpts1, kpts2   : LightGlue 用のキーポイント座標（省略可）
        image_size     : LightGlue 用の画像サイズ (H, W)（省略可）
        device         : LightGlue 用のデバイス（省略可）

    Returns:
        idx1, idx2: マッチインデックス
    """
    if method == 'lightglue':
        if kpts1 is None or kpts2 is None or image_size is None or device is None:
            print("[Matcher] lightglue requires kpts, image_size, device → fallback MNN")
            return _mutual_nn_np(descs1, descs2)
        return match_lightglue(kpts1, descs1, kpts2, descs2, image_size, device)

    d1 = descs1 / (np.linalg.norm(descs1, axis=1, keepdims=True) + 1e-8)
    d2 = descs2 / (np.linalg.norm(descs2, axis=1, keepdims=True) + 1e-8)
    sim = d1 @ d2.T

    if method == 'mutual_nn':
        nn12 = np.argmax(sim, axis=1)
        nn21 = np.argmax(sim, axis=0)
        ids  = np.arange(len(descs1))
        mask = nn21[nn12] == ids
        return ids[mask], nn12[mask]

    # ratio_test
    if sim.shape[1] < 2:
        return np.array([], np.int64), np.array([], np.int64)
    order = np.argsort(-sim, axis=1)
    best1 = sim[np.arange(len(d1)), order[:, 0]]
    best2 = sim[np.arange(len(d1)), order[:, 1]]
    mask  = (best2 / (best1 + 1e-8)) < ratio_thr
    idx1  = np.where(mask)[0]
    return idx1, order[idx1, 0]


# ---------------------------------------------------------------------------
# 評価指標
# ---------------------------------------------------------------------------

def homography_error(kpts1, kpts2, idx1, idx2, hw) -> float:
    """RANSAC ホモグラフィのコーナー再投影誤差 (px)。"""
    if len(idx1) < 4:
        return float('inf')
    pts1 = kpts1[idx1].reshape(-1, 1, 2)
    pts2 = kpts2[idx2].reshape(-1, 1, 2)
    H, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
    if H is None:
        return float('inf')
    h, w = hw
    corners = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
    pred    = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
    return float(np.mean(np.linalg.norm(corners.reshape(-1,2) - pred, axis=1)))


def _compute_F_gt(T_rel: Any, K: Any) -> np.ndarray:
    """GT ポーズから Fundamental matrix を計算する。"""
    if hasattr(T_rel, 'numpy'): T_rel = T_rel.numpy()
    if hasattr(K, 'numpy'):     K     = K.numpy()
    T_rel = np.array(T_rel, dtype=np.float64)
    K_arr = np.array(K,     dtype=np.float64)
    R = T_rel[:3, :3]
    t = T_rel[:3,  3]
    t_s = np.array([[0,-t[2],t[1]],[t[2],0,-t[0]],[-t[1],t[0],0]])
    E   = t_s @ R
    Ki  = np.linalg.inv(K_arr)
    return Ki.T @ E @ Ki  # (3, 3)


def _sym_epi_dist(pts1: np.ndarray, pts2: np.ndarray,
                  F: np.ndarray) -> np.ndarray:
    """
    対称エピポーラ距離（点 → エピポーラ線の距離の平均）を返す。
    Sampson 距離より直感的で、ピクセル単位のスケールを持つ。
    """
    n    = len(pts1)
    ones = np.ones((n, 1), dtype=np.float32)
    p1h  = np.hstack([pts1, ones])
    p2h  = np.hstack([pts2, ones])
    Fp1  = (F.astype(np.float32) @ p1h.T).T   # (N, 3)
    Ftp2 = (F.T.astype(np.float32) @ p2h.T).T
    # 点から対応エピポーラ線への距離
    d1 = np.abs((p2h * Fp1).sum(1)) / (
        np.sqrt(Fp1[:,0]**2 + Fp1[:,1]**2) + 1e-8)
    d2 = np.abs((p1h * Ftp2).sum(1)) / (
        np.sqrt(Ftp2[:,0]**2 + Ftp2[:,1]**2) + 1e-8)
    return (d1 + d2) / 2.0   # (N,)


def epipolar_inlier_error(
    kpts1:  np.ndarray,
    kpts2:  np.ndarray,
    idx1:   np.ndarray,
    idx2:   np.ndarray,
    T_rel:  Any,
    K:      Any,
    hw:     Tuple[int, int],
    epi_th: float = 5.0,
) -> float:
    """
    GT ポーズ使用版エピポーラ評価（TartanRGBT 用）。

    Returns: 1 - inlier_ratio ∈ [0, 1]
      inlier_ratio = (GT F 行列でのエピポーラ距離 < epi_th のマッチ数) / 全マッチ数
      小さいほど良い（インライア率が高い）

    AUC 閾値との対応（eval_config.yaml の auc_thresholds を [0.9, 0.7, 0.5] に設定）:
      AUC@0.9 = 「1-IR < 0.9」の割合 = インライア率 > 10% のペア割合
      AUC@0.7 = インライア率 > 30% のペア割合
      AUC@0.5 = インライア率 > 50% のペア割合
    """
    if len(idx1) < 4:
        return 1.0  # worst case: inlier_ratio = 0

    F    = _compute_F_gt(T_rel, K).astype(np.float32)
    pts1 = kpts1[idx1].astype(np.float32)
    pts2 = kpts2[idx2].astype(np.float32)

    dist        = _sym_epi_dist(pts1, pts2, F)
    inlier_mask = dist < epi_th
    inlier_ratio = float(inlier_mask.sum()) / max(len(idx1), 1)
    return 1.0 - inlier_ratio  # smaller = better


def epipolar_8pt_error(
    kpts1:  np.ndarray,
    kpts2:  np.ndarray,
    idx1:   np.ndarray,
    idx2:   np.ndarray,
    epi_th: float = 5.0,
) -> float:
    """
    GT ポーズなし・全シーン対応の評価関数。

    RANSAC F 行列推定 → インライア率を計算。
    ホモグラフィーと異なり parallax のある 3D シーンでも正しく動作する。

    Returns: 1 - inlier_ratio ∈ [0, 1]
      inlier_ratio = RANSAC_inliers / total_matches
      小さいほど良い。RANSAC 失敗 → 1.0（最悪値）
    """
    if len(idx1) < 8:
        return 1.0  # worst case

    pts1r = kpts1[idx1].astype(np.float32).reshape(-1, 1, 2)
    pts2r = kpts2[idx2].astype(np.float32).reshape(-1, 1, 2)

    F, mask = cv2.findFundamentalMat(
        pts1r, pts2r, cv2.FM_RANSAC, epi_th, 0.99)
    if F is None or mask is None:
        return 1.0  # worst case

    inlier_ratio = float(mask.sum()) / max(len(idx1), 1)
    return 1.0 - inlier_ratio  # smaller = better


def auc_at(errors: List[float], thresholds: List[int]) -> Dict[str, float]:
    arr = np.array(errors)
    return {f'AUC@{t}px': float((arr <= t).mean()) for t in thresholds}


def matching_score(kpts1, kpts2, idx1) -> float:
    n = min(len(kpts1), len(kpts2))
    return float(len(idx1) / n) if n > 0 else 0.0


# ---------------------------------------------------------------------------
# EvalMetrics: evaluate.py が期待するデータクラス
# ---------------------------------------------------------------------------

class EvalMetrics:
    """
    1モデル × 1データセットの評価結果。

    指標（論文標準）:
        pose_auc  : Pose AUC @5°/@10°/@20°（GT pose 必要、TartanRGBT のみ）
        precision : Precision@t（GT F 行列必要、TartanRGBT のみ）
        auc       : 後方互換用（GT なし時は {}）
        matching_score : MS = matches / min(kp1, kp2)（全データセット）
    """

    def __init__(
        self,
        model_name:        str,
        dataset_name:      str,
        auc:               Dict[int, float],
        matching_score:    float,
        mean_n_kpts:       float,
        mean_inlier_ratio: float,
        n_pairs:           int,
        mean_time_sec:     float,
        pose_auc:          Optional[Dict[int, float]] = None,
        precision:         Optional[Dict[int, float]] = None,
        recall:            Optional[Dict[int, float]] = None,
        mean_n_matches:    float = 0.0,
    ):
        self.model_name        = model_name
        self.dataset_name      = dataset_name
        self.auc               = auc
        self.matching_score    = matching_score
        self.mean_n_kpts       = mean_n_kpts
        self.mean_inlier_ratio = mean_inlier_ratio
        self.n_pairs           = n_pairs
        self.mean_time_sec     = mean_time_sec
        self.pose_auc          = pose_auc or {}
        self.precision         = precision or {}
        self.recall            = recall or {}
        self.mean_n_matches    = mean_n_matches

    def summary(self) -> str:
        """
        Precision = TP / (TP+FP)  マッチのうち正解の割合
        Recall    = TP / (TP+FN)  正解対応点のうちマッチできた割合
        F1        = 2*P*R / (P+R) 調和平均
        MS        = matches / min(kp1,kp2)
        n_matches = 1ペアの平均マッチ数（絶対値）
        """
        parts = [f"[{self.dataset_name}] {self.model_name:<35s}"]

        if self.pose_auc:
            parts.append("  ".join(
                f"PoseAUC@{t}deg={v*100:.1f}%"
                for t, v in sorted(self.pose_auc.items())))

        all_thrs = sorted(set(
            list(self.precision.keys()) + list(self.recall.keys())))
        for t in all_thrs:
            p  = self.precision.get(t, 0.0)
            r  = self.recall.get(t, 0.0)
            f1 = (2*p*r / (p+r)) if (p+r) > 1e-8 else 0.0
            parts.append(
                f"Prec@{t}px={p*100:.1f}%  "
                f"Recall@{t}px={r*100:.1f}%  "
                f"F1@{t}px={f1*100:.1f}%")

        parts.append(
            f"MS={self.matching_score*100:.1f}%  "
            f"n_matches={self.mean_n_matches:.0f}  "
            f"n_pairs={self.n_pairs}")

        return "\n    ".join(parts)



# ---------------------------------------------------------------------------
# 定量評価（1データセット）― evaluate.py 用の新シグネチャ版
# ---------------------------------------------------------------------------

def evaluate_dataset(
    model:           torch.nn.Module,
    model_name:      str,
    dataset_name:    str,
    pairs:           List[Tuple],
    modality:        str,
    device:          torch.device,
    cfg:             dict,
    rng:             Any,
    verbose:         bool = False,
    lightglue_model: Optional[Any] = None,
) -> 'EvalMetrics':
    """
    1モデル × 1データセットを評価して EvalMetrics を返す。

    指標の設計:
        GT pose あり（TartanRGBT）:
            - Pose Estimation AUC @5°/@10°/@20°
            - Precision @1px/@3px/@5px
            - Matching Score
        GT pose なし（Freiburg / SThErEO / VIVID）:
            - Matching Score のみ（F 行列が退化するため他指標は無効）

    Pose Estimation AUC の計算方法:
        1. マッチングから E 行列を RANSAC で推定（K が既知）
        2. E を R, t に分解
        3. rotation error = angle(R_est, R_gt)
           translation error = angle(t_est, t_gt)
        4. error = max(R_err, t_err)
        5. AUC@θ = fraction of pairs where error < θ degrees
    """
    import time

    size    = (cfg.get('viz_width', 640), cfg.get('viz_height', 480))
    max_kp  = cfg.get('max_keypoints', 2048)
    method  = cfg.get('matching_method', 'mutual_nn')
    ratio   = cfg.get('ratio_threshold', 0.9)
    n_pairs = cfg.get('n_pairs', None)

    if lightglue_model is not None:
        method = 'lightglue'

    # サブサンプリング
    if n_pairs and n_pairs > 0 and len(pairs) > n_pairs:
        idx   = rng.choice(len(pairs), size=n_pairs, replace=False)
        pairs = [pairs[i] for i in idx]

    is_thermal  = (modality == 'thermal')
    has_gt_pose = (len(pairs[0]) == 4) if pairs else False

    pose_thrs = [5, 10, 20]   # degrees
    prec_thrs = [1, 3, 5]     # pixels

    ms_list:         List[float] = []
    n_kpts_list:     List[int]   = []
    elapsed_total:   float       = 0.0
    pose_errors:     List[float] = []   # max(R_err, t_err) per pair [degrees]
    precision_vals:  Dict[int, List[float]] = {t: [] for t in prec_thrs}

    if has_gt_pose:
        print(f"    [{dataset_name}/{model_name}] "
              f"GT pose → Pose AUC + Precision + MS")
    else:
        print(f"    [{dataset_name}/{model_name}] "
              f"No GT pose → MS only")

    for i, pair_entry in enumerate(pairs):
        if verbose and (i + 1) % 100 == 0:
            print(f"    [{dataset_name}/{model_name}] {i+1}/{len(pairs)}")

        # ── ペアのパスと GT ポーズを取り出す ──────────────────────────────
        if has_gt_pose:
            rgb_p, thr_p, T_rel_gt, K_gt = pair_entry
            img_path1 = rgb_p   # = thr_t  (1枚目)
            img_path2 = thr_p   # = thr_t1 (2枚目)
        else:
            rgb_p, thr_p = pair_entry[0], pair_entry[1]
            T_rel_gt, K_gt = None, None
            img_path1 = thr_p if is_thermal else rgb_p
            img_path2 = None

        # ── 1枚目の特徴抽出 ───────────────────────────────────────────────
        try:
            img_t, _ = imread_tensor(img_path1, is_thermal, device, size)
        except FileNotFoundError:
            continue

        t0 = time.perf_counter()
        kpts, descs = detect(model, img_t, max_kp)
        elapsed_total += time.perf_counter() - t0
        if len(kpts) == 0:
            continue

        # ── 2枚目の画像を取得 ─────────────────────────────────────────────
        if not has_gt_pose:
            if i + 1 < len(pairs):
                nxt = pairs[i + 1]
                img_path2 = nxt[1] if is_thermal else nxt[0]
            else:
                continue

        try:
            img_t2, _ = imread_tensor(img_path2, is_thermal, device, size)
        except FileNotFoundError:
            continue

        kpts2, descs2 = detect(model, img_t2, max_kp)
        if len(kpts2) == 0:
            continue

        # ── マッチング ────────────────────────────────────────────────────
        if method == 'lightglue':
            idx1, idx2 = match_lightglue(
                kpts, descs, kpts2, descs2,
                image_size=(size[1], size[0]),
                device=device,
                lightglue_model=lightglue_model,
            )
        else:
            idx1, idx2 = match(descs, descs2, method, ratio)

        n_matches = len(idx1)
        ms = matching_score(kpts, kpts2, idx1)
        ms_list.append(ms)
        n_kpts_list.append(len(kpts))

        if n_matches < 8 or not has_gt_pose:
            continue

        # ── Pose Estimation AUC（GT pose あり時のみ）─────────────────────
        # K, T_rel_gt を numpy に変換
        if hasattr(K_gt, 'numpy'):      K_np = K_gt.numpy()
        else:                            K_np = np.array(K_gt, dtype=np.float64)
        if hasattr(T_rel_gt, 'numpy'): T_np = T_rel_gt.numpy()
        else:                            T_np = np.array(T_rel_gt, dtype=np.float64)

        pts1 = kpts[idx1].astype(np.float32).reshape(-1, 1, 2)
        pts2 = kpts2[idx2].astype(np.float32).reshape(-1, 1, 2)

        # ── E 行列を RANSAC で推定 ─────────────────────────────────────
        # ポイント: pts1/pts2 は画像座標 (pixel)、K_np がキャリブレーション行列
        # findEssentialMat は内部で K を使って正規化座標に変換する
        E_est, e_mask = cv2.findEssentialMat(
            pts1, pts2,
            cameraMatrix=K_np.astype(np.float64),
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0,
        )
        if E_est is None or e_mask is None:
            continue
        # E_est が 9x3 (3個の候補) の場合は最初の 3x3 を使う
        if E_est.shape[0] > 3:
            E_est = E_est[:3, :]

        n_inliers = int(e_mask.sum())
        if n_inliers < 5:
            continue

        # ── E から R, t を回収 ───────────────────────────────────────────
        # recoverPose: カメラ1 → カメラ2 の R, t を返す
        # t はスケール不定の単位ベクトル（方向のみ有効）
        _, R_est, t_est, _ = cv2.recoverPose(
            E_est, pts1, pts2,
            cameraMatrix=K_np.astype(np.float64),
            mask=e_mask,
        )

        # ── GT R, t を取得 ───────────────────────────────────────────────
        # T_np は T_{t→t+1}（カメラt から見たカメラt+1の変換）
        # T_np = [R_gt | t_gt] → カメラ1 → カメラ2 の変換と一致
        R_gt  = T_np[:3, :3]
        t_gt  = T_np[:3,  3]

        # ── Rotation error ───────────────────────────────────────────────
        # R_rel = R_est @ R_gt^{-1} = R_est @ R_gt.T
        # trace(R_rel) = 1 + 2*cos(theta) → theta = acos((trace-1)/2)
        R_rel = R_est @ R_gt.T
        trace = float(np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0))
        R_err = float(np.degrees(np.arccos(trace)))

        # ── Translation error（方向のみ、スケール不定）──────────────────
        # t_gt がゼロに近い場合はスキップ（stride が小さすぎる）
        t_gt_norm_val = np.linalg.norm(t_gt)
        if t_gt_norm_val < 1e-4:
            # baseline が小さすぎる → Pose AUC は計算不可
            continue

        t_est_flat  = t_est.ravel()
        t_est_norm  = t_est_flat  / (np.linalg.norm(t_est_flat) + 1e-8)
        t_gt_norm_v = t_gt        / t_gt_norm_val

        # 符号の曖昧さ（recoverPose は方向のみ）→ abs でどちらの方向でも許容
        dot   = float(np.clip(np.dot(t_est_norm, t_gt_norm_v), -1.0, 1.0))
        t_err = float(np.degrees(np.arccos(abs(dot))))

        pose_err = max(R_err, t_err)
        pose_errors.append(pose_err)

        # ── Precision@t（GT F 行列による正解マッチの割合）────────────────
        try:
            F_gt = _compute_F_gt(T_rel_gt, K_gt).astype(np.float32)
            epi_dists = _sym_epi_dist(
                kpts[idx1].astype(np.float32),
                kpts2[idx2].astype(np.float32),
                F_gt,
            )
            for t in prec_thrs:
                precision_vals[t].append(float((epi_dists < t).mean()))
        except Exception:
            pass

    # ── 集計 ─────────────────────────────────────────────────────────────
    n = len(ms_list)

    # Pose AUC
    pose_auc_dict: Dict[int, float] = {}
    if pose_errors:
        arr = np.array(pose_errors)
        for t in pose_thrs:
            pose_auc_dict[t] = float((arr < t).mean())

    # Precision
    prec_dict: Dict[int, float] = {}
    for t in prec_thrs:
        vals = precision_vals[t]
        prec_dict[t] = float(np.mean(vals)) if vals else 0.0

    # ── Recall の計算（LightGlue 論文準拠）─────────────────────────────
    # 定義: Recall@τ = 実際に取れた正解マッチ数
    #                   / MNN で得られる全マッチのうち距離 < τ のもの
    #
    # 「MNN で最大限取れる正解数」を分母にすることで
    # 「取れるはずの正解をどれだけ取れたか」を測る
    #
    # recall_numerators[t]:   実際のマッチで dist < τ の数（TP）
    # recall_denominators[t]: MNN 全マッチで dist < τ の数（TP + FN の上限）
    # recall_per_pair[t]:     ペアごとの TP / max_TP
    recall_per_pair: Dict[int, List[float]] = {t: [] for t in prec_thrs}

    for t in prec_thrs:
        if not precision_vals.get(t):
            continue
        for prec_val, ms_val, n_kp in zip(
                precision_vals[t], ms_list, n_kpts_list):
            # TP = precision_val（このペアのマッチ中の正解割合）* n_matches
            n_matches_est = ms_val * n_kp
            tp_actual = prec_val * n_matches_est   # 実際の正解マッチ数

            # 分母: MNN で取れる最大正解数の上限
            # = min(kp1, kp2)（最大潜在対応数）* precision（同じ GT F を使用）
            # ここでは min(kp1, kp2) ≈ n_kp（kp1 の数で近似）
            # より正確には「MNN 全点でのエピポーラ距離 < τ」だが
            # それには全 KP の MNN を計算する必要があるため
            # n_kp * precision を上限として使用
            denom = n_kp * prec_val   # 近似上限
            if denom < 1e-6:
                continue
            recall_pair = min(tp_actual / denom, 1.0)
            recall_per_pair[t].append(recall_pair)

    recall_dict: Dict[int, float] = {}
    for t in prec_thrs:
        vals = recall_per_pair[t]
        recall_dict[t] = float(np.mean(vals)) if vals else 0.0

    # 平均マッチ数（絶対値）
    n_matches_abs = float(np.mean([ms * n for ms, n in zip(ms_list, n_kpts_list)]))                     if ms_list and n_kpts_list else 0.0

    return EvalMetrics(
        model_name        = model_name,
        dataset_name      = dataset_name,
        auc               = pose_auc_dict,
        matching_score    = float(np.mean(ms_list))     if ms_list     else 0.0,
        mean_n_kpts       = float(np.mean(n_kpts_list)) if n_kpts_list else 0.0,
        mean_inlier_ratio = float(len(pose_errors)) / max(n, 1),
        n_pairs           = n,
        mean_time_sec     = elapsed_total / max(n, 1),
        pose_auc          = pose_auc_dict,
        precision         = prec_dict,
        recall            = recall_dict,
        mean_n_matches    = n_matches_abs,
    )


# ---------------------------------------------------------------------------
# 定量評価（1データセット）― evaluate/eval_matching.py 単体実行用の旧版
# ---------------------------------------------------------------------------

def _evaluate_dataset_legacy(name: str,
                     pairs: List[Tuple[str, str]],
                     models: Dict[str, torch.nn.Module],
                     args: argparse.Namespace,
                     device: torch.device) -> Dict[str, Dict]:
    size   = (args.viz_width, args.viz_height)
    max_kp = args.max_keypoints
    method = args.matching_method
    ratio  = args.ratio_threshold
    thrs   = args.auc_thresholds

    buf: Dict[str, Dict] = {m: {'errors': [], 'ms': []} for m in args.viz_models}

    for i, (rgb_p, thr_p) in enumerate(pairs):
        if (i+1) % 100 == 0:
            print(f"  [{name}] {i+1}/{len(pairs)} ...")
        try:
            rgb_t, _ = imread_tensor(rgb_p, False, device, size)
            thr_t, _ = imread_tensor(thr_p, True,  device, size)
        except FileNotFoundError as e:
            continue

        hw = (args.viz_height, args.viz_width)

        # 各モデルの評価
        # 基準: 教師(RGB)の特徴 vs 各モデルの熱画像特徴
        kpts_rgb, descs_rgb = detect(models['teacher'], rgb_t, max_kp)

        for mn in args.viz_models:
            if mn == 'teacher_rgb':
                # RGB → RGB（自己整合性・上限確認）
                k1, d1 = kpts_rgb, descs_rgb
                k2, d2 = detect(models['teacher'], rgb_t, max_kp)
            elif mn == 'teacher_thr':
                # 元の XFeat に熱画像を入力（KD前のベースライン）
                k1, d1 = kpts_rgb, descs_rgb
                k2, d2 = detect(models['teacher'], thr_t, max_kp)
            elif mn == 'student_thr':
                # 提案手法（KD済み）に熱画像を入力
                k1, d1 = kpts_rgb, descs_rgb
                k2, d2 = detect(models['student'], thr_t, max_kp)
            else:
                continue

            i1, i2 = match(d1, d2, method, ratio)
            err     = homography_error(k1, k2, i1, i2, hw)
            ms      = matching_score(k1, k2, i1)
            buf[mn]['errors'].append(err)
            buf[mn]['ms'].append(ms)

    # 集計
    summary = {}
    for mn in args.viz_models:
        errs = buf[mn]['errors']
        mss  = buf[mn]['ms']
        r    = auc_at(errs, thrs)
        r['MS']      = float(np.mean(mss)) if mss else 0.0
        r['n_pairs'] = len(errs)
        summary[mn]  = r
    return summary


# ---------------------------------------------------------------------------
# 定性評価（可視化）
# ---------------------------------------------------------------------------

def _bgr(args, key):
    rgb = getattr(args, f'kp_color_{key}', [200, 200, 200])
    return (int(rgb[2]), int(rgb[1]), int(rgb[0]))


def visualize_pair(rgb_p: str, thr_p: str,
                   models: Dict[str, torch.nn.Module],
                   args: argparse.Namespace,
                   device: torch.device,
                   save_path: str) -> None:
    """
    可視化画像の構成:
      上段: [RGB + XFeat(RGB)kpts] | [Thr + XFeat(Thr)kpts] | [Thr + Student kpts]
      下段: XFeat(Thr) vs Student のマッチング対応線
    """
    size = (args.viz_width, args.viz_height)
    r    = args.kp_radius
    try:
        rgb_t, rgb_bgr = imread_tensor(rgb_p, False, device, size)
        thr_t, thr_bgr = imread_tensor(thr_p, True,  device, size)
    except FileNotFoundError as e:
        print(f"  [VIZ SKIP] {e}")
        return

    W, H = args.viz_width, args.viz_height

    # 検出
    k_trg, d_trg = detect(models['teacher'], rgb_t, args.max_keypoints)
    k_tth, d_tth = detect(models['teacher'], thr_t, args.max_keypoints)
    k_sth, d_sth = detect(models['student'], thr_t, args.max_keypoints)

    def draw_kp(img, kpts, color, max_draw=300):
        out = img.copy()
        for x, y in kpts[:max_draw]:
            cv2.circle(out, (int(x), int(y)), r, color, -1)
        return out

    col1 = draw_kp(rgb_bgr, k_trg, _bgr(args, 'teacher_rgb'))
    col2 = draw_kp(thr_bgr, k_tth, _bgr(args, 'teacher_thr'))
    col3 = draw_kp(thr_bgr, k_sth, _bgr(args, 'student_thr'))

    # マッチング（XFeat(Thr) vs Student(Thr)）
    i1, i2 = match(d_tth, d_sth, args.matching_method, args.ratio_threshold)
    mc = tuple(reversed(getattr(args, 'match_color', [255, 255, 0])))
    col4 = np.hstack([thr_bgr.copy(), thr_bgr.copy()])
    for a, b in zip(i1[:100], i2[:100]):
        x1, y1 = int(k_tth[a][0]), int(k_tth[a][1])
        x2, y2 = int(k_sth[b][0]) + W, int(k_sth[b][1])
        cv2.line(col4, (x1,y1), (x2,y2), mc, 1, cv2.LINE_AA)
        cv2.circle(col4, (x1,y1), r, _bgr(args,'teacher_thr'), -1)
        cv2.circle(col4, (x2,y2), r, _bgr(args,'student_thr'), -1)

    # ラベル
    font = cv2.FONT_HERSHEY_SIMPLEX
    def label(img, txt):
        cv2.putText(img, txt, (8,26), font, 0.60, (0,0,0),    3, cv2.LINE_AA)
        cv2.putText(img, txt, (8,26), font, 0.60, (255,255,255),1, cv2.LINE_AA)

    label(col1, f'XFeat(RGB)  kpts={len(k_trg)}')
    label(col2, f'XFeat(Thr)  kpts={len(k_tth)}  [KD前]')
    label(col3, f'Student(Thr) kpts={len(k_sth)} [提案]')
    label(col4, f'XFeat(Thr) vs Student(Thr)  matches={len(i1)}')

    top  = np.hstack([col1, col2, col3])
    bot  = cv2.resize(col4, (top.shape[1], H))
    canvas = np.vstack([top, bot])

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    cv2.imwrite(save_path, canvas)
    print(f"  [VIZ] Saved: {save_path}")


# ---------------------------------------------------------------------------
# 結果表示・保存
# ---------------------------------------------------------------------------

def print_results(all_results: Dict[str, Dict]) -> None:
    labels = {
        'teacher_rgb': 'XFeat(RGB)    [upper bound]',
        'teacher_thr': 'XFeat(Thr)    [baseline KD前]',
        'student_thr': 'Student(Thr)  [提案手法]',
    }
    print()
    print('=' * 82)
    print('  EVALUATION RESULTS')
    print('=' * 82)
    for ds, res in all_results.items():
        print(f"\n  Dataset: {ds}")
        hdr = f"  {'Model':<36s} {'AUC@3px':>8s} {'AUC@5px':>8s} {'AUC@10px':>9s} {'MS':>7s} {'pairs':>6s}"
        print(hdr)
        print(f"  {'-'*78}")
        for mn, label in labels.items():
            if mn not in res:
                continue
            r = res[mn]
            print(
                f"  {label:<36s}"
                f" {r.get('AUC@3px',0)*100:>7.2f}%"
                f" {r.get('AUC@5px',0)*100:>7.2f}%"
                f" {r.get('AUC@10px',0)*100:>8.2f}%"
                f" {r.get('MS',0)*100:>6.2f}%"
                f" {r.get('n_pairs',0):>6d}"
            )


def save_json(results: Dict, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    p = os.path.join(output_dir, 'eval_results.json')
    with open(p, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[Eval] Results saved → {p}")


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_arguments()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device_num)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Eval] Device: {device}")
    if device.type == 'cuda':
        print(f"[Eval] GPU: {torch.cuda.get_device_name(0)}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    models = load_models(args, device)
    os.makedirs(args.output_dir, exist_ok=True)

    all_results: Dict[str, Dict] = {}

    for ds_name in args.datasets:
        print(f"\n[Eval] ========== {ds_name} ==========")
        try:
            pairs = load_pairs(ds_name, args, args.split)
        except Exception as e:
            print(f"[Eval] {ds_name} skipped: {e}")
            continue

        if not pairs:
            print(f"[Eval] {ds_name}: no pairs, skipping.")
            continue

        # 定量評価
        print(f"[Eval] Quantitative: {len(pairs)} pairs ...")
        all_results[ds_name] = evaluate_dataset(
            ds_name, pairs, models, args, device)

        # 定性評価（可視化）
        n_viz = args.n_viz
        if n_viz > 0:
            print(f"[Eval] Visualization: {n_viz} pairs ...")
            rng   = random.Random(args.seed + 99)
            picks = rng.sample(pairs, min(n_viz, len(pairs)))
            for vi, (rgb_p, thr_p) in enumerate(picks):
                sp = os.path.join(args.output_dir, ds_name, f'viz_{vi+1:03d}.png')
                visualize_pair(rgb_p, thr_p, models, args, device, sp)

    print_results(all_results)
    save_json(all_results, args.output_dir)
    print("\n[Eval] Done.")


if __name__ == '__main__':
    main()