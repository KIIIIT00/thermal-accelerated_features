"""
evaluate.py
キーポイント・マッチング精度 評価エントリポイント。

使用方法:
    # デフォルト設定で評価
    python evaluate.py --config configs/eval_config.yaml

    # データセットを CLI で指定（yaml の eval_dataset を上書き）
    python evaluate.py --config configs/eval_config.yaml \\
        --eval_dataset freiburg ms2

    # モデルの重みを CLI で指定
    python evaluate.py --config configs/eval_config.yaml \\
        --proposed_weights checkpoints/thermal_kd/default/thermal_kd_student_best.pth

    # 評価ペア数を制限（デバッグ用）
    python evaluate.py --config configs/eval_config.yaml \\
        --n_pairs 50
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml


# ---------------------------------------------------------------------------
# 設定ファイルの読み込み
# ---------------------------------------------------------------------------

def _load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _merge_args(cfg: dict, args: argparse.Namespace) -> argparse.Namespace:
    """YAML 設定を args のデフォルトとして適用（CLI が優先）。"""
    parser_defaults = vars(args).copy()
    for k, v in cfg.items():
        if k not in parser_defaults or parser_defaults[k] is None:
            setattr(args, k, v)
    return args


# ---------------------------------------------------------------------------
# 引数パーサー
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Thermal XFeat Matching Accuracy Evaluation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('--config', type=str, default='configs/eval_config.yaml',
                        help='評価設定 YAML ファイルのパス')

    # ── データセット ──────────────────────────────────────────────────────
    parser.add_argument(
        '--eval_dataset', nargs='+', default=None,
        metavar='NAME',
        help='評価するデータセット名（例: freiburg ms2 sthereo tartanrgbt vivid）'
             ' 指定しない場合は yaml の eval_dataset を使用',
    )
    parser.add_argument('--n_pairs', type=int, default=None,
                        help='評価ペア数（-1 で全件。デバッグ時は 50 程度を推奨）')

    # ── モデル重みの上書き ────────────────────────────────────────────────
    parser.add_argument('--proposed_weights', type=str, default=None,
                        help='提案手法の重みファイルパス（yaml の設定を上書き）')
    parser.add_argument('--baseline_weights', type=str, default=None,
                        help='ベースライン（XFeat）の重みファイルパス')

    # ── データルートの上書き ──────────────────────────────────────────────
    parser.add_argument('--freiburg_root',   type=str, default=None)
    parser.add_argument('--tartanrgbt_root', type=str, default=None)
    parser.add_argument('--vivid_root',      type=str, default=None)
    parser.add_argument('--sthereo_root',    type=str, default=None)
    parser.add_argument('--ms2_root',        type=str, default=None)

    # ── 出力 ─────────────────────────────────────────────────────────────
    parser.add_argument('--output_dir', type=str, default=None,
                        help='結果の保存先ディレクトリ（yaml の output_dir を上書き）')
    parser.add_argument('--no_vis', action='store_true',
                        help='可視化画像を生成しない（高速化）')
    parser.add_argument('--matching_method', type=str, default=None,
                        choices=['mutual_nn', 'lightglue'],
                        help='マッチング手法（yaml の matching_method を上書き）')

    # ── デバイス ─────────────────────────────────────────────────────────
    parser.add_argument('--device_num', type=str, default=None,
                        help='使用する GPU 番号（例: 0）')

    return parser


# ---------------------------------------------------------------------------
# データセットのペアリスト取得
# ---------------------------------------------------------------------------

def get_pairs_from_dataset(
    name: str,
    args: Any,
) -> List[Tuple]:
    """
    データセット名からペアリストを取得する。

    返り値の形式:
      Freiburg / SThErEO / VIVID : [(rgb_path, thr_path), ...]
      TartanRGBT                 : [(rgb_path, thr_path, T_rel, K), ...]
                                    ← GT ポーズ付き（エピポーラ評価に使用）
    """
    from modules.dataset.thermal.loader import (
        _resolve_data_root,
        _resolve_splits_dir,
    )
    from modules.dataset.thermal.freiburg   import FreiburgDataset
    from modules.dataset.thermal.tartanrgbt import TartanRGBTDataset
    from modules.dataset.thermal.vivid      import VividDataset
    from modules.dataset.thermal.sthereo    import SthEreoDataset
    from modules.dataset.thermal.ms2        import MS2Dataset

    _CLS = {
        'freiburg':   FreiburgDataset,
        'tartanrgbt': TartanRGBTDataset,
        'vivid':      VividDataset,
        'sthereo':    SthEreoDataset,
        'ms2':        MS2Dataset,
    }

    name_l = name.lower()
    if name_l not in _CLS:
        raise ValueError(f"Unknown dataset: {name!r}. Valid: {list(_CLS)}")

    data_root  = _resolve_data_root(name_l, args)
    splits_dir = _resolve_splits_dir(name_l, args)

    # TartanRGBT は GT ポーズ付きの Sequential 版を使う
    if name_l == 'tartanrgbt':
        try:
            from modules.dataset.thermal.sequential import TartanRGBTSequentialDataset
            # stride=1 は連続フレームで運動量が極小 → trivial（全マッチが閾値内）
            # stride=5 以上で非 trivial な評価ペアを生成する
            eval_stride = getattr(args, 'tartanrgbt_eval_stride', None) or                           cfg.get('tartanrgbt_eval_stride', 5)                           if hasattr(args, 'cfg') else 5
            # args から cfg にアクセスできない場合のフォールバック
            try:
                import yaml as _yaml
                _cfg = _yaml.safe_load(open(args.config)) or {}
                eval_stride = _cfg.get('tartanrgbt_eval_stride', 5)
            except Exception:
                eval_stride = 5

            ds = TartanRGBTSequentialDataset(
                data_root  = data_root,
                splits_dir = splits_dir,
                stride     = eval_stride,
            )
            print(f"[Eval] tartanrgbt: using stride={eval_stride}")
            pairs = list(ds._pairs)
            # sequential の _pairs は (thr_t, thr_t1, T_rel, K) 形式
            # evaluate.py は (rgb, thr, T_rel, K) を期待するため変換
            # TartanRGBT は RGB パスが別ディレクトリにあるため thr を rgb 代わりに使う
            pairs_out = []
            for p in pairs:
                if len(p) == 4:
                    thr_t, thr_t1, T_rel, K = p
                    # rgb_path = thr_path（モダリティは thermal で評価）
                    pairs_out.append((thr_t, thr_t1, T_rel, K))
                else:
                    pairs_out.append(p)
            print(f"[Eval] tartanrgbt: {len(pairs_out)} val pairs loaded"
                  f" (with GT pose) | data={data_root}")
            return pairs_out
        except Exception as e:
            print(f"[Eval] tartanrgbt: Sequential failed ({e}), "
                  f"falling back to standard pairs")

    ds = _CLS[name_l](
        data_root=data_root,
        splits_dir=splits_dir,
        split='val',
        augment=False,
    )
    pairs = list(ds._pairs)
    print(f"[Eval] {name}: {len(pairs)} val pairs loaded"
          f" | data={data_root}")
    return pairs


# ---------------------------------------------------------------------------
# モデルのロード
# ---------------------------------------------------------------------------

def load_model(
    weights_path: Optional[str],
    device: torch.device,
) -> torch.nn.Module:
    """
    XFeatModel を weights_path からロードして返す。
    weights_path が None の場合はデフォルト重みを使用。
    """
    from modules.model import XFeatModel

    model = XFeatModel().to(device).eval()
    if weights_path and os.path.isfile(weights_path):
        state = torch.load(weights_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        print(f"[Eval] Loaded weights: {weights_path}")
    elif weights_path:
        print(f"[Eval] WARNING: weights not found: {weights_path!r}"
              f"  → using random weights")
    else:
        print("[Eval] Using default model initialization")
    return model


# ---------------------------------------------------------------------------
# 結果の保存
# ---------------------------------------------------------------------------

def save_results(
    results: dict,
    all_errors: dict,
    output_dir: str,
    cfg: dict,
) -> None:
    """数値結果を CSV / JSON で保存する。"""
    os.makedirs(output_dir, exist_ok=True)
    auc_thrs = cfg.get('auc_thresholds', [1, 3, 5, 10])

    # ── JSON ─────────────────────────────────────────────────────────────
    if cfg.get('save_metrics_json', True):
        json_data = {}
        for ds_name, ds_res in results.items():
            json_data[ds_name] = {}
            for model_name, m in ds_res.items():
                json_data[ds_name][model_name] = {
                    'auc':              m.auc,
                    'matching_score':   m.matching_score,
                    'mean_n_kpts':      m.mean_n_kpts,
                    'mean_inlier_ratio': m.mean_inlier_ratio,
                    'n_pairs':          m.n_pairs,
                    'mean_time_ms':     m.mean_time_sec * 1000,
                }
        json_path = os.path.join(output_dir, 'metrics.json')
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"[Eval] Saved: {json_path}")

    # ── CSV ──────────────────────────────────────────────────────────────
    if cfg.get('save_metrics_csv', True):
        csv_path = os.path.join(output_dir, 'metrics.csv')
        headers = (['model', 'dataset'] +
                   [f'AUC@{t}px' for t in sorted(auc_thrs)] +
                   ['matching_score', 'mean_n_kpts',
                    'mean_inlier_ratio', 'n_pairs', 'mean_time_ms'])
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for ds_name, ds_res in results.items():
                for model_name, m in ds_res.items():
                    row = ([model_name, ds_name] +
                           [f"{m.auc.get(t, 0):.6f}"
                            for t in sorted(auc_thrs)] +
                           [f"{m.matching_score:.6f}",
                            f"{m.mean_n_kpts:.1f}",
                            f"{m.mean_inlier_ratio:.6f}",
                            str(m.n_pairs),
                            f"{m.mean_time_sec * 1000:.2f}"])
                    writer.writerow(row)
        print(f"[Eval] Saved: {csv_path}")


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    # YAML 設定を読み込んで args にマージ
    cfg = _load_yaml(args.config)
    print(f"[Eval] Config: {args.config}")

    # data_roots を args に展開
    data_roots = cfg.get('data_roots', {})
    if not hasattr(args, 'data_roots') or args.data_roots is None:
        args.data_roots = data_roots
    # CLI 上書き
    for name in ['freiburg', 'tartanrgbt', 'vivid', 'sthereo', 'ms2']:
        cli_root = getattr(args, f'{name}_root', None)
        if cli_root:
            args.data_roots[name] = cli_root

    # eval_dataset の決定（CLI > yaml）
    if args.eval_dataset is None:
        args.eval_dataset = cfg.get('eval_dataset', ['freiburg'])
    if args.n_pairs is None:
        args.n_pairs = cfg.get('n_pairs', 1000)

    # output_dir の決定
    output_dir = args.output_dir or cfg.get('output_dir', 'evaluate/results')
    os.makedirs(output_dir, exist_ok=True)

    # matching_method の CLI 上書き（yaml より CLI が優先）
    if getattr(args, 'matching_method', None):
        cfg['matching_method'] = args.matching_method
        print(f"[Eval] matching_method overridden by CLI: {args.matching_method}")

    # デバイスの設定
    device_num = args.device_num or str(cfg.get('device_num', '0'))
    os.environ['CUDA_VISIBLE_DEVICES'] = device_num
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Eval] Device: {device}")

    # ── モデルの準備 ──────────────────────────────────────────────────────
    models_cfg: List[dict] = cfg.get('models', [])

    # CLI でモデル重みが指定された場合は上書き
    for m in models_cfg:
        if m['name'] == 'thermal_xfeat_proposed' and args.proposed_weights:
            m['weights'] = args.proposed_weights
        if m['name'] in ('xfeat_rgb', 'xfeat_thermal_baseline') \
                and args.baseline_weights:
            m['weights'] = args.baseline_weights

    print(f"\n[Eval] Loading {len(models_cfg)} models ...")
    models: Dict[str, torch.nn.Module] = {}
    for m_cfg in models_cfg:
        model = load_model(m_cfg.get('weights'), device)
        models[m_cfg['name']] = model
        print(f"  {m_cfg['name']:35s} modality={m_cfg['modality']}"
              f"  desc={m_cfg.get('description','')}")

    # ── LightGlue のロード ─────────────────────────────────────────────────
    lightglue_model = None
    if cfg.get('matching_method') == 'lightglue':
        from eval.eval_matching import load_lightglue
        lg_weights = cfg.get('lightglue_weights', None)
        print(f"\n[Eval] Loading LightGlue"
              f" (weights={lg_weights or 'pretrained'}) ...")
        lightglue_model = load_lightglue(lg_weights, device)
        if lightglue_model is not None:
            print("[Eval] LightGlue ready")
        else:
            print("[Eval] WARNING: LightGlue failed → fallback to MNN")

    # ── 評価ループ ────────────────────────────────────────────────────────
    from eval.eval_matching import evaluate_dataset, EvalMetrics

    results:    Dict[str, Dict[str, EvalMetrics]]   = {}
    all_errors: Dict[str, Dict[str, np.ndarray]]    = {}

    for ds_name in args.eval_dataset:
        print(f"\n{'='*60}")
        print(f"  Dataset: {ds_name}")
        print(f"{'='*60}")

        try:
            pairs = get_pairs_from_dataset(ds_name, args)
        except Exception as e:
            print(f"[Eval] {ds_name}: skipped — {e}")
            continue

        results[ds_name]    = {}
        all_errors[ds_name] = {}

        rng = np.random.default_rng(cfg.get('seed', 42))

        for m_cfg in models_cfg:
            model_name = m_cfg['name']
            modality   = m_cfg['modality']
            model      = models[model_name]

            print(f"\n  -- {model_name} ({modality}) --")
            metrics = evaluate_dataset(
                model=model,
                model_name=model_name,
                dataset_name=ds_name,
                pairs=pairs,
                modality=modality,
                device=device,
                cfg={**cfg, 'n_pairs': args.n_pairs},
                rng=rng,
                verbose=True,
                lightglue_model=lightglue_model,
            )
            results[ds_name][model_name] = metrics

            # 誤差配列を保存（ヒストグラム用）
            # ※ evaluate_dataset は現状 errors を返さないため空
            all_errors[ds_name][model_name] = np.array([])

    # ── 結果の保存 ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Results Summary")
    print(f"{'='*60}")
    for ds_name, ds_res in results.items():
        for model_name, m in ds_res.items():
            print(f"  {m.summary()}")

    save_results(results, all_errors, output_dir, cfg)

    # ── 可視化 ────────────────────────────────────────────────────────────
    if not args.no_vis and cfg.get('save_figures', True):
        from eval.eval_visualize import (
            plot_auc_curves,
            plot_error_histogram,
            plot_summary_table,
            save_match_images,
        )
        auc_thrs = cfg.get('auc_thresholds', [1, 3, 5, 10])

        print("\n[Vis] Generating figures ...")
        plot_auc_curves(results, output_dir, auc_thrs)
        plot_summary_table(results, output_dir, auc_thrs)

        if cfg.get('save_match_images', True):
            for ds_name in results:
                try:
                    pairs = get_pairs_from_dataset(ds_name, args)
                    save_match_images(
                        models_cfg=models_cfg,
                        models=models,
                        pairs=pairs,
                        device=device,
                        output_dir=output_dir,
                        dataset_name=ds_name,
                        cfg=cfg,
                        n_samples=cfg.get('n_vis_samples', 10),
                        seed=cfg.get('seed', 42),
                        lightglue_model=lightglue_model,
                    )
                except Exception as e:
                    print(f"[Vis] {ds_name}: skipped — {e}")

    print(f"\n[Eval] Done. Results saved to: {output_dir}/")


if __name__ == '__main__':
    main()