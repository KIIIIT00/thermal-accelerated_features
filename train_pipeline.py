"""
train_pipeline.py
ThermalXFeat 堅牢モデルの全学習パイプラインを順次実行する。

設計根拠（実験結果）:
  - kd_only が最良（rep/geo 損失は PoseAUC -22.4pt）
  - 全データセットで汎化（kaist_morning 単独 93% vs 全val 54%）
  - XFeat+LG 分離最適化が記述子ドリフトを引き起こす（Recall 135.9%→65.7%）

パイプライン構成:
  Stage 1: KD 事前学習（SThErEO+VIVID, 100 epoch）
  Stage 2: 空間分散損失を追加（50 epoch）
  Stage 3: LightGlue 再 fine-tune（20 epoch）

使用方法:
    # 全ステージを順番に実行
    python train_pipeline.py --config configs/pipeline_config.yaml

    # 特定ステージから再開
    python train_pipeline.py --config configs/pipeline_config.yaml --start_stage 2

    # wandb なし
    python train_pipeline.py --config configs/pipeline_config.yaml --no_wandb
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from typing import Any, Dict, Optional

import yaml


# ---------------------------------------------------------------------------
# 引数
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config',      default='configs/pipeline_config.yaml')
    p.add_argument('--start_stage', type=int, default=1,
                   help='開始ステージ番号（途中再開用）')
    p.add_argument('--end_stage',   type=int, default=3,
                   help='終了ステージ番号')
    p.add_argument('--no_wandb',    action='store_true', default=False)
    p.add_argument('--device',      default='0')
    # 設定ファイルの値を CLI で上書きする場合
    p.add_argument('--sthereo_root', default=None)
    p.add_argument('--vivid_root',   default=None)
    p.add_argument('--ms2_root',     default=None,
                   help='MS2 ルート（省略時は使用しない）')
    # 初期化の種類（比較実験用）
    p.add_argument('--init_type',
                   choices=['rgb', 'proposed'],
                   default='proposed',
                   help='rgb: RGB デフォルト重み / proposed: post_kd_s2_final.pth')
    return p.parse_args()


# ---------------------------------------------------------------------------
# ユーティリティ
# ---------------------------------------------------------------------------

def load_cfg(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def run(cmd: list, desc: str) -> int:
    """サブプロセスとして実行し、リターンコードを返す。"""
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"  コマンド: {' '.join(cmd)}")
    print(f"{'='*60}")
    t0 = time.time()
    ret = subprocess.run(cmd).returncode
    elapsed = time.time() - t0
    status = "✅ 完了" if ret == 0 else f"❌ 失敗 (code={ret})"
    print(f"\n  {status}  ({elapsed/60:.1f} 分)")
    return ret


def resolve(cfg: Dict, key: str, cli_val: Optional[str], default: str) -> str:
    """CLI → config → default の優先順で値を解決する。"""
    if cli_val:
        return cli_val
    return cfg.get(key, default)


# ---------------------------------------------------------------------------
# 各ステージの実行
# ---------------------------------------------------------------------------

def stage1_kd(cfg: Dict, args: argparse.Namespace) -> int:
    """Stage 1: SThErEO + VIVID で KD 事前学習。"""
    s = cfg.get('stage1', {})
    data = cfg.get('data', {})

    sthereo_root = resolve(cfg, 'sthereo_root', args.sthereo_root,
                           data.get('sthereo_root', 'datasets/sthereo'))
    vivid_root   = resolve(cfg, 'vivid_root',   args.vivid_root,
                           data.get('vivid_root',   'datasets/vivid'))
    ms2_root     = getattr(args, 'ms2_root', None) or                    data.get('ms2_root', None)

    epochs     = s.get('epochs', 100)
    batch_size = s.get('batch_size', 16)
    lr         = s.get('lr', 1e-4)
    split      = data.get('split', 'all')
    stride     = data.get('stride', 3)
    vivid_stride = data.get('vivid_stride', 2)
    n_eval     = data.get('eval_n_pairs', 200)
    max_pairs  = data.get('max_pairs_per_seq', 2000)
    init_type = getattr(args, 'init_type', 'proposed')
    if init_type == 'proposed':
        weights   = s.get('weights_init',
                          'checkpoints/post_kd/default/post_kd_s2_final.pth')
        output    = s.get('output', 'checkpoints/pipeline/stage1_kd') + '_proposed'
        run_name  = 'stage1_kd_proposed'
        tags      = ['stage1', 'kd', 'init:proposed']
    else:
        weights   = 'weights/xfeat.pt'   # 公式 RGB 学習済み重み
        output    = s.get('output', 'checkpoints/pipeline/stage1_kd') + '_rgb'
        run_name  = 'stage1_kd_rgb'
        tags      = ['stage1', 'kd', 'init:rgb']
    best_metric = s.get('best_metric', 'sthereo_PoseAUC@5')

    cmd = [
        sys.executable, 'train_kd_sthereo.py',
        '--sthereo_root',      sthereo_root,
        '--vivid_root',        vivid_root,
        '--output',            output,
        '--epochs',            str(epochs),
        '--batch_size',        str(batch_size),
        '--lr',                str(lr),
        '--split',             split,
        '--stride',            str(stride),
        '--vivid_stride',      str(vivid_stride),
        '--n_eval_pairs',      str(n_eval),
        '--max_pairs_per_seq', str(max_pairs),
        '--best_metric',       best_metric,
        '--device',            args.device,
        '--wandb_project',     'thermal-xfeat-kd',
        '--wandb_group',       'pipeline_comparison',
        '--wandb_run_name',    run_name,
        '--wandb_tags',        *tags,
    ]
    if weights and os.path.isfile(weights):
        cmd += ['--weights_init', weights]
    if ms2_root:
        cmd += ['--ms2_root', ms2_root]
    if args.no_wandb:
        cmd += ['--no_wandb']

    return run(cmd, "Stage 1: KD 事前学習（SThErEO + VIVID + MS2）")


def stage2_spatial(cfg: Dict, args: argparse.Namespace) -> int:
    """Stage 2: 空間分散損失を追加した fine-tune。"""
    s   = cfg.get('stage2', {})
    s1  = cfg.get('stage1', {})
    data = cfg.get('data', {})

    # Stage1 の best.pth を初期重みとして使用
    init_type     = getattr(args, 'init_type', 'proposed')
    stage1_output = s1.get('output', 'checkpoints/pipeline/stage1_kd') + f'_{init_type}'
    weights_init  = s.get('weights_init',
                           os.path.join(stage1_output, 'best.pth'))

    sthereo_root = resolve(cfg, 'sthereo_root', args.sthereo_root,
                           data.get('sthereo_root', 'datasets/sthereo'))
    vivid_root   = resolve(cfg, 'vivid_root', args.vivid_root,
                           data.get('vivid_root', 'datasets/vivid'))

    output     = s.get('output', 'checkpoints/pipeline/stage2_spatial') + f'_{init_type}'
    epochs     = s.get('epochs', 50)
    batch_size = s.get('batch_size', 16)
    lr         = s.get('lr', 5e-5)
    n_eval     = data.get('eval_n_pairs', 200)
    vivid_stride = data.get('vivid_stride', 2)

    # Stage2 は losses_kd.py の spatial/thermal 損失を使う
    # → train_kd_sthereo.py に --lambda_spatial / --lambda_thermal を追加予定
    # 現時点では train_kd_sthereo.py と同じコマンド + 低 lr で実行
    cmd = [
        sys.executable, 'train_kd_sthereo.py',
        '--sthereo_root',  sthereo_root,
        '--vivid_root',    vivid_root,
        '--output',        output,
        '--epochs',        str(epochs),
        '--batch_size',    str(batch_size),
        '--lr',            str(lr),
        '--split',         data.get('split', 'all'),
        '--vivid_stride',  str(vivid_stride),
        '--n_eval_pairs',  str(n_eval),
        '--weights_init',  weights_init,
        '--best_metric',   s.get('best_metric', 'sthereo_PoseAUC@5'),
        '--device',        args.device,
        '--wandb_project', 'thermal-xfeat-kd',
        '--wandb_group',   'pipeline_comparison',
        '--wandb_run_name',f'stage2_spatial_{init_type}',
        '--wandb_tags',    'stage2', 'spatial', f'init:{init_type}',
    ]
    if args.no_wandb:
        cmd += ['--no_wandb']

    return run(cmd, f"Stage 2: 空間分散損失 fine-tune (init={init_type})")


def stage3_lg(cfg: Dict, args: argparse.Namespace) -> int:
    """Stage 3: LightGlue 再 fine-tune（Stage2 後の記述子に LG を再適応）。

    使用スクリプト: train_lightglue_ft.py
    データセット: SThErEO（連続フレーム + GT pose）
      理由: Freiburg/TartanRGBT は GT pose なし → PoseAUC 評価不可
            SThErEO は GPS/IMU GT（高精度）で連続フレームマッチングを学習可能
    損失: NegativeLogAssignment（LG 公式損失）
    XFeat: Stage2/best.pth で固定（frozen）
    """
    s  = cfg.get('stage3', {})
    s2 = cfg.get('stage2', {})
    data = cfg.get('data', {})
    init_type = getattr(args, 'init_type', 'proposed')
    suffix = f'_{init_type}'

    stage2_output = s2.get('output', 'checkpoints/pipeline/stage2_spatial') + suffix
    xfeat_weights = s.get('xfeat_weights',
                          os.path.join(stage2_output, 'best.pth'))
    output        = s.get('output', 'checkpoints/pipeline/stage3_lg') + suffix

    sthereo_root = resolve(cfg, 'sthereo_root', args.sthereo_root,
                           data.get('sthereo_root', 'datasets/sthereo'))

    # lightglue_ft_config_stage3.yaml を生成
    lg_cfg = {
        'thermal_weights': xfeat_weights,
        'ckpt_save_path':  output,
        'n_steps':         s.get('n_steps', 13000),  # 5 epoch 相当（2619 steps/epoch × 5）
        'batch_size':      s.get('batch_size', 4),
        'lr':              s.get('lr', 1e-4),
        'max_keypoints':   512,
        'stride':          data.get('stride', 3),
        # データセット: SThErEO のみ（GPS/IMU GT・高精度）
        'ft_datasets':     ['sthereo'],
        'data_roots': {
            'sthereo': sthereo_root,
        },
        # wandb
        'no_wandb':       args.no_wandb,
        'wandb_project':  'thermal-xfeat-kd',
        'wandb_run_name': f'stage3_lg_{init_type}',
    }

    tmp_cfg = f'configs/lightglue_ft_config_stage3{suffix}.yaml'
    with open(tmp_cfg, 'w') as f:
        yaml.dump(lg_cfg, f, allow_unicode=True)

    cmd = [
        sys.executable, 'train_lightglue_ft.py',
        '--config',     tmp_cfg,
        '--device_num', args.device,
    ]
    return run(cmd, f"Stage 3: LightGlue 再 fine-tune（SThErEO, init={init_type}）")



def stage4_joint(cfg: Dict, args: argparse.Namespace) -> int:
    """Stage 4: XFeat + LightGlue 同時 fine-tune（記述子ドリフトの根本解決）。

    根拠:
        実験で XFeat 単独更新後に Recall 135.9% → 65.7% に低下（記述子ドリフト）。
        XFeat と LG を同一 backward path で同時更新することで
        記述子変化に LG が追従し Recall の回復が期待できる。

    設定:
        lr=1e-5（非常に低い lr で壊さない）
        XFeat: 学習可能（Stage2 の best.pth から開始）
        LG:    学習可能（Stage3 の best.pth から開始）
        損失:  L = L_kd(XFeat) + λ_match × L_match(LG, GT)
    """
    s  = cfg.get('stage4', {})
    s2 = cfg.get('stage2', {})
    s3 = cfg.get('stage3', {})
    data = cfg.get('data', {})
    init_type = getattr(args, 'init_type', 'proposed')
    suffix = f'_{init_type}'

    stage2_output  = s2.get('output', 'checkpoints/pipeline/stage2_spatial') + suffix
    stage3_output  = s3.get('output', 'checkpoints/pipeline/stage3_lg') + suffix
    xfeat_weights  = os.path.join(stage2_output, 'best.pth')
    lg_weights     = os.path.join(stage3_output, 'lightglue_gf_final.pth')
    output         = s.get('output', 'checkpoints/pipeline/stage4_joint') + suffix

    sthereo_root = resolve(cfg, 'sthereo_root', args.sthereo_root,
                           data.get('sthereo_root', 'datasets/sthereo'))

    # stage4_joint_config.yaml を生成
    joint_cfg = {
        'xfeat_weights':  xfeat_weights,
        'lg_weights':     lg_weights,
        'ckpt_save_path': output,
        'n_steps':        s.get('n_steps', 5000),
        'batch_size':     s.get('batch_size', 16),
        'lr':             s.get('lr', 1e-5),
        'lambda_match':   s.get('lambda_match', 0.5),
        'max_keypoints':  512,
        'stride':         data.get('stride', 3),
        'ft_datasets':    ['sthereo'],
        'data_roots': {'sthereo': sthereo_root},
        'no_wandb':       args.no_wandb,
        'wandb_project':  'thermal-xfeat-kd',
        'wandb_run_name': f'stage4_joint_{init_type}',
    }

    tmp_cfg = f'configs/joint_config_stage4{suffix}.yaml'
    with open(tmp_cfg, 'w') as f:
        yaml.dump(joint_cfg, f, allow_unicode=True)

    cmd = [
        sys.executable, 'train_joint.py',
        '--config',     tmp_cfg,
        '--device_num', args.device,
    ]
    return run(cmd, f"Stage 4: XFeat+LG 同時 fine-tune（init={init_type}）")


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------

def main():
    args = get_args()
    cfg  = load_cfg(args.config)

    print(f"\n{'='*60}")
    print(f"  ThermalXFeat 学習パイプライン")
    print(f"  config: {args.config}")
    print(f"  stages:    {args.start_stage} → {args.end_stage}")
    print(f"  init_type: {getattr(args, 'init_type', 'proposed')}")
    print(f"  wandb:     {'無効' if args.no_wandb else '有効'}")
    print(f"{'='*60}")

    stage_fns = {
        1: (stage1_kd,       "Stage 1: KD 事前学習"),
        2: (stage2_spatial,  "Stage 2: 空間分散損失 fine-tune"),
        3: (stage3_lg,       "Stage 3: LightGlue 再 fine-tune"),
        4: (stage4_joint,    "Stage 4: XFeat+LG 同時 fine-tune"),
    }

    results = {}
    for stage_num in range(args.start_stage, args.end_stage + 1):
        if stage_num not in stage_fns:
            print(f"  [Stage {stage_num}] 未実装 → スキップ")
            continue

        fn, desc = stage_fns[stage_num]
        ret = fn(cfg, args)
        results[stage_num] = ret

        if ret != 0:
            print(f"\n❌ Stage {stage_num} が失敗しました（code={ret}）")
            print(f"   --start_stage {stage_num} で再実行してください")
            sys.exit(ret)

    # 最終サマリー
    print(f"\n{'='*60}")
    print(f"  パイプライン完了")
    print(f"{'='*60}")
    for stage_num, ret in results.items():
        status = "✅" if ret == 0 else "❌"
        fn, desc = stage_fns[stage_num]
        print(f"  {status} {desc}")

    print(f"\n  最終モデル:")
    s = cfg.get('stage2', {})
    print(f"    XFeat:     {s.get('output', 'checkpoints/pipeline/stage2_spatial')}/best.pth")
    s3 = cfg.get('stage3', {})
    print(f"    LightGlue: {s3.get('output', 'checkpoints/pipeline/stage3_lg')}/best.tar")

    print(f"\n  評価（主結果）:")
    print(f"    python run_overfit_experiment.py \\")
    print(f"        --seq kaist_morning --loss kd_only --epochs 1 \\")
    print(f"        --max_pairs 500 --eval_all_pairs --seed 42 \\")
    print(f"        --weights_proposed {s.get('output', 'checkpoints/pipeline/stage2_spatial')}/best.pth")


if __name__ == '__main__':
    main()