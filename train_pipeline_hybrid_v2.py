"""
train_pipeline_hybrid.py
ハイブリッド熱画像特徴抽出パイプライン 統合オーケストレーター

【設計思想】
1. config_master.yaml を読み込み、各ステージの有効/無効を判定して順番に実行する。
2. 前のステージで保存された「Bestな重み (Loss最小)」を自動的に次のステージへ渡す。
3. 万が一途中でエラーが発生した場合は、その時点で安全にパイプラインを停止する。
"""

import yaml
import argparse
import subprocess
import os
import sys

def run_stage_cmd(cmd, desc):
    """ サブプロセスとしてコマンドを実行し、エラーをハンドリングする """
    print(f"\n{'='*80}")
    print(f"🚀 [START]: {desc}")
    print(f"Executing command:\n  {' '.join(cmd)}")
    print(f"{'='*80}")
    
    # リアルタイムで標準出力を表示しつつ実行
    ret = subprocess.run(cmd).returncode
    if ret != 0:
        print(f"\n❌ [FAILED]: {desc} (Error Code: {ret})")
        print("パイプラインを停止します。ログを確認してください。")
        sys.exit(ret)
    print(f"\n✅ [SUCCESS]: {desc}\n")

def main():
    parser = argparse.ArgumentParser(description="Master Pipeline Runner")
    parser.add_argument('--config', type=str, default='config_master.yaml', help='マスター設定ファイルのパス')
    # 🎯 Stage 1 で必須となる XFeatの公式事前学習済み重みへのパス
    parser.add_argument('--teacher_weights', type=str, default=None, help='XFeat公式RGB事前学習済み重みのパス')
    parser.add_argument('--start_stage', type=int, default=1, choices=[1, 2, 3], help='実行を開始するステージ')
    parser.add_argument('--check_datasets', action='store_true', help='学習前に各データセットのサンプルを保存する')
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"❌ Error: Config {args.config} not found.")
        sys.exit(1)
        
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    common = cfg.get('common', {})
    output_root = common.get('output_root', 'checkpoints/hybrid_pipeline')
    roots = common.get('data_roots', {})
    device = str(common.get('device', '0'))
    no_wandb = common.get('no_wandb', False)

    teacher_weights = args.teacher_weights or common.get('teacher_weights', 'weights/xfeat.pt')

    os.makedirs(output_root, exist_ok=True)

    # =====================================================================
    # Stage 1: XFeat Knowledge Distillation
    # =====================================================================
    s1 = cfg.get('stage1', {})
    s1_out = os.path.join(output_root, "stage1_kd")
    # 🎯 最終モデルではなく、検証で最も良かった重みを次へ渡す
    s1_best_weights = os.path.join(s1_out, "model_stage1_best.pth")

    if s1.get('enabled', False) and args.start_stage <= 1:
        cmd1 = [
            'python', 'train_kd_hybrid.py',
            '--teacher_weights', teacher_weights,
            '--sthereo_root', roots.get('sthereo', ''),
            '--ms2_root', roots.get('ms2', ''),
            '--vivid_root', roots.get('vivid', ''),
            '--tartanrgbt_root', roots.get('tartanrgbt', ''),
            '--freiburg_root', roots.get('freiburg', ''),
            '--output', s1_out,
            '--batch_size', str(s1.get('batch_size', 16)),
            '--epochs', str(s1.get('epochs', 10)),
            '--lr', str(s1.get('lr', 1e-4)),
            '--device', device,
            '--wandb_run_name', s1.get('wandb_run_name', 'stage1_kd')
        ]
        
        if args.check_datasets: cmd1.append('--save_dataset_samples')
        # アブレーション Loss の設定
        if s1.get('use_kd_loss', True): cmd1.append('--use_kd_loss')
        if s1.get('use_hybrid_loss', False): cmd1.append('--use_hybrid_loss')
        if s1.get('use_spatial_loss', False): cmd1.append('--use_spatial_loss')
        cmd1.extend([
            '--lambda_kd', str(s1.get('lambda_kd', 1.0)),
            '--lambda_hybrid', str(s1.get('lambda_hybrid', 0.1)),
            '--lambda_spatial', str(s1.get('lambda_spatial', 0.01))
        ])

        # データ拡張 (Ablation) の設定
        aug_cfg = s1.get('augmentation', {})
        if aug_cfg.get('enabled', False):
            methods_str = ",".join(aug_cfg.get('methods', []))
            crop_h, crop_w = aug_cfg.get('crop_size', [256, 256])
            cmd1.extend([
                '--aug_list', methods_str,
                '--crop_size', f"{crop_h},{crop_w}"
            ])

        if no_wandb: cmd1.append('--no_wandb')

        run_stage_cmd(cmd1, "Stage 1: XFeat Knowledge Distillation")
    else:
        print("\n⏭️  Stage 1 is disabled in config. Skipping.")

    # =====================================================================
    # Stage 2: LightGlue Geometry-Safe Finetuning
    # =====================================================================
    s2 = cfg.get('stage2', {})
    s2_out = os.path.join(output_root, "stage2_lightglue")
    # 🎯 Stage 2で最もMatch Lossが低かったLightGlueの重み
    s2_best_weights = os.path.join(s2_out, "lg_stage2_best.pth")

    s2_input_weights = s2.get('xfeat_weights', s2_best_weights)
    print("s2_input_weights", s2_input_weights)
    s2_lg_weights_path = s2.get('lg_weights', None)
    

    s2_datasets = s2.get('datasets', [])

    if s2.get('enabled', False) and args.start_stage <= 2:
        if not os.path.exists(s2_input_weights):
            print(f"❌ Error: Stage 1 の Best重みが見つかりません: {s1_best_weights}")
            print("Stage 1 を先に実行するか、パスを確認してください。")
            sys.exit(1)

        cmd2 = [
            'python', 'train_lg_ft_hybrid.py',
            '--config', args.config,
            '--xfeat_weights', s2_input_weights,
            '--lg_weights', s2_lg_weights_path,
            '--sthereo_root', roots.get('sthereo', '') if 'sthereo' in s2_datasets else '',
            '--ms2_root', roots.get('ms2', '') if 'ms2' in s2_datasets else '',
            '--vivid_root', roots.get('vivid', '') if 'vivid' in s2_datasets else '',
            '--tartanrgbt_root', roots.get('tartanrgbt', '') if 'tartanrgbt' in s2_datasets else '',
            '--freiburg_root', roots.get('freiburg', '') if 'freiburg' in s2_datasets else '',
            '--output', s2_out,
            '--batch_size', str(s2.get('batch_size', 8)),
            '--lr', str(s2.get('lr', 1e-4)),
            '--n_steps', str(s2.get('n_steps', 10000)),
            '--device', device,
            '--wandb_run_name', s2.get('wandb_run_name', 'stage2_lg')
        ]
        if args.check_datasets: cmd2.append('--save_dataset_samples')
        if no_wandb: cmd2.append('--no_wandb')

        run_stage_cmd(cmd2, "Stage 2: LightGlue Geometry-Safe Finetuning")
    else:
        print("\n⏭️  Stage 2 is disabled in config. Skipping.")

    # =====================================================================
    # Stage 3: End-to-End Joint Training
    # =====================================================================
    s3 = cfg.get('stage3', {})
    s3_out = os.path.join(output_root, "stage3_joint")

    s3_xfeat_weights = s3.get('xfeat_weights', s1_best_weights)
    s3_lg_weights = s3.get('lg_weights', s2_best_weights)
    
    if s3.get('enabled', False) and args.start_stage <= 3:
        if not os.path.exists(s3_xfeat_weights) or not os.path.exists(s3_lg_weights):
            print(f"❌ Error: Stage 1 または Stage 2 の Best重みが見つかりません。")
            sys.exit(1)

        cmd3 = [
            'python', 'train_joint_hybrid.py',
            '--xfeat_weights', s3_xfeat_weights,
            '--lg_weights', s3_lg_weights,
            '--vivid_root', roots.get('vivid', ''),
            '--ms2_root', roots.get('ms2', ''),
            '--tartanrgbt_root', roots.get('tartanrgbt', ''),
            '--freiburg_root', roots.get('freiburg', ''),
            '--sthereo_root', roots.get('sthereo', ''),
            '--output', s3_out,
            '--batch_size', str(s3.get('batch_size', 16)),
            '--lr', str(s3.get('lr', 1e-5)),
            '--n_steps', str(s3.get('n_steps', 5000)),
            '--lambda_match', str(s3.get('lambda_match', 0.5)),
            '--device', device,
            '--wandb_run_name', s3.get('wandb_run_name', 'stage3_joint')
        ]
        if args.check_datasets: cmd3.append('--save_dataset_samples')
        if no_wandb: cmd3.append('--no_wandb')

        run_stage_cmd(cmd3, "Stage 3: Joint End-to-End Training")
    else:
        print("\n⏭️  Stage 3 is disabled in config. Skipping.")

    print(f"\n🎉 すべてのパイプライン処理が正常に完了しました！ (Output: {output_root})")

if __name__ == '__main__':
    main()