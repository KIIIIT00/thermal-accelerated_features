# # """
# # train_pipeline_hybrid.py
# # 統合コンフィグ (config_master.yaml) を用いたマルチステージ学習オーケストレーター
# # """

# # import yaml
# # import argparse
# # import subprocess
# # import os
# # import sys

# # def run_stage_cmd(cmd, desc):
# #     print(f"\n{'='*80}")
# #     print(f"🚀 開始: {desc}")
# #     print(f"実行コマンド:\n  {' '.join(cmd)}")
# #     print(f"{'='*80}")
# #     ret = subprocess.run(cmd).returncode
# #     if ret != 0:
# #         print(f"\n❌ 失敗: {desc} (エラーコード: {ret})")
# #         sys.exit(ret)
# #     print(f"\n✅ 成功: {desc}\n")

# # def main():
# #     parser = argparse.ArgumentParser(description="Master Pipeline Runner")
# #     parser.add_argument('--config', type=str, default='config_master.yaml', help='マスター設定ファイルのパス')
# #     args = parser.parse_args()

# #     # マスターコンフィグの読み込み
# #     if not os.path.exists(args.config):
# #         print(f"❌ エラー: コンフィグファイル {args.config} が見つかりません。")
# #         sys.exit(1)
        
# #     with open(args.config, 'r', encoding='utf-8') as f:
# #         cfg = yaml.safe_load(f)

# #     common = cfg['common']
# #     output_root = common['output_root']
# #     roots = common['data_roots']
# #     os.makedirs(output_root, exist_ok=True)
# #     os.makedirs('configs/auto_generated', exist_ok=True) # Stage 2,3 用のYAML保存先

# #     # =====================================================================
# #     # Stage 1: 特徴抽出器のKD学習
# #     # =====================================================================
# #     s1 = cfg['stage1']
# #     s1_out = os.path.join(output_root, "stage1_kd")
# #     s1_final_weights = os.path.join(s1_out, "diagnostic_model.pth") # Stage1の最終出力
    
# #     if s1['enabled']:
# #         cmd = [
# #             sys.executable, 'train_kd_hybrid.py',
# #             '--output', s1_out,
# #             '--epochs', str(s1['epochs']),
# #             '--batch_size', str(s1['batch_size']),
# #             '--lr', str(s1['lr']),
# #             '--lambda_kd', str(s1['lambda_kd']),
# #             '--lambda_hybrid', str(s1['lambda_hybrid']),
# #             '--lambda_spatial', str(s1['lambda_spatial']),
# #             '--tau_fixed', str(s1['tau_fixed']),
# #             '--device', str(common['device']),
# #             '--wandb_run_name', s1['wandb_run_name']
# #         ]
        
# #         # データセットパスの付与
# #         if 'sthereo' in s1['datasets']: cmd.extend(['--sthereo_root', roots['sthereo']])
# #         if 'ms2' in s1['datasets']: cmd.extend(['--ms2_root', roots['ms2']])
# #         if 'vivid' in s1['datasets']: cmd.extend(['--vivid_root', roots['vivid']])
# #         if 'tartanrgbt' in s1['datasets']: cmd.extend(['--tartanrgbt_root', roots['tartanrgbt']])
# #         if 'freiburg' in s1['datasets']: cmd.extend(['--freiburg_root', roots['freiburg']])
        
# #         # 🔬 アブレーションフラグの付与
# #         if s1['no_kd_loss']: cmd.append('--no_kd_loss')
# #         if s1['use_hybrid_loss']: cmd.append('--use_hybrid_loss')
# #         if s1['use_spatial_loss']: cmd.append('--use_spatial_loss')
# #         if common['no_wandb']: cmd.append('--no_wandb')

# #         run_stage_cmd(cmd, "Stage 1: Feature Extractor KD (XFeat)")

# #     # =====================================================================
# #     # Stage 2: マッチャーのドメイン適応 (LightGlue FT)
# #     # =====================================================================
# #     s2 = cfg['stage2']
# #     s2_out = os.path.join(output_root, "stage2_lg")
# #     s2_final_weights = os.path.join(s2_out, "lightglue_gf_final.pth")
    
# #     if s2['enabled']:
# #         if not os.path.exists(s1_final_weights):
# #             print(f"❌ エラー: Stage 1 の重みが見つかりません: {s1_final_weights}")
# #             sys.exit(1)
            
# #         # Stage 2用のYAMLを動的生成
# #         s2_yaml_path = 'configs/auto_generated/stage2_lg.yaml'
# #         s2_cfg = {
# #             'thermal_weights': s1_final_weights,
# #             'ckpt_save_path': s2_out,
# #             'ft_datasets': s2['datasets'],
# #             'data_roots': {k: roots[k] for k in s2['datasets']},
# #             'batch_size': s2['batch_size'],
# #             'lr': s2['lr'],
# #             'n_steps': s2['n_steps'],
# #             'max_keypoints': s2['max_keypoints'],
# #             'wandb_project': common['wandb_project'],
# #             'wandb_run_name': s2['wandb_run_name'],
# #             'no_wandb': common['no_wandb']
# #         }
# #         with open(s2_yaml_path, 'w') as f: yaml.dump(s2_cfg, f)
        
# #         cmd = [sys.executable, 'train_lightglue_ft.py', '--config', s2_yaml_path, '--device_num', str(common['device'])]
# #         run_stage_cmd(cmd, "Stage 2: LightGlue Fine-tuning")

# #     # =====================================================================
# #     # Stage 3: エンドツーエンド同時学習 (Joint FT)
# #     # =====================================================================
# #     s3 = cfg['stage3']
# #     s3_out = os.path.join(output_root, "stage3_joint")
    
# #     if s3['enabled']:
# #         if not os.path.exists(s2_final_weights):
# #             print(f"❌ エラー: Stage 2 の重みが見つかりません: {s2_final_weights}")
# #             sys.exit(1)
            
# #         # Stage 3用のYAMLを動的生成
# #         s3_yaml_path = 'configs/auto_generated/stage3_joint.yaml'
# #         s3_cfg = {
# #             'xfeat_weights': s1_final_weights,
# #             'lg_weights': s2_final_weights,
# #             'ckpt_save_path': s3_out,
# #             'ft_datasets': s3['datasets'],
# #             'data_roots': {k: roots[k] for k in s3['datasets']},
# #             'batch_size': s3['batch_size'],
# #             'lr': s3['lr'],
# #             'lambda_match': s3['lambda_match'],
# #             'n_steps': s3['n_steps'],
# #             'wandb_project': common['wandb_project'],
# #             'wandb_run_name': s3['wandb_run_name'],
# #             'no_wandb': common['no_wandb']
# #         }
# #         with open(s3_yaml_path, 'w') as f: yaml.dump(s3_cfg, f)
        
# #         cmd = [sys.executable, 'train_joint.py', '--config', s3_yaml_path, '--device_num', str(common['device'])]
# #         run_stage_cmd(cmd, "Stage 3: End-to-End Joint Fine-tuning")

# #     print(f"\n🎉 全ての有効なパイプラインステージが正常に完了しました！")
# #     print(f"最終的な重みは {output_root} 以下の各ディレクトリに保存されています。")

# # if __name__ == '__main__':
# #     main()

# """
# train_pipeline_hybrid.py (Research Grade)
# 統合設定に基づき、Stage 1～3 の学習を統制するマスター・オーケストレーター。
# 解決策B（動的パディングによる幾何学保護）と、AnyThermal パス解決を完全統合。
# """

# import yaml
# import argparse
# import subprocess
# import os
# import sys

# def run_stage_cmd(cmd, desc):
#     print(f"\n{'='*80}")
#     print(f"🚀 [START]: {desc}")
#     print(f"Executing command:\n  {' '.join(cmd)}")
#     print(f"{'='*80}")
#     ret = subprocess.run(cmd).returncode
#     if ret != 0:
#         print(f"\n❌ [FAILED]: {desc} (Error Code: {ret})")
#         sys.exit(ret)
#     print(f"\n✅ [SUCCESS]: {desc}\n")

# def main():
#     parser = argparse.ArgumentParser(description="Thermal-XFeat Master Pipeline")
#     parser.add_argument('--config', type=str, default='config_master.yaml', help='Path to config_master.yaml')
#     args = parser.parse_args()

#     # --- 1. 設定のロードとディレクトリ準備 ---
#     if not os.path.exists(args.config):
#         print(f"❌ Error: Config {args.config} not found.")
#         sys.exit(1)
        
#     with open(args.config, 'r', encoding='utf-8') as f:
#         cfg = yaml.safe_load(f)

#     common = cfg['common']
#     output_root = common['output_root']
#     roots = common['data_roots']
#     os.makedirs(output_root, exist_ok=True)
#     os.makedirs('configs/auto_generated', exist_ok=True)

#     # AnyThermal スプリットのベースパス
#     any_base = 'third_party/anythermal/custom_datasets'

#     # =====================================================================
#     # Stage 1: XFeat 知識蒸留 (KD)
#     # =====================================================================
#     s1 = cfg['stage1']
#     s1_out = os.path.join(output_root, "stage1_kd")
#     s1_final_weights = os.path.join(s1_out, "model_xfeat_final.pth")

#     if s1['enabled']:
#         cmd1 = [
#             'python', 'train_kd_hybrid.py',
#             '--sthereo_root', roots['sthereo'],
#             '--sthereo_splits', os.path.join(any_base, 'sthereo', 'splits', 'frame_lists'),
#             '--ms2_root', roots['ms2'],
#             '--ms2_splits', os.path.join(any_base, 'ms2', 'splits', 'frame_lists'),
#             '--vivid_root', roots['vivid'],
#             '--vivid_splits', os.path.join(any_base, 'vivid', 'splits', 'frame_lists'),
#             '--tartanrgbt_root', roots['tartanrgbt'],
#             '--tartanrgbt_splits', os.path.join(any_base, 'tartanRGBT', 'splits'),
#             '--freiburg_root', roots['freiburg'],
#             '--freiburg_splits', os.path.join(any_base, 'freiburg', 'splits', 'frame_list'),
#             '--output', s1_out,
#             '--dataset_choice', s1['dataset_choice'],
#             '--batch_size', str(s1['batch_size']),
#             '--epochs', str(s1['epochs']),
#             '--lr', str(s1['lr']),
#             '--lambda_hybrid', str(s1['lambda_hybrid']),
#             '--wandb_project', common['wandb_project'],
#             '--wandb_run_name', s1['wandb_run_name']
#         ]
#         run_stage_cmd(cmd1, "Stage 1: XFeat Knowledge Distillation")

#     # =====================================================================
#     # Stage 2: LightGlue ファインチューニング (FT)
#     # =====================================================================
#     s2 = cfg['stage2']
#     s2_out = os.path.join(output_root, "stage2_lightglue")
#     s2_final_weights = os.path.join(s2_out, "model_lightglue_final.pth")

#     if s2['enabled']:
#         # 幾何学的整合性を守るため、解決策B(パディング)を使用することを暗示
#         if not os.path.exists(s1_final_weights) and not s1['enabled']:
#              print(f"⚠️ Warning: Stage 1 weights not found at {s1_final_weights}. Using pretrained XFeat.")

#         # Stage 2用の引数構築 (データセットパスも追加)
#         cmd2 = [
#             'python', 'train_lg_ft_hybrid.py',  # 幾何学安全版のスクリプトを想定
#             '--xfeat_weights', s1_final_weights if os.path.exists(s1_final_weights) else "pretrained",
#             '--sthereo_root', roots['sthereo'],
#             '--ms2_root', roots['ms2'],
#             '--vivid_root', roots['vivid'],
#             '--freiburg_root', roots['freiburg'],
#             '--output', s2_out,
#             '--batch_size', str(s2['batch_size']),
#             '--lr', str(s2['lr']),
#             '--n_steps', str(s2['n_steps']),
#             '--wandb_project', common['wandb_project'],
#             '--wandb_run_name', s2['wandb_run_name']
#         ]
#         run_stage_cmd(cmd2, "Stage 2: LightGlue Geometry-Safe Finetuning")

#     # =====================================================================
#     # Stage 3: エンドツーエンド同時学習 (Joint FT)
#     # =====================================================================
#     s3 = cfg['stage3']
#     s3_out = os.path.join(output_root, "stage3_joint")
    
#     if s3['enabled']:
#         if not os.path.exists(s2_final_weights):
#             print(f"❌ Error: Stage 2 weights not found: {s2_final_weights}")
#             sys.exit(1)
            
#         # Stage 3用のYAMLを動的生成 (解決策Bの反映)
#         s3_yaml_path = 'configs/auto_generated/stage3_joint.yaml'
#         s3_cfg = {
#             'xfeat_weights': s1_final_weights,
#             'lg_weights': s2_final_weights,
#             'ckpt_save_path': s3_out,
#             'ft_datasets': s3['datasets'],
#             'data_roots': {k: roots[k] for k in s3['datasets']},
#             # スプリットパスも注入
#             'splits_dirs': {
#                 'sthereo': os.path.join(any_base, 'sthereo', 'splits', 'frame_lists'),
#                 'ms2': os.path.join(any_base, 'ms2', 'splits', 'frame_lists'),
#                 'vivid': os.path.join(any_base, 'vivid', 'splits', 'frame_lists'),
#                 'freiburg': os.path.join(any_base, 'freiburg', 'splits', 'frame_list'),
#             },
#             'batch_size': s3['batch_size'],
#             'lr': s3['lr'],
#             'lambda_match': s3['lambda_match'],
#             'n_steps': s3['n_steps'],
#             'wandb_project': common['wandb_project'],
#             'wandb_run_name': s3['wandb_run_name'],
#             'no_wandb': common['no_wandb']
#         }
#         with open(s3_yaml_path, 'w') as f:
#             yaml.dump(s3_cfg, f, default_flow_style=False)
        
#         cmd3 = ['python', 'train_joint_hybrid.py', '--config', s3_yaml_path]
#         run_stage_cmd(cmd3, "Stage 3: Joint End-to-End Training")

# if __name__ == '__main__':
#     main()

"""
train_pipeline_hybrid.py (Master Pipeline)
Stage 1～3 の学習を統制するマスター・オーケストレーター。
"""
import yaml
import argparse
import subprocess
import os
import sys

def run_stage_cmd(cmd, desc):
    print(f"\n{'='*80}")
    print(f"🚀 [START]: {desc}")
    print(f"Executing command:\n  {' '.join(cmd)}")
    print(f"{'='*80}")
    ret = subprocess.run(cmd).returncode
    if ret != 0:
        print(f"\n❌ [FAILED]: {desc} (Error Code: {ret})")
        sys.exit(ret)
    print(f"\n✅ [SUCCESS]: {desc}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config_master.yaml')
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"❌ Error: Config {args.config} not found.")
        sys.exit(1)
        
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    common = cfg['common']
    output_root = common['output_root']
    roots = common['data_roots']
    os.makedirs(output_root, exist_ok=True)
    os.makedirs('configs/auto_generated', exist_ok=True)

    any_base = 'third_party/anythermal/custom_datasets'

    # =====================================================================
    # Stage 1: XFeat Knowledge Distillation
    # =====================================================================
    s1 = cfg.get('stage1', {})
    s1_out = os.path.join(output_root, "stage1_kd")
    s1_final_weights = os.path.join(s1_out, "diagnostic_model.pth")

    if s1.get('enabled', False):
        cmd1 = [
            'python', 'train_kd_hybrid.py',
            '--sthereo_root', roots['sthereo'],
            '--ms2_root', roots['ms2'],
            '--vivid_root', roots['vivid'],
            '--tartanrgbt_root', roots['tartanrgbt'],
            '--freiburg_root', roots['freiburg'],
            '--output', s1_out,
            '--dataset_choice', 'all',
            '--batch_size', str(s1['batch_size']),
            '--epochs', str(s1['epochs']),
            '--lr', str(s1['lr']),
            '--device', str(common['device']),
            '--wandb_run_name', s1['wandb_run_name']
        ]
        
        # 🎯 データ拡張 (Ablation) の引数を追加
        aug_cfg = s1.get('augmentation', {})
        if aug_cfg.get('enabled', False):
            methods_str = ",".join(aug_cfg.get('methods', []))
            crop_h, crop_w = aug_cfg.get('crop_size', [256, 256])
            cmd1.extend([
                '--aug_list', methods_str,
                '--crop_size', f"{crop_h},{crop_w}"
            ])

        run_stage_cmd(cmd1, "Stage 1: XFeat Knowledge Distillation")

    # =====================================================================
    # Stage 2: LightGlue Geometry-Safe Finetuning
    # =====================================================================
    s2 = cfg.get('stage2', {})
    s2_out = os.path.join(output_root, "stage2_lightglue")
    s2_final_weights = os.path.join(s2_out, "lg_stage2_final.pth")

    if s2.get('enabled', False):
        cmd2 = [
            'python', 'train_lg_ft_hybrid.py',
            '--xfeat_weights', s1_final_weights,
            '--sthereo_root', roots['sthereo'],
            '--vivid_root', roots['vivid'],
            '--tartanrgbt_root', roots['tartanrgbt'],
            '--output', s2_out,
            '--batch_size', str(s2['batch_size']),
            '--lr', str(s2['lr']),
            '--n_steps', str(s2['n_steps']),
            '--device', str(common['device']),
            '--wandb_run_name', s2['wandb_run_name']
        ]
        run_stage_cmd(cmd2, "Stage 2: LightGlue Geometry-Safe Finetuning")

    # =====================================================================
    # Stage 3: End-to-End Joint Training
    # =====================================================================
    s3 = cfg.get('stage3', {})
    s3_out = os.path.join(output_root, "stage3_joint")
    
    if s3.get('enabled', False):
        cmd3 = [
            'python', 'train_joint_hybrid.py',
            '--xfeat_weights', s1_final_weights,
            '--lg_weights', s2_final_weights,
            '--sthereo_root', roots['sthereo'],
            '--vivid_root', roots['vivid'],
            '--output', s3_out,
            '--batch_size', str(s3['batch_size']),
            '--lr', str(s3['lr']),
            '--n_steps', str(s3['n_steps']),
            '--device', str(common['device']),
            '--wandb_run_name', s3['wandb_run_name']
        ]
        run_stage_cmd(cmd3, "Stage 3: Joint End-to-End Training")

if __name__ == '__main__':
    main()