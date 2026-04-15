"""
train_lightglue_gf.py
glue-factory を使った LightGlue Fine-tuning エントリーポイント。

XFeat が LightGlue（LighterGlue）を学習したのと同じ方式:
    - gluefactory の TwoViewPipeline
    - 合成ホモグラフィー + 熱画像コーパス
    - NegativeLogAssignment 損失
    - ThermalXFeat は frozen、LightGlue だけを学習

学習データ : Freiburg (train) + TartanRGBT (train)
評価データ : SThErEO, VIVID（一切使用しない）

事前準備:
    pip install git+https://github.com/cvg/glue-factory.git

使用方法:
    python train_lightglue_gf.py \\
        --config configs/lightglue_gf_config.yaml \\
        --experiment thermal_xfeat_lightglue
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


# ── CUDA_VISIBLE_DEVICES を import torch より前に設定 ──────────────────────
def _set_cuda_early() -> None:
    device_num = '0'
    for i, arg in enumerate(sys.argv):
        if arg == '--device_num' and i + 1 < len(sys.argv):
            device_num = sys.argv[i + 1]
            break
    if device_num == '0':
        for i, arg in enumerate(sys.argv):
            if arg == '--config' and i + 1 < len(sys.argv):
                try:
                    import yaml
                    cfg = yaml.safe_load(open(sys.argv[i+1])) or {}
                    device_num = str(cfg.get('device_num', '0'))
                except Exception:
                    pass
                break
    os.environ['CUDA_VISIBLE_DEVICES'] = device_num
    print(f'[Device] CUDA_VISIBLE_DEVICES={device_num}')


_set_cuda_early()
# ──────────────────────────────────────────────────────────────────────────

# プロジェクトルートを sys.path に追加（gf_modules を importable にする）
_REPO = os.path.dirname(os.path.abspath(__file__))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def _check_gluefactory() -> None:
    try:
        import gluefactory  # noqa: F401
    except ImportError:
        print('\n[ERROR] gluefactory が見つかりません。')
        print('  pip install git+https://github.com/cvg/glue-factory.git\n')
        sys.exit(1)


def _register_custom_modules() -> None:
    """
    gluefactory の動的ローダーに
    ThermalXFeat と ThermalHomographyDataset を登録する。

    gluefactory は model/dataset 名を Python モジュールパスとして解釈する。
    _REPO を sys.path に追加済みなので
    'gf_modules.models.thermal_xfeat.ThermalXFeat' として解決される。
    """
    # 事前インポートしてエラーを早期検出
    from gf_modules.models.thermal_xfeat import ThermalXFeat  # noqa: F401
    from gf_modules.datasets.thermal_homography import ThermalHomographyDataset  # noqa: F401
    print('[Register] ThermalXFeat OK')
    print('[Register] ThermalHomographyDataset OK')


def _patch_gluefactory_model_loader() -> None:
    """
    gluefactory の get_model / get_dataset が外部モジュールを
    ロードできるように動的ローダーをパッチする。
    """
    import importlib
    import gluefactory.models as gf_models
    import gluefactory.datasets as gf_datasets

    _orig_get_model = gf_models.get_model if hasattr(gf_models, 'get_model') else None

    def _patched_get_model(name: str):
        # gf_modules.* の場合は直接 importlib で解決
        if '.' in name:
            parts     = name.rsplit('.', 1)
            module    = importlib.import_module(parts[0])
            cls       = getattr(module, parts[1])
            return cls
        # それ以外は元の実装に委譲
        if _orig_get_model is not None:
            return _orig_get_model(name)
        raise ValueError(f'Model not found: {name}')

    # gluefactory の get_model を置き換え
    if hasattr(gf_models, 'get_model'):
        gf_models.get_model = _patched_get_model
        print('[Patch] gluefactory.models.get_model patched')

    _orig_get_dataset = gf_datasets.get_dataset if hasattr(gf_datasets, 'get_dataset') else None

    def _patched_get_dataset(name: str):
        if '.' in name:
            parts  = name.rsplit('.', 1)
            module = importlib.import_module(parts[0])
            cls    = getattr(module, parts[1])
            return cls
        if _orig_get_dataset is not None:
            return _orig_get_dataset(name)
        raise ValueError(f'Dataset not found: {name}')

    if hasattr(gf_datasets, 'get_dataset'):
        gf_datasets.get_dataset = _patched_get_dataset
        print('[Patch] gluefactory.datasets.get_dataset patched')


def _build_conf(config_path: str, overrides: list) -> object:
    """
    YAML と CLI オーバーライドから OmegaConf の設定を構築する。
    """
    from omegaconf import OmegaConf

    base = OmegaConf.load(config_path)

    # CLI オーバーライド: 'key=value' 形式
    cli  = OmegaConf.from_dotlist(overrides)
    conf = OmegaConf.merge(base, cli)
    return conf


def main() -> None:
    parser = argparse.ArgumentParser(
        description='LightGlue Fine-tuning with glue-factory',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--config', type=str,
        default='configs/lightglue_gf_config.yaml',
        help='設定 YAML のパス',
    )
    parser.add_argument(
        '--experiment', type=str,
        default='thermal_xfeat_lightglue',
        help='実験名（outputs/{name}/ に保存される）',
    )
    parser.add_argument(
        '--device_num', type=str, default=None,
        help='GPU 番号',
    )
    parser.add_argument(
        '--overrides', nargs='*', default=[],
        help='設定の上書き（例: train.batch_size=16）',
    )
    args = parser.parse_args()

    # gluefactory の確認
    _check_gluefactory()

    # カスタムモジュールの登録
    _register_custom_modules()
    _patch_gluefactory_model_loader()

    # 設定の構築
    if not os.path.isfile(args.config):
        print(f'[ERROR] config not found: {args.config}')
        sys.exit(1)

    conf = _build_conf(args.config, args.overrides or [])

    # 出力ディレクトリ
    output_dir = Path('outputs') / args.experiment
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f'[Train] Output dir: {output_dir}')
    print(f'[Train] Experiment: {args.experiment}')

    # ── gluefactory の訓練ループを使用 ───────────────────────────────────
    try:
        from gluefactory import train as gf_train

        # gluefactory の training() 関数を呼ぶ
        # glue-factory の CLI と同じ引数形式
        import types
        gf_args = types.SimpleNamespace(
            experiment        = args.experiment,
            conf              = conf,
            output_dir        = str(output_dir),
            distributed       = False,
            restore           = False,
            print_conf        = True,
            overwrite         = False,
        )

        # gluefactory の training 関数のシグネチャを確認して呼ぶ
        import inspect
        sig = inspect.signature(gf_train.training)
        params = list(sig.parameters.keys())

        if 'rank' in params:
            # 新しい API: training(rank, conf, output_dir, args)
            gf_train.training(
                rank       = 0,
                conf       = conf,
                output_dir = str(output_dir),
                args       = gf_args,
            )
        else:
            # 旧い API: training(conf, output_dir, args)
            gf_train.training(
                conf       = conf,
                output_dir = str(output_dir),
                args       = gf_args,
            )

    except (ImportError, AttributeError) as e:
        print(f'[WARNING] gluefactory.train API が異なります: {e}')
        print('[INFO] gluefactory の CLI を直接使用してください:')
        print(f'  python -m gluefactory.train {args.experiment} '
              f'--conf {args.config}')
        print()
        print('[INFO] または以下でも実行できます:')
        print(f'  bash scripts/train_lightglue_gf.sh')
        sys.exit(0)


if __name__ == '__main__':
    main()