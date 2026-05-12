# modules/matchers/factory.py
import sys
import torch
import torch.nn as nn
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
LGS_PATH = ROOT_DIR / "third_party" / "lightgluestick"
GSTICK_PATH = ROOT_DIR / "third_party" / "gluestick"

if str(LGS_PATH) not in sys.path:
    sys.path.insert(0, str(LGS_PATH))
if str(GSTICK_PATH) not in sys.path:
    sys.path.insert(0, str(GSTICK_PATH))

def build_matcher(config):
    """
    config['name'] に応じて Matcher を生成する Factory 関数。
    """
    matcher_name = config.get('name', 'lightgluestick').lower()

    if matcher_name == 'lightgluestick':
        # third_party/lightgluestick にパスが通っている前提
        from lightgluestick.lightgluestick import LightGlueStick
        
        # XFeatの64次元入力を256次元に射影する設定を強制上書き
        lgs_conf = {
            'input_dim': 64,
            'descriptor_dim': 256,
            'filter_threshold': config.get('filter_threshold', 0.1),
            'flash': config.get('flash', True), # RTX 5090なのでTrue推奨
        }
        # pre-trained weight を使う場合は config で指定
        if 'weights' in config:
            lgs_conf['weights'] = config['weights']
            
        print("[Matcher Factory] Initializing LightGlueStick...")
        return LightGlueStick(lgs_conf)

    elif matcher_name == 'gluestick':
        # third_party/gluestick にパスが通っている前提
        from gluestick.models.gluestick import GlueStick
        
        gs_conf = {
            'input_dim': 64,
            'descriptor_dim': 256,
            'filter_threshold': config.get('filter_threshold', 0.2),
        }
        if 'weights' in config:
            gs_conf['weights'] = config['weights']
            
        print("[Matcher Factory] Initializing GlueStick...")
        return GlueStick(gs_conf)

    else:
        raise ValueError(f"Unknown matcher name: {matcher_name}")