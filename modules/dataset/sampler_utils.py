import torch
from torch.utils.data import WeightedRandomSampler, ConcatDataset

def parse_dataset_weights(weight_str: str):
    """
    コマンドライン引数の文字列 "ms2:0.1500,tartanrgbt:0.8500" をパースして辞書を返す
    """
    if not weight_str:
        return None
        
    weights = {}
    for item in weight_str.split(','):
        if ':' in item:
            name, w = item.split(':')
            weights[name.strip().lower()] = float(w)
    return weights

def build_weighted_sampler(dataset_list: list, dataset_names: list, weight_str: str):
    """
    複数のデータセットに対して、指定された重みに基づく WeightedRandomSampler を生成する
    
    Args:
        dataset_list: [ms2_dataset, tartanrgbt_dataset, ...] (Datasetオブジェクトのリスト)
        dataset_names: ['ms2', 'tartanrgbt', ...] (リストの順番に対応するデータセット名)
        weight_str: "ms2:0.1500,tartanrgbt:0.8500"
        
    Returns:
        WeightedRandomSampler (重み指定がない場合は None を返す)
    """
    weights_dict = parse_dataset_weights(weight_str)
    if not weights_dict:
        return None

    sample_weights = []
    
    # 各データセットをループし、その長さに応じて重みを配列に展開する
    for dataset, name in zip(dataset_list, dataset_names):
        name_lower = name.lower()
        
        # 引数で指定されたドメインの重みを取得（指定がない場合は0.0にして抽出させない）
        domain_weight = weights_dict.get(name_lower, 0.0)
        
        # そのデータセット内の全サンプルに対して同じ重みを割り当てる
        num_samples = len(dataset)
        sample_weights.extend([domain_weight] * num_samples)
        
    sample_weights_tensor = torch.DoubleTensor(sample_weights)
    
    # 🌟 研究者視点の工夫: 
    # replacement=True にすることで、重みの高い(枚数が少ない)データセットから
    # 1エポック中に何度も「復元抽出」されるようになり、完全なバランスが達成されます。
    sampler = WeightedRandomSampler(
        weights=sample_weights_tensor,
        num_samples=len(sample_weights_tensor), # 1エポックあたりの総ステップ数は元の合計枚数を維持
        replacement=True 
    )
    
    return sampler