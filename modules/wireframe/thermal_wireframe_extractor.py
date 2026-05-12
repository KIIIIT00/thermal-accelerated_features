# modules/wireframe/thermal_wireframe_extractor.py
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import DBSCAN
import pytlsd # pip install pytlsd が必要です

from modules.xfeat import XFeat

def sample_descriptors(keypoints, dense_descriptors, scale_factor=8):
    """
    密な特徴マップ(1/8解像度)から、指定された座標の特徴量を双一次補間でサンプリングする。
    keypoints: [B, N, 2] (元の画像座標)
    dense_descriptors: [B, C, H_feat, W_feat]
    """
    b, c, h_f, w_f = dense_descriptors.shape
    # グリッドサンプリングのために座標を [-1, 1] に正規化
    # 特徴マップのサイズに対してスケーリング
    norm_kpts = keypoints / (keypoints.new_tensor([w_f * scale_factor, h_f * scale_factor]))
    norm_kpts = norm_kpts * 2.0 - 1.0 # [0, 1] -> [-1, 1]
    
    # grid_sample は [B, C, H, W] と [B, H_out, W_out, 2] を取る
    grid = norm_kpts.view(b, 1, -1, 2)
    sampled_desc = F.grid_sample(dense_descriptors, grid, mode='bilinear', align_corners=False)
    
    # [B, C, 1, N] -> [B, N, C]
    sampled_desc = sampled_desc.squeeze(2).transpose(1, 2)
    # 再度L2正規化
    sampled_desc = F.normalize(sampled_desc, p=2, dim=-1)
    return sampled_desc

class ThermalWireframeExtractor(nn.Module):
    """
    ThermalXFeat (点・密特徴) と pytlsd (線分) を統合し、
    LightGlueStick が要求するパディング済みの Dict[Tensor] を出力するラッパー。
    """
    def __init__(self, xfeat_weights, max_keypoints=1024, max_lines=250, min_line_length=15, nms_radius=3):
        super().__init__()
        # XFeatの初期化 (前回の拡張が反映されている前提)
        self.xfeat = XFeat(weights=xfeat_weights, top_k=max_keypoints)
        self.max_keypoints = max_keypoints
        self.max_lines = max_lines
        self.min_line_length = min_line_length
        self.nms_radius = nms_radius

    def extract_lines_lsd(self, images_tensor):
        """ pytlsdを用いてバッチ内の各画像から線分を抽出する """
        b_size = images_tensor.shape[0]
        device = images_tensor.device
        
        lines_list, scores_list = [], []
        
        for b in range(b_size):
            # テンソル [1, H, W] -> numpy [H, W] の uint8 に変換
            img_np = (images_tensor[b].squeeze().cpu().numpy() * 255).astype(np.uint8)
            
            # LSD実行
            img_np = cv2.GaussianBlur(img_np, (5, 5), 1.5) # ブラーを追加
            segs = pytlsd.lsd(img_np)
            if len(segs) == 0:
                lines_list.append(torch.zeros((0, 2, 2), device=device))
                scores_list.append(torch.zeros((0,), device=device))
                continue
                
            # 長さでフィルタリング
            lengths = np.linalg.norm(segs[:, 2:4] - segs[:, 0:2], axis=1)
            valid = lengths >= self.min_line_length
            segs = segs[valid]
            lengths = lengths[valid]
            
            # スコアの計算 (LSDの信頼度 * 長さの平方根 等のヒューリスティック)
            scores = segs[:, -1] * np.sqrt(lengths)
            
            # 上位 max_lines 個を選択
            if len(segs) > self.max_lines:
                idx = np.argsort(-scores)[:self.max_lines]
                segs = segs[idx]
                scores = scores[idx]
                
            # フォーマット整形: [N, 4] -> [N, 2, 2]
            segs_tensor = torch.from_numpy(segs[:, :4].reshape(-1, 2, 2)).to(device).float()
            scores_tensor = torch.from_numpy(scores).to(device).float()
            
            lines_list.append(segs_tensor)
            scores_list.append(scores_tensor)
            
        return lines_list, scores_list

    @torch.inference_mode()
    def forward(self, images):
        """
        images: [B, 1, H, W] (グレースケールの熱画像テンソル)
        """
        device = images.device
        b_size = images.shape[0]
        
        # 1. ThermalXFeat による点と密な特徴の抽出
        # 戻り値は List[Dict] を想定
        xfeat_outs = self.xfeat.detectAndCompute(images, top_k=self.max_keypoints)
        
        # 2. pytlsd による線分抽出
        lines_list, line_scores_list = self.extract_lines_lsd(images)
        
        out_keypoints, out_scores, out_descriptors = [], [], []
        out_lines, out_line_scores, out_junc_idx = [], [], []
        
        # 3. 画像ごとに点と線を統合し、端点(Junction)の特徴をサンプリング
        for b in range(b_size):

            
            kpts = xfeat_outs[b]['keypoints']      # [Nk, 2]
            kpt_sc = xfeat_outs[b]['scores']       # [Nk]
            kpt_desc = xfeat_outs[b]['descriptors']# [Nk, 64]
            dense_desc = xfeat_outs[b]['dense_descriptors'].unsqueeze(0) # [1, 64, H/8, W/8]

            margin = 5
            h_img, w_img = images.shape[2], images.shape[3]

            # 有効な点のマスクを作成
            valid_mask = (kpts[:, 0] > margin) & (kpts[:, 0] < w_img - margin) & \
                         (kpts[:, 1] > margin) & (kpts[:, 1] < h_img - margin)

            kpts = kpts[valid_mask]
            kpt_sc = kpt_sc[valid_mask]
            kpt_desc = kpt_desc[valid_mask]

            lines = lines_list[b]          # [Nl, 2, 2]
            line_sc = line_scores_list[b]  # [Nl]
            num_lines = lines.shape[0]
            
            if num_lines > 0:
                # 線の端点を1Dに展開 [Nl*2, 2]
                endpoints = lines.reshape(-1, 2)
                
                # DBSCANで近接する端点をマージ (Junctionの生成)
                db = DBSCAN(eps=self.nms_radius, min_samples=1).fit(endpoints.cpu().numpy())
                clusters = db.labels_
                n_clusters = len(set(clusters))
                
                # クラスターごとの平均座標を計算して新しいJunctionとする
                clusters_t = torch.tensor(clusters, dtype=torch.long, device=device)
                junctions = torch.zeros((n_clusters, 2), dtype=torch.float, device=device)
                junctions.scatter_reduce_(0, clusters_t.unsqueeze(1).expand(-1, 2), endpoints, reduce='mean', include_self=False)
                
                # トポロジー情報 (どの線がどのJunctionに繋がっているか)
                lines_junc_idx = clusters_t.reshape(-1, 2) # [Nl, 2]
                
                # Junctionの特徴量を密マップからサンプリング
                junc_desc = sample_descriptors(junctions.unsqueeze(0), dense_desc, scale_factor=8).squeeze(0) # [Nj, 64]
                
                # Junctionのスコア (対応する線分の平均スコア)
                junc_sc = torch.zeros(n_clusters, dtype=torch.float, device=device)
                junc_sc.scatter_reduce_(0, clusters_t, line_sc.repeat_interleave(2), reduce='mean', include_self=False)
                
            else:
                junctions = torch.zeros((1, 2), dtype=torch.float, device=device)
                junc_desc = torch.zeros((1, 64), dtype=torch.float, device=device)
                junc_sc = torch.zeros((1,), dtype=torch.float, device=device)
                lines_junc_idx = torch.zeros((1, 2), dtype=torch.long, device=device)
                
            # --- 点(Keypoints)と端点(Junctions)の結合 ---
            combined_kpts = torch.cat([junctions, kpts], dim=0)
            combined_sc = torch.cat([junc_sc, kpt_sc], dim=0)
            combined_desc = torch.cat([junc_desc, kpt_desc], dim=0)
            
            out_keypoints.append(combined_kpts)
            out_scores.append(combined_sc)
            out_descriptors.append(combined_desc)
            out_lines.append(lines)
            out_line_scores.append(line_sc)
            out_junc_idx.append(lines_junc_idx)
            
        # 4. バッチ化のためのゼロパディング処理
        # LightGlueStickに渡すために、最大要素数で揃えて [B, N, ...] のテンソルにする
        def pad_tensors(tensor_list, pad_val=0):
            if not tensor_list: return torch.empty(0)
            # 各テンソルの最初の次元（要素数）の最大値を取得
            max_len = max([t.shape[0] for t in tensor_list])
            
            # 🌟 修正1: 最大要素数が0（全画像で特徴ゼロ）の場合の安全策
            if max_len == 0:
                # 少なくとも1つの要素があるように見せかける（後段のクラッシュ防止）
                max_len = 1

            padded = []
            for t in tensor_list:
                pad_size = max_len - t.shape[0]
                if pad_size > 0:
                    pad_shape = list(t.shape)
                    pad_shape[0] = pad_size
                    # 🌟 修正2: インデックスのパディングは 0 に統一
                    pad_tensor = torch.full(pad_shape, pad_val, dtype=t.dtype, device=t.device)
                    padded.append(torch.cat([t, pad_tensor], dim=0))
                else:
                    padded.append(t)
            return torch.stack(padded, dim=0)

        # 🌟 修正3: Junctionが0個のケースを救済するためのダミーデータの挿入
        # 各画像の Junction が空の場合、[0, 0] 座標のダミーを1つ入れることで
        # LightGlueStick内部の「サイズ0」によるクラッシュを物理的に防ぐ
        for i in range(len(out_keypoints)):
            if out_keypoints[i].shape[0] == 0:
                out_keypoints[i] = torch.zeros((1, 2), device=device)
                out_scores[i] = torch.zeros((1,), device=device)
                out_descriptors[i] = torch.zeros((1, 64), device=device)
            if out_junc_idx[i].shape[0] == 0:
                out_junc_idx[i] = torch.zeros((1, 2), dtype=torch.long, device=device)

        # パディングの実行
        batched_data = {
            'keypoints': pad_tensors(out_keypoints, pad_val=0.0),
            'keypoint_scores': pad_tensors(out_scores, pad_val=0.0),
            'descriptors': pad_tensors(out_descriptors, pad_val=0.0),
            'lines': pad_tensors(out_lines, pad_val=0.0),
            'line_scores': pad_tensors(out_line_scores, pad_val=0.0),
            # 🌟 重要: インデックスのパディングは 0 で行う。
            # 全ての画像が最低1つの Junction (index 0) を持つようになったため、
            # pad_val=0 は常に安全（範囲内）になります。
            'lines_junc_idx': pad_tensors(out_junc_idx, pad_val=0) 
        }
        
        return batched_data