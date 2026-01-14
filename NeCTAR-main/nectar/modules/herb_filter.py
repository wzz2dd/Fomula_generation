import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
import dill
import os
import torch.nn as nn



class ResidualBlock(nn.Module):
    def __init__(self, size, dropout_p=0.4):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(size, size), nn.BatchNorm1d(size), nn.PReLU(),
            nn.Dropout(dropout_p), nn.Linear(size, size), nn.BatchNorm1d(size)
        )
        self.activation = nn.PReLU()
    def forward(self, x): return self.activation(x + self.block(x))

class AdvancedPredictor(nn.Module):
    def __init__(self, input_size=10014, output_size=500):
        super(AdvancedPredictor, self).__init__()
        self.entry = nn.Sequential(nn.Linear(input_size, 4096), nn.BatchNorm1d(4096), nn.PReLU(), nn.Dropout(0.3))
        self.compress = nn.Sequential(nn.Linear(4096, 2048), nn.BatchNorm1d(2048), nn.PReLU())
        self.res1 = ResidualBlock(2048)
        self.res2 = ResidualBlock(2048)
        self.res3 = ResidualBlock(2048)
        self.head = nn.Sequential(nn.Linear(2048, 1024), nn.PReLU(), nn.Linear(1024, output_size))
    def forward(self, x):
        x = self.entry(x)
        x = self.compress(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        return self.head(x)

def load_model(model_path, input_size, output_size, device):
    model = AdvancedPredictor(input_size=input_size, output_size=output_size).to(device)
    state_dict = torch.load(model_path, map_location=device)
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model

def preprocess_data(input_data):
    if isinstance(input_data, (pd.DataFrame, pd.Series)): input_data = input_data.values
    if isinstance(input_data, np.ndarray): input_tensor = torch.from_numpy(input_data).float()
    else: input_tensor = input_data.float()
    return input_tensor

def herb_filter(combined_nes):
    INPUT_DIM = 10014
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 注意：模型本身还是输出500维（因为权重矩阵没变），我们是在“推理后”做筛选
    model_path = os.path.join(os.path.dirname(script_dir), "data", "自适应Top400平衡.pth") 
    
    # 1. 数据预处理
    input_tensor = preprocess_data(combined_nes)
    if input_tensor.dim() == 1: input_tensor = input_tensor.unsqueeze(0)
    if input_tensor.shape[0] > 1 and input_tensor.shape[1] == 1: input_tensor = input_tensor.T

    # 维度对齐
    if input_tensor.shape[1] != INPUT_DIM:
        if input_tensor.shape[1] < INPUT_DIM:
            padding = torch.zeros(input_tensor.shape[0], INPUT_DIM - input_tensor.shape[1])
            input_tensor = torch.cat([input_tensor, padding], dim=1)
        else:
            input_tensor = input_tensor[:, :INPUT_DIM]

    # ==========================================
    # 【Step 1: 输入端的降噪】(保留你原有的逻辑，这很好)
    # 作用：只保留 NES 信号最强的 Top-400 通路，减少输入噪声
    # ==========================================
    INPUT_LIMIT = 400
    clean_tensor = input_tensor.clone()
    data_np = clean_tensor.cpu().numpy().flatten()
    mask = np.zeros_like(data_np, dtype=bool)
    
    # 正向特征保留 Top 400
    pos_idx = np.where(data_np > 0)[0]
    if len(pos_idx) > INPUT_LIMIT:
        top_pos = pos_idx[np.argsort(data_np[pos_idx])[-INPUT_LIMIT:]]
        mask[top_pos] = True
    else:
        mask[pos_idx] = True 
        
    # 负向特征保留 Top 400
    neg_idx = np.where(data_np < 0)[0]
    if len(neg_idx) > INPUT_LIMIT:
        top_neg = neg_idx[np.argsort(data_np[neg_idx])[:INPUT_LIMIT]]
        mask[top_neg] = True
    else:
        mask[neg_idx] = True
        
    clean_tensor[0, ~mask] = 0
    input_tensor = clean_tensor.to(device)
    # ==========================================

    # 2. 模型推理
    if not os.path.exists(model_path): return np.random.rand(1, 500)
    
    try:
        model = load_model(model_path, input_size=INPUT_DIM, output_size=500, device=device)
        with torch.no_grad(): 
            logits = model(input_tensor) # 此时输出的是 [1, 500] 的原始分数
            scores = logits.cpu().numpy()

        # ==========================================
        # 【Step 2: 输出端的动态 Top-400 池】(这是你要新增的核心修改)
        # 作用：从400个预测结果中，只保留得分最高的400个，其余强制归零或设为负无穷
        # ==========================================
        POOL_SIZE = 400 # 这里是调整超参数的地方
        
        flat_scores = scores.flatten()
        
        # 找到第 500 大的分数作为阈值
        # np.partition 比 sort 快，适合找 Top K
        if len(flat_scores) > POOL_SIZE:
            # 找到第 -POOL_SIZE 位置的元素，它左边都比它小，右边都比它大
            threshold_val = np.partition(flat_scores, -POOL_SIZE)[-POOL_SIZE]
            
            # 核心逻辑：低于阈值的药物，分数设为 -10000 (代表极度不推荐)
            # 这样在后续排序中它们会直接沉底，实际上就等于被剔除出了候选池
            scores[scores < threshold_val] = -10000.0
            
        return scores # 返回修改后的分数矩阵
        # ==========================================

    except Exception as e:
        print(f"Model Error: {e}")
        return np.random.rand(1, 500)
    except: return np.random.rand(1, 500)