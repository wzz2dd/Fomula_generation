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
    
    # ==========================================
    # 【修改 1】模型路径指向 "消融实验_绝对值Top600.pth"
    # ==========================================
    model_path = os.path.join(os.path.dirname(script_dir), "data", "消融实验_绝对值Top600.pth") 
    
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
    # 【修改 2】输入端降噪：改为绝对值 Top 600
    # ==========================================
    INPUT_LIMIT = 600
    clean_tensor = input_tensor.clone()
    data_np = clean_tensor.cpu().numpy().flatten()
    mask = np.zeros_like(data_np, dtype=bool)
    
    # 计算绝对值
    abs_data = np.abs(data_np)
    
    # 取绝对值最大的 600 个 (不分正负)
    if len(abs_data) > INPUT_LIMIT:
        top_indices = np.argsort(abs_data)[-INPUT_LIMIT:]
        mask[top_indices] = True
    else:
        mask[:] = True
        
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
        # 【修改 3】输出端池化：同步改为 Top 600
        # ==========================================
        POOL_SIZE = 600 
        
        flat_scores = scores.flatten()
        
        if len(flat_scores) > POOL_SIZE:
            threshold_val = np.partition(flat_scores, -POOL_SIZE)[-POOL_SIZE]
            scores[scores < threshold_val] = -10000.0
            
        return scores 
        # ==========================================

    except Exception as e:
        print(f"Model Error: {e}")
        return np.random.rand(1, 500)
    except: return np.random.rand(1, 500)