# inference.py (使用建议)
import torch
import pandas as pd
import numpy as np
from torch import nn
import torch.nn.functional as F

# 必须把原来的模型类定义复制过来，或者 import 进来
class PrescriptionNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(PrescriptionNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 4096),
            nn.BatchNorm1d(4096),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(4096, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(2048, output_size)
        )
    def forward(self, x):
        return F.softplus(self.net(x))

def predict():
    # 1. 加载数据结构信息（为了对其维度）
    df_herb = pd.read_csv("通路-中药组合(top500)_NES矩阵.csv", index_col=0)
    input_size = df_herb.shape[0] # 16925
    output_size = df_herb.shape[1] # 500
    
    # 2. 加载并对齐疾病数据 (这里用之前修好的逻辑)
    df_disease_raw = pd.read_csv("disease_nes.csv", header=None, names=['pathway', 'nes'], dtype=str)
    df_disease_raw['nes'] = pd.to_numeric(df_disease_raw['nes'], errors='coerce')
    df_disease_raw.dropna(subset=['nes'], inplace=True)
    df_disease_raw = df_disease_raw.groupby('pathway')['nes'].mean()
    # 对齐！
    df_disease_aligned = df_disease_raw.reindex(df_herb.index, fill_value=0)
    
    # 3. 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PrescriptionNet(input_size, output_size).to(device)
    model.load_state_dict(torch.load("best_herb_model.pth", map_location=device))
    model.eval()
    
    # 4. 预测
    input_tensor = torch.tensor(df_disease_aligned.values, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        print("Input Tensor Max:", input_tensor.max().item())
        print("Input Tensor Min:", input_tensor.min().item())
        print("Input Tensor Sum:", input_tensor.sum().item())
        weights = model(input_tensor).cpu().numpy().flatten()
    
    # 5. 输出结果
    top_indices = weights.argsort()[-20:][::-1]
    print("Top 10 Recommended Herbs:")
    for idx in top_indices[:10]:
        print(f"{df_herb.columns[idx]}: {weights[idx]:.4f}")

if __name__ == "__main__":
    predict()