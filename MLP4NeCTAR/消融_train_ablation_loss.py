import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os

# ==========================================
# 辅助：动态 Top-300 掩码 (保持与主模型一致，控制变量)
# ==========================================
def apply_dynamic_mask_torch(vector, limit=300):
    mask = torch.zeros_like(vector, dtype=torch.bool)
    
    # 1. 正向部分
    pos_mask = vector > 0
    pos_indices = torch.nonzero(pos_mask).squeeze()
    if pos_indices.numel() > limit:
        pos_values = vector[pos_indices]
        _, top_k_indices = torch.topk(pos_values, limit)
        mask[pos_indices[top_k_indices]] = True
    else:
        mask[pos_indices] = True

    # 2. 负向部分
    neg_mask = vector < 0
    neg_indices = torch.nonzero(neg_mask).squeeze()
    if neg_indices.numel() > limit:
        neg_values = vector[neg_indices]
        _, top_k_indices = torch.topk(neg_values, limit, largest=False)
        mask[neg_indices[top_k_indices]] = True
    else:
        mask[neg_indices] = True
        
    return vector * mask

# ==========================================
# 主程序
# ==========================================
def main():
    # 1. 单卡设置
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print(f"🔥 [Ablation] Loss Ablation Experiment (Single GPU): {torch.cuda.get_device_name(0)}")
    else:
        raise RuntimeError("❌ Error: GPU not found!")
    
    seed = 2025
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    data_path = "通路-中药组合_NES矩阵.csv" 
    print(f"📚 [Data] Loading Matrix from {data_path}...")
    
    try:
        df = pd.read_csv(data_path, index_col=0)
    except:
        df = pd.read_csv(data_path, sep='\t', index_col=0)
    df.fillna(0, inplace=True)
    
    if df.shape[0] > df.shape[1]: 
        print("⚠️ Detected [Rows < Cols], transposing...")
        df = df.T

    herb_matrix_tensor = torch.tensor(df.values.astype(np.float32)).to(device)
    num_herbs, num_pathways = herb_matrix_tensor.shape

    # 2. 全 GPU Dataset
    class AdversarialDatasetGPU(Dataset):
        def __init__(self, herb_matrix, num_samples=60000, max_mix=12):
            self.herb_matrix = herb_matrix 
            self.num_samples = num_samples
            self.max_mix = max_mix
            self.num_herbs = herb_matrix.shape[0]
            self.num_features = herb_matrix.shape[1]

        def __len__(self): return self.num_samples

        def __getitem__(self, idx):
            k = torch.randint(1, self.max_mix + 1, (1,), device=self.herb_matrix.device).item()
            indices = torch.randperm(self.num_herbs, device=self.herb_matrix.device)[:k]
            coeffs = torch.rand(k, device=self.herb_matrix.device) * (3.0 - 0.3) + 0.3
            
            selected_vectors = self.herb_matrix[indices]
            clean_effect = torch.matmul(coeffs, selected_vectors)
            
            raw_disease = -1.0 * clean_effect
            masked_disease = apply_dynamic_mask_torch(raw_disease, limit=300)
            
            max_val = torch.max(torch.abs(masked_disease))
            if max_val == 0: max_val = 1.0
            
            normalized_disease = masked_disease / max_val
            real_amplitude = torch.rand(1, device=self.herb_matrix.device) * (5.0 - 1.5) + 1.5
            final_input = normalized_disease * real_amplitude
            
            global_noise = torch.randn(self.num_features, device=self.herb_matrix.device) * 0.1
            final_input += global_noise
            
            target = torch.ones(self.num_herbs, device=self.herb_matrix.device) * 0.01
            target[indices] = 0.99 
            
            return final_input, target, clean_effect

    # 3. 模型定义
    class ResidualBlock(nn.Module):
        def __init__(self, size, dropout_p=0.4):
            super(ResidualBlock, self).__init__()
            self.block = nn.Sequential(
                nn.Linear(size, size), nn.BatchNorm1d(size), nn.PReLU(),
                nn.Dropout(dropout_p), nn.Linear(size, size), nn.BatchNorm1d(size),
            )
            self.activation = nn.PReLU()
        def forward(self, x): return self.activation(x + self.block(x))

    class AdvancedPredictor(nn.Module):
        def __init__(self, input_size, output_size):
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

    # 4. 训练流程
    print("🚀 Initializing Training (Loss Ablation Mode)...")
    
    train_dataset = AdversarialDatasetGPU(herb_matrix_tensor, num_samples=60000, max_mix=12)
    test_dataset = AdversarialDatasetGPU(herb_matrix_tensor, num_samples=2000, max_mix=12)
    
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=0)
    
    model = AdvancedPredictor(num_pathways, num_herbs).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    epochs = 60
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for inputs, targets, ideal_effects in loop:
            optimizer.zero_grad()
            logits = model(inputs)
            
            # ==========================================
            # 【🔥核心消融点🔥】只保留分类 Loss
            # ==========================================
            cls_loss = F.binary_cross_entropy_with_logits(logits, targets)
            
            # 我们移除了 Spearman Loss (rev_loss) 和 Penalty
            # 这意味着模型只会去"猜药名"，而不知道这些药组合起来的 NES 效果对不对
            
            total_loss = cls_loss 
            # ==========================================
            
            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()
            loop.set_postfix(loss=total_loss.item())
            
        avg_loss = epoch_loss / len(train_loader)
        scheduler.step(avg_loss)

    # 5. 保存模型 (命名为：无Spearman)
    save_name = '消融实验_Top300_无Spearman.pth'
    torch.save(model.state_dict(), save_name)
    print(f"💾 Ablation Model Saved as: {save_name}")

if __name__ == "__main__":
    main()