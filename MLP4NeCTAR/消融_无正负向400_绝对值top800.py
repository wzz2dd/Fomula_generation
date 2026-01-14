import torch
import numpy as np
import pandas as pd
import random
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
# 【🔥消融实验开关🔥】
# True = 训练“绝对值Top800”的对照组模型 (不分正负)
# False = 训练“标准Top400”的最优模型 (正负各400)
# ==========================================
ABLATION_MODE = True  
print(f"⚠️ 当前模式: {'【消融实验：绝对值Top800 (去除正负平衡策略)】' if ABLATION_MODE else '【标准模式：阴阳平衡Top400】'}")

# ==========================================
# 辅助：可微 Spearman Loss
# ==========================================
def differentiable_spearman_loss(pred, target):
    cos_sim = F.cosine_similarity(pred, target, dim=1)
    loss_cos = 1.0 + cos_sim.mean()
    
    n = pred.shape[1]
    idx1 = torch.randint(0, n, (2000,), device=pred.device)
    idx2 = torch.randint(0, n, (2000,), device=pred.device)
    
    diff_target = target[:, idx1] - target[:, idx2]
    diff_pred = pred[:, idx1] - pred[:, idx2]
    
    rank_loss = F.relu(diff_pred * diff_target).mean()
    
    return loss_cos + 2.0 * rank_loss

# ==========================================
# 1. 标准策略：正负平衡动态掩码 (Top 400 Pos + Top 400 Neg)
# ==========================================
def apply_dynamic_mask_torch(vector, limit=400):
    mask = torch.zeros_like(vector, dtype=torch.bool)
    
    # 正向部分
    pos_mask = vector > 0
    pos_indices = torch.nonzero(pos_mask).squeeze()
    if pos_indices.numel() > limit:
        pos_values = vector[pos_indices]
        _, top_k_indices = torch.topk(pos_values, limit)
        mask[pos_indices[top_k_indices]] = True
    else:
        mask[pos_indices] = True

    # 负向部分
    neg_mask = vector < 0
    neg_indices = torch.nonzero(neg_mask).squeeze()
    if neg_indices.numel() > limit:
        neg_values = vector[neg_indices]
        _, top_k_indices = torch.topk(neg_values, limit, largest=False) # 找最小(最负)的
        mask[neg_indices[top_k_indices]] = True
    else:
        mask[neg_indices] = True
        
    return vector * mask

# ==========================================
# 2. 消融策略：绝对值掩码 (Top 800 Abs)
# ==========================================
def apply_absolute_mask_torch(vector, limit=800):
    """
    不区分正负，只保留绝对值最大的 limit 个特征
    """
    mask = torch.zeros_like(vector, dtype=torch.bool)
    abs_vector = torch.abs(vector)
    
    # 直接找绝对值最大的 limit 个
    if abs_vector.numel() > limit:
        _, topk_indices = torch.topk(abs_vector, k=limit)
        mask[topk_indices] = True
    else:
        mask[:] = True
        
    return vector * mask

# ==========================================
# 主程序
# ==========================================
def main():
    # 1. 强制单卡 GPU
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print(f"🔥 [System] GPU Mode: {torch.cuda.get_device_name(0)}")
    else:
        raise RuntimeError("❌ 错误：未检测到 GPU！")
    
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

    # 全量数据放入 GPU
    herb_matrix_tensor = torch.tensor(df.values.astype(np.float32)).to(device)
    
    num_herbs, num_pathways = herb_matrix_tensor.shape
    print(f"✨ [Dimension] Input: {num_pathways} | Output: {num_herbs}")

    # --- Dataset (包含消融逻辑) ---
    class AdversarialDatasetGPU(Dataset):
        def __init__(self, herb_matrix, num_samples=60000, max_mix=12, is_ablation=False):
            self.herb_matrix = herb_matrix
            self.num_samples = num_samples
            self.max_mix = max_mix
            self.num_herbs = herb_matrix.shape[0]
            self.is_ablation = is_ablation

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            # 1. 随机组合
            k = torch.randint(1, self.max_mix + 1, (1,), device=self.herb_matrix.device).item()
            indices = torch.randperm(self.num_herbs, device=self.herb_matrix.device)[:k]
            coeffs = torch.rand(k, device=self.herb_matrix.device) * (3.0 - 0.3) + 0.3
            
            selected_vectors = self.herb_matrix[indices]
            clean_effect = torch.matmul(coeffs, selected_vectors)
            
            # 2. 原始疾病信号
            raw_disease = -1.0 * clean_effect
            
            # ==========================================
            # 【🔥消融实验核心分支🔥】
            # ==========================================
            if self.is_ablation:
                # ❌ 消融模式：使用绝对值 Top 800
                # 这里的逻辑是：只保留信号最强的800个点，其余置0 (和标准模式一样的背景处理，只是选择标准不同)
                # 这样可以公平对比"选择策略"的优劣，无需注入额外噪音
                final_disease = apply_absolute_mask_torch(raw_disease, limit=800)
            else:
                # ✅ 标准模式：使用正负各 Top 400 (共800)
                final_disease = apply_dynamic_mask_torch(raw_disease, limit=400)
            # ==========================================
            
            # 4. 数值缩放
            max_val = torch.max(torch.abs(final_disease))
            if max_val == 0: max_val = 1.0
            
            normalized_disease = final_disease / max_val
            real_amplitude = torch.rand(1, device=self.herb_matrix.device) * (5.0 - 1.5) + 1.5
            final_input = normalized_disease * real_amplitude
            
            # 5. 添加少量测量误差
            global_noise = torch.randn(num_pathways, device=self.herb_matrix.device) * 0.1
            final_input += global_noise
            
            # 6. 标签
            target = torch.ones(self.num_herbs, device=self.herb_matrix.device) * 0.01
            target[indices] = 0.99
            
            return final_input, target, clean_effect

    # --- Model ---
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

    # --- Training ---
    print(f"🚀 Initializing Training (Full GPU Acceleration)...")
    
    train_dataset = AdversarialDatasetGPU(herb_matrix_tensor, num_samples=60000, max_mix=12, is_ablation=ABLATION_MODE)
    test_dataset = AdversarialDatasetGPU(herb_matrix_tensor, num_samples=2000, max_mix=12, is_ablation=ABLATION_MODE)
    
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
            
            cls_loss = F.binary_cross_entropy_with_logits(logits, targets)
            
            pred_probs = torch.sigmoid(logits)
            pred_effect = torch.matmul(pred_probs, herb_matrix_tensor)
            rev_loss = differentiable_spearman_loss(pred_effect, inputs)
            
            bad_dir = F.relu(inputs * pred_effect)
            penalty = bad_dir.mean() * 10.0
            
            total_loss = cls_loss + 0.5 * rev_loss + 0.2 * penalty
            
            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()
            loop.set_postfix(loss=total_loss.item())
            
        avg_loss = epoch_loss / len(train_loader)
        scheduler.step(avg_loss)

    # 保存模型
    if ABLATION_MODE:
        save_name = '消融实验_绝对值Top800.pth'
    else:
        save_name = '自适应Top400平衡.pth'
        
    torch.save(model.state_dict(), save_name)
    print(f"💾 Model Saved as: {save_name}")

if __name__ == "__main__":
    main()