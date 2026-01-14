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
# 辅助：可微 Spearman Loss
# ==========================================
def differentiable_spearman_loss(pred, target):
    # Cosine 相似度 (方向)
    cos_sim = F.cosine_similarity(pred, target, dim=1)
    loss_cos = 1.0 + cos_sim.mean()
    
    # 软排序一致性 (Rank)
    n = pred.shape[1]
    idx1 = torch.randint(0, n, (2000,), device=pred.device)
    idx2 = torch.randint(0, n, (2000,), device=pred.device)
    
    diff_target = target[:, idx1] - target[:, idx2]
    diff_pred = pred[:, idx1] - pred[:, idx2]
    
    rank_loss = F.relu(diff_pred * diff_target).mean()
    
    return loss_cos + 2.0 * rank_loss

# ==========================================
# 辅助：动态 Top-500 掩码
# ==========================================
def apply_dynamic_mask_np(vector, limit=500):
    mask = np.zeros_like(vector, dtype=bool)
    
    # 1. 正向部分
    pos_idx = np.where(vector > 0)[0]
    if len(pos_idx) > limit:
        top_pos = pos_idx[np.argsort(vector[pos_idx])[-limit:]]
        mask[top_pos] = True
    else:
        mask[pos_idx] = True # 不足500全取

    # 2. 负向部分
    neg_idx = np.where(vector < 0)[0]
    if len(neg_idx) > limit:
        top_neg = neg_idx[np.argsort(vector[neg_idx])[:limit]]
        mask[top_neg] = True
    else:
        mask[neg_idx] = True # 不足500全取
        
    return vector * mask

# ==========================================
# 主程序
# ==========================================
def main():
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"🔥 [System] Detected {gpu_count} GPUs. High-Performance Mode.")
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
        gpu_count = 0
    
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
    
    herb_matrix_np = df.values.T.astype(np.float32)
    herb_matrix_tensor = torch.tensor(herb_matrix_np).to(device)
    
    num_herbs, num_pathways = herb_matrix_np.shape
    print(f"✨ [Dimension] Input: {num_pathways} | Output: {num_herbs}")

    # --- Dataset ---
    class AdversarialDataset(Dataset):
        def __init__(self, herb_matrix, num_samples=60000, max_mix=12):
            self.herb_matrix = herb_matrix
            self.num_samples = num_samples
            self.max_mix = max_mix
            self.num_herbs = herb_matrix.shape[0]
            self.num_features = herb_matrix.shape[1]

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            # 1. 随机组合构建 (原始大数值)
            k = np.random.randint(1, self.max_mix + 1)
            indices = np.random.choice(self.num_herbs, k, replace=False)
            coeffs = np.random.uniform(0.3, 3.0, size=k).astype(np.float32)
            
            selected_vectors = self.herb_matrix[indices]
            clean_effect = np.dot(coeffs, selected_vectors)
            
            # 2. 原始疾病信号 (反向)
            raw_disease = -1.0 * clean_effect
            
            # 3. 【策略一】应用动态 Top-500 规则
            # 这一步去除了尾部噪声，只保留核心特征
            #masked_disease = apply_dynamic_mask_np(raw_disease, limit=500)
            fake_noise = np.zeros_like(raw_disease)
            noise_idx = np.random.choice(self.num_features, 1000, replace=False)
            fake_noise[noise_idx] = np.random.uniform(-3.0, 3.0, size=1000)
            dirty_disease = raw_disease + fake_noise

            masked_disease = dirty_disease  # 直接把噪声传下去
            # 4. 【策略二】数值仿真缩放 (Simulation Scaling)
            # 这一步把几十上百的数值，压缩到真实疾病的 [-5, 5] 区间
            max_val = np.max(np.abs(masked_disease))
            if max_val == 0: max_val = 1.0
            
            # 归一化到 [-1, 1]
            normalized_disease = masked_disease / max_val
            
            # 赋予真实的 NES 强度 (1.5 ~ 5.0)
            real_amplitude = np.random.uniform(1.5, 5.0)
            final_input = normalized_disease * real_amplitude
            
            # 5. 添加少量对抗噪声
            noise = np.random.normal(0, 0.1, size=self.num_features).astype(np.float32)
            final_input += noise
            
            # 6. 标签
            target = np.ones(self.num_herbs, dtype=np.float32) * 0.01
            target[indices] = 0.99 
            
            return (torch.from_numpy(final_input.astype(np.float32)), 
                    torch.from_numpy(target),
                    torch.from_numpy(clean_effect.astype(np.float32)))

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
    print("🚀 Initializing Dynamic-Simulation Training...")
    
    train_dataset = AdversarialDataset(herb_matrix_np, num_samples=60000, max_mix=12)
    test_dataset = AdversarialDataset(herb_matrix_np, num_samples=2000, max_mix=12)
    
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=8, pin_memory=True)
    
    model = AdvancedPredictor(num_pathways, num_herbs).to(device)
    if gpu_count > 1: model = nn.DataParallel(model)
    
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    for epoch in range(60):
        model.train()
        epoch_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for inputs, targets, ideal_effects in loop:
            inputs, targets, ideal_effects = inputs.to(device), targets.to(device), ideal_effects.to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            
            # Loss 1: 分类
            cls_loss = F.binary_cross_entropy_with_logits(logits, targets)
            
            # Loss 2: 逆转能力 (Spearman)
            pred_probs = torch.sigmoid(logits)
            pred_effect = torch.matmul(pred_probs, herb_matrix_tensor)
            rev_loss = differentiable_spearman_loss(pred_effect, inputs)
            
            # Loss 3: 同向惩罚
            bad_dir = F.relu(inputs * pred_effect)
            penalty = bad_dir.mean() * 10.0
            
            total_loss = cls_loss + 0.5 * rev_loss + 0.2 * penalty
            
            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()
            loop.set_postfix(loss=total_loss.item())
            
        avg_loss = epoch_loss / len(train_loader)
        scheduler.step(avg_loss)

    torch.save(model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(), 'w_o_Dynamic_Model.pth')
    print("💾 Model Saved!")

if __name__ == "__main__":
    main()