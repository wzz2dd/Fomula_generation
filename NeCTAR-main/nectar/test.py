import os
import numpy as np
import pandas as pd
import torch
import sys

# 引入你的数据读取逻辑
# 假设你的 modules 路径是对的，如果报错请检查 pythonpath
try:
    from modules.data_io import load_disease_data
except ImportError:
    # 兼容性 Fallback：如果找不到模块，定义一个简单的读取函数
    def load_disease_data(path):
        try:
            return pd.read_csv(path)
        except:
            return pd.read_csv(path, sep='\t')

def apply_dynamic_mask(vector, limit=500):
    """保持与训练逻辑一致的动态截断"""
    if isinstance(vector, torch.Tensor):
        vector_np = vector.cpu().numpy().flatten()
    else:
        vector_np = vector.flatten()
    
    # 确保是浮点数，防止字符串混入
    vector_np = vector_np.astype(float)
        
    mask = np.zeros_like(vector_np, dtype=bool)
    pos_idx = np.where(vector_np > 0)[0]
    if len(pos_idx) > limit:
        top_pos = pos_idx[np.argsort(vector_np[pos_idx])[-limit:]]
        mask[top_pos] = True
    else:
        mask[pos_idx] = True
    neg_idx = np.where(vector_np < 0)[0]
    if len(neg_idx) > limit:
        top_neg = neg_idx[np.argsort(vector_np[neg_idx])[:limit]]
        mask[top_neg] = True
    else:
        mask[neg_idx] = True
    return vector_np * mask

def diagnose(herb_info_path, disease_data_path):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    herb_nes_path = os.path.join(BASE_DIR, "data", "通路-中药组合_NES矩阵.csv")
    
    print("="*60)
    print("🚑 疾病数据与药库匹配度诊断 (Diagnostic Tool V2)")
    print("="*60)

    # ----------------------------------------------------
    # 1. 加载药库 (基准)
    # ----------------------------------------------------
    print(f"[1] Loading Herb Matrix...")
    try:
        # 读取药库，第一列作为索引 (Pathway IDs)
        df_herb_nes = pd.read_csv(herb_nes_path, index_col=0)
        df_herb_nes.fillna(0, inplace=True)
        
        # 检查药库格式
        print(f"   药库维度: {df_herb_nes.shape} (Rows=Pathways, Cols=Herbs)")
        
        # 获取标准的通路 ID 列表
        standard_pathways = df_herb_nes.index.astype(str).tolist()
        
    except Exception as e:
        print(f"❌ Error loading herb matrix: {e}")
        return

    # ----------------------------------------------------
    # 2. 加载疾病数据 (并对齐)
    # ----------------------------------------------------
    print(f"\n[2] Loading Disease Data: {os.path.basename(disease_data_path)}")
    try:
        df_disease = load_disease_data(disease_data_path)
        
        # 2.1 智能寻找 NES 列和 ID 列
        cols = df_disease.columns.tolist()
        id_col = None
        nes_col = None
        
        # 找 ID 列
        if 'ID' in cols: id_col = 'ID'
        elif 'Term' in cols: id_col = 'Term'
        else: id_col = cols[0] # 盲猜第一列
        
        # 找 NES 列
        if 'NES' in cols: nes_col = 'NES'
        elif 'score' in cols.lower(): nes_col = next(c for c in cols if 'score' in c.lower())
        else: nes_col = cols[1] if len(cols) > 1 else None # 盲猜第二列
        
        print(f"   识别列名: ID='{id_col}', NES='{nes_col}'")
        
        # 2.2 设置索引并提取 NES
        df_disease.set_index(id_col, inplace=True)
        # 确保索引也是字符串，方便匹配
        df_disease.index = df_disease.index.astype(str)
        
        if nes_col not in df_disease.columns:
            raise ValueError(f"无法找到数值列。现有列: {df_disease.columns}")
            
        disease_series = df_disease[nes_col]
        
        # 2.3 【核心修正】按药库 ID 对齐 (Reindex)
        # 这步操作会自动把药库里有、但疾病里没有的通路填 0
        # 也会把疾病里有、但药库里没有的通路丢弃
        aligned_disease = disease_series.reindex(standard_pathways, fill_value=0.0)
        
        disease_vec_raw = aligned_disease.values.astype(float)
        
        print(f"   对齐后维度: {len(disease_vec_raw)} (与药库 100% 一致)")
        
    except Exception as e:
        print(f"❌ Error loading/aligning disease data: {e}")
        import traceback
        traceback.print_exc()
        return

    # ----------------------------------------------------
    # 3. 准备计算矩阵
    # ----------------------------------------------------
    # 药库: (Pathways, Herbs) -> 转置为 (Herbs, Pathways)
    # 确保类型为 float32 加速计算
    herb_matrix = df_herb_nes.values.T.astype(np.float32)
    herb_names = df_herb_nes.columns.tolist()
    
    # ----------------------------------------------------
    # 4. 数据体检
    # ----------------------------------------------------
    print(f"\n[3] 疾病数据体检")
    print(f"   数值范围: [{np.min(disease_vec_raw):.4f}, {np.max(disease_vec_raw):.4f}]")
    
    # 统计有多少个有效匹配 (非0值)
    valid_overlap = np.count_nonzero(disease_vec_raw)
    print(f"   有效重叠通路数: {valid_overlap} / 10014")
    
    if valid_overlap < 50:
        print("⚠️ [严重警告] 你的疾病数据与药库的通路 ID 几乎匹配不上！")
        print("   请检查：疾病数据的 ID 格式（如 'hsa04060'）是否与药库一致？")
        print("   如果 ID 格式不同（例如一个是基因名，一个是通路ID），模型完全无效。")
    
    # ----------------------------------------------------
    # 5. 应用动态截断
    # ----------------------------------------------------
    print(f"\n[4] 应用动态 Top-500 截断...")
    disease_vec_clean = apply_dynamic_mask(disease_vec_raw, limit=500)
    print(f"   截断后保留特征数: {np.count_nonzero(disease_vec_clean)}")
    
    # ----------------------------------------------------
    # 6. 暴力匹配测试
    # ----------------------------------------------------
    print(f"\n[5] 药库全扫描 (Brute-force Screening)")
    
    # 归一化
    d_norm = np.linalg.norm(disease_vec_clean) + 1e-9
    d_unit = disease_vec_clean / d_norm
    
    h_norms = np.linalg.norm(herb_matrix, axis=1) + 1e-9
    h_unit = herb_matrix / h_norms[:, np.newaxis]
    
    # Cosine 相似度 (Dot Product of Unit Vectors)
    scores = np.dot(h_unit, d_unit)
    
    # 排序 (从小到大，越负越好)
    sorted_idx = np.argsort(scores)
    
    print("-" * 60)
    print(f"{'Rank':<5} | {'Herb Name':<30} | {'Score (Cosine)':<10}")
    print("-" * 60)
    
    top_10_scores = []
    for i in range(10):
        idx = sorted_idx[i]
        score = scores[idx]
        name = herb_names[idx]
        top_10_scores.append(score)
        # 高亮强相关
        mark = "🌟" if score < -0.15 else ""
        print(f"{i+1:<5} | {name:<30} | {score:.4f} {mark}")
    
    print("-" * 60)
    
    # ----------------------------------------------------
    # 7. 诊断结论
    # ----------------------------------------------------
    best_score = top_10_scores[0]
    
    print(f"\n[6] 最终诊断 (Diagnosis)")
    print(f"   理论最强单药分数: {best_score:.4f}")
    
    if valid_overlap < 50:
        print("\n❌ [结论: ID匹配失败]")
        print("   问题不在模型，在于数据ID对不上。")
        print("   药库用的是 KEGG ID (如 hsa123) 还是 Reactome？请确保输入数据一致。")
    elif best_score > -0.05:
        print("\n❌ [结论: 无药可救 (Coverage Gap)]")
        print("   即使在 10014 个特征对齐后，依然没有药物能与疾病呈负相关。")
        print("   这说明该疾病的核心病理机制不在现有 500 味中药的靶点范围内。")
    elif best_score > -0.15:
        print("\n⚠️ [结论: 效果微弱 (Weak Signal)]")
        print("   有药能治，但对口度不高。AI 可能会推荐，但置信度低。")
    else:
        print("\n✅ [结论: 药库匹配良好 (Good Match)]")
        print("   存在强力对症药物！如果之前的模型跑不出结果，那是模型训练的问题。")
        print("   建议：使用现在的训练代码重新训练，应该能抓到这些药。")

if __name__ == "__main__":
    disease_path = "nectar/data/disease_nes.csv" 
    if len(sys.argv) > 1:
        disease_path = sys.argv[1]
    diagnose(None, disease_path)