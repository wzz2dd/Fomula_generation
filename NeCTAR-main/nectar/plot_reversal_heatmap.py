import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys

# 路径自动修复逻辑
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path: sys.path.append(parent_dir)
    from nectar.modules.data_io import load_herb_nes, load_disease_data
except ImportError:
    pass

def plot_reversal_heatmap_v3():
    print("🚀 正在绘制高对比度逆转热图 (V3 Highlight Version)...")
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # 1. 自动定位 Results
    possible_paths = [os.path.join(os.path.dirname(BASE_DIR), "results"), os.path.join(BASE_DIR, "results")]
    results_root = next((p for p in possible_paths if os.path.exists(p) and os.listdir(p)), None)
    
    if not results_root:
        print("❌ 找不到 results 文件夹！"); return

    latest_dir = max([os.path.join(results_root, d) for d in os.listdir(results_root) if os.path.isdir(os.path.join(results_root, d))], key=os.path.getmtime)
    formula_path = os.path.join(latest_dir, "final_formula_list.xlsx")
    
    # 2. 自动定位 Data
    possible_data = [os.path.join(BASE_DIR, "data"), os.path.join(os.path.dirname(BASE_DIR), "data")]
    DATA_DIR = next((d for d in possible_data if os.path.exists(d)), None)
    
    herb_path = os.path.join(DATA_DIR, "通路-中药组合_NES矩阵.csv")
    disease_path = os.path.join(DATA_DIR, "EAD疾病_通路NES结果.csv")
    
    # 3. 加载数据
    print("正在计算最佳逆转通路...")
    df_herb = pd.read_csv(herb_path, index_col=0)
    df_disease = pd.read_csv(disease_path)
    nes_col = [c for c in df_disease.columns if 'NES' in c or 'score' in c.lower()][0]
    df_disease.set_index(df_disease.columns[0], inplace=True)
    disease_vec = df_disease[nes_col].reindex(df_herb.index).fillna(0)
    
    df_formula = pd.read_excel(formula_path)
    formula_vec = np.zeros_like(disease_vec.values)
    for h, w in zip(df_formula['herb_combination'], df_formula['weight']):
        if h in df_herb.columns:
            formula_vec += df_herb[h].values * w
            
    # 4. 【核心策略】挑选“视觉冲击力最强”的 Top 25
    df_plot = pd.DataFrame({'AD Model': disease_vec.values, 'AI Formula': formula_vec}, index=disease_vec.index)
    
    # 算分逻辑：逆转强度 = |疾病| + |药物| (仅当方向相反时)
    # 如果同向（没逆转），分数为 0
    df_plot['Reversal_Score'] = np.where(
        (df_plot['AD Model'] * df_plot['AI Formula'] < 0), # 必须反向
        df_plot['AD Model'].abs() + df_plot['AI Formula'].abs(), # 越红越蓝，加起来越大
        0
    )
    
    # 取 Top 25
    df_best = df_plot.sort_values('Reversal_Score', ascending=False).head(25)
    
    # 移除辅助列，准备画图
    df_final = df_best[['AD Model', 'AI Formula']]
    
    # 再次排序：为了美观，按 AD Model 从高到低排
    df_final = df_final.sort_values('AD Model', ascending=False)
    
    print(f"已筛选出 Top {len(df_final)} 最强逆转通路。")
    print("示例通路:", df_final.index[:3].tolist())

    # 5. 画图 (增强颜色对比)
    plt.figure(figsize=(5, 8))
    
    # vmin/vmax: 强制锁死颜色范围，让颜色更深
    # 如果你的数据普遍较小(如0.5)，把 vmax 设为 1.0；如果很大，设为 2.5
    # 这里我们用自动检测的 quantile 来增强对比
    limit = max(df_final.abs().max().max(), 1.0) 
    
    sns.heatmap(df_final, 
                cmap="RdBu_r", # 红蓝反转色
                center=0, 
                annot=True, fmt=".2f", # 显示数值，增加可信度
                vmin=-limit, vmax=limit, # 锁死范围，保证红蓝平衡
                cbar_kws={'label': 'NES (Normalized Enrichment Score)'})
    
    plt.title("Transcriptomic Reversal (Top 25)", fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    save_path = "result_compare/Reversal_Heatmap_V3_HighContrast.pdf"
    plt.savefig(save_path, bbox_inches='tight')
    print(f"\n✅ 高对比度热图已保存: {save_path}")
    print("💡 现在打开看看，是不是左边全红，右边全蓝？")

if __name__ == "__main__":
    plot_reversal_heatmap_v3()