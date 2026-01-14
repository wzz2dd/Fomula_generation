import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.preprocessing import MinMaxScaler
import os
# 兼容 Linux
mpl.rcParams['pdf.fonttype'] = 42

def get_dynamic_mask_indices(disease_vec, limit=400):
    """获取动态截断的索引"""
    pos_indices = np.where(disease_vec > 0)[0]
    neg_indices = np.where(disease_vec < 0)[0]
    
    selected_indices = []
    
    if len(pos_indices) > limit:
        top_pos = pos_indices[np.argsort(disease_vec[pos_indices])[-limit:]]
        selected_indices.extend(top_pos)
    else:
        selected_indices.extend(pos_indices)
        
    if len(neg_indices) > limit:
        top_neg = neg_indices[np.argsort(disease_vec[neg_indices])[:limit]]
        selected_indices.extend(top_neg)
    else:
        selected_indices.extend(neg_indices)
    
    return np.array(selected_indices, dtype=int)

def calculate_split_score(input_tensor, weights_cpu_list, result_folder=None):
    """
    通用评分函数 (带安全性分析版)
    """
    test = input_tensor[:, 0]
    herb_matrix = input_tensor[:, 1:]
    weights = np.array([max(0, w) for w in weights_cpu_list])
    sum_adjusted = np.dot(herb_matrix, weights)
    
    # 默认使用 Top-400
    CURRENT_LIMIT = 400 
    
    # 自动识别筛选策略
    if 'get_absolute_mask_indices' in globals():
        valid_indices = get_absolute_mask_indices(test, limit=CURRENT_LIMIT)
        strategy_name = f"绝对值 (Absolute) Top-{CURRENT_LIMIT}"
    else:
        # 假设你有 get_dynamic_mask_indices 函数
        valid_indices = get_dynamic_mask_indices(test, limit=CURRENT_LIMIT)
        strategy_name = f"平衡 (Balanced) Top-{CURRENT_LIMIT}"
    
    # 提取核心区数据
    valid_test = test[valid_indices]      # 疾病向量
    valid_sum = sum_adjusted[valid_indices] # 药物向量
    
    # 基础统计
    count_pos = np.sum(valid_test > 0)
    count_neg = np.sum(valid_test < 0)
    total_selected = len(valid_test)
    ratio = count_pos / count_neg if count_neg > 0 else 0.0

    # ==========================================
    # 🚑 【新增】安全性与副作用分析
    # 逻辑：如果 (疾病 * 药物) > 0，说明方向相同，属于"助纣为虐"
    # ==========================================
    # 1. 计算同向恶化的掩码 (True 代表恶化)
    worsening_mask = (valid_test * valid_sum) > 0
    
    # 2. 统计数量
    total_worsened = np.sum(worsening_mask)
    worsening_rate = (total_worsened / total_selected) * 100 if total_selected > 0 else 0.0
    
    # 3. 细分：正向恶化 vs 负向恶化
    pos_worsened = np.sum(worsening_mask & (valid_test > 0)) # 本来亢进，还在补
    neg_worsened = np.sum(worsening_mask & (valid_test < 0)) # 本来衰退，还在泻
    # ==========================================

    # 计算分数
    corr_core_total, _ = spearmanr(valid_sum, valid_test)
    
    pos_mask_core = valid_test > 0
    neg_mask_core = valid_test < 0
    
    corr_pos_core, _ = spearmanr(valid_sum[pos_mask_core], valid_test[pos_mask_core]) if np.sum(pos_mask_core) > 5 else 0.0
    corr_neg_core, _ = spearmanr(valid_sum[neg_mask_core], valid_test[neg_mask_core]) if np.sum(neg_mask_core) > 5 else 0.0

    # ==========================================
    # 📝 生成报告内容
    # ==========================================
    report_lines = []
    report_lines.append("="*50)
    report_lines.append(f"🧐 [Pathways Distribution Analysis]")
    report_lines.append(f"Strategy: {strategy_name}")
    report_lines.append("-" * 50)
    report_lines.append(f"🔴 Positive (Hyperactive): {count_pos}")
    report_lines.append(f"🔵 Negative (Suppressed) : {count_neg}")
    report_lines.append(f"∑  Total Selected      : {total_selected}")
    report_lines.append("="*50)
    
    # 新增安全性板块
    report_lines.append(f"🚑 [Safety & Side Effect Analysis]")
    report_lines.append(f"⚠️ Total Worsened Pathways : {total_worsened} / {total_selected}")
    report_lines.append(f"💀 Worsening Rate (Risk)   : {worsening_rate:.2f}%")
    report_lines.append(f"   - Aggravated Excess (正向恶化): {pos_worsened}")
    report_lines.append(f"   - Aggravated Deficiency (负向恶化): {neg_worsened}")
    report_lines.append("="*50)
    
    report_lines.append(f"🎯 Core Score (Total)   : {corr_core_total:.4f}")
    report_lines.append(f"🔥 Core Positive Score  : {corr_pos_core:.4f}")
    report_lines.append(f"❄️ Core Negative Score  : {corr_neg_core:.4f}")
    report_lines.append("="*50 + "\n")
    
    report_text = "\n".join(report_lines)

    # 1. 打印到控制台
    print(report_text)
    
    # 2. 保存到文件
    if result_folder:
        save_path = os.path.join(result_folder, "final_score_analysis.txt")
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(report_text)
        print(f"📄 [Report Saved] Analysis saved to: {save_path}")

    return corr_core_total, corr_pos_core, corr_neg_core

def calculateScore(input_tensor, weights_cpu_list):
    test = input_tensor[:, 0]
    herb_matrix = input_tensor[:, 1:]
    weights = np.array([max(0, w) for w in weights_cpu_list])
    sum_adjusted = np.dot(herb_matrix, weights)
    
    # 1. 获取保留下来的索引
    valid_indices = get_dynamic_mask_indices(test, limit=400)
    
    valid_test = test[valid_indices]
    valid_sum = sum_adjusted[valid_indices]
    
    # 2. 【优化】只计算非零区域的相关性 (剔除背景噪声)
    # 这会让分数更聚焦于"由于药物作用而产生的变化"
    # 如果 valid_sum 全是 0 (还没开始优化)，相关性为 0
    if np.std(valid_sum) < 1e-9:
        corr = 0.0
    else:
        corr, _ = spearmanr(valid_sum, valid_test)
        
    combined = test + sum_adjusted
    return combined.reshape(-1, 1), corr

def calculateScore_plot(formula, input_tensor, weights_cpu_list, result_folder, plot=0):
    test = input_tensor[:, 0]
    herb_matrix = input_tensor[:, 1:]
    weights = np.array([max(0, w) for w in weights_cpu_list])
    sum_adjusted = np.dot(herb_matrix, weights)

    valid_indices = get_dynamic_mask_indices(test, limit=400)
    valid_test = test[valid_indices]
    valid_sum = sum_adjusted[valid_indices]
    
    if np.std(valid_sum) < 1e-9:
        corr = 0.0
    else:
        corr, _ = spearmanr(valid_sum, valid_test)

    # Scatter Plot
    scaler = MinMaxScaler(feature_range=(-1, 1))
    x_data = scaler.fit_transform(valid_sum.reshape(-1, 1)).flatten()
    y_data = scaler.fit_transform(valid_test.reshape(-1, 1)).flatten()

    plt.figure(figsize=(8, 6))
    sns.set(style="whitegrid", font_scale=1.2)
    sns.regplot(x=x_data, y=y_data, 
                scatter_kws={'alpha':0.6, 's':20, 'color': 'dodgerblue'}, 
                line_kws={'color': 'darkorange', 'linewidth': 2})
                
    # 【已修正标题】
    plt.title(f'Dynamic Top-400 Correlation: {corr:.2f}', fontsize=16)
    plt.xlabel('Formula Score', fontsize=14)
    plt.ylabel('Disease Score', fontsize=14)
    plt.savefig(f'{result_folder}/spearman_scatter_{plot}.pdf', format='pdf', bbox_inches='tight')
    plt.close()

    # Histogram
    plt.figure(figsize=(10, 6))
    plt.hist(test, bins=50, alpha=0.5, label='Disease', color='blue')
    plt.hist(sum_adjusted, bins=50, alpha=0.5, label='Formula', color='red')
    plt.title(f'Distribution Analysis (Iter {plot})') # 修正标题
    plt.legend()
    plt.savefig(f'{result_folder}/distribution_{plot}.png')
    plt.close()

    # Heatmap
    hotmap_plot_balanced(test, sum_adjusted, valid_indices, f"{result_folder}/heatmap_{plot}.pdf")

def normalize_data(data):
    max_abs = np.max(np.abs(data))
    if max_abs == 0: return data
    return data / max_abs

def hotmap_plot_balanced(test_data, sum_adjusted_data, indices, save_path):
    # 提取数据
    raw_disease = test_data[indices]
    raw_formula = sum_adjusted_data[indices]
    
    # 归一化
    test_mini = normalize_data(raw_disease)
    sum_mini = normalize_data(raw_formula)
    
    # 排序
    sort_indices = np.argsort(test_mini)[::-1]
    
    # 组合
    combined = np.column_stack((test_mini[sort_indices], sum_mini[sort_indices]))

    plt.figure(figsize=(8, 10))
    sns.heatmap(
        combined, 
        center=0, 
        cmap='coolwarm', 
        vmin=-1, vmax=1, 
        cbar_kws={'label': 'Normalized NES'}, 
        xticklabels=['Disease', 'Formula']
    )
    # 【已修正标题】
    plt.title('Dynamic Balanced Heatmap', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close()