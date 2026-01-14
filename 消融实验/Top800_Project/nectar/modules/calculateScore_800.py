import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.preprocessing import MinMaxScaler
import os
# 兼容 Linux
mpl.rcParams['pdf.fonttype'] = 42

# ==========================================
# 【关键修改】绝对值筛选函数 (消融实验专用)
# ==========================================
def get_absolute_mask_indices(disease_vec, limit=600):
    """
    获取绝对值最大的 Top-N 索引 (不分正负)
    用于消融实验，验证绝对值策略是否会导致严重的偏科
    """
    abs_vec = np.abs(disease_vec)
    
    if len(abs_vec) > limit:
        # argsort 从小到大排，取最后 limit 个 (即绝对值最大的 limit 个)
        return np.argsort(abs_vec)[-limit:]
    else:
        return np.arange(len(disease_vec))

def calculate_split_score(input_tensor, weights_cpu_list, result_folder=None):
    test = input_tensor[:, 0]
    herb_matrix = input_tensor[:, 1:]
    weights = np.array([max(0, w) for w in weights_cpu_list])
    sum_adjusted = np.dot(herb_matrix, weights)
    
    # ==========================================
    # ⚠️ 注意：如果你是 Top-600 文件，这里记得改成 600
    # ⚠️ 如果是 Top-400 文件，这里记得改成 400
    # ==========================================
    CURRENT_LIMIT = 600 
    
    # 自动识别使用哪种筛选 (兼容你的两个文件)
    if 'get_absolute_mask_indices' in globals():
        valid_indices = get_absolute_mask_indices(test, limit=CURRENT_LIMIT)
        strategy_name = f"绝对值 (Absolute) Top-{CURRENT_LIMIT}"
    else:
        valid_indices = get_dynamic_mask_indices(test, limit=CURRENT_LIMIT)
        strategy_name = f"平衡 (Balanced) Top-{CURRENT_LIMIT}"
    
    # 提取核心区数据
    valid_test = test[valid_indices]
    valid_sum = sum_adjusted[valid_indices]
    
    # 统计数据
    count_pos = np.sum(valid_test > 0)
    count_neg = np.sum(valid_test < 0)
    total_selected = len(valid_test)
    ratio = count_pos / count_neg if count_neg > 0 else 0.0

    # 计算分数
    corr_core_total, _ = spearmanr(valid_sum, valid_test)
    
    pos_mask_core = valid_test > 0
    neg_mask_core = valid_test < 0
    
    corr_pos_core, _ = spearmanr(valid_sum[pos_mask_core], valid_test[pos_mask_core]) if np.sum(pos_mask_core) > 5 else 0.0
    corr_neg_core, _ = spearmanr(valid_sum[neg_mask_core], valid_test[neg_mask_core]) if np.sum(neg_mask_core) > 5 else 0.0

    # ==========================================
    # 📝 生成报告内容 (String Buffer)
    # ==========================================
    report_lines = []
    report_lines.append("="*50)
    report_lines.append(f"🧐 [Pathways Distribution Analysis]")
    report_lines.append(f"Strategy: {strategy_name}")
    report_lines.append("-" * 50)
    report_lines.append(f"🔴 Positive (Hyperactive): {count_pos}")
    report_lines.append(f"🔵 Negative (Suppressed) : {count_neg}")
    report_lines.append(f"∑  Total Selected      : {total_selected}")
    report_lines.append(f"⚖️  Pos/Neg Ratio       : {ratio:.2f}")
    report_lines.append("="*50)
    report_lines.append(f"🎯 Core Score (Total)   : {corr_core_total:.4f}")
    report_lines.append(f"🔥 Core Positive Score  : {corr_pos_core:.4f}")
    report_lines.append(f"❄️ Core Negative Score  : {corr_neg_core:.4f}")
    report_lines.append("="*50 + "\n")
    
    report_text = "\n".join(report_lines)

    # 1. 打印到控制台
    print(report_text)
    
    # 2. 保存到文件 (如果有路径)
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
    
    # 【修改 3】使用绝对值筛选 Top 600
    valid_indices = get_absolute_mask_indices(test, limit=600)
    
    valid_test = test[valid_indices]
    valid_sum = sum_adjusted[valid_indices]
    
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

    # 【修改 4】使用绝对值筛选 Top 600
    valid_indices = get_absolute_mask_indices(test, limit=600)
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
                
    # 【修改 5】标题改为 Ablation Absolute-600
    plt.title(f'Ablation Absolute-600 Correlation: {corr:.2f}', fontsize=16)
    plt.xlabel('Formula Score', fontsize=14)
    plt.ylabel('Disease Score', fontsize=14)
    plt.savefig(f'{result_folder}/spearman_scatter_{plot}.pdf', format='pdf', bbox_inches='tight')
    plt.close()

    # Histogram
    plt.figure(figsize=(10, 6))
    plt.hist(test, bins=50, alpha=0.5, label='Disease', color='blue')
    plt.hist(sum_adjusted, bins=50, alpha=0.5, label='Formula', color='red')
    plt.title(f'Distribution Analysis (Iter {plot})')
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
    # 【修改 6】标题改为 Ablation Absolute-600
    plt.title('Ablation Absolute-600 Heatmap', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close()