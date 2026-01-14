import os
import copy
import numpy as np
import pandas as pd
import torch
import sys
import warnings

# 忽略警告
warnings.filterwarnings("ignore")

# 设置路径
current_dir = os.path.dirname(os.path.abspath(__file__)) # .../nectar
parent_dir = os.path.dirname(current_dir)              # .../Top800_Project
sys.path.append(parent_dir)                            # 把父目录加入搜索路径


from nectar.modules import (
    herb_ratio_optimization,
    herb_filter,
    calculateScore_800
)
from nectar.modules.data_io import (
    load_herb_nes,
    load_disease_data
)
from nectar.modules.data_preprocessing import prepare_input_data
from nectar.modules.seed_utils import set_random_seeds
from nectar.modules.utils import create_result_folders

# 全局屏蔽列表
stopherbs = []

# ==========================================
# 【关键修改】绝对值 Top-600 掩码函数
# ==========================================
def apply_absolute_mask(vector, limit=600):
    """
    对输入向量应用绝对值 Top-N 截断。
    - 不分正负，只保留绝对值最大的 Top N
    - 用于消融实验
    """
    # 确保是 numpy 数组
    if isinstance(vector, torch.Tensor):
        vector_np = vector.cpu().numpy().flatten()
    else:
        vector_np = vector.flatten()
        
    mask = np.zeros_like(vector_np, dtype=bool)
    abs_vec = np.abs(vector_np)
    
    # 找到绝对值最大的 limit 个
    if len(vector_np) > limit:
        top_indices = np.argsort(abs_vec)[-limit:]
        mask[top_indices] = True
    else:
        mask[:] = True
        
    # 应用 Mask (未选中的置为 0)
    cleaned_vector = vector_np * mask
    return cleaned_vector


def add_top_herbs(result, df_herb_nes, top_num=15):
    if result is None: return []
    logits_list = result.flatten().tolist()
    sorted_indices = sorted(range(len(logits_list)), key=lambda i: logits_list[i], reverse=True)
    column_names = df_herb_nes.columns.tolist()[1:]
    return [column_names[i] for i in sorted_indices[:top_num]]

# ==========================================
# 辅助函数：动态数量剪枝
# ==========================================
def remove_low_weight_herbs(formula, weights_list, threshold=None):
    if not weights_list: return [], []
    if threshold is not None and threshold < 0: return formula, weights_list

    paired = sorted(zip(formula, weights_list), key=lambda x: x[1], reverse=True)
    if not paired: return [], []
    
    max_w = paired[0][1]
    quality_cutoff = max_w * 0.01 
    
    final_combos = []
    final_weights = []
    unique_herbs = set()
    
    MIN_HERBS = 10
    MAX_HERBS = 16
    
    for f, w in paired:
        if w < quality_cutoff:
            if len(unique_herbs) >= MIN_HERBS: continue
        
        current_parts = set()
        if '+' in f:
            parts = f.split('+')
            for p in parts: current_parts.add(p.strip())
        else:
            current_parts.add(f.strip())
            
        potential_herbs = unique_herbs.union(current_parts)
        if len(potential_herbs) > MAX_HERBS: continue
        
        unique_herbs.update(current_parts)
        final_combos.append(f)
        final_weights.append(w)
    
    if len(unique_herbs) < MIN_HERBS and len(final_combos) < len(paired):
        for f, w in paired:
            if f in final_combos: continue
            current_parts = set(f.split('+')) if '+' in f else {f.strip()}
            potential_herbs = unique_herbs.union(current_parts)
            if len(potential_herbs) <= MAX_HERBS:
                unique_herbs.update(current_parts)
                final_combos.append(f)
                final_weights.append(w)
                if len(unique_herbs) >= MIN_HERBS: break

    return final_combos, final_weights

# ==========================================
# 核心函数：优化权重 (应用绝对值截断)
# ==========================================
def optimize_formula_weights(df_herb_nes, formula, df_disease, result_folder, filter_threshold=1e-5):
    if not formula:
        _, temp_input = prepare_input_data(df_herb_nes, [], df_disease)
        return [], [], temp_input[:, 0], float('inf')

    herb_count = len(formula)
    _, input_data = prepare_input_data(df_herb_nes, formula, df_disease) # input_data: [pathways, 1+herbs]

    # --- 【关键一致性】应用绝对值 Top-600 截断 ---
    # 提取疾病向量
    disease_vec = input_data[:, 0]
    
    # 清洗数据 (训练时怎么做，这里就怎么做)
    # 【修改点 1】改为 apply_absolute_mask, limit=600
    cleaned_disease = apply_absolute_mask(disease_vec, limit=600)
    
    # 将清洗后的数据放回 input_data
    # 注意：优化器会基于这个清洗后的数据计算 Loss
    input_data[:, 0] = torch.from_numpy(cleaned_disease) if isinstance(cleaned_disease, np.ndarray) else cleaned_disease
    # --------------------------------

    weights_tensor, _ = herb_ratio_optimization.optimize_weights(input_data, herb_count, result_folder)
    weights_list = weights_tensor.cpu().tolist()
    
    formula, weights_list = remove_low_weight_herbs(formula, weights_list, threshold=filter_threshold)
    
    if not formula:
        _, temp_input = prepare_input_data(df_herb_nes, [], df_disease)
        return [], [], temp_input[:, 0], float('inf')

    # 计算最终分数时，使用【原始数据】(不做截断) 来评估真实效果
    _, raw_input_data = prepare_input_data(df_herb_nes, formula, df_disease)
    combined_nes, score = calculateScore_800.calculateScore(raw_input_data, weights_list)
    
    return formula, weights_list, combined_nes, score

def optimize_loop(df_herb_nes, formula, df_disease, result_folder, filter_threshold=1e-5):
    best_score = float('inf')
    no_improve_count = 0
    max_patience = 2 
    current_formula = formula
    best_internal_formula = formula
    best_internal_weights = []
    best_internal_combined_nes = None
    
    for _ in range(3):
        prev_set = set(current_formula)
        current_formula, current_weights, current_combined_nes, current_score = optimize_formula_weights(
            df_herb_nes, current_formula, df_disease, result_folder, 
            filter_threshold=filter_threshold
        )
        
        if not current_formula: break
        if current_score < best_score:
            best_score = current_score
            best_internal_formula = current_formula
            best_internal_weights = current_weights
            best_internal_combined_nes = current_combined_nes
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        if set(current_formula) == prev_set: break
        if no_improve_count >= max_patience: break
            
    if best_internal_combined_nes is None:
         _, temp_input = prepare_input_data(df_herb_nes, [], df_disease)
         best_internal_combined_nes = temp_input[:, 0]
    if not best_internal_weights and current_weights:
        best_internal_formula = current_formula
        best_internal_weights = current_weights
        
    return best_internal_formula, best_internal_weights, best_internal_combined_nes, best_score

# ==========================================
# 主程序
# ==========================================
def nectar(herb_info_path, disease_data_path):
    set_random_seeds()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    herb_nes_path = os.path.join(BASE_DIR, "data", "通路-中药组合_NES矩阵.csv")
    print(f"Loading Data...")
    
    try:
        df_herb_nes = load_herb_nes(herb_nes_path)
    except:
        df_herb_nes = pd.read_csv(herb_nes_path, index_col=0)
        df_herb_nes.reset_index(inplace=True)
        df_herb_nes.rename(columns={df_herb_nes.columns[0]: 'ID'}, inplace=True)
        df_herb_nes.fillna(0, inplace=True)

    df_disease = load_disease_data(disease_data_path)
    df_herb_info_excel = pd.read_excel(herb_info_path)
    formula = list(df_herb_info_excel["name"])
    
    DISEASE_NAME = "阿尔兹海默疾病"
    # 【修改点 2】加载绝对值 Top-600 的消融模型
    MODEL_FILENAME = "消融实验_绝对值Top600.pth"
    result_folder = create_result_folders(model_name=DISEASE_NAME+MODEL_FILENAME)
    
    print(f"Initial formula: {formula}")

    # Phase 1
    formula, weights, combined_nes, score = optimize_loop(
        df_herb_nes, formula, df_disease, result_folder, filter_threshold=-1.0 
    )
    best_global_score = score
    best_global_formula = formula
    best_global_weights = copy.deepcopy(weights)
    
    # Phase 2
    max_outer_loops = 20
    force_growth_rounds = 5
    
    print("="*60)
    print("STARTING ITERATIVE OPTIMIZATION (Ablation Absolute-600 Mode)")
    print("="*60)

    for i in range(max_outer_loops):
        print(f"\n>>> Iteration {i+1}")
        
        # --- 【关键一致性】AI 调用前的数据清洗 ---
        # 【修改点 3】改为 apply_absolute_mask, limit=600
        cleaned_nes_for_ai = apply_absolute_mask(combined_nes, limit=600)
        
        #  AI 推荐
        ai_suggested = []
        try:
            # 传入清洗后的数据
            pred_logits = herb_filter.herb_filter(cleaned_nes_for_ai)
            ai_suggested = add_top_herbs(pred_logits, df_herb_nes, top_num=100)
        except Exception as e:
            pass

        
        suggested_herbs = list(set(ai_suggested))

        new_formula = list(set(formula + suggested_herbs))
        new_formula = [h for h in new_formula if h not in stopherbs]
        new_formula = sorted(new_formula)
        
        is_force_period = (i < force_growth_rounds)
        current_threshold = -1.0 if is_force_period else 1.0
            
        formula, weights, combined_nes, score = optimize_loop(
            df_herb_nes, new_formula, df_disease, result_folder, 
            filter_threshold=current_threshold
        )
        
        print(f"  [Status] Score: {score:.5f} | Combo Count: {len(formula)}")
        
        is_score_improved = (score < best_global_score)
        
        if is_force_period:
            best_global_score = score
            best_global_formula = formula
            best_global_weights = copy.deepcopy(weights)
            print(f"  [Result] FORCED ACCEPT (Growth).")
        else:
            if is_score_improved:
                best_global_score = score
                best_global_formula = formula
                best_global_weights = copy.deepcopy(weights)
                print(f"  [Result] ACCEPT (Improved).")
            else:
                print(f"  [Result] REJECT. Rolling back.")
                formula = best_global_formula 

    print("\n" + "="*50)
    print(f"OPTIMIZATION FINISHED")
    print(f"Final Score: {best_global_score}")
    
    if len(best_global_weights) != len(best_global_formula):
        if len(best_global_weights) > len(best_global_formula):
             best_global_weights = best_global_weights[:len(best_global_formula)]
        else:
             best_global_weights += [0] * (len(best_global_formula) - len(best_global_weights))

    paired = sorted(zip(best_global_formula, best_global_weights), key=lambda x: x[1], reverse=True)
    final_formula_clean = [x[0] for x in paired]
    final_weights_clean = [x[1] for x in paired]
    
    print("-" * 50)
    print(f"{'Combo Name':<35} | {'Weight':<10}")
    print("-" * 50)
    for h, w in zip(final_formula_clean, final_weights_clean):
        print(f"{h:<35} | {w:.6f}")
    print("-" * 50)

    single_herbs = set()
    for item in final_formula_clean:
        if '+' in item:
            parts = item.split('+')
            for part in parts:
                single_herbs.add(part.strip())
        else:
            single_herbs.add(item.strip())
            
    sorted_single_herbs = sorted(list(single_herbs))
    
    print("\n" + "="*50)
    print(f"🌿 最终拆解单味中药 (Unique Single Herbs: {len(sorted_single_herbs)})")
    print("="*50)
    for i in range(0, len(sorted_single_herbs), 5):
        print(", ".join(sorted_single_herbs[i:i+5]))
    print("="*50 + "\n")

    df_final = pd.DataFrame({
        "herb_combination": final_formula_clean,
        "weight": final_weights_clean
    })
    output_path = os.path.join(result_folder, "final_formula_list.xlsx")
    df_final.to_excel(output_path, index=False)
    
    df_single = pd.DataFrame({"single_herb": sorted_single_herbs})
    output_single_path = os.path.join(result_folder, "final_single_herbs.xlsx")
    df_single.to_excel(output_single_path, index=False)
    
    print(f"Results saved to:\n1. {output_path}\n2. {output_single_path}")

    try:
        _, input_for_plot = prepare_input_data(df_herb_nes, final_formula_clean, df_disease)
        
        # 1. 原有的绘图 (保持不变)
        calculateScore_800.calculateScore_plot(
            final_formula_clean, 
            input_for_plot, 
            final_weights_clean, 
            result_folder, 
            plot="final"
        )
        
        # ==========================================
        # 2. 调用拆解评分 (配合修改后的 calculateScore_600.py 会自动打印 Top 600 统计)
        # ==========================================
        calculateScore_800.calculate_split_score(
            input_for_plot, 
            final_weights_clean,
            result_folder=result_folder
        )
        # ==========================================
            
    except Exception as e:
            print(f"[Warning] Plot/Score error: {e}")
    
    return {
        "final_formula": final_formula_clean,
        "single_herbs": sorted_single_herbs,
        "final_score": best_global_score,
        "result_folder": result_folder
    }

if __name__ == "__main__":
    import argparse
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    set_random_seeds()

    parser = argparse.ArgumentParser()
    parser.add_argument("--herb_info_path", type=str, default=os.path.join(BASE_DIR, "data", "info_input_herbs.xlsx"))
    parser.add_argument("--disease_data_path", type=str, default=os.path.join(BASE_DIR, "data", "EAD疾病_通路NES结果.csv"))
    
    args = parser.parse_args()
    nectar(args.herb_info_path, args.disease_data_path)