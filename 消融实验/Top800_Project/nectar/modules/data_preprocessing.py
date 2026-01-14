# nectar/modules/data_preprocessing.py (Clean Version)

import pandas as pd
import numpy as np
import warnings

# 忽略 Pandas 的 SettingWithCopyWarning
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

def normalize_column(df):
    """标准化列数据到-1到1的范围"""
    try:
        return 2 * (df - df.min()) / (df.max() - df.min()) - 1
    except:
        return df

def prepare_herb_data(df_herb_nes, formula, nor=False):
    """处理中药数据"""
    # 【修复】使用 .copy() 创建副本，彻底解决 SettingWithCopyWarning
    df_herbs = df_herb_nes[formula].copy()
    df_herbs.fillna(0, inplace=True)
    
    if nor:
        df_herbs = df_herbs.apply(normalize_column, axis=0)
    return df_herbs

def merge_disease_data(df_herb_nes, df_disease):
    """合并疾病与中药数据"""
    result = pd.merge(df_herb_nes, df_disease, how='left', on='ID')
    result.fillna(0, inplace=True)
    # 【修复】删除烦人的 print(result.shape)
    return result

def process_disease_data(result, nor=False):
    """处理疾病数据"""
    result_df = pd.DataFrame(result["NES"])
    if nor:
        result_df = result_df.apply(normalize_column, axis=0)
    return np.array(result_df)

def prepare_input_data(df_herb_nes, formula, df_disease):
    """准备最终输入数据"""
    
    # 1. 归一化版本
    df_herbs_norm = prepare_herb_data(df_herb_nes, formula, nor=True)
    result_norm = merge_disease_data(df_herb_nes, df_disease)
    result_nes_norm = process_disease_data(result_norm, nor=True)
    input_array1 = np.column_stack((result_nes_norm, df_herbs_norm))
    
    # 2. 原始版本 (无归一化)
    df_herbs_raw = prepare_herb_data(df_herb_nes, formula, nor=False)
    result_raw = merge_disease_data(df_herb_nes, df_disease)
    result_nes_raw = process_disease_data(result_raw, nor=False)
    input_array2 = np.column_stack((result_nes_raw, df_herbs_raw))
    
    return input_array1, input_array2