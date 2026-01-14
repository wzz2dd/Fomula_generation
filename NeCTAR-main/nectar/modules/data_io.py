# data_io.py
import pandas as pd
import pickle



def load_herb_info(file_path):
    """加载中药信息"""
    return pd.read_excel(file_path)

def load_herb_nes(file_path):
    """加载中药 NES 数据(自动识别 csv / tsv)"""
    # 如果是 csv 文件，用逗号分隔
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path, encoding='utf-8', engine='python')
    # 否则按 txt/tsv 读取
    return pd.read_csv(file_path, sep='\t')

def load_dosage_info(file_path):
    """加载中药的计量范围"""
    return pd.read_csv(file_path, sep='\t')
'''
def load_disease_data(file_path):
    """加载疾病 NES 数据"""
    with open(file_path, 'rb') as f:
        resultList = pickle.load(f)
    return pd.DataFrame(resultList)[['ID', "NES"]]
'''

def load_disease_data(file_path):
    """加载疾病 NES 数据"""
    try:
        # 首先尝试读取CSV文件
        df = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(file_path, encoding='gbk')
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='latin1')
    
    # 打印所有列名以便调试
    print(f"CSV文件中的列名: {df.columns.tolist()}")
    print(f"数据框形状: {df.shape}")
    print("前几行数据:")
    print(df.head())
    
    # 检查是否有ID和NES列
    if 'ID' not in df.columns:
        print("警告: 未找到'ID'列")
        # 尝试找到可能是ID的列
        potential_id_cols = [col for col in df.columns if 'id' in col.lower() or 'pathway' in col.lower()]
        if potential_id_cols:
            print(f"可能是ID的列: {potential_id_cols}")
            # 使用第一个可能的ID列
            df = df.rename(columns={potential_id_cols[0]: 'ID'})
        else:
            # 如果没有找到，使用第一列作为ID
            print(f"使用第一列 '{df.columns[0]}' 作为ID")
            df = df.rename(columns={df.columns[0]: 'ID'})
    
    if 'NES' not in df.columns:
        print("警告: 未找到'NES'列")
        # 尝试找到可能是NES的列
        potential_nes_cols = [col for col in df.columns if 'nes' in col.lower() or 'enrichment' in col.lower()]
        if potential_nes_cols:
            print(f"可能是NES的列: {potential_nes_cols}")
            # 使用第一个可能的NES列
            df = df.rename(columns={potential_nes_cols[0]: 'NES'})
        else:
            # 如果没有找到，查看数值列
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                print(f"数值列: {numeric_cols}")
                # 使用第一个数值列作为NES
                df = df.rename(columns={numeric_cols[0]: 'NES'})
            else:
                print("错误: 未找到数值列，无法确定NES列")
                return None
    
    # 确保ID列是字符串类型
    df['ID'] = df['ID'].astype(str)
    
    print(f"处理后列名: {df.columns.tolist()}")
    
    # 返回ID和NES列
    return df[['ID', 'NES']]