import pandas as pd
import gseapy as gp
import os

# ==============================================================================
# 1. 读取数据
# ==============================================================================
# 确保文件都在当前目录下，或者填写绝对路径
try:
    # 读取疾病数据
    df_ead = pd.read_csv("/home/wzz/Disease_nes/OriginSource/AD_single疾病数据.csv")
    
    # 读取通路-靶标关联
    df_pathway_target = pd.read_excel("OriginSource/通路Pahway-靶标target关联.xlsx")
    
    # 读取全部靶标信息
    df_target_info = pd.read_excel("OriginSource/全部靶标信息.xlsx")

    print("数据读取成功！")

except FileNotFoundError as e:
    print(f"错误：找不到文件，请检查文件名和路径。\n{e}")
    exit()

# ==============================================================================
# 2. 构建自定义通路背景集 (Gene Sets)
# ==============================================================================
# 目标：构建一个字典，格式为 { 'Pathway_ID_1': ['GeneA', 'GeneB'...], ... }

# 第一步：通过 TCM_TarID 将 "通路表" 和 "靶标信息表" 合并
# 这样我们就把 Pathway_ID 和 Symbol 关联起来了
merged_bg = pd.merge(
    df_pathway_target[['Pathway_ID', 'TCM_TarID']], 
    df_target_info[['Symbol', 'TCM_TarID']], 
    on='TCM_TarID', 
    how='inner'
)

# 去除 Symbol 为空的行和重复行
merged_bg = merged_bg.dropna(subset=['Symbol', 'Pathway_ID']).drop_duplicates()

# 第二步：转换为 gseapy 需要的字典格式
# key 是 Pathway_ID，value 是该通路下所有 Symbol 组成的列表
gene_sets_dict = merged_bg.groupby('Pathway_ID')['Symbol'].apply(list).to_dict()

print(f"背景集构建完成，共包含 {len(gene_sets_dict)} 条通路。")

# ==============================================================================
# 3. 准备排序基因列表 (Ranked List)
# ==============================================================================
# GSEA 需要根据 Log2FC 对基因进行降序排列

# 处理重复基因：如果有相同的 SYMBOL，取 Log2FC 的平均值
rnk = df_ead.groupby('SYMBOL')['LOG2FC'].mean()

# 极其重要：必须按值从大到小排序
rnk = rnk.sort_values(ascending=False)

print(f"排序基因列表准备完成，共包含 {len(rnk)} 个基因。")

# ==============================================================================
# 4. 运行 GSEA (Prerank 模式)
# ==============================================================================
print("正在运行 GSEA 分析，请稍候...")

# gseapy.prerank 专门用于你这种提供了排序列表(rnk)的情况
pre_res = gp.prerank(
    rnk=rnk,                # 排序好的基因 Series
    gene_sets=gene_sets_dict, # 自定义的通路字典
    processes=4,            # 线程数，可根据电脑配置调整
    permutation_num=1000,   # 置换次数，通常 1000
    outdir=None,            # 不自动输出复杂报告文件夹，后面手动保存
    min_size=5,             # 最小通路基因数
    max_size=1000,          # 最大通路基因数
    seed=1234               # 随机种子，保证结果可复现
)

# ==============================================================================
# 5. 整理并导出结果
# ==============================================================================

# 提取结果 DataFrame
# gseapy 的结果在 res2d 中
results_df = pre_res.res2d.copy()

# 筛选需要的列。gseapy 输出的通路名称列叫 'Term'
final_df = results_df[['Term', 'NES']].copy()

# 重命名列以符合你的要求：第一列 ID，第二列 NES
final_df.columns = ['ID', 'NES']

# 保存为 CSV
output_filename = "result/AD_single疾病_通路NES结果.csv"
final_df.to_csv(output_filename, index=False)

print("-" * 30)
print(f"分析完成！")
print(f"结果已保存为: {output_filename}")
print(f"结果预览:")
print(final_df.head())