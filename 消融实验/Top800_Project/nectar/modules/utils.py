def create_result_folders(base_dir='results', model_name="best_herb_model_advanced"):
    from datetime import datetime
    import os
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # 【修改点】文件名 = 时间戳 + 模型名
    folder_name = f"{model_name.replace('.pth', '')}_{timestamp}"
    result_folder = f'{base_dir}/{folder_name}'
    
    os.makedirs(f'{result_folder}/weights', exist_ok=True)
    os.makedirs(f'{result_folder}/plots', exist_ok=True)
    return result_folder