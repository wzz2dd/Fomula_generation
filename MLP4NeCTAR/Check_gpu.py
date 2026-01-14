import torch

print("=== GPU检测报告 ===")
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA是否可用: {torch.cuda.is_available()}")
print(f"CUDA版本: {torch.version.cuda}")

if torch.cuda.is_available():
    print(f"GPU数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  内存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
else:
    print("CUDA不可用，可能的原因:")
    print("1. 没有安装GPU版本的PyTorch")
    print("2. CUDA驱动未安装")
    print("3. GPU不支持CUDA")