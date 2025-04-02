import torch
print(torch.cuda.is_available())       # True여야 GPU 사용 가능
print(torch.cuda.get_device_name(0))   # 예: 'NVIDIA GeForce RTX 4060'
