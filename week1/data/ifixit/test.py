import torch
print(torch.__version__)            # 2.3.1
print(torch.cuda.is_available())    # True
print(torch.cuda.get_device_name(0)) # Quadro RTX 5000
