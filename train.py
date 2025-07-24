from net import AlexNet
from torch import nn

model = AlexNet()

n_params = sum(p.numel() for p in model.parameters())

print(f'Number of paramters: {n_params / 1e6:.2f}M') # ~60M