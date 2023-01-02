import torch
from torch import nn

# L1 损失函数
inputs = torch.tensor([1, 2, 3], dtype=torch.float32)
targets = torch.tensor([2, 3, 5], dtype=torch.float32)

loss1 = nn.L1Loss()
print(loss1(inputs, targets))

loss2 = nn.L1Loss(reduction='none')
print(loss2(inputs, targets))

loss3 = nn.L1Loss(reduction='sum')
print(loss3(inputs, targets))

# 交叉熵损失函数
inputs = torch.tensor([[0.1, 0.2, 0.3],
                       [0.2, 0.1, 0.7]])
targets = torch.tensor([1, 2])
cross_entropy_loss = nn.CrossEntropyLoss()
print(cross_entropy_loss(inputs, targets))
