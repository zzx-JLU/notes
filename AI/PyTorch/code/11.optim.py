import torchvision
from torch import nn
import torch
from torch.utils.data import DataLoader


class Cifar10(nn.Module):
    def __init__(self):
        super(Cifar10, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, (5, 5), padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, (5, 5), padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, (5, 5), padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.model(x)


model = Cifar10()  # 模型
loss = nn.CrossEntropyLoss()  # 损失函数
optim = torch.optim.SGD(model.parameters(), lr=0.01)  # 优化器

# 加载数据
test_set = torchvision.datasets.CIFAR10('./data/CIFAR10', train=False, download=True,
                                        transform=torchvision.transforms.ToTensor())
test_loader = DataLoader(test_set, batch_size=64)

# 训练模型
for epoch in range(20):
    # 每个 epoch 将所有数据看一遍
    running_loss = 0.0
    for imgs, targets in test_loader:
        optim.zero_grad()  # 将梯度值清零
        outputs = model.forward(imgs)  # 前向传播
        error = loss(outputs, targets)  # 计算误差
        error.backward()  # 反向传播，得到梯度值
        optim.step()  # 执行优化算法
        running_loss = running_loss + error
    print(running_loss)
