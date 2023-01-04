from torch import nn


# 搭建神经网络
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
