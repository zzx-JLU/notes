import torch
from torch import nn


class PoolTest(nn.Module):
    def __init__(self):
        super(PoolTest, self).__init__()
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, ceil_mode=True)

    def forward(self, x):
        return self.maxpool1(x)


model = PoolTest()

x = torch.tensor([[1, 2, 0, 3, 1],
                  [0, 1, 2, 3, 1],
                  [1, 2, 1, 0, 0],
                  [5, 2, 3, 1, 1],
                  [2, 1, 0, 1, 1]], dtype=torch.float32)
x = torch.reshape(x, (1, 1, 5, 5))

output = model(x)
print(output)
