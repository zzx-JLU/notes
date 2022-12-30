import torch
from torch import nn


class Demo(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        output = input + 1
        return output


demo = Demo()
x = torch.tensor(1.0)
y = demo.forward(x)
print(y)
