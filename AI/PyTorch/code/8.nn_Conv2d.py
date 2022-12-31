import torchvision
from torch import nn
from torch.utils.data import DataLoader


class ConvTest(nn.Module):
    def __init__(self):
        super(ConvTest, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=(3, 3))

    def forward(self, x):
        return self.conv1(x)


test_set = torchvision.datasets.CIFAR10('./data/CIFAR10', train=False, download=True,
                                        transform=torchvision.transforms.ToTensor())
test_loader = DataLoader(test_set, batch_size=64)

model = ConvTest()
for data in test_loader:
    imgs, targets = data
    output = model.forward(imgs)
    print(output.shape)
