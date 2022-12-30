import torchvision
from torch.utils.data import DataLoader

test_set = torchvision.datasets.CIFAR10('./data/CIFAR10', train=False, download=True,
                                        transform=torchvision.transforms.ToTensor())

test_loader = DataLoader(test_set, batch_size=64)

for batch in test_loader:
    imgs, targets = batch
    print(imgs.shape)
    print(targets)
