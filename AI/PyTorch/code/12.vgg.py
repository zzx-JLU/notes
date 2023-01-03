import torchvision
from torch import nn

vgg16 = torchvision.models.vgg16()
print(vgg16)
print(vgg16.classifier)
print(vgg16.classifier[6])

# vgg16.classifier.add_module('output', nn.Linear(1000, 10))
# print(vgg16.classifier)

vgg16.classifier[6] = nn.Linear(4096, 10)
print(vgg16.classifier)
