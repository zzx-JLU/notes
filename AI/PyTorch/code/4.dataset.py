import torchvision

train_set = torchvision.datasets.CIFAR10('./data/CIFAR10', train=True, download=True)
test_set = torchvision.datasets.CIFAR10('./data/CIFAR10', train=False, download=True)

print(len(train_set))
print(train_set[0])

img, target = train_set[0]
print(img)
print(target)

print(train_set.classes)
print(train_set.classes[target])
