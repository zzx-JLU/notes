---
title: PyTorch入门
chrome:
    format: "A4"
    headerTemplate: '<div></div>'
    footerTemplate: '<div style="width:100%; text-align:center; border-top: 1pt solid #eeeeee; margin: 10px 10px 10px; font-size: 8pt;">
    <span class=pageNumber></span> / <span class=totalPages></span></div>'
    displayHeaderFooter: true
    margin:
        top: '40px'
        bottom: '65px'
        left: '40px'
        right: '40px'
---

<h1>PyTorch入门</h1>

<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->

- [1 常用工具](#1-常用工具)
  - [1.1 `Dataset`类](#11-dataset类)
  - [1.2 TensorBoard](#12-tensorboard)
  - [1.3 Transforms](#13-transforms)
  - [1.4 torchvision中的数据集](#14-torchvision中的数据集)
  - [1.5 `DataLoader`类](#15-dataloader类)
- [2 神经网络](#2-神经网络)
  - [2.1 神经网络的基本骨架](#21-神经网络的基本骨架)
  - [2.2 卷积层](#22-卷积层)
  - [2.3 池化层](#23-池化层)
  - [2.4 非线性激活函数](#24-非线性激活函数)
  - [2.5 线性层](#25-线性层)
  - [2.6 `Sequential`](#26-sequential)
  - [2.7 损失函数](#27-损失函数)
  - [2.8 优化器](#28-优化器)
  - [2.9 现有网络模型的使用及修改](#29-现有网络模型的使用及修改)
  - [2.10 网络模型的保存与读取](#210-网络模型的保存与读取)

<!-- /code_chunk_output -->

# 1 常用工具

## 1.1 `Dataset`类

`Dataset`类用于导入数据。使用方法为：

1. 导入`Dataset`类：`from torch.utils.data import Dataset`
2. 建立`Dataset`的子类。
3. 重写`__int__()`、`__getitem__()`、`__len__()`方法。
4. 创建类对象，使用`[index]`获得数据。当使用索引时，会自动调用`__getitem__()`方法。

例如，hymenoptera 数据集将文件夹名作为标签名，处理方法如下：

```python
from torch.utils.data import Dataset
import os
from PIL import Image


# 建立 Dataset 的子类
class MyData(Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir  # 根目录
        self.label_dir = label_dir  # 标签目录，既是文件夹名，也是标签名
        self.path = os.path.join(self.root_dir, self.label_dir)  # 文件夹路径
        self.files = os.listdir(self.path)  # 文件名

    def __getitem__(self, index):
        file_name = self.files[index]  # 获得指定索引的文件名
        file_path = os.path.join(self.path, file_name)  # 文件完整路径
        img = Image.open(file_path)  # 读取文件。文件类型不同，处理方法也不同
        label = self.label_dir  # 标签
        return img, label

    def __len__(self):
        return len(self.files)


if __name__ == '__main__':
    train_dir = './data/hymenoptera_data/train/'  # 训练集根目录
    ants_dir = 'ants'
    bees_dir = 'bees'

    ants_data = MyData(train_dir, ants_dir)  # 获取蚂蚁数据
    img, label = ants_data[0]  # 获取第一条蚂蚁数据
    img.show()

    bees_data = MyData(train_dir, bees_dir)  # 获取蜜蜂数据
```

## 1.2 TensorBoard

`SummaryWriter`类用于向指定目录下的事件文件写入数据，供 TenserBoard 使用。通常用于保存训练过程中的数据和图像，便于观察训练过程。

要使用`SummaryWriter`类，需要安装 tensorboard 包。

> `AttributeError: module 'distutils' has no attribute 'version'`解决方法
>
> 报错原因：setuptools 版本过高
>
> 解决方案：
>
> 1. 卸载 setuptools：`pip uninstall setuptools`。注意要用 pip 卸载，不能用 conda。
> 2. 安装低版本的 setuptools：`pip install setuptools==58.0.4`

创建`SummaryWriter`对象时，需要指定一个路径。如果不指定，默认路径为`runs/**CURRENT_DATETIME_HOSTNAME**`。例如：

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs")
```

`add_scalar()`方法用于添加标量数据。参数为：

1. `tag`：标题
2. `scalar_value`：要保存的标量值，作为 y 轴
3. `global_step`：步骤数，作为 x 轴

```python
for i in range(100):
    writer.add_scalar('y=2x', 2 * i, i)
```

要想查看事件文件，可以在命令行输入`tensorboard --logdir=*** --port=****`，然后点击生成的 URL。

`add_image()`方法用于添加图像。参数为：

1. `tag`：标题
2. `img_tensor`：图像数据。类型可以是`torch.Tensor`、`numpy.array`或`string`。
3. `global_step`：步骤数
4. `dataformats`：数据格式。默认值为`'CHW'`，表示通道、高度、宽度三个维度的顺序。可以根据实际数据的格式进行设置。

```python
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np

writer = SummaryWriter("logs")

img = Image.open('./data/hymenoptera_data/train/ants/0013035.jpg')
img_array = np.array(img)

writer.add_image('train', img_array, 1, dataformats='HWC')

# 使用完毕后需要关闭
writer.close()
```

## 1.3 Transforms

Transforms 用于处理图像，通常用于数据预处理。

`torchvision.transforms`包中提供了很多类，具有不同的功能。这些类的使用方法为：

1. 创建类实例。
2. 直接调用类实例，传入要求的参数。

`ToTensor`类用于将`PIL.Image`或`numpy.ndarray`类型的图像数据转换成`Tensor`类型。例如：

```python
from torchvision import transforms
from PIL import Image

img = Image.open('./data/hymenoptera_data/train/ants/0013035.jpg')
print(type(img))  # <class 'PIL.JpegImagePlugin.JpegImageFile'>

tensor_trans = transforms.ToTensor()  # 创建类实例
img_tensor = tensor_trans(img)  # 调用类实例，执行转换
print(type(img_tensor))  # <class 'torch.Tensor'>
```

`ToPILImage`类用于将`Tensor`或`numpy.ndarray`类型的图像数据转换成`PIL.Image`类型。

`Normalize`类用均值和标准差对`Tensor`类型的图像数据进行归一化。例如：

```python
from torchvision import transforms
from PIL import Image

img = Image.open('./data/hymenoptera_data/train/ants/0013035.jpg')

# ToTensor: 转换为 Tensor 类型
trans_tensor = transforms.ToTensor()
img_tensor = trans_tensor(img)
print(img_tensor)
# tensor([[[0.3137, 0.3137, 0.3137,  ..., 0.3176, 0.3098, 0.2980],
#          [0.3176, 0.3176, 0.3176,  ..., 0.3176, 0.3098, 0.2980],
#          [0.3216, 0.3216, 0.3216,  ..., 0.3137, 0.3098, 0.3020],
#          ...,
#          [0.3412, 0.3412, 0.3373,  ..., 0.1725, 0.3725, 0.3529],
#          [0.3412, 0.3412, 0.3373,  ..., 0.3294, 0.3529, 0.3294],
#          [0.3412, 0.3412, 0.3373,  ..., 0.3098, 0.3059, 0.3294]],
#
#         [[0.5922, 0.5922, 0.5922,  ..., 0.5961, 0.5882, 0.5765],
#          [0.5961, 0.5961, 0.5961,  ..., 0.5961, 0.5882, 0.5765],
#          [0.6000, 0.6000, 0.6000,  ..., 0.5922, 0.5882, 0.5804],
#          ...,
#          [0.6275, 0.6275, 0.6235,  ..., 0.3608, 0.6196, 0.6157],
#          [0.6275, 0.6275, 0.6235,  ..., 0.5765, 0.6275, 0.5961],
#          [0.6275, 0.6275, 0.6235,  ..., 0.6275, 0.6235, 0.6314]],
#
#         [[0.9137, 0.9137, 0.9137,  ..., 0.9176, 0.9098, 0.8980],
#          [0.9176, 0.9176, 0.9176,  ..., 0.9176, 0.9098, 0.8980],
#          [0.9216, 0.9216, 0.9216,  ..., 0.9137, 0.9098, 0.9020],
#          ...,
#          [0.9294, 0.9294, 0.9255,  ..., 0.5529, 0.9216, 0.8941],
#          [0.9294, 0.9294, 0.9255,  ..., 0.8863, 1.0000, 0.9137],
#          [0.9294, 0.9294, 0.9255,  ..., 0.9490, 0.9804, 0.9137]]])

# Normalize: 对 Tensor 类型的图像数据进行归一化
trans_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
img_norm = trans_norm(img_tensor)
print(img_norm)
# tensor([[[-0.3725, -0.3725, -0.3725,  ..., -0.3647, -0.3804, -0.4039],
#          [-0.3647, -0.3647, -0.3647,  ..., -0.3647, -0.3804, -0.4039],
#          [-0.3569, -0.3569, -0.3569,  ..., -0.3725, -0.3804, -0.3961],
#          ...,
#          [-0.3176, -0.3176, -0.3255,  ..., -0.6549, -0.2549, -0.2941],
#          [-0.3176, -0.3176, -0.3255,  ..., -0.3412, -0.2941, -0.3412],
#          [-0.3176, -0.3176, -0.3255,  ..., -0.3804, -0.3882, -0.3412]],
#
#         [[ 0.1843,  0.1843,  0.1843,  ...,  0.1922,  0.1765,  0.1529],
#          [ 0.1922,  0.1922,  0.1922,  ...,  0.1922,  0.1765,  0.1529],
#          [ 0.2000,  0.2000,  0.2000,  ...,  0.1843,  0.1765,  0.1608],
#          ...,
#          [ 0.2549,  0.2549,  0.2471,  ..., -0.2784,  0.2392,  0.2314],
#          [ 0.2549,  0.2549,  0.2471,  ...,  0.1529,  0.2549,  0.1922],
#          [ 0.2549,  0.2549,  0.2471,  ...,  0.2549,  0.2471,  0.2627]],
#
#         [[ 0.8275,  0.8275,  0.8275,  ...,  0.8353,  0.8196,  0.7961],
#          [ 0.8353,  0.8353,  0.8353,  ...,  0.8353,  0.8196,  0.7961],
#          [ 0.8431,  0.8431,  0.8431,  ...,  0.8275,  0.8196,  0.8039],
#          ...,
#          [ 0.8588,  0.8588,  0.8510,  ...,  0.1059,  0.8431,  0.7882],
#          [ 0.8588,  0.8588,  0.8510,  ...,  0.7725,  1.0000,  0.8275],
#          [ 0.8588,  0.8588,  0.8510,  ...,  0.8980,  0.9608,  0.8275]]])
```

`Resize`类用于将`PIL.Image`类型的图像数据修改为指定的大小。例如：

```python
from torchvision import transforms
from PIL import Image

img = Image.open('./data/hymenoptera_data/train/ants/0013035.jpg')
print(img)  # <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=768x512 at 0x1B743128F40>

trans_resize = transforms.Resize((512, 512))
img_resize = trans_resize(img)
print(img_resize)  # <PIL.Image.Image image mode=RGB size=512x512 at 0x1B74B862B80>
```

`Compose`类可以将多个 Transforms 类组合在一起。构造`Compose`类对象时传入 Transforms 类对象的列表，当调用`Compose`类对象时，将依次调用列表中的 Transforms 类对象。例如：

```python
from torchvision import transforms
from PIL import Image

img = Image.open('./data/hymenoptera_data/train/ants/0013035.jpg')
print(img.size)  # (768, 512)

trans_compose = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor()
])
img_compose = trans_compose(img)
print(type(img_compose))  # <class 'torch.Tensor'>
print(img_compose.size())  # torch.Size([3, 256, 384])
```

`RandomCrop`类用于对`PIL`类型的图像数据进行随机裁剪，在随机位置裁剪出指定大小的子图像。

## 1.4 torchvision中的数据集

`torchvision.datasets`模块中提供了一些标准数据集，可以直接使用。

例如，CIFAR10 是一个用于物体识别的数据集。`torchvision.datasets.CIFAR10()`用于获取 CIFAR10 数据集，参数为：

1. `root`：保存数据集的根目录。
2. `train`：布尔值，为`True`则创建训练集，为`False`则创建测试集。可选，默认为`True`。
3. `transform`：一个函数或 Transforms 对象，要求参数为`PIL`类型，对图像进行处理。可选，默认为`None`。
4. `target_transform`：一个函数，作用于 target。可选，默认为`None`。
5. `download`：布尔值，如果为`True`，则下载数据集并保存到`root`参数指定的路径。可选，默认为`False`。

对于返回的数据集，可以使用整数索引获取每条数据。整数索引返回一个元组，第一个元素为`PIL`图像数据；第二个元素为 target，是一个整数，表示类别。`classes`属性是一个列表，保存了整数与类别名的对应关系，可以使用 target 作为索引获取相应类别。例如：

```python
import torchvision

train_set = torchvision.datasets.CIFAR10('./data/CIFAR10', train=True, download=True)
test_set = torchvision.datasets.CIFAR10('./data/CIFAR10', train=False, download=True)

print(len(train_set))  # 50000
print(train_set[0])  # (<PIL.Image.Image image mode=RGB size=32x32 at 0x1E39F0A98E0>, 6)

img, target = train_set[0]
print(img)  # <PIL.Image.Image image mode=RGB size=32x32 at 0x1E39F0A98E0>
print(target)  # 6

print(train_set.classes)
# ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

print(train_set.classes[target])  # frog
```

## 1.5 `DataLoader`类

`torch.utils.data.DataLoader`类用于加载数据，作为人工智能模型的输入。

使用`DataLoader()`创建类对象，参数为：

1. `dataset`：`Dataset`对象，从该数据集中加载数据。
2. `batch_size`：每批次加载的样本数量。可选，默认值为 1。
3. `shuffle`：布尔值，若为`True`则将每个 epoch 的数据打乱。可选，默认值为`False`。
4. `num_workers`：用于加载数据的子进程数量，如果为 0 则在主进程中加载数据。可选，默认值为 0。
5. `drop_last`：布尔值，如果为`True`，则当数据集大小无法被`batch_size`整除时，舍去最后一个不完整的批次；如果为`False`，则不会舍去，将最后一个批次的大小变小。可选，默认值为`False`。

可以用 for 循环遍历`DataLoader`对象，获取各个批次，每个批次的数据封装在一起。例如：

```python
import torchvision
from torch.utils.data import DataLoader

test_set = torchvision.datasets.CIFAR10('./data/CIFAR10', train=False, download=True,
                                        transform=torchvision.transforms.ToTensor())

test_loader = DataLoader(test_set, batch_size=64)

for batch in test_loader:
    # CIFAR10 的 Dataset 对象返回 img 和 target
    # 在 DataLoader 中，将每个批次的 img 和 target 分别包装到一起
    imgs, targets = batch
    print(imgs.shape)
    print(targets)
```

# 2 神经网络

## 2.1 神经网络的基本骨架

`torch.nn.Module`类是所有神经网络模块的基类，自定义的模型需要继承`Module`类。`forward()`方法定义了前向传播操作，子类必须重写该方法。

神经网络模型的基本结构如下所示：

```python
import torch
from torch import nn


class Demo(nn.Module):
    # 定义模型结构
    def __init__(self):
        super(Demo, self).__init__()

    # 前向传播
    def forward(self, input):
        output = input + 1
        return output


demo = Demo()
x = torch.tensor(1.0)  # 输入数据
y = demo.forward(x)  # 执行前向传播，得到输出
print(y)  # tensor(2.)
```

## 2.2 卷积层

`torch.nn.functional`模块提供了神经网络中常用的函数，其中`conv2d()`函数实现了二维矩阵的卷积运算。参数为：

1. `input`：输入数据，`Tensor`类型，形状为 $(\text{minibatch}, \text{in\_channels}, iH, iW)$。
2. `weight`：卷积核，`Tensor`类型，形状为 $(\text{out\_channels}, \frac{\text{in\_channels}}{\text{groups}}, kH, kW)$。
3. `bias`：偏移量，`Tensor`类型，形状为 $(\text{out\_channels})$。可选，默认值为`None`。
4. `stride`：卷积核移动的步长。如果取值为单个整数，则将垂直和水平方向的步长设置为相同值；如果取值为元组`(sH, sW)`，则分别设置垂直和水平方向的步长。可选，默认值为 1。
5. `padding`：在输入矩阵周围补零的宽度，可以为单个整数，也可以为元组`(padH, padW)`。可选，默认值为 0。
6. `dilation`：卷积核元素之间的距离，可以为单个整数，也可以为元组`(dH, dW)`。可选，默认值为 1。
7. `groups`：将输入数据的通道分组，$\text{in\_channels}$ 必须能够被`groups`值整除。可选，默认值为 1。

```python
import torch
import torch.nn.functional as F

x = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]])

kernel = torch.tensor([[1, 2, 1],
                       [0, 1, 0],
                       [2, 1, 0]])

# 调整形状
x = torch.reshape(x, (1, 1, 5, 5))
kernel = torch.reshape(kernel, (1, 1, 3, 3))

output = F.conv2d(x, kernel)
print(output)
# tensor([[[[10, 12, 12],
#           [18, 16, 16],
#           [13,  9,  3]]]])

output2 = F.conv2d(x, kernel, stride=2)
print(output2)
# tensor([[[[10, 12],
#           [13,  3]]]])

output3 = F.conv2d(x, kernel, padding=1)
print(output3)
# tensor([[[[ 1,  3,  4, 10,  8],
#           [ 5, 10, 12, 12,  6],
#           [ 7, 18, 16, 16,  8],
#           [11, 13,  9,  3,  4],
#           [14, 13,  9,  7,  4]]]])
```

`torch.nn.Conv2d`类定义了二维卷积层，实例化参数为：

1. `in_channels`：输入图像的通道数。
2. `out_channels`：输出通道数。卷积核的个数与输出通道数相同。
3. `kernel_size`：元组，卷积核的形状。
4. `stride`：卷积核移动的步长。可选，默认值为 1。
5. `padding`：输入图像的边距。可选，默认值为 0。
6. `dilation`：卷积核元素之间的距离。可选，默认值为 1。
7. `groups`：分组数。可选，默认值为 1。
8. `bias`：布尔值，设置是否偏置。可选，默认值为`True`。
9. `padding_mode`：边距的填充方式。可选，默认值为`'zeros'`，表示在填充位置补零。

设输入数据的形状为 $(N, C_{\text{in}}, H_{\text{in}}, W_{\text{in}})$，输出数据的形状为 $(N, C_{\text{out}}, H_{\text{out}}, W_{\text{out}})$，则输出数据的形状可以按照下列公式确定：

$$
H_{\text{out}} = \left\lfloor \dfrac{H_{\text{in}} + 2 \times \text{padding}[0] - \text{dilation}[0] \times (\text{kernel\_size}[0] - 1) - 1}{\text{stride}[0]} + 1 \right\rfloor \\[0.5em]
W_{\text{out}} = \left\lfloor \dfrac{W_{\text{in}} + 2 \times \text{padding}[1] - \text{dilation}[1] \times (\text{kernel\_size}[1] - 1) - 1}{\text{stride}[1]} + 1 \right\rfloor
$$

网络示例：

```python
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
```

## 2.3 池化层

`torch.nn.MaxPool2d`类定义了最大池化层，实例化参数为：

1. `kernel_size`：池化核的形状。
2. `stride`：池化核移动的步长。可选，默认值为`None`，此时取值与`kernel_size`相同。
3. `padding`：输入数据的边距宽度。可选，默认值为 0。
4. `dilation`：池化核中元素的间距。可选，默认值为 1。
5. `return_indices=False`
6. `ceil_mode`：取值为`True`时，输出数据的形状用 $\lceil \, \rceil$ 计算，此时当池化核窗口内有空数据时，保留该次取得的数据；取值为`False`时，输出数据的形状用 $\lfloor \, \rfloor$ 计算，此时当池化核窗口内有空数据时，舍去该次取得的数据。可选，默认为`False`。

在最简单的情况下，设输入数据的形状为 $(N,C,H,W)$，输出数据的形状为 $(N, C, H_{\text{out}}, W_{\text{out}})$，池化核的形状为 $(kH, kW)$，则输出数据的计算方法为：

$$
\begin{aligned}
    \operatorname{out}(N_i, C_j, h, w) = & \max_{m=0,\cdots,kH-1} \max_{n=0,\cdots,kH-1} \\
    & \operatorname{input}(N_i, C_j, \text{stride}[0] \times h + m, \text{stride}[1] \times w + n)
\end{aligned}
$$

即每次选取池化核窗口范围内的最大值。

输出数据的形状可以按照下列公式确定：

$$
H_{\text{out}} = \left\lfloor \dfrac{H + 2 \times \text{padding}[0] - \text{dilation}[0] \times (\text{kernel\_size}[0] - 1) - 1}{\text{stride}[0]} + 1 \right\rfloor \\[0.5em]
W_{\text{out}} = \left\lfloor \dfrac{W + 2 \times \text{padding}[1] - \text{dilation}[1] \times (\text{kernel\_size}[1] - 1) - 1}{\text{stride}[1]} + 1 \right\rfloor
$$

示例代码：

```python
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
# tensor([[[[2., 3.],
#           [5., 1.]]]])
```

## 2.4 非线性激活函数

`torch.nn.ReLU`类定义了 ReLU 激活函数。

- 实例化参数为`inplace`，布尔值，设置是否直接修改输入数据，可选，默认值为`False`。
- 对输入数据的所有元素应用 ReLU 激活函数，输入数据与输出数据形状相同。
- ReLU 激活函数的数学表达式为
  $$
  \operatorname{ReLU}(x) = \max(0, x) = \begin{cases}
    x & x > 0 \\
    0 & x \leqslant 0
  \end{cases}
  $$

`torch.nn.Sigmoid`类定义了 Sigmoid 激活函数，数学表达式为

$$
\operatorname{sigmoid}(x) = \sigma(x) = \dfrac{1}{1 + e^{-x}}
$$

## 2.5 线性层

`torch.nn.Linear`类定义了线性层，即全连接层。实例化参数为：

1. `in_features`：输入数据的特征数，即每个输入样本的长度。
2. `out_features`：输出数据的特征数，即每个输出样本的长度。
3. `bias`：布尔值，设置是否训练偏置。可选，默认值为`True`。

## 2.6 `Sequential`

`torch.nn.Sequential`类是一个序列容器，可以将多个模块按照构造器中指定的顺序添加到容器中。调用容器时，将会按顺序依次执行各个模块。

## 2.7 损失函数

`torch.nn.L1Loss`类定义了平均绝对误差（mean absolute error，MAE），也称为 L1 损失函数。实例化参数为：

1. `reduction`：指定输出结果的压缩策略，可能的取值有`none`、`mean`、`sum`。可选，默认值为`mean`。

设预测结果为 $\hat{y}$，实际值为 $y$，批次大小为 $N$。当`reduction`参数为`none`时，返回值为

$$
l(\hat{y}, y) = (|\hat{y}_1-y_1|, |\hat{y}_2-y_2|, \cdots, |\hat{y}_N-y_N|)
$$

当`reduction`参数为`mean`时，返回值为

$$
l(\hat{y}, y) = \dfrac{1}{N} \sum_{i=1}^N |\hat{y}_i-y_i|
$$

当`reduction`参数为`sum`时，返回值为

$$
l(\hat{y}, y) = \sum_{i=1}^N |\hat{y}_i-y_i|
$$

调用`L1Loss`对象时，要求 input 和 target 形状相同，例如：

```python
import torch
from torch import nn

inputs = torch.tensor([1, 2, 3], dtype=torch.float32)
targets = torch.tensor([2, 3, 5], dtype=torch.float32)

loss1 = nn.L1Loss()
print(loss1(inputs, targets))  # tensor(1.3333)

loss2 = nn.L1Loss(reduction='none')
print(loss2(inputs, targets))  # tensor([1., 1., 2.])

loss3 = nn.L1Loss(reduction='sum')
print(loss3(inputs, targets))  # tensor(4.)
```

`torch.nn.MSELoss`类定义了均方误差（mean square error，MSE）。当`reduction`参数为`none`时，返回值为

$$
\operatorname{MSE}(\hat{y}, y) = ((\hat{y}_1-y_1)^2, (\hat{y}_2-y_2)^2, \cdots, (\hat{y}_N-y_N)^2)
$$

当`reduction`参数为`mean`时，返回值为

$$
\operatorname{MSE}(\hat{y}, y) = \dfrac{1}{N} \sum_{i=1}^N (\hat{y}_i-y_i)^2
$$

当`reduction`参数为`sum`时，返回值为

$$
\operatorname{MSE}(\hat{y}, y) = \sum_{i=1}^N (\hat{y}_i-y_i)^2
$$

`torch.nn.CrossEntropyLoss`类定义了交叉熵损失函数，常用于分类问题。设类别数为 $C$，批次大小为 $N$，则 input 的形状为 $(N,C)$ 或 $(N,C,d_1,d_2,\cdots,d_K)$，target 的形状为 $(N)$ 或 $(N,d_1,d_2,\cdots,d_K)$。例如：

```python
import torch
from torch import nn

inputs = torch.tensor([[0.1, 0.2, 0.3],
                       [0.2, 0.1, 0.7]])
targets = torch.tensor([1, 2])
cross_entropy_loss = nn.CrossEntropyLoss()
print(cross_entropy_loss(inputs, targets))  # tensor(0.9349)
```

对损失函数的返回结果调用`backward()`方法，可以执行反向传播，获得模型参数的梯度值。

## 2.8 优化器

`torch.optim`模块包含多种优化器。优化器的使用方法：

1. 构造优化器对象。优化器会保存当前状态，并根据梯度值更新参数。构造优化器时，必须传递模型参数，之后可以指定优化器特定的可选参数。
2. 调用`zero_grad()`方法清空梯度值。
3. 计算梯度值。
4. 调用`step()`方法，执行优化算法。

`Optimizer`类是所有优化器的基类，定义了优化器的基本方法。

使用优化器训练模型的示例如下：

```python
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
```

## 2.9 现有网络模型的使用及修改

PyTorch 提供了很多成熟的模型。例如，`torchvision.models`模块提供了计算机视觉常用的模型，下面以 VGG 模型为例，介绍现有网络模型的使用及修改方法。

`torchvision.models.vgg16()`函数可以获得 VGG16 模型。参数为：

1. `pretrained`：布尔值，如果为`True`，则返回在 ImageNet 数据集上预训练的模型。可选，默认值为`False`。
2. `progress`：布尔值，如果为`True`，则在`stderr`显示下载进度条。可选，默认值为`True`。

用`print()`输出模型，可以看到模型的结构，例如：

```python
import torchvision

vgg16 = torchvision.models.vgg16()
print(vgg16)
# VGG(
#   (features): Sequential(
#     (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (1): ReLU(inplace=True)
#     (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (3): ReLU(inplace=True)
#     (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (6): ReLU(inplace=True)
#     (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (8): ReLU(inplace=True)
#     (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (11): ReLU(inplace=True)
#     (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (13): ReLU(inplace=True)
#     (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (15): ReLU(inplace=True)
#     (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (18): ReLU(inplace=True)
#     (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (20): ReLU(inplace=True)
#     (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (22): ReLU(inplace=True)
#     (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (25): ReLU(inplace=True)
#     (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (27): ReLU(inplace=True)
#     (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (29): ReLU(inplace=True)
#     (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   )
#   (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
#   (classifier): Sequential(
#     (0): Linear(in_features=25088, out_features=4096, bias=True)
#     (1): ReLU(inplace=True)
#     (2): Dropout(p=0.5, inplace=False)
#     (3): Linear(in_features=4096, out_features=4096, bias=True)
#     (4): ReLU(inplace=True)
#     (5): Dropout(p=0.5, inplace=False)
#     (6): Linear(in_features=4096, out_features=1000, bias=True)
#   )
# )
```

模型中的每个模块都有一个名字，可以像访问属性一样访问这些模块。例如：

```python
import torchvision

vgg16 = torchvision.models.vgg16()
print(vgg16.classifier)
# Sequential(
#   (0): Linear(in_features=25088, out_features=4096, bias=True)
#   (1): ReLU(inplace=True)
#   (2): Dropout(p=0.5, inplace=False)
#   (3): Linear(in_features=4096, out_features=4096, bias=True)
#   (4): ReLU(inplace=True)
#   (5): Dropout(p=0.5, inplace=False)
#   (6): Linear(in_features=4096, out_features=1000, bias=True)
# )
```

`add_module()`方法用于向模块中添加子模块。参数为：

1. `name`：名称，可以使用此名称访问子模块。
2. `module`：要添加的模块。

```python
import torchvision
from torch import nn

vgg16 = torchvision.models.vgg16()
vgg16.classifier.add_module('output', nn.Linear(1000, 10))
print(vgg16.classifier)
# Sequential(
#   (0): Linear(in_features=25088, out_features=4096, bias=True)
#   (1): ReLU(inplace=True)
#   (2): Dropout(p=0.5, inplace=False)
#   (3): Linear(in_features=4096, out_features=4096, bias=True)
#   (4): ReLU(inplace=True)
#   (5): Dropout(p=0.5, inplace=False)
#   (6): Linear(in_features=4096, out_features=1000, bias=True)
#   (output): Linear(in_features=1000, out_features=10, bias=True)
# )
```

`Sequential`容器中的模块具有整数索引，可以使用索引访问它们。例如：

```python
import torchvision

vgg16 = torchvision.models.vgg16()
print(vgg16.classifier[6])  # Linear(in_features=4096, out_features=1000, bias=True)
```

可以通过赋值修改模块。例如：

```python
import torchvision
from torch import nn

vgg16 = torchvision.models.vgg16()
vgg16.classifier[6] = nn.Linear(4096, 10)
print(vgg16.classifier)
# Sequential(
#   (0): Linear(in_features=25088, out_features=4096, bias=True)
#   (1): ReLU(inplace=True)
#   (2): Dropout(p=0.5, inplace=False)
#   (3): Linear(in_features=4096, out_features=4096, bias=True)
#   (4): ReLU(inplace=True)
#   (5): Dropout(p=0.5, inplace=False)
#   (6): Linear(in_features=4096, out_features=10, bias=True)
# )
```

## 2.10 网络模型的保存与读取

`torch.save()`函数用于保存模型，`torch.load()`函数用于加载模型。

要想保存模型的完整信息，包括模型结构和参数，可以直接将模型对象传入`save()`函数。加载模型时直接使用`load()`函数就能得到完整的模型。例如：

```python
import torchvision
import torch

vgg16 = torchvision.models.vgg16()
torch.save(vgg16, './model/vgg16.pt')

saved_model = torch.load('./model/vgg16.pt')
```

如果只保存模型参数，可以对模型调用`state_dict()`方法，将返回值传入`save()`函数。此时通过`load()`函数只能得到模型参数，还需要对模型对象调用`load_state_dict()`方法，将参数导入模型。例如：

```python
import torchvision
import torch

vgg16 = torchvision.models.vgg16()
torch.save(vgg16.state_dict(), './model/vgg16_param.pt')

params = torch.load('./model/vgg16_param.pt')
vgg16_load = torchvision.models.vgg16()
vgg16_load.load_state_dict(params)
```
