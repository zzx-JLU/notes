---
title: PyTorch
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

<h1>PyTorch</h1>

<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->

- [1 常用工具](#1-常用工具)
  - [1.1 `Dataset`类](#11-dataset类)
  - [1.2 TensorBoard](#12-tensorboard)
  - [1.3 Transforms](#13-transforms)
  - [1.4 torchvision中的数据集](#14-torchvision中的数据集)

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
