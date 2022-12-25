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
