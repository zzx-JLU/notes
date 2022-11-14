---
title: Matplotlib
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

<h1>Matplotlib</h1>

<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->

- [1 Matplotlib简介](#1-matplotlib简介)
- [2 PyLab模块](#2-pylab模块)
- [3 `matplotlib.pyplot`模块](#3-matplotlibpyplot模块)
  - [3.1 `matplotlib.pyplot` API](#31-matplotlibpyplot-api)
  - [3.2 简单绘图](#32-简单绘图)

<!-- /code_chunk_output -->

# 1 Matplotlib简介

Matplotlib 是用于数据可视化的 Python 包，它是一个跨平台库，用于根据数组中的数据制作 2D 图。

Matplotlib 是用 Python 编写的，并使用了 NumPy。

Matplotlib 提供了一个面向对象的 API，有助于使用 Python GUI 工具包（如 PyQt）在应用程序中嵌入绘图。Matplotlib 也可以用于 Python、IPython shell、Jupyter Notebook 和 Web 应用程序服务器。

Matplotlib + NumPy 可以视作 MATLAB 的开源等价物。

# 2 PyLab模块

PyLab 是一个面向 Matplotlib 的绘图库接口，其语法和 MATLAB 十分接近。它和`matplotlib.pyplot`模块都能实现 Matplotlib 的绘图功能。

PyLab 是一个单独的模块，随 Matplotlib 软件包一起安装。

基本绘图：提供两个长度相同的数组或序列，用`pylab.plot()`函数绘制曲线，然后用`pylab.show()`函数显示图像。例如：

```python
import numpy as np
import pylab as plb

x = np.linspace(-3, 3, 30)
y = x ** 2

plb.plot(x, y)
plb.show()
```

<div align="center">
    <img src="https://cdn.staticaly.com/gh/zzx-JLU/images_for_markdown@main/Matplotlib/1.png" width=50% style="margin-top: -15px">
</div>

`pylab.plot()`函数的第三个参数是一个字符串，用于设置曲线的样式。可选的设置有：

- 颜色：`b`（蓝色）、`g`（绿色）、`r`（红色）、`c`（青色）、`m`（品红）、`y`（黄色）、`k`（黑色）、`w`（白色）
- 形状：`-`、`--`、`-.`、`:`、`.`、`,`、`_`、`|`、`^`、`v`、`<`、`>`、`s`、`+`、`x`、`D`、`d`、`o`、`h`、`H`、`p`、`1`、`2`、`3`、`4`

```python
import numpy as np
import pylab as plb

x = np.linspace(-3, 3, 30)
y = x ** 2

plb.plot(x, y, "r:")
plb.show()
```

<div align="center">
    <img src="https://cdn.staticaly.com/gh/zzx-JLU/images_for_markdown@main/Matplotlib/2.png" width=50% style="margin-top: -15px">
</div>

如果需要在同一绘图区域内绘制多个图形，只需要使用多个绘图命令。例如：

```python
import numpy as np
import pylab as plb

x = np.linspace(-3, 3, 30)

plb.plot(x, np.sin(x))
plb.plot(x, np.cos(x), 'r:')
plb.plot(x, -np.sin(x), 'b--')
plb.show()
```

<div align="center">
    <img src="https://cdn.staticaly.com/gh/zzx-JLU/images_for_markdown@main/Matplotlib/3.png" width=50% style="margin-top: -15px">
</div>

`pylab.clf()`函数用于清空图像。

# 3 `matplotlib.pyplot`模块

## 3.1 `matplotlib.pyplot` API

`matplotlib.pyplot`是 Matplotlib 中的一个模块，是命令样式函数的集合，使 Matplotlib 像 MATLAB 一样工作。

绘制函数：

| 函数 | 描述 |
| :--: | :--: |
| `bar` | 绘制条形图（柱状图） |
| `barh` | 绘制水平条形图 |
| `boxplot` | 绘制箱型图 |
| `hist` | 绘制直方图 |
| `hist2d` | 绘制 2D 直方图 |
| `pie` | 绘制饼图 |
| `plot` | 绘制平面曲线和/或标记 |
| `polar` | 绘制极坐标图 |
| `scatter` | 绘制 x 和 y 的散点图 |
| `stackplot` | 绘制堆叠图 |
| `stem` | 绘制杆图 |
| `step` | 绘制阶梯图 |
| `quiver` | 绘制二维矢量场 |

图像函数：

| 函数 | 描述 |
| :--: | :--: |
| `imread` | 将文件中的图像读入数组 |
| `imsave` | 将数组保存为图像文件 |
| `imshow` | 将数据显示为图像 |

轴函数：

| 函数 | 描述 |
| :--: | :--: |
| `axes` | 向图像添加轴 |
| `text` | 向轴添加文本 |
| `title` | 设置当前图像的标题 |
| `xlabel` | 设置 x 轴的标签 |
| `xlim` | 获取或设置当前轴的 x 限制 |
| `xscale` | 设置 x 轴的缩放比例 |
| `xticks` | 获取或设置 x 轴的刻度位置和标签 |
| `ylabel` | 设置 y 轴的标签 |
| `ylim` | 获取或设置当前轴的 y 限制 |
| `yscale` | 设置 y 轴的缩放比例 |
| `yticks` | 获取或设置 y 轴的刻度位置和标签 |

图形函数：

| 函数 | 描述 |
| :--: | :--: |
| `figtext` | 将文字添加到图形 |
| `figure` | 创建一个新的图形 |
| `show` | 显示所有打开的图形 |
| `savefig` | 保存当前图形 |
| `close` | 关闭一个图窗口 |

## 3.2 简单绘图

与 PyLab 模块类似，提供两个长度相同的数组或序列，用`plot()`函数绘制曲线，然后用`show()`函数显示图像。例如：

```python
import numpy as np
import matplotlib.pyplot as plt
import math

x = np.arange(0, math.pi * 2, 0.05)
y = np.sin(x)

plt.plot(x, y)
plt.show()
```

<div align="center">
    <img src="https://cdn.staticaly.com/gh/zzx-JLU/images_for_markdown@main/Matplotlib/4.png" width=50% style="margin-top: -15px">
</div>

`title()`函数设置图像的标题，`xlabel()`函数设置 x 轴的标签，`ylabel()`函数设置 y 轴的标签。例如：

```python
import numpy as np
import matplotlib.pyplot as plt
import math

x = np.arange(0, math.pi * 2, 0.05)
y = np.sin(x)

plt.plot(x, y)
plt.title('sine wave')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
```

<div align="center">
    <img src="https://cdn.staticaly.com/gh/zzx-JLU/images_for_markdown@main/Matplotlib/5.png" width=50% style="margin-top: -15px">
</div>

`xlabel()`函数和`ylabel()`函数可以通过`color`参数设置标签颜色，用`fontsize`参数设置字体大小，用`rotation`设置角度。例如：

```python
import numpy as np
import matplotlib.pyplot as plt
import math

x = np.arange(0, math.pi * 2, 0.05)
y = np.sin(x)

plt.plot(x, y)
plt.title('sine wave')
plt.xlabel('x', color='r')
plt.ylabel('y', fontsize=18, rotation=0)
plt.show()
```

<div align="center">
    <img src="https://cdn.staticaly.com/gh/zzx-JLU/images_for_markdown@main/Matplotlib/6.png" width=50% style="margin-top: -15px">
</div>

在图像中使用中文时，默认情况下无法正常显示，需要配置中文字体。步骤为：

1. 下载 ttf 字体文件（如 Micr.ttf）。
2. 调用`matplotlib.matplotlib_fname()`函数，获得配置文件的路径，例如：D:\Anaconda\Anaconda\envs\exercise\lib\site-packages\matplotlib\mpl-data\matplotlibrc
3. matplotlibrc 是配置文件的名字，进入 mpl-data 文件夹，可以看到 fonts 文件夹。进入 mpl-data\fonts\ttf 文件夹，将下载的 ttf 文件复制到此文件夹下。
4. 用记事本打开 matplotlibrc 配置文件，用 ctrl+F 查找 font.family，找到`#font.family:  sans-serif`这一行，将前面的`#`去掉。
5. 在 matplotlibrc 文件中，用 ctrl+F 查找 font.sans-serif，找到`#font.sans-serif: DejaVu Sans,...`这一行，将之前下载的 ttf 文件的名字（Micr）添加进来。
6. 在 matplotlibrc 文件中，用 ctrl+F 查找 axes.unicode_minus，找到`#axes.unicode_minus: True`，删除前面的`#`，将`True`改为`False`。保存 matplotlibrc 文件。
7. 删除 Matplotlib 缓冲。
   - Windows：删除 C:\Users\用户名\.matplotlib 文件夹
   - Mac 与 Linux：执行命令`rm -rf ~/.cache/matplotlib`
8. 在 Python 代码中添加设置代码：`plt.rcParams['font.sans-serif'] = ['Micr']`

经过如上配置后，就可以在图像中正常显示中文。
