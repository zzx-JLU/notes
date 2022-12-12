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
- [4 使用面向对象思想画图](#4-使用面向对象思想画图)
  - [4.1 `Figure`类](#41-figure类)
  - [4.2 `Axes`类](#42-axes类)
  - [4.3 `Figure`与`Axes`的关系](#43-figure与axes的关系)

<!-- /code_chunk_output -->

# 1 Matplotlib简介

Matplotlib 是用于数据可视化的 Python 包，它是一个跨平台库，用于根据数组中的数据制作 2D 图。

Matplotlib 是用 Python 编写的，并使用了 NumPy。

Matplotlib 提供了一个面向对象的 API，有助于使用 Python GUI 工具包（如 PyQt）在应用程序中嵌入绘图。Matplotlib 也可以用于 Python、IPython shell、Jupyter Notebook 和 Web 应用程序服务器。

Matplotlib + NumPy 可以视作 MATLAB 的开源等价物。

# 2 PyLab模块

PyLab 是一个面向 Matplotlib 的绘图库接口，其语法和 MATLAB 十分接近。它和`matplotlib.pyplot`模块都能实现 Matplotlib 的绘图功能。

PyLab 是一个单独的模块，随 Matplotlib 软件包一起安装。

不建议使用 PyLab 模块。

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

| 颜色 | 描述 |
| :--: | :--: |
| `b` | 蓝色 |
| `g` | 绿色 |
| `r` | 红色 |
| `c` | 青色 |
| `m` | 品红 |
| `y` | 黄色 |
| `k` | 黑色 |
| `w` | 白色 |

| 点标记 | 描述 |
| :--: | :--: |
| `.` | 点状 |
| `,` | 像素 |
| `o` | 圆形 |
| `v` | 朝下的三角形 |
| `^` | 朝上的三角形 |
| `<` | 朝左的三角形 |
| `>` | 朝右的三角形 |
| `s` | 正方形 |
| `p` | 五角星 |
| `*` | 星形 |
| `h` | 1 号六角形 |
| `H` | 2 号六角形 |
| `+` | 加号 |
| `D` | 钻石形 |
| `d` | 小版钻石形 |
| `|` | 竖直线形 |
| `_` | 水平线形 |
| `1` | 下箭头 |
| `2` | 上箭头 |
| `3` | 左箭头 |
| `4` | 右箭头 |
| `x` | X 形 |

| 线条样式 | 描述 |
| :--: | :--: |
| `-` | 实线 |
| `--` | 虚线 |
| `-.` | 点划线 |
| `:` | 点虚线 |

```python
import numpy as np
import pylab as plb

x = np.linspace(-3, 3, 30)
y = x ** 2

plb.plot(x, y, "r:")  # 红色，点虚线
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

与 PyLab 模块类似，`matplotlib.pyplot`模块中也有`plot()`和`show()`函数。例如：

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

# 4 使用面向对象思想画图

虽然使用`matplotlib.pyplot`模块很容易快速生成绘图，但建议使用面向对象的方法，因为它可以更好地控制和自定义绘图。`matplotlib.axes.Axes`类中提供了大多数函数。

面向对象方法的主要思想是创建图形对象，然后只调用该对象的方法或属性，这种方式有助于更好地处理其上有多个绘图的画布。

## 4.1 `Figure`类

`matplotlib.figure.Figure`类是所有 plot 元素的顶级容器，从`pyplot`模块调用`figure()`函数来实例化`Figure`对象。参数为：

1. `figsize`：`(width, height)`，以英寸为单位的元组。
2. `dpi`：每英寸点数
3. `facecolor`：图的背景颜色
4. `edgecolor`：图的边缘颜色
5. `linewidth`：边线宽度

```python
import matplotlib.pyplot as plt

fig = plt.figure()
print(fig)  # Figure(640x480)
```

## 4.2 `Axes`类

`Axes`对象是具有数据空间的图像区域。给定的图形可以包含许多`Axes`对象，但给定的`Axes`对象只能在一个图中。

`Axes`类及其成员函数是使用面向对象接口的主要入口点。

`Figure`对象通过调用`add_axes()`方法将`Axes`对象添加到图中，返回`Axes`对象。参数为：

1. `rect`：长度为 4 的元组`(left, bottom, width, height)`，其中`left`和`bottom`分别指定坐标轴与图像左侧和底部的距离，`width`和`height`分别指定坐标轴的宽度和高度。
2. `projection`：坐标轴的投影类型。可选，默认值为`None`。
3. `polar`：布尔值，可选，默认为`False`。取值为`True`时，相当于`projection`参数取值为`'polar'`。

添加`Axes`对象后，所有的绘图操作都通过`Axes`对象来进行。

`Axes.plot()`方法是`Axes`类的基本方法，用于绘制曲线。`Axes.set_title()`方法用于设置标题，`Axes.set_xlabel()`和`Axes.set_ylabel()`方法分别用于设置 x 轴和 y 轴的标签。例如：

```python
import numpy as np
import matplotlib.pyplot as plt
import math

x = np.arange(0, math.pi * 2, 0.05)
y = np.sin(x)

fig = plt.figure(facecolor='r')  # 背景颜色为红色
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])

axes.plot(x, y)
axes.set_title('sine wave')
axes.set_xlabel('x')
axes.set_ylabel('y')

fig.show()
```

<div align="center">
    <img src="https://raw.githubusercontent.com/zzx-JLU/images_for_markdown/main/Matplotlib/7.png" style="margin-top: -10px">
</div>

`Axes.plot()`方法的其他参数：

| 参数 | 说明 |
| :--: | :--: |
| `color` | 颜色 |
| `alpha` | 透明度 |
| `linestyle`或`ls` | 线型 |
| `linewidth`或`lw` | 线宽 |
| `marker` | 点类型 |
| `markersize` | 点大小 |
| `markeredgewidth` | 点边缘的宽度 |
| `markeredgecolor` | 点边缘的颜色 |
| `markerfacecolor` | 点内部的颜色 |

```python
import numpy as np
import matplotlib.pyplot as plt
import math

x = np.arange(0, math.pi * 2, 0.05)
y = np.sin(x)

fig = plt.figure()
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])

axes.plot(x, y, color='r', alpha=0.2, ls='--', lw=5)

fig.show()
```

<div align="center">
    <img src="https://cdn.staticaly.com/gh/zzx-JLU/images_for_markdown@main/Matplotlib/9.png" style="margin-top: -15px" width=50%>
</div>

`Axes.legend()`方法在坐标轴上添加图例。它有多种原型：

1. `legend()`：自动检测图例中显示的绘图元素。只有设置过标签的绘图元素才会显示。
2. `legend(handles)`：明确列出图例中的绘图元素，每个绘图元素的标签在绘制曲线时指定。
3. `legend(handles, labels)`：明确列出图例中的绘图元素和标签。
4. `legend(labels)`：将所有绘图元素添加到图例中，指定每个绘图元素的标签。

关键字参数`loc`指定图例的位置。取值可以为字符串，或者为字符串对应的数字代码，或者用一个长度为 2 的元组指定图例左下角的坐标。位置字符串及代码如下表所示。

| 位置字符串 | 位置代码 | 说明 |
| :--: | :--: | :--: |
| `'best'` | 0 | 将图例放置在其他 9 个位置中与绘图元素重叠最小的位置 |
| `'upper right'` | 1 | 右上角 |
| `'upper left'` | 2 | 左上角 |
| `'lower left'` | 3 | 左下角 |
| `'lower right'` | 4 | 右下角 |
| `'right'` | 5 | 等价于`'center right'` |
| `center left` | 6 | 水平居左，垂直居中 |
| `center right` | 7 | 水平居右，垂直居中 |
| `lower center` | 8 | 水平居中，垂直居下 |
| `upper center` | 9 | 水平居中，垂直居上 |
| `center` | 10 | 居中 |

```python
import matplotlib.pyplot as plt

x1 = [1, 16, 30, 42, 55, 68, 77, 88]
x2 = [1, 6, 12, 18, 28, 40, 52, 65]
y = [1, 4, 9, 16, 25, 36, 49, 64]

fig = plt.figure(figsize=(10, 5))
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])

axes.plot(x1, y, 'rs-')  # 红色，正方形点，实线
axes.plot(x2, y, 'bo--')  # 蓝色，圆形点，虚线
axes.legend(labels=('tv', 'phone'), loc='lower right')  # 图例，位于右下角

fig.show()
```

<div align="center">
    <img src="https://cdn.staticaly.com/gh/zzx-JLU/images_for_markdown@main/Matplotlib/8.png" style="margin-top: -15px">
</div>

## 4.3 `Figure`与`Axes`的关系

类比：在纸上画图，可以选定纸上的多个区域，在不同的区域画不同的图，而这些画图区域必须在纸上才有意义。`Figure`对象就像一张纸，`Axes`对象是纸上的画图区域。

`Figure`对象是一个画布，`Axes`对象用于在画布上确定画图区域和作图方式。所谓画图，就是在当前的活动`Figure`对象中的一个`Axes`对象上作图。

一个`Figure`对象可以有多个`Axes`对象，`Axes`对象必须在`Figure`对象上，要画图必须要有`Axes`对象。

<div align="center" style="margin-bottom: 10px">
    <img src="https://cdn.staticaly.com/gh/zzx-JLU/images_for_markdown@main/Matplotlib/图4.1-Figure与Axes的关系.png">
    <br>
    图 4.1&nbsp;&nbsp;&nbsp;&nbsp;<code>Figure</code>与<code>Axes</code>的关系
</div>

`Figure`是级别最高的对象，它对应于整个图形表示，通常可以包含多个`Axes`对象。

`Axes`称为坐标轴，每个`Axes`对象只属于一个`Figure`对象。二维坐标轴由 2 个`Axis`对象表示，三维情况下为 3 个。此外，标题、x 轴标签和 y 轴标签也属于`Axes`对象。

`Axis`对象管理要在轴上表示的数值，定义坐标范围，并管理刻度和刻度标签。标签的位置由`Locator`对象调整，刻度标签的格式由`Formatter`对象调整。

## 4.4 在画布上创建多个子图

`pyplot.subplot(nrows, nclos, index)`函数返回给定网格位置的`Axes`对象。具体说明如下：

1. 该函数在当前图中创建并返回一个`Axes`对象，
