---
title: Pandas
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

<h1>Pandas</h1>

<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->

- [1 Pandas简介](#1-pandas简介)
- [2 Pandas数据结构](#2-pandas数据结构)
  - [2.1 Series](#21-series)
  - [2.2 DataFrame](#22-dataframe)
- [3 处理CSV文件](#3-处理csv文件)

<!-- /code_chunk_output -->

# 1 Pandas简介

Pandas 是 Python 语言的一个扩展程序库，用于数据分析。

Pandas 是一个强大的分析结构化数据的工具集，基础是 Numpy。

Pandas 可以从各种文件格式（如 CSV、JSON、SQL、Microsoft Excel）导入数据。

导入 Pandas 一般使用别名`pd`来代替。

# 2 Pandas数据结构

## 2.1 Series

Pandas Series 类似表格中的一个列，类似于一维数组，可以保存任何数据类型。

Series 由索引（index）和列组成。用`pandas.Series()`函数创建`Series`对象，参数为：

1. `data`：一组数据，可以是序列或数组。
2. `index`：数据索引标签。如果不指定，默认从 0 开始。
3. `dtype`：数据类型。默认会自己判断。
4. `name`：设置名称。
5. `copy`：布尔值，设置是否拷贝数据。默认为`False`。

```python
import pandas as pd

a = [1, 2, 3]
var = pd.Series(a)
print(var)
# 0    1
# 1    2
# 2    3
# dtype: int64
```

可以指定索引值。例如：

```python
import pandas as pd

a = ["Google", "Runoob", "Wiki"]
var = pd.Series(a, index=["x", "y", "z"])
print(var)
# x    Google
# y    Runoob
# z      Wiki
# dtype: object
```

可以根据索引值读取数据。例如：

```python
import pandas as pd

a = [1, 2, 3]
varA = pd.Series(a)
print(varA[1])  # 2

b = ["Google", "Runoob", "Wiki"]
varB = pd.Series(b, index=["x", "y", "z"])
print(varB["y"])  # Runoob
```

也可以使用字典来创建`Series`对象。例如：

```python
import pandas as pd

sites = {1: "Google", 2: "Runoob", 3: "Wiki"}
var = pd.Series(sites)
print(var)
# 1    Google
# 2    Runoob
# 3      Wiki
# dtype: object
```

如果只需要字典中的一部分数据，可以指定需要数据的索引。例如：

```python
import pandas as pd

sites = {1: "Google", 2: "Runoob", 3: "Wiki"}
var = pd.Series(sites, index=[1, 2])
print(var)
# 1    Google
# 2    Runoob
# dtype: object
```

设置`name`参数：

```python
import pandas as pd

sites = {1: "Google", 2: "Runoob", 3: "Wiki"}
var = pd.Series(sites, index=[1, 2], name="RUNOOB-Series-TEST")
print(var)
# 1    Google
# 2    Runoob
# Name: RUNOOB-Series-TEST, dtype: object
```

## 2.2 DataFrame

DataFrame 是一个表格型的数据结构，它含有一组有序的列，每列可以是不同的值类型（数值、字符串、布尔型值）。DataFrame 既有行索引也有列索引，它可以被看做由 Series 组成的字典（共同用一个索引）。

DataFrame 构造方法为`pandas.DataFrame(data, index, columns, dtype, copy)`，参数为：

1. `data`：一组数据，可以是 Ndarray、Series、map、list、dict 等类型。
2. `index`：行标签。
3. `columns`：列标签。默认为从 0 开始的整数。
4. `dtype`：数据类型。
5. `copy`：布尔值，设置是否拷贝数据。默认为`False`。

使用列表创建 DataFrame：

```python
import pandas as pd

data = [['Google', 10], ['Runoob', 12], ['Wiki', 13]]
df = pd.DataFrame(data, columns=['Site', 'Age'])
print(df)
#      Site  Age
# 0  Google   10
# 1  Runoob   12
# 2    Wiki   13
```

使用 Ndarrays 创建 DataFrame 时，每个 Ndarray 的长度必须相同。如果传递了`index`参数，则索引的长度应等于数组的长度；如果没有传递索引，则默认情况下，索引将是`range(n)`，其中`n`是数组长度。

```python
import pandas as pd

data = {'Site': ['Google', 'Runoob', 'Wiki'], 'Age': [10, 12, 13]}
df = pd.DataFrame(data)
print(df)
#      Site  Age
# 0  Google   10
# 1  Runoob   12
# 2    Wiki   13
```

还可以使用字典创建 DataFrame，其中字典的 key 为列名。例如：

```python
import pandas as pd

data = [{'a': 1, 'b': 2}, {'a': 5, 'b': 10, 'c': 20}]
df = pd.DataFrame(data)
print(df)
#    a   b     c
# 0  1   2   NaN
# 1  5  10  20.0
```

DataFrame 可以使用`loc`属性返回指定行的数据，返回结果是一个`Series`对象。例如：

```python
import pandas as pd

data = {
  "calories": [420, 380, 390],
  "duration": [50, 40, 45]
}

df = pd.DataFrame(data)

# 返回第一行
print(df.loc[0])
# calories    420
# duration     50
# Name: 0, dtype: int64

# 返回第二行
print(df.loc[1])
# calories    380
# duration     40
# Name: 1, dtype: int64
```

也可以返回多行数据，使用`[[ ... ]]`格式，`...`为各行的索引，以逗号隔开，返回结果是一个`DataFrame`对象。例如：

```python
import pandas as pd

data = {
  "calories": [420, 380, 390],
  "duration": [50, 40, 45]
}

df = pd.DataFrame(data)

# 返回第一行和第二行
print(df.loc[[0, 1]])
#    calories  duration
# 0       420        50
# 1       380        40
```

可以使用`index`参数指定行索引。例如：

```python
import pandas as pd

data = {
  "calories": [420, 380, 390],
  "duration": [50, 40, 45]
}

df = pd.DataFrame(data, index=["day1", "day2", "day3"])
print(df)
#       calories  duration
# day1       420        50
# day2       380        40
# day3       390        45
```

# 3 处理CSV文件

CSV（Comma-Separated Values，逗号分隔值）文件以纯文本形式存储表格数据。
