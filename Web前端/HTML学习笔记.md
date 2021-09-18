# 1 HTML简介
## 1.1 基本概念

 1. HTML 全称为超文本标记语言（Hyper Text Markup Language），是一种用来描述网页的语言。HTML 不是一种编程语言，而是一种**标记语言**，它使用一套标记标签来描述网页。
 2. HTML 文档包含 HTML 标签和文本。HTML 文档也叫做 web 页面。
 3. HTML 由浏览器解析执行，由上往下，由左往右。
 4. HTML 由**标签**组成，标签是由`<>`包含的关键词。
标签分为两种：双标签、单标签。
双标签由开始标记`<>`和结束标记`</>`组成，内容包括在开始标记和结束标记之间。例如：`<p>内容</p>`。
单标签没有结束标记，在右尖括号之前加一个`/`表示结束，没有内容。例如：`<meta/>`。
 5. HTML5 标准下，HTML 标签不区分大小写，建议小写。
 6. 从开始标记到结束标记之间的所有内容叫做**元素**。元素之间可以嵌套。
例如：`<p>content</p>`是一个元素，其中`<p>`是元素的开始，`content`是元素的内容，`</p>`是元素的结束。
元素分为两种：块级元素、行内元素。
 7. HTML 标签具有**属性**，用来描述标签。属性的语法为`属性名="属性值"`。
双标签的属性写在开始标记中，单标签的属性写在`/`之前。
一个标签可以有多个属性，多个属性之间用空格隔开，不区分前后顺序。
 8. HTML 注释的语法为`<!--注释内容-->`，注释内容可以换行。注释之间不能相互嵌套。

## 1.2 基本结构

```html
<!DOCTYPE html>
<html>
<head>
	<mata charset="UTF-8"/>
	<title></title>
</head>
<body></body>
</html>
```
`<!DOCTYPE html>`：文档类型声明。

`<html></html>`：HTML文档的所有内容写在`html`标签之间。

`<head></head>`：网页的描述性信息。

`<body></body>`：网页的主体内容。

### 1.2.1 DOCTYPE
`<!DOCTYPE html>`：文档类型声明，不是HTML 标签，作用是让浏览器按照当前标准解析代码。

HTML5 的文档类型声明：`<!DOCTYPE html>`。
### 1.2.2 head标签
`head`标签中可以包含的标签如下所示。

 1. `<title></title>`标签：设置网页标题。
 2. `<meta/>`标签。
`charset`属性：设置网页的字符集。中文开发常用`UTF-8`。
`name`、`content`属性：设置网页的描述信息，不会显示在页面上，可以用于提高网页在搜索引擎中的排名。
 3. `<link/>`标签：引入外部资源。
`rel`属性：设置引入内容的类型。`icon`表示标题上的图标，`stylesheet`表示样式表。
`href`属性：设置要引入的内容的路径。

# 2 HTML基本内容
## 2.1 标题
标题标签：`<h1></h1>`、`<h2></h2>`、`<h3></h3>`、`<h4></h4>`、`<h5></h5>`、`<h6></h6>`

`h1`字号最大，从`h1`往`h6`逐级递减。默认水平居左、加粗。

`align`属性：设置元素内容的水平对齐方式，取值可以为`left`、`center`、`rignt`。
## 2.2 段落
段落标签：`<p></p>`

`align`属性：设置元素内容的水平对齐方式，默认值为`left`。
## 2.3 换行
换行标签：`<br/>`
## 2.4 水平线
水平线标签：`<hr/>`

`hr`标签的属性：

 1. `color`：设置线条颜色
 2. `width`：设置线条的水平长度
 3. `size`：设置水平线的垂直高度
 4. `align`：设置水平线的水平对齐方式，默认值为`center`


## 2.5 图片
图片标签：`<img/>`

`img`标签的属性：

 1. `src`：设置图片路径，不可省略
 2. `alt`：设置图片无法显示时的提示文字
 3. `title`：设置鼠标悬停在图片上时的提示文字
 4. `width`：设置图片宽度
 5. `height`：设置图片高度

当`width`和`height`只设置其中一个时，另一个属性等比例变化。
## 2.6 超链接
超链接标签：`<a></a>`

`a`标签中的内容是超链接在网页中显示的内容，可以是文字、图片等。

`a`标签的属性：

 1. `href`：跳转路径，不可省略
 2. `target`：设置跳转方式。`_self`表示在原窗口打开，`_blank`表示在新窗口打开。默认值是`_self`。
 3. `name`：设置锚点

`href`属性值为`#`时叫做空链接，表示跳转到页面顶部。

锚点用于跳转到页面中的特定位置。锚点有以下两种类型。
 1. 从`a`标签跳转到`a`标签
在要跳转到的`a`标签设置`name`属性值。在要点击的超链接处，在`href`属性中加入`#+name值`。例如：

```html
<a href="#here">点击</a>
<a href="" name="here">跳转到我</a>
```
 2. 从`a`标签跳转到块级元素
在要跳转到的块级元素设置`id`属性值。在要点击的超链接处，在`href`属性中加入`#+id值`。例如：

```html
<a href="#here">点击</a>
<p id="here">跳转到我</p>
```
## 2.7 文本格式化标签
`<b></b>`：**粗体**
`<i></i>`：*斜体*
`<u></u>`：<u>下划线</u>
`<em></em>`：*强调（斜体）*
`<strong></strong>`：**强调（加粗）**
`<small></small>`：<small>小号字</small>
`<big></big>`：<big>大号字</big>
`<sub></sub>`：<sub>下标</sub>
`<sup></sup>`：<sup>上标</sup>
`<ins></ins>`：<ins>插入字（下划线）</ins>
`<del></del>`、`<s></s>`：<s>删除线</s>

## 2.8 无序列表
无序列表标签：`<ul></ul>`
列表项标签：`<li></li>`

`ul`标签表示一个无序列表。在`ul`标签内部，每一行的内容包括在`li`标签中，每个`li`标签的内容之前有一个项目符号。例如：

```html
<ul>
	<li>第一行</li>
	<li>第二行</li>
	<li>第三行</li>
</ul>
```

`type`属性：设置项目符号的类型。`disc`表示实心圆，`circle`表示空心圆，`square`表示实心矩形，`none`表示不显示。默认值为`disc`。

`ul`标签的`type`属性作用于内部的所有`li`标签，`li`标签的`type`属性只作用于此`li`标签。

无序列表的嵌套：内层`ul`标签要写在外层`ul`标签中的`li`标签内部。例如：

```html
<ul>
	<li>
		中国
		<ul>
			<li>北京</li>
			<li>天津</li>
		</ul>
	</li>
	<li>美国</li>
</ul>
```
## 2.9 有序列表
有序列表标签：`<ol></ol>`

有序列表与无序列表类似，只是显示在页面上时，项目符号为数字。

`ol`标签的属性：

 1. `type`：设置项目符号的类型。`1`表示阿拉伯数字，`A`表示大写字母，`a`表示小写字母，`I`表示大写罗马数字，`i`表示小写罗马数字。
 2. `start`：设置起始序号，取值为阿拉伯数字。
 3. `reversed`：取值为`reversed`时表示倒序，不设置此属性时默认为正序。当属性名和属性值相同时，属性值可以省略。

## 2.10 自定义列表
自定义列表标签：`<dl></dl>`
主题标签：`<dt></dt>`
描述标签：`<dd></dd>`

`dt`标签用于表示主题。`dd`标签跟在`dt`标签之后，表示对主题的描述和注释。每个`dt`标签后面可以有多个`dd`标签。例如：

```html
<dl>
	<dt>主题一</dt>
	<dd>1.1</dd>
	<dd>1.2</dd>
	<dt>主题二</dt>
	<dd>2.1</dd>
</dl>
```
## 2.11 表格
表格标签：`<table></table>`
行标签：`<tr></tr>`
单元格标签：`<td></td>`
特殊的单元格：`<th></th>`。其中的内容默认加粗、水平居中。
表格标题：`<caption></caption>`
表格头部：`<thead></thead>`
表格主体：`<tbody></tbody>`
表格脚部：`<tfoot></tfoot>`

每个`table`标签中可以有多个`tr`标签，每个`tr`标签中可以有多个`td`标签。`thead`、`tbody`、`tfoot`标签用于为表格分区，包裹在`tr`标签外面。如果没有分区，默认将所有`tr`标签放在`tbody`标签中。例如：

```html
<table>
	<caption>标题</caption>
	<thead>
		<tr>
			<td></td>
			<td></td>
		</tr>
	</thead>
	<tbody>
		<tr>
			<td></td>
			<td></td>
		</tr>
		<tr>
			<td></td>
			<td></td>
		</tr>
	</tbody>
</table>
```
`table`标签的属性：

 1. `border`：设置外边框宽度
 2. `width`：表格的宽度
 3. `height`：表格的高度
 4. `align`：设置表格的水平对齐方式
 5. `bgcolor`：背景颜色
 6. `background`：背景图片
 7. `bordercolor`：边框颜色
 8. `cellpadding`：设置内容与单元格边框的距离
 9. `cellspacing`：设置单元格之间的距离

`tr`标签的属性：

 1. `height`：设置行高度
 2. `bgcolor`：行背景颜色
 3. `background`：行背景图片
 4. `align`：设置行内容的水平对齐方式，默认值为`left`
 5. `valign`：设置行内容的垂直对齐方式，可以取值为`top`、`middle`、`bottom`，默认值为`middle`

`td`、`th`标签的属性：

 1. `width`：设置单元格宽度
 2. `height`：设置单元格高度
 3. `bgcolor`：单元格背景颜色
 4. `background`：单元格背景图片
 5. `align`：设置单元格内容的水平对齐方式
 6. `valign`：设置单元格内容的垂直对齐方式，可以取值为`top`、`middle`、`bottom`，默认值为`middle`
 7. `colspan`：水平合并单元格，取值为合并单元格的数量（设置单元格跨越的列数）
 8. `rowspan`：垂直合并单元格，取值为合并单元格的数量（设置单元格跨越的行数）

## 2.12 实体字符
一些特殊符号不能识别，可以用实体字符替代。

空格：`&nbsp;`
左尖括号：`&lt;`
右尖括号：`&gt;`

## 2.13 表单
表单用于用户填写信息，使网页具有交互性。一般将表单设计在 HTML 文档中，当用户填写完信息后做提交操作，将表单的内容从浏览器传送到服务器上，经过服务器上的处理程序处理后，再将用户所需信息传送回浏览器上。

一个完整的表单包含三个基本组成部分：表单标签、表单域、表单按钮。其中表单域和表单按钮统称为表单元素。

![表单](https://img-blog.csdnimg.cn/bcf75a6e84fd436cbcc20d9ee4a66cd9.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQ4ODE0MjA1,size_16,color_FFFFFF,t_70#pic_center)

### 2.13.1 表单标签
表单标签：`<form></form>`

`form`标签不能相互嵌套，一个页面可以有多个`form`标签。

`form`标签的属性：

 1. `name`：表单的名称
 2. `action`：提交的地址
 3. `method`：提交的方式，可以取`get`或`post`，默认值为`get`。
`get`提交的数据会在地址中显示，不安全；`post`提交的数据不会在地址中显示，更加安全。
`get`只能提交少量数据（2KB），`post`理论上没有限制。

### 2.13.2 表单域
表单域是`form`标签中用来收集用户输入的每一项。表单域有不同的类型，对应不同的用户数据。

表单域有以下几种：

 1. 文本框：`<input type="text"/>`
 2. 密码框：`<input type="password"/>`
 3. 单选按钮：`<input type="radio"/>`
`name`属性相同的单选按钮为同一组，同一组的单选按钮最多只能选中一个。
`checked`属性：`checked="checked"`设置默认选中。
 4. 多选按钮：`<input type="checkbox"/>`
`checked`属性：`checked="checked"`设置默认选中。
 5. `<label></label>`标签与单选按钮、多选按钮配合，使得当用户点击文字时选中该文字对应的按钮，提升用户体验。
`label`标签的`for`属性与`input`标签的`id`属性配合，属性值相等的按钮与文字相关联。例如：

```html
<input type="radio" id="man" name="sex" value="male"/>
<label for="man">男</label>
<input type="radio" id="woman" name="sex" value="female"/>
<label for="woman">女</label>
```
 6. 下拉列表：`<select></select>`，`size`属性表示列表框的高度占几个列表项。
`<option></option>`表示列表中的一项，`value`属性表示当前值，`selected="selected"`属性值设置该项默认选中。
`<optgroup></optgroup>`标签为`option`标签分组，`label`属性设置组名。

```html
<select>
	<optgroup label="河北">
		<option value="sjz">石家庄</option>
	</optgroup>
	<optgroup label="北京">
		<option value="cy">朝阳</option>
	</optgroup>
</select>
```
 7. 多行文本框：`<textarea></textarea>`
`rows`属性表示行数，`cols`属性表示列数。
 8. 文件框：`<input type="file"/>`
`form`表单默认不支持上传文件，需要将`form`的编码格式修改为二进制：`enctype="multipart/form-data"`。
 9. 隐藏域：`<input type="hidden"/>`
隐藏域不会显示在页面上，但是其中的数据可以正常提交给服务器。

### 2.13.3 表单按钮
表单按钮用来将表单中的所有信息提交到服务器。

表单按钮有以下几种：

 1. 提交按钮：`<input type="submit"/>`或`<button type="submit"></button>`
 2. 重置按钮：`<input type="reset"/>`或`<button type="reset"></button>`
 3. 没有功能的按钮：`<input type="button"/>`或`<button type="button"></button>`
 4. 图片提交按钮： `<input type="image" src=""/>`或`<button><img src=""/></button>`

`input`标签是单标签，按钮上显示的文字由`value`属性指定。`button`标签是双标签，按钮上显示的内容放在`button`标签中，可以是文字、图片等。
### 2.13.4 表单元素的属性

 1. `type`：设置表单元素的类型
 2. `name`：表单元素的名称
 3. `value`：当前值
 4. `disabled`：禁用
 5. `readonly`：只读

### 2.13.5 HTML5新增的表单元素

 1. 邮箱：`<input type="email"/>`
 2. 搜索框：`<input type="search"/>`
 3. 电话：`<input type="tel"/>`
 4. URL：`<input type="url"/>`
 5. 颜色选择框：`<input type="color"/>`
 6. 数字输入框：`<input type="number"/>`
`min`属性指定最小值，`max`属性指定最大值，`step`属性规定数字间隔。
 7. 范围选择框：`<input type="range"/>`
类似于滑动条，用于精确值不重要的输入数字的控件。
`min`属性指定最小值，`max`属性指定最大值，`step`属性规定数字间隔，`value`属性设置默认值。
 8. 日期控件：`<input type="date"/>`
 9. 月份控件：`<input type="month"/>`
 10. 星期控件：`<input type="week"/>` 

### 2.13.6 HTML5新增的表单元素属性

 1. `placeholder`：提供提示信息，描述输入域所期待的值。
 2. `autofocus`：在页面加载时，该表单域自动获得焦点。适用于所有`input`标签的类型。推荐写在第一个表单元素上。
 3. `multiple`：规定输入域中可以选择多个值。只适用于`<input type="email"/>`和`<input type="file"/>`。
 4. `required`：规定输入域不能为空。
 5. `minlength`：最小长度
 6. `maxlength`：最大长度

## 2.14 布局标签
`<div></div>`、`<span></span>`表示无语义的容器，用于页面布局。

`<div></div>`是块级元素，`<span></span>`是行内元素。
## 2.15 HTML5新增标签
注意：IE 8 及以下不支持 HTML5。
### 2.15.1 语义化布局标签
在 HTML5 之前，通常采用 DIV+CSS 的布局方式。由于`div`标签是无语义的，使得文档结构不清晰，而且不利于搜索引擎爬虫对页面的爬取。为了解决上述缺点，HTML5 新增了很多语义化布局标签。

`<header></header>`：头部
`<nav></nav>`：导航栏
`<article></article>`：文章、帖子、博客等独立的一块
`<section></section>`：文章中的章节
`<aside></aside>`：侧边栏
`<footer></footer>`：页脚

这些标签都是块级元素。
### 2.15.2 视频
视频标签（行内标签）：`<video></video>`

标签中的内容是提示信息，当浏览器不支持视频时，显示标签中的内容。

支持的视频格式：mp4、ogg、webM。

`video`标签的属性：

 1. `src`：要播放的视频的 URL。
 2. `autoplay`：视频在就绪后自动播放。
 3. `controls`：显示控件，如播放按钮等。
 4. `height`：设置视频播放器的高度。
 5. `width`：设置视频播放器的宽度。
 6. `loop`：循环播放。
 7. `muted`：静音。
 8. `poster`：规定视频下载时的图像，或者在用户点击播放按钮前的图像。取值为图像的 URL。
 9. `preload`：如果出现该属性，则视频在页面加载时进行加载，并预备播放。如果使用`autoplay`属性，则忽略`preload`属性。

### 2.15.3 音频
音频标签：`<audio></audio>`

标签中的内容是提示信息，当浏览器不支持音频时，显示标签中的内容。

`audio`标签的属性：

 1. `src`：要播放的音频的 URL。
 2. `autoplay`：音频在就绪后自动播放。
 3. `controls`：显示控件，如播放按钮等。
 4. `loop`：循环播放。
 5. `muted`：静音。

### 2.15.4 资源
资源标签：`<source src=""/>`

`source`标签为媒介元素（如`video`和`audio`标签）定义媒介资源。`source`标签允许程序员规定可替换的视频/音频文件，供浏览器根据它对媒体类型或者解编码器的支持进行选择。

`source`标签写在`video`标签和`audio`标签内部，可以有多个，浏览器选择其中支持的一个进行显示。如果都不支持，则显示提示文字。例如：

```html
<video controls>
	<source src="video1.mp4"/>
	<source src="video1.ogg"/>
	<source src="video1.webM"/>
	您的浏览器不支持视频，请升级
</video>
```
