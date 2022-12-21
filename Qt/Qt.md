---
title: Qt
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

<h1>Qt</h1>

<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->

- [1 Qt入门](#1-qt入门)
  - [1.1 创建项目](#11-创建项目)
  - [1.2 项目结构](#12-项目结构)
  - [1.3 窗口坐标系](#13-窗口坐标系)
  - [1.4 窗口常用API](#14-窗口常用api)
  - [1.5 按钮控件](#15-按钮控件)
  - [1.6 对象树](#16-对象树)
  - [1.7 资源文件](#17-资源文件)
- [2 信号和槽](#2-信号和槽)
  - [2.1 信号和槽简介](#21-信号和槽简介)
  - [2.2 自定义信号和槽](#22-自定义信号和槽)
  - [2.3 信号槽的扩展](#23-信号槽的扩展)
  - [2.4 Qt4版本的连接方式](#24-qt4版本的连接方式)
  - [2.5 Lambda表达式](#25-lambda表达式)
- [3 `QMainWindow`](#3-qmainwindow)
  - [3.1 菜单栏](#31-菜单栏)
  - [3.2 工具栏](#32-工具栏)
  - [3.3 状态栏](#33-状态栏)
  - [3.4 铆接部件](#34-铆接部件)
  - [3.5 中心部件](#35-中心部件)
- [4 对话框](#4-对话框)
  - [4.1 基本概念](#41-基本概念)
  - [4.2 标准对话框](#42-标准对话框)
    - [4.2.1 消息对话框](#421-消息对话框)
    - [4.2.2 颜色对话框](#422-颜色对话框)
    - [4.2.3 文件对话框](#423-文件对话框)
    - [4.2.4 字体对话框](#424-字体对话框)
- [5 控件](#5-控件)
  - [5.1 按钮组](#51-按钮组)
  - [5.2 QListWidget](#52-qlistwidget)
  - [5.3 QTreeWidget](#53-qtreewidget)
  - [5.4 QTableWidget](#54-qtablewidget)
  - [5.5 下拉框](#55-下拉框)

<!-- /code_chunk_output -->

# 1 Qt入门

## 1.1 创建项目

示例版本：Qt 5.14.2，Qt Creator 4.11.1。

第一步：选择模板。这里选择 Qt Widgets Application。

<div align="center" style="margin-bottom: 10px">
    <img src="https://cdn.jsdelivr.net/gh/zzx-JLU/images_for_markdown@main/Qt/图1.1-选择模板.64gba1s00tk0.png">
    <br>
    图 1.1&nbsp;&nbsp;&nbsp;&nbsp;选择模板
</div>

第二步：指定项目名称和路径。项目名称和路径中不能含有中文或空格。

<div align="center" style="margin-bottom: 10px">
    <img src="https://cdn.jsdelivr.net/gh/zzx-JLU/images_for_markdown@main/Qt/图1.2-指定项目名称和路径.hspywtk4rdk.png">
    <br>
    图 1.2&nbsp;&nbsp;&nbsp;&nbsp;指定项目名称和路径
</div>

第三步：指定编译器。这里使用 qmake。

<div align="center" style="margin-bottom: 10px">
    <img src="https://cdn.jsdelivr.net/gh/zzx-JLU/images_for_markdown@main/Qt/图1.3-指定编译器.3xoqw05bkma0.png">
    <br>
    图 1.3&nbsp;&nbsp;&nbsp;&nbsp;指定编译器
</div>

第四步：设置类信息。指定主窗口类名和主窗口的基类，头文件名、源文件名和界面文件名根据类名自动改变，不需要手动设置。

基类有三种选择：`QMainWindow`表示一个有菜单栏、状态栏的窗口，`QWidget`表示一个空窗口，`QDialog`表示一个对话框。

<div align="center" style="margin-bottom: 10px">
    <img src="https://cdn.jsdelivr.net/gh/zzx-JLU/images_for_markdown@main/Qt/图1.4-设置类信息.413mogzzqa60.png">
    <br>
    图 1.4&nbsp;&nbsp;&nbsp;&nbsp;设置类信息
</div>

第五步：如果要为项目的用户接口提供翻译，需要指定语言。

<div align="center" style="margin-bottom: 10px">
    <img src="https://cdn.jsdelivr.net/gh/zzx-JLU/images_for_markdown@main/Qt/图1.5-选择翻译语言.4zvzrsd0nf40.png">
    <br>
    图 1.5&nbsp;&nbsp;&nbsp;&nbsp;选择翻译语言
</div>

第六步：选择工具包。

<div align="center" style="margin-bottom: 10px">
    <img src="https://cdn.jsdelivr.net/gh/zzx-JLU/images_for_markdown@main/Qt/图1.6-选择工具包.6pb03q9jj9s0.png">
    <br>
    图 1.6&nbsp;&nbsp;&nbsp;&nbsp;选择工具包
</div>

第七步：选择是否将该项目作为子项目添加到另一个项目中，是否将该项目添加到版本管理系统。

<div align="center" style="margin-bottom: 10px">
    <img src="https://cdn.jsdelivr.net/gh/zzx-JLU/images_for_markdown@main/Qt/图1.7-总结.39w3ynwisco0.png">
    <br>
    图 1.7&nbsp;&nbsp;&nbsp;&nbsp;总结
</div>

点击“完成”，创建新项目。

## 1.2 项目结构

项目结构如图 1.8 所示。

<div align="center" style="margin-bottom: 10px">
    <img src="https://cdn.jsdelivr.net/gh/zzx-JLU/images_for_markdown@main/Qt/图1.8-项目结构.6uaztmc06lc0.png">
    <br>
    图 1.8&nbsp;&nbsp;&nbsp;&nbsp;项目结构
</div>

`test.pro`：工程文件。

<div align="center" style="margin-bottom: 10px">
    <img src="https://cdn.jsdelivr.net/gh/zzx-JLU/images_for_markdown@main/Qt/图1.9-项目文件.6si3v24oc100.png">
    <br>
    图 1.9&nbsp;&nbsp;&nbsp;&nbsp;工程文件
</div>

`widget.h`：主窗口的头文件。

```c++{.line-numbers}
#ifndef WIDGET_H
#define WIDGET_H

#include <QWidget> // 包含 QWidget 类的头文件

class Widget : public QWidget
{
    Q_OBJECT // 宏，允许类中使用信号和槽的机制

public:
    Widget(QWidget *parent = nullptr);
    ~Widget();
}
#endif // WIDGET_H
```

`widget.cpp`：主窗口的源文件。

```c++{.line-numbers}
#include "widget.h"

Widget::Widget(QWidget *parent)
    : QWidget(parent)
{
}

Widget::~Widget()
{
}
```

`main.cpp`：主程序。

```c++{.line-numbers}
#include "widget.h"

#include <QApplication> // 包含一个应用程序类的头文件

// 程序入口
int main(int argc, char *argv[])
{
    QApplication a(argc, argv); // 应用程序对象，有且只有一个
    Widget w; // 窗口对象
    w.show(); // 显示窗口
    return a.exec(); // 让应用程序对象进入消息循环
}
```

`test_zh_CN.ts`：由于创建项目时选择了翻译成中文，所以自动生成了这个文件。

## 1.3 窗口坐标系

以窗口左上角为原点，向右为 X 轴正方向，向下为 Y 轴正方向。

## 1.4 窗口常用API

1. `resize(int w, int h)`：重新设置窗口大小。
2. `setWindowTitle(const QString &)`：设置窗口标题。
3. `setFixedSize(int w, int h)`：设置窗口固定大小。

## 1.5 按钮控件

1. `QPushButton *button = new QPushButton;`：创建按钮控件。
2. `QPushButton(QWidget *parent = nullptr)`：创建按钮控件，并指定父对象。
3. `QPushButton(const QString &text, QWidget *parent = nullptr)`：创建按钮控件，并设置文字和父对象。
4. `setParent(QWidget *parent)`：设置按钮控件的父对象。
5. `setText(const QString &text)`：设置按钮上的文字。
6. `move(int ax, int ay)`：设置按钮的位置。

## 1.6 对象树

Qt 中`QObject`类是所有类的基类。`QObject`是以对象树的形式组织起来的。创建`QObject`对象时，`QObject`的构造函数接收一个`QObject`指针作为参数，这个参数是`parent`，也就是父对象指针。在创建`QObject`对象时，可以提供一个父对象，新创建的对象会自动添加到其父对象的`children`列表。当父对象析构的时候，它的所有子对象也会析构。

`QObject`类的所有子类都会继承这种对象树关系。对象树简化了内存回收机制。

`QWidget`是能够在屏幕上显示的一切组件的父类。`QWidget`继承自`QObject`，因此也继承了这种对象树关系。当窗口关闭时，窗口对象被析构，它的所有子组件也一起被析构。

当一个`QObject`对象在堆上创建时，Qt 会同时为其创建一个对象树。任何对象树中的`QObject`对象被析构时，如果这个对象有父对象，则自动将其从父对象的`children`列表中删除；如果有子对象，则自动析构每一个子对象。

如果`QObject`对象在栈上创建，Qt 保持同样的行为。

<div align="center">
    <img src="https://raw.githubusercontent.com/zzx-JLU/images_for_markdown/main/Qt/%E5%9B%BE1.10-%E5%AF%B9%E8%B1%A1%E6%A0%91.png">
    <br>
    图 1.10&nbsp;&nbsp;&nbsp;&nbsp;对象树
</div>

## 1.7 资源文件

添加资源文件的步骤如下：

1. 将要添加的资源文件拷贝到项目位置下。
2. 鼠标右键单击项目，选择“添加新建项”。

<div align="center" style="margin-bottom: 10px">
    <img src="https://raw.githubusercontent.com/zzx-JLU/images_for_markdown/main/Qt/%E5%9B%BE1.11-%E6%B7%BB%E5%8A%A0%E6%96%B0%E5%BB%BA%E9%A1%B9.png">
    <br>
    图 1.11&nbsp;&nbsp;&nbsp;&nbsp;添加新建项
</div>

3. 在新建文件对话框中，选择 Qt→Qt 资源文件。

<div align="center" style="margin-bottom: 10px">
    <img src="https://raw.githubusercontent.com/zzx-JLU/images_for_markdown/main/Qt/%E5%9B%BE1.12-%E9%80%89%E6%8B%A9Qt%E8%B5%84%E6%BA%90%E6%96%87%E4%BB%B6.png">
    <br>
    图 1.12&nbsp;&nbsp;&nbsp;&nbsp;选择 Qt 资源文件
</div>

4. 设置资源文件的名称和路径，点击“下一步”。

<div align="center" style="margin-bottom: 10px">
    <img src="https://raw.githubusercontent.com/zzx-JLU/images_for_markdown/main/Qt/%E5%9B%BE1.13-%E8%AE%BE%E7%BD%AE%E8%B5%84%E6%BA%90%E6%96%87%E4%BB%B6%E7%9A%84%E5%90%8D%E7%A7%B0%E5%92%8C%E8%B7%AF%E5%BE%84.png">
    <br>
    图 1.13&nbsp;&nbsp;&nbsp;&nbsp;设置资源文件的名称和路径
</div>

5. 选择要添加到哪个项目和版本控制系统，点击“完成”。

<div align="center" style="margin-bottom: 10px">
    <img src="https://raw.githubusercontent.com/zzx-JLU/images_for_markdown/main/Qt/%E5%9B%BE1.14-%E6%B7%BB%E5%8A%A0%E5%88%B0%E9%A1%B9%E7%9B%AE.png">
    <br>
    图 1.14&nbsp;&nbsp;&nbsp;&nbsp;添加到项目
</div>

6. 右键单击资源文件，选择“在编辑器中打开”。

<div align="center" style="margin-bottom: 10px">
    <img src="https://raw.githubusercontent.com/zzx-JLU/images_for_markdown/main/Qt/%E5%9B%BE1.15-%E6%89%93%E5%BC%80%E8%B5%84%E6%BA%90%E6%96%87%E4%BB%B6.png">
    <br>
    图 1.15&nbsp;&nbsp;&nbsp;&nbsp;打开资源文件
</div>

7. 添加前缀和文件。文件必须位于项目文件夹下。

使用资源文件时，需要指定文件的路径，路径的格式为`": + 前缀名 + 文件名"`。

# 2 信号和槽

## 2.1 信号和槽简介

信号和槽的本质是观察者模式。当某个事件发生后，就会发出一个信号（signal），这种发送是没有目的的，类似于广播。如果有对象对这个信号感兴趣，它就会使用`connect()`函数，将想要处理的信号和自己的一个槽（slot）绑定来处理这个信号。也就是说，当信号发出时，被连接的槽函数会自动被回调。

槽的本质是类的成员函数，和普通的成员函数几乎没有区别。槽和普通的成员函数唯一的区别在于，槽可以与信号连接在一起，每当和槽连接的信号被发出的时候，就会调用这个槽。

信号和槽是 Qt 特有的信息传输机制，是 Qt 程序设计的重要基础，它可以让互不干扰的对象建立一种联系。

使用`connect()`函数将信号和槽连接在一起，`connect()`函数最常用的一般形式为：

```c++
connect(QObject *sender, Func1 signal, QObject *receiver, Func1 slot)
```

4 个参数分别为：

1. `sender`：发出信号的对象。
2. `signal`：发出的信号（信号函数的地址）。
3. `receiver`：接收信号的对象。
4. `slot`：槽函数（槽函数的地址）。

例如，点击按钮使窗口关闭的功能可以实现如下：

```c++
// 窗口类
class Widget : public QWidget
{
    Q_OBJECT
public:
    Widget(QWidget *parent = nullptr);
    ~Widget();
};

Widget::Widget(QWidget *parent)
    : QWidget(parent)
{
    // 创建按钮
    QPushButton *button = new QPushButton();
    // 指定窗口为按钮的父对象
    button->setParent(this);
    // 信号的发出者为按钮，发出的信号为点击按钮
    // 信号的接收者为窗口，收到信号后执行关闭窗口的槽函数
    connect(button, &QPushButton::clicked, this, &QWidget::close);
}
```

## 2.2 自定义信号和槽

一个类只有继承了`QObject`类之后才具有信号槽的能力。`QObject`类及其子类应该在类定义的第一行写上`Q_OBJECT`，这是一个宏，为类提供了信号槽机制、国际化机制和反射能力。

自定义信号写在类声明中的`signals`关键字之下，返回值是`void`，只需要声明，不需要实现，可以有参数，可以重载。

自定义槽函数时，早期 Qt 版本必须要写到`public slots`之下，Qt 5.4 之后可以写到`public`或全局下。槽函数返回值为`void`，需要声明，也需要实现，可以有参数，可以重载。

定义信号和槽函数后，要在适当位置使用`emit`关键字触发信号。

例如，定义老师类和学生类，当“下课”事件发生时，老师发出“饥饿”信号，学生接收信号并响应“请客”。代码如下：

```c++{.line-numbers}
/* teacher.h */
class Teacher : public QObject
{
    Q_OBJECT
public:
    explicit Teacher(QObject *parent = 0);
signals:
    // 自定义信号写在 signals 之下，返回值是 void
    // 只需要声明，不需要实现
    void hungry();
};
```

```c++{.line-numbers}
/* student.h */
class Student : public QObject
{
    Q_OBJECT
public:
    explicit Student(QObject *parent = 0);
public slots:
    // 自定义槽函数写在 public slots 之下
    // 返回值是 void，需要声明，也需要实现
    void treat();
};
```

```c++{.line-numbers}
/* student.cpp */
void Student::treat()
{
    qDebug() << "请老师吃饭"; // 向控制台输出信息
}
```

```c++{.line-numbers}
/* widget.h */
class Widget : public QWidget
{
    Q_OBJECT
public:
    Widget(QWidget *parent = nullptr);
    ~Widget();
private:
    Teacher *teacher;
    Student *student;
    void classIsOver(); // 下课事件
};
```

```c++{.line-numbers}
/* widget.cpp */
Widget::Widget(QWidget *parent)
    : QWidget(parent)
{
    // 创建老师和学生对象
    this->teacher = new Teacher(this);
    this->student = new Student(this);
    // 连接
    connect(teacher, &Teacher::hungry, student, &Student::treat);
    // 发生“下课”事件
    classIsOver();
}

void Widget::classIsOver()
{
    emit teacher->hungry(); // 触发老师发出“饥饿”信号
}
```

当信号和槽发生重载时，在连接时需要指出使用的重载版本，通过函数指针指定具体的重载版本。例如：

```c++{.line-numbers}
/* teacher.h */
class Teacher : public QObject
{
    Q_OBJECT
public:
    explicit Teacher(QObject *parent = 0);
signals:
    void hungry();
    void hungry(QString foodName); // 重载的信号
};
```

```c++{.line-numbers}
/* student.h */
class Student : public QObject
{
    Q_OBJECT
public:
    explicit Student(QObject *parent = 0);
public slots:
    void treat();
    void treat(QString foodName); // 重载的槽函数
};
```

```c++{.line-numbers}
/* student.cpp */
void Student::treat()
{
    qDebug() << "请老师吃饭";
}

void Student::treat(QString foodName)
{
    qDebug() << "请老师吃饭，吃的是" << foodName;
}
```

```c++{.line-numbers}
/* widget.h */
class Widget : public QWidget
{
    Q_OBJECT
public:
    Widget(QWidget *parent = nullptr);
    ~Widget();
private:
    Teacher *teacher;
    Student *student;
    void classIsOver(); // 下课事件
};
```

```c++{.line-numbers}
/* widget.cpp */
Widget::Widget(QWidget *parent)
    : QWidget(parent)
{
    // 创建老师和学生对象
    this->teacher = new Teacher(this);
    this->student = new Student(this);
    // 连接
    void (Teacher::*teacherSignal)(QString) = &Teacher::hungry;
    void (Student::*studentSlot)(QString) = &Student::treat;
    connect(teacher, teacherSignal, student, studentSlot);
    // 发生“下课”事件
    classIsOver();
}

void Widget::classIsOver()
{
    emit teacher->hungry("宫保鸡丁"); // 触发老师发出“饥饿”信号
}
```

这里`treat()`函数的参数类型为`QString`，当输出到控制台时，`QString`类型的字符串会带有引号。为了去掉引号，可以将`QString`类型的变量转换成`char*`类型。转换方法为，先调用`toUtf8()`函数转为`QByteArray`，再调用`data()`函数转为`char*`类型。例如：

```c++
void Student::treat(QString foodName)
{
    qDebug() << "请老师吃饭，吃的是" << foodName.toUtf8().data();
}
```

## 2.3 信号槽的扩展

1. 信号可以连接信号

信号和信号也可以连接。当两个信号连接后，第一个信号的发出会触发第二个信号，第二个信号会进一步触发后面的动作。通过这种方式可以实现信号的传递。

例如，点击“下课”按钮后触发下课事件，老师发出“饥饿”信号，学生接收信号并响应“请客”。代码如下：

```c++
Widget::Widget(QWidget *parent)
    : QWidget(parent)
{
    // 创建老师和学生对象
    this->teacher = new Teacher(this);
    this->student = new Student(this);
    // 创建按钮
    QPushButton *button = new QPushButton("下课", this);
    // 按钮的“点击”信号连接老师的“饥饿”信号，老师的“饥饿”信号连接学生的“请客”槽函数
    connect(button, &QPushButton::clicked, teacher, &Teacher::hungry);
    connect(teacher, &Teacher::hungry, student, &Student::treat);
}
```

2. 一个信号可以连接多个槽。
3. 多个信号可以连接同一个槽。
4. 信号和槽函数的参数类型必须一一对应。
5. 信号的参数个数可以多于槽函数，但不能少于槽函数。
6. 信号槽可以断开连接。使用`disconnect()`函数断开连接，参数与`connect()`函数一致。

## 2.4 Qt4版本的连接方式

在`connect()`函数中，使用`SIGNAL`和`SLOT`宏将函数名转换成字符串。例如：

```c++
// 无参
connect(teacher, SIGNAL(hungry()), student, SLOT(treat()));
// 有参
connect(teacher, SIGNAL(hungry(QString)), student, SLOT(treat(QString)));
```

优点：参数直观。

缺点：不做参数检测，一旦连接失败，不会在编译时报错，而是在运行时给出错误，增加了程序的不稳定性。

## 2.5 Lambda表达式

要想在 Qt 中使用 Lambda 表达式，需要在项目文件中添加如下内容：`CONFIG += C++11`。Qt 5.4 版本之后，项目文件中会自带该内容，不需要手动添加。

C++11 中的 Lambda 表达式用于定义并创建匿名的函数对象。Lambda 表达式的基本构成如下：

```c++
[capture](parameters)mutable->returnType
{
    statement
}
```

1. `[capture]`：函数对象参数，标识一个 Lambda 表达式的开始，不能省略。函数对象参数是传递给编译器自动生成的函数对象类的构造函数的。函数对象参数只能使用那些到定义 Lambda 表达式为止时 Lambda 表达式所在作用范围内可见的局部变量。函数对象参数有以下形式：
   （1）空，没有函数对象参数。
   （2）`=`：函数体内可以使用 Lambda 表达式所在作用范围内所有可见的局部变量，并且是值传递方式。
   （3）`&`：函数体内可以使用 Lambda 表达式所在作用范围内所有可见的局部变量，并且是引用传递方式。
   （4）`this`：函数体内可以使用 Lambda 表达式所在类中的成员变量。
   （5）`a`：将`a`按值进行传递。按值进行传递时，函数体内不能修改传递进来的`a`的拷贝，因为默认情况下函数是`const`的。
   （6）`&a`：将`a`按引用进行传递。
   （7）`a, &b`：将`a`按值进行传递，`b`按引用进行传递。
   （8）`=, &a, &b`：除`a`和`b`按引用进行传递外，其他参数都按值进行传递。
   （9）`&, a, b`：除`a`和`b`按值进行传递外，其他参数都按引用进行传递。
2. `(parameters)`：操作符重载函数参数，标识重载的`()`操作符的参数。没有参数时，这部分可以省略。参数可以通过按值和按引用两种方式进行传递。
3. `mutable`：可修改标识符，可以省略。加上`mutable`后，可以修改按值传递进来的函数对象参数的拷贝。
4. `->returnType`：函数返回值类型。当返回值类型为`void`，或者函数体中只有一处`return`语句时，这部分可以省略。
5. `{statement}`：函数体，不能省略，但是可以为空。

可以在`connect()`函数中直接使用 Lambda 表达式，例如：

```c++
Widget::Widget(QWidget *parent)
    : QWidget(parent)
{
    // 创建按钮
    QPushButton *button = new QPushButton("关闭", this);
    // 点击按钮关闭窗口
    connect(button, &QPushButton::clicked, this, [=](){
        this->close();
    });
}
```

# 3 `QMainWindow`

`QMainWindow`是一个为用户提供主窗口的类，包含一个菜单栏（menu bar）、多个工具栏（tool bar）、多个铆接部件（dock widget）、一个状态栏（status bar）以及一个中心部件（central widget）。

<div align="center">
    <img src="https://cdn.jsdelivr.net/gh/zzx-JLU/images_for_markdown@main/Qt/图3.1-QMainWindow.papjrcwoi9s.png">
    <br>
    图 3.1&nbsp;&nbsp;&nbsp;&nbsp;<code>QMainWindow</code>
</div>

## 3.1 菜单栏

菜单栏最多只能有一个。菜单栏的相关操作如下：

```c++
MainWindow::MainWindow(QWidget *parent): QMainWindow(parent)
{
    // 创建菜单栏
    QMenuBar *menuBar = menuBar();

    // 将菜单栏添加到窗口中
    setMenuBar(menuBar);

    // 创建菜单
    QMenu *fileMenu = menuBar->addMenu("文件");

    // 创建菜单项
    QAction *newAction = fileMenu->addAction("新建");

    // 添加分割线
    fileMenu->addSeparator();
}
```

## 3.2 工具栏

工具栏可以有多个，用户可以在窗口中移动工具栏。工具栏的相关操作如下：

```c++
MainWindow::MainWindow(QWidget *parent): QMainWindow(parent)
{
    // 创建工具栏
    QToolBar *toolBar = new QToolBar(this);

    // 将工具栏添加到窗口中，默认位于上方
    addToolBar(toolBar);
    
    /* 将工具栏添加到窗口中时，可以指定位置，位置的取值包括：
     *   Qt::LeftToolBarArea, Qt::RightToolBarArea, Qt::TopToolBarArea,
     *   Qt::BottomToolBarArea, Qt::AllToolBarArea, Qt::NoToolBarArea
     * 例如：addToolBar(Qt::LeftToolBarArea, toolBar);
     */
    
    // 设置工具栏允许的停靠范围，默认值为 Qt::AllToolBarArea
    // 如果有多个取值，使用或运算符分隔
    toolBar->setAllowedAreas(Qt::LeftToolBarArea | Qt::RightToolBarArea);

    // 设置浮动。参数为 true 表示允许浮动，参数为 false 表示不允许浮动
    toolBar->setFloatable(false);

    // 设置移动。参数为 true 表示允许移动，参数为 false 表示不允许移动
    toolBar->setMovable(false);

    // 添加菜单项
    QAction *newAction = new QAction("新建", this);
    toolBar->addAction(newAction);

    // 添加分割线
    toolBar->addSeparator();

    // 添加控件
    QPushButton *button = new QPushButton("按钮", this);
    toolBar->addWidget(button);
}
```

## 3.3 状态栏

状态栏最多有一个。状态栏的相关操作如下：

```c++
MainWindow::MainWindow(QWidget *parent): QMainWindow(parent)
{
    // 创建状态栏
    QStatusBar* statusBar = statusBar();

    // 将状态栏设置到窗口中
    setStatusBar(statusBar);

    // 创建标签控件
    QLabel* label = new QLabel("提示信息", this);

    // 将标签控件放入状态栏
    statusBar->addWidget(label);

    // 将标签控件放在状态栏右侧
    QLabel* label2 = new QLabel("右侧提示信息", this);
    statusBar->addPermanentWidget(label2);
}
```

## 3.4 铆接部件

铆接部件可以有多个。铆接部件的相关操作如下：

```c++
MainWindow::MainWindow(QWidget *parent): QMainWindow(parent)
{
    // 创建铆接部件，指定窗口标题和父元素
    QDockWidget* dockWidget = new QDockWidget("浮动", this);

    // 将铆接部件加入窗口中
    // 第一个参数是停靠位置，位置的取值包括：
    //   Qt::LeftDockWidgetArea, Qt::RightDockWidgetArea,
    //   Qt::TopDockWidgetArea, Qt::BottomDockWidgetArea,
    //   Qt::AllDockWidgetArea, Qt::NoDockWidgetArea
    addDockWidget(Qt::BottomDockWidgetArea, dockWidget);

    // 设置允许的停靠范围
    dockWidget->setAllowedAreas(Qt::TopDockWidgetArea | Qt::BottomDockWidgetArea);
}
```

## 3.5 中心部件

中心部件只能有一个。中心部件的相关操作如下：

```c++
MainWindow::MainWindow(QWidget *parent): QMainWindow(parent)
{
    // 创建文本控件
    QTextEdit* edit = new QTextEdit(this);
    // 设置中心部件
    setCentralWidget(edit);
}
```

# 4 对话框

## 4.1 基本概念

Qt 中使用`QDialog`类实现对话框。`QDialog`类及其子类的`parent`指针有额外的含义：如果`parent`为`NULL`，则该对话框会作为一个顶层窗口；否则作为其父组件的子对话框，此时对话框默认出现的位置是父组件的中心。顶层窗口与非顶层窗口的区别在于，顶层窗口在任务栏中会有自己的位置，而非顶层窗口会共享其父组件的位置。

对话框分为模态对话框和非模态对话框。模态对话框会阻塞代码，打开模态对话框后，不可以对其他窗口进行操作；而非模态对话框打开后，仍然可以对其他窗口进行操作。

模态对话框的创建：

```c++
MainWindow::MainWindow(QWidget *parent): QMainWindow(parent)
{
    QDialog dlg(this); // 创建对话框
    dlg.resize(200, 100); // 指定对话框的大小
    dlg.exec(); // 弹出模态对话框
}
```

非模态对话框的创建：

```c++
MainWindow::MainWindow(QWidget *parent): QMainWindow(parent)
{
    QDialog* dlg = new QDialog(this); // 创建对话框
    dlg->setAttribute(Qt::WA_DeleteOnClose); // 当对话框关闭时释放对象，避免内存泄露
    dlg->show(); // 弹出非模态对话框
}
```

## 4.2 标准对话框

标准对话框是 Qt 内置的一系列对话框，用于简化开发。Qt 的标准对话框分为以下几类：

- `QColorDialog`：选择颜色。
- `QFileDialog`：选择文件或目录。
- `QFontDialog`：选择字体。
- `QInputDialog`：允许用户输入一个值，并将其返回。
- `QMessageBox`：消息对话框，用于显示信息、询问问题等，属于模态对话框。
- `QPageSetupDialog`：为打印机提供纸张相关的选项。
- `QPrintDialog`：打印机配置。
- `QPrintPreviewDialog`：打印预览。
- `QProgressDialog`：显示操作过程。

### 4.2.1 消息对话框

消息对话框有不同种类，可以通过不同的静态函数创建不同类型的消息对话框：

- 错误提示对话框：`QMessageBox::critical()`
- 警告对话框：`QMessageBox::warning()`
- 询问对话框：`QMessageBox::question()`
- 提示信息对话框：`QMessageBox::information()`

它们的参数相同，如下所示：

1. `QWidget *parent`：指定父组件。
2. `const QString &title`：对话框的标题。
3. `const QString &text`：对话框中的文字。
4. `StandardButtons buttons`：对话框中的按钮。对于`critical`、`warning`和`information`对话框，默认值为`Ok`；对于`question`对话框，默认值为`StandardButtons(Yes | No)`。
5. `StandardButton defaultButton`：默认关联回车的按钮，默认值为`NoButton`。

这些函数的返回值是`StandardButton`类型，返回的是用户点击的按钮。

### 4.2.2 颜色对话框

颜色对话框的创建：`QColorDialog::getColor()`。

参数：

1. `const QColor &initial`：初始颜色值，默认值为`Qt::white`。
2. `QWidget *parent`：指定父组件，默认值为`nullptr`。
3. `const QString &title`：对话框的标题，默认值为`QString()`。
4. `ColorDialogOptions options`：对话框的选项，默认值为`ColorDialogOptions()`。

返回值：`QColor`类型，返回用户选择的颜色值。

### 4.2.3 文件对话框

文件对话框的创建：`QFileDialog::getOpenFileName()`。

参数：

1. `QWidget *parent`：指定父组件，默认值为`nullptr`。
2. `const QString &caption`：对话框的标题，默认值为`QString()`。
3. `const QString &dir`：默认打开的路径，默认值为`QString()`。
4. `const QString &filter`：过滤文件类型，默认值为`QString()`。书写格式为`"(*.文件类型)"`
5. `QString *selectedFiler`：默认值为`nullptr`。
6. `Options options`：默认值为`Options()`。

返回值：`QString`类型，返回用户选择的文件的路径。

### 4.2.4 字体对话框

字体对话框的创建：`QFontDialog::getFont()`。

参数：

1. `bool *ok`。
2. `const QFont &initial`：默认字体。
3. `QWidget *parent`：指定父组件，默认值为`nullptr`。
4. `const QString &title`：对话框的标题，默认值为`QString()`。
5. `FontDialogOptions options`：默认值为`FontDialogOptions()`。

返回值：`QFont`类型，返回字体信息。

# 5 控件

## 5.1 按钮组

1. `QPushButton`：常用按钮。
2. `QToolButton`：工具按钮，常用于显示图片。
3. radioButton：单选按钮。
4. checkbox：多选按钮。

## 5.2 QListWidget

`QListWidget`是一个列表容器，其中的一行为`QListWidgetItem`类型。

向容器中添加一行：`addItem(QListWidgetItem *item)`。

向容器中添加多行：`addItems(const QStringList &labels)`。

设置一行中的文本对齐方式：`setTextAlignment(int alignment)`。

## 5.3 QTreeWidget

`QTreeWidget`是一个树控件，将若干个节点组织成树形结构，其中的每一个节点为`QTreeWidgetItem`类型。

设置列头：`setHeaderLabels(const QStringList &labels)`。

添加顶层节点：`addTopLevelItem(QTreeWidgetItem *item)`。

添加子节点：`addChild(QTreeWidgetItem *child)`。

## 5.4 QTableWidget

`QTableWidget`是一个表格控件。

设置列数：`setColumnCount(int columns)`。

设置水平表头：`setHorizontalHeaderLabels(const QStringList &labels)`。

设置行数：`setRowCount(int rows)`。

设置单元格内容：`setItem(int row, int column, QTableWidgetItem *item)`。

## 5.5 下拉框

添加项目：`addItem(const QString &text, const QVarient &userData = QVarient())`。

设置当前选中项：`setCurrentIndex(int index)`、`setCurrentText(const QString &text)`。
