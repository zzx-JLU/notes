---
chrome:
    format: "A4"
    headerTemplate: '<div></div>'
    footerTemplate: '<div style="width:100%; text-align:center; border-top: 1pt solid #eeeeee; margin: 10px 10px 20px; font-size: 8pt;">
    <span class=pageNumber></span> / <span class=totalPages></span></div>'
    displayHeaderFooter: true
    margin:
        top: '40px'
        bottom: '80px'
        left: '60px'
        right: '60px'
---

# 1 Java程序范例：Hello World

```java{.line-numbers}
public class HelloWorld
{
    public static void main(String[] args)
    {
        System.out.println("Hello World");
    }
}
```

第一行：`public class HelloWorld`
`public`称为访问修饰符，用于控制程序的其他部分对这段代码的访问级别。
关键字`class`表示一个类，**Java 应用程序中的全部内容都必须放置在类中**。
`class`后面紧跟类名。类名必须以字母开头，后面可以跟字母和数字的任意组合。
类名的命名规范：**骆驼命名法**。类名是以大写字母开头的名词，如果名字由多个单词组成，每个单词的第一个字母都应该大写。
源代码的文件名必须与公共类的名字相同，并用`.java`作为扩展名。

第三行：`public static void main(String[] args)`
运行程序时，总是从指定类中的`main`方法的代码开始执行。
`main`方法必须声明为`public static`。

第五行：`System.out.println("Hello World");`
使用`System.out`对象，调用它的`println`方法，输出字符串“Hello World”，并自动输出换行符。
每条语句用分号结束。

# 2 注释

Java 有 3 种标记注释的方式，展示如下

```java
// System.out.println("Hello World");
单行注释，其内容从双斜线开始，到本行末尾

/* System.out.println("Hello World"); */
多行注释，将一段比较长的注释括起来，可以换行

/**
* This is the firsh java program
* @version 1.01 2020-10-31
* @author All秃
*/
文档注释，可以用来自动生成文档
```

注意：`/* */`注释不能嵌套。
