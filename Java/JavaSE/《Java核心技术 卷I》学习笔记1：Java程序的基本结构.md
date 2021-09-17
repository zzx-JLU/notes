[toc]
# 1 Java程序范例：Hello World

```java
public class HelloWorld
{
	public static void main(String[] args)
	{
		System.out.println("Hello World");
	}
}
```
第一行：```public class HelloWorld```<br>
`public`称为访问修饰符，用于控制程序的其他部分对这段代码的访问级别。<br>
关键字`class`表示一个类，**Java应用程序中的全部内容都必须放置在类中**。<br>
`class`后面紧跟类名。类名必须以字母开头，后面可以跟字母和数字的任意组合。<br>
类名的命名规范：**骆驼命名法**。类名是以大写字母开头的名词，如果名字由多个单词组成，每个单词的第一个字母都应该大写。<br>
源代码的文件名必须与公共类的名字相同，并用`.java`作为扩展名。

第三行：```public static void main(String[] args)```<br>
运行程序时，总是从指定类中的`main`方法的代码开始执行。<br>
`main`方法必须声明为`public static`。

第五行：```System.out.println("Hello World");```<br>
使用`System.out`对象，调用它的`println`方法，输出字符串“Hello World”，并自动输出换行符。<br>
每条语句用分号结束。
# 2 注释
Java有3中标记注释的方式，展示如下

```java
// System.out.println("Hello World");
第一种注释方式，其内容从双斜线开始，到本行末尾

/* System.out.println("Hello World"); */
第二种注释方式，将一段比较长的注释括起来，可以换行

/**
* This is the firsh java program
* @version 1.01 2020-10-31
* @author All秃
*/
第三种注释方式，可以用来自动生成文档
```
注意：`/* */`注释不能嵌套。