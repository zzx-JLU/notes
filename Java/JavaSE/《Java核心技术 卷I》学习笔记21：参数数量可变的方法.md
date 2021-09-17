我们常用的`printf`方法是一个参数数量可变的方法，它的定义为：

```java
public PrintStream printf(String fmt, Object... args)
{
	return format(fmt, args);
}
```
这里的省略号`...`是Java代码的一部分，它表明这个方法可以接收任意数量的对象。

`printf`方法接收两个参数，一个是格式字符串，另一个是`Object[]`数组。换言之，`Object...`参数类型与`Object[]`完全一样。编译器需要转换每个`printf`调用，将参数绑定到数组中，并在必要的时候进行自动装箱。

可以自定义有可变参数的方法，可以为参数指定任意类型。下面是一个示例，计算若干个数值中的最大值：

```java
public static double max(double... values)
{
	double largest = Double.NEGATIVE_INFINITY; // 浮点型的特殊值，表示负无穷
	for (double v : values)
		if (v > largest)
			largest = v;
	return largest;
}
```
允许将数组作为最后一个参数传递给有可变参数的方法，例如：

```java
System.out.printf("%d %s", new Object[] {new Integer(1), "widgets"});
```
因此，如果一个方法的最后一个参数是数组，可以把它重新定义为有可变参数的方法，而不会破坏任何已有的代码。