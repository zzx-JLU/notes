`java.math`包中有两个很有用的类：`BigInteger`和`BigDecimal`。`BigInteger`类实现任意精度的整数运算，`BigDecimal`类实现任意精度的浮点数运算。

下面列出这两个类的常用API。

`BigInteger`类：

```java
/* java.math.BigInteger */
BigInteger add(BigInteger other)
	// 返回这个大整数与 other 的和
BigInteger subtract(BigInteger other)
	// 返回这个大整数与 other 的差
BigInteger multiply(BigInteger other)
	// 返回这个大整数与 other 的积
BigInteger divide(BigInteger other)
	// 返回这个大整数与 other 的商
BigInteger mod(BigInteger other)
	// 返回这个大整数与 other 的余数
BigInteger sqrt()
	// 得到这个大整数的平方根
int compareTo(BigInteger other)
	// 如果这个大整数与 other 相等，返回 0；如果小于 other，返回负数；如果大于 other，返回正数
static BigInteger valueOf(long x)
	// 将普通正数转换为大整数
BigInteger(String val)
	// 将一个字符串转换为大整数
```
`BigDecimal`类：

```java
/* java.math.BigDecimal */
BigDecimal add(BigDecimal other)
	// 返回这个大实数与 other 的和
BigDecimal subtract(BigDecimal other)
	// 返回这个大实数与 other 的差
BigDecimal multiply(BigDecimal other)
	// 返回这个大实数与 other 的积
BigDecimal divide(BigDecimal other)
	// 返回这个大实数与 other 的商。如果商是无限循环小数，会抛出异常
BigDecimal divide(BigDecimal other, RoundingMode mode)
	// 得到一个舍入结果，第二个参数使用 RoundingMode.HALF_UP 采用四舍五入方式
int compareTo(BigDecimal other)
	// 如果这个大实数与 other 相等，返回 0；如果小于 other，返回负数；如果大于 other，返回正数
static BigDemical valueOf(long x)
	// 将 long 类型整数转换为大实数
static BigDemical valueOf(long x, int scale)
	// 返回值等于 x/10^scale 的大实数
```
