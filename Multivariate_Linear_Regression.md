# 机器学习

## 多元线性回归
### 多特征
一个简单的例子： 估算房屋价格。

具有多个变量的线性回归也被称为多元线性回归。

$`x_j^{(i)}`$ 表示第 i 个训练样本中的第 j 个特征。

$`x^{(i)}`$ 表示第 i 个训练样本中的输入。

$`m`$ 表示训练样本的数量。

$`n`$ 表示特征的数量。

适应这些多个特征的假设函数的多元形式如下：

```math
h_{\theta}(x) = \theta_0 + \theta_1x_1 + \theta_2x_2 + \theta_3x_3 + ... + \theta_nx_n
```

为了更直观的解释这个函数，我们认为 $`\theta_0`$ 作为房屋的基础价格，$`\theta_1`$ 是每平米的价格， $`\theta_1`$是每层的价格等等。那么， $`\x_1`$就是房屋有多少平方米，$`x_2`$就是房屋的层数。



为了简洁明了得描述监督式学习， 我们的目的就是： 给出训练集， 去学习一个函数 $`h:X \rightarrow Y`$，使得$`h(x)`$成为一个好的关于$`y`$的预测器。 由于某些历史原因，这个函数h是“假设”(hypothesis)的意思。 如图所示，过程如下：

运用矩阵乘法的定义， 我们的多元假设函数可以简洁的表示为：
```math
h_\theta(x) =[\begin{array}{}
\theta_0&&\theta_1&&...&&\theta_n]\end{array}\left[
\begin{array}{c}
  x_0\\
  x_1\\
  .\\
  .\\
  .\\
  x_n
\end{array}
\right]  = \theta^Tx
```

这是一个训练样本的假设函数的向量化，了解更多可以看向量化的课程。

注意：为了方便， 课程中假定$`x_0^{(i)}`$为1（$`i \in 1, ..., m`$）。这使得我们可以对参数和x做矩阵运算，也使得，两个向量$`\theta`$和$`x`$的元素相匹配（都是n+1个元素）。

### 多变量的梯度下降
梯度下降本身是相同的形式，我们只需要为n个特征重复。

```math
\begin{array}{l}
  \text{repeat until convergence:}\lbrace\\
  \theta_0 := \theta_0 - \alpha \frac{1}{m} \sum\limits_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_0^{(i)}\\
  \theta_1 := \theta_1 - \alpha \frac{1}{m} \sum\limits_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_1^{(i)}\\
  \theta_2 := \theta_2 - \alpha \frac{1}{m} \sum\limits_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_2^{(i)}\\
  ...\\
  \rbrace
\end{array}
```

换句话说:

```math
\begin{array}{l}
  \text{repeat until convergence:}\lbrace\\
  \theta_j := \theta_j - \alpha \frac{1}{m} \sum\limits_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_j^{(i)} \; & \text{for j := 0...n}\\
  ...\\
  \rbrace
\end{array}
```

下图展示了单元和多元的梯度下降：

![梯度下降对比](https://git.yizhoucp.cn/xiaoyu/droplet/uploads/f7abd3ea3debf7baf9d67ff2bf550191/gradient_desecent.png)