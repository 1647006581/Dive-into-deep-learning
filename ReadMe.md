# 线性回归学习笔记

        线性回归是一种简单的回归模型，其作用是预测一个线性的连续

<img src="file:///C:/Users/ASUS/AppData/Roaming/marktext/images/2024-01-26-15-40-08-image.png" title="" alt="" width="590">

## 线性模型

        假设n维输入向量$ \mathbf{x} = [x_1, x_2, \ldots, x_n]^T $

        输出结果为 $\mathbf{y}=w_1 * x_1 + w_2 *x_2 + ...+w_n*x_n + b$    

        $w$表示 weight 权重   $b$表示bias 偏移

               其中 $\mathbf{w} = [w_1, w_2, \ldots, w_n]^T$

         这里 线性回归要做的任务就是通过已有的一些离散的点去拟合出一组$\mathbf{w},b$ 得到一条曲线用以预测关于对应特征的关系

        $\hat{y}=<\mathbf{w},\mathbf{x}>+  b$

## 训练数据

        假设有n个样本记为:(<mark>注意这里$X$ 和前面所述的向量 $x$不一样 </mark>，<mark>此处为样本 可以理解为下方的$x_i$为一个前面说的向量$x$)</mark>
        $X = [x_1, x_2, …, x_n]^T  ,y = [y_1, y_2, …, y_n]^T$

## 损失函数

       评估模型训练好坏最直观的方式就是查看预测值和真实值的差值

            平方损失函数定义如下：~~(二分之一是因为可以求导消去，其实无所谓是多少)~~

            $ℓ(\hat{y}, y) = \frac{1}{2} (\hat{y} - y)^2$

## 优化算法

        有了模型和损失函数，相当于有了材料和工具，我们可以通过损失函数评估目前模型训练好坏，但是还需要一个达到我们可接受的损失的方法  ~~（主观能动性）~~，

        对于线性回归，我们回到损失函数    $ℓ(\hat{y}, y) = \frac{1}{2} (\hat{y} - y)^2$

        也就是$\frac{1}{2} (\hat{y} - y)^2 = \frac{1}{2} (w_1 * x_1 + w_2 *x_2 + ...+w_n*x_n + b - y)^2$

        为了方便推导,简写为内积格式$\frac{1}{2} (\hat{y} - y)^2 = \frac{1}{2} (<\mathit{\mathbf{x}},\mathbf{w}> + b - y)^2$

         

        有了训练数据 以及等待预测参数$\mathbf{w}$和$b$，损失函数可以具体写为：

        $ℓ(\hat{y}, y) = ℓ(\mathbf{X},\mathbf{w},b,\mathbf{y} )$

                       $=\frac{1}{2n}\sum_{i=1}^{n}{(<\mathit{\mathbf{x_i}},\mathbf{w}> + b - y_i)^2}$ 

                       $=\frac{1}{2n}\|\mathbf{X}\mathbf{w}+b-\mathbf{y}  \|^2$

>          为什么要➗n ？ 为了获得平均损失，在评估精度的时候好一些。

        此时我们得到最终的损失函数了，因为我们要训练出$\mathbf{w}$和$b$的值，优化是针对这两个参数的，不要陷入思维定势盯着*X* 和 *y*。

>         这里的逻辑是，训练并找出出一对精度可以接受的$\mathbf{w}$和$b$，用于拟合**y**与**X**的线性关系 

### GD梯度下降算法

        观察损失函数，$\ell$  关于$w$的偏导，$\ell$ 关于$b$的偏导  ~~(太麻烦不加粗了，而且不应该叫偏导应该叫梯度)~~

        把这两个向量看成两个变量话，可以得到$\ell$有只一个极值点，不用判断了只能是这个点了，所以现在的目标就是不断逼近这个“点”

>         这里是推导过程：<img src="file:///C:/Users/ASUS/AppData/Roaming/marktext/images/2024-01-26-18-00-42-image.png" title="" alt="" width="594">

        接下来就引出了梯度下降优化方法，用$w$举例，选定一个初始值$w_0$ 此时一定是和这个最优或者说较优解是比较远的，而快速向解逼近的方向就是梯度方向（梯度性质），此时我们只需要不断用上一个向量$w$减去当前向量$w$

> 梯度的方向指向函数在给定点上增加最快的方向。梯度的反方向指向函数减小最快的方向。

> <img title="" src="file:///C:/Users/ASUS/AppData/Roaming/marktext/images/2024-01-26-18-02-27-image.png" alt="" width="491">
> 
> 将过程画出来：
> 
> <img src="file:///C:/Users/ASUS/AppData/Roaming/marktext/images/2024-01-26-18-24-51-image.png" title="" alt="" width="378">

        当然，为了迭代可控，可以引入一个参数，学习率来控制步长。学习率过大过小都可能引发问题，学习率过小可能会导致计算量过大会占用大量的资源，学习率过大可能会因为步长过大无法获得较精确解，或者是导致求导过程中出现➗0。

## 关键代码实现

pytorch实现，用到的函数如下

```python
def linreg(X,w,b):
"""建立模型"""
    return torch.matmul(X, w) + b
```

```python
def square_loss(y_hat,y):
"""损失函数"""
    return (y_hat - y.reshape(y_hat.shape))**2 / 2
```

#### 数据迭代器

    定义一个数据迭代器实现随机分批次，*yield*关键字的作用：用于定义生成器函数。生成器函数与普通函数不同，它的执行是延迟的，只有在需要时才会产生一个值。下面是一个对比

```python
def data_iter(batch_size,features,labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0,num_examples,batch_size):
        batch_indices = torch.tensor(indices [i : min(i+batch_size,num_examples)])
        yield features[batch_indices],labels[batch_indices]

```

    两者的不同将在调用时体现

```python
def data_iter(batch_size,features,labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0,num_examples,batch_size):
        batch_indices = torch.tensor(indices [i : min(i+batch_size,num_examples)])
        return features[batch_indices],labels[batch_indices]
#调用必须有申请空间全部存下
a = data_iter(batch_size,features,labels)
```



#### 小批量随机梯度下降

```python
def sgd(params,lr,batch_size):
    """小批量梯度下降"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


```

#### 训练

```python
"""初始化"""
batch_size = 10
w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

num_epoch = 5
lr = 0.03
net = linreg
loss = square_loss
for epoch in range(num_epoch):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)
        l.sum().backward()
        sgd([w, b], lr, batch_size)
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

```

    `for X, y in data_iter(batch_size, features, labels):`这里体现出了生成器函数的作用，如果你不使用 `yield`，而是在函数内部直接返回一个包含所有批次数据的数据结构，比如列表，那么你需要一次性将整个数据集加载到内存中。这可能会导致内存不足的问题，特别是当处理大规模数据集时。

    `with torch.no_grad()`用于指定一段代码块，在这个代码块中，PyTorch会关闭梯度计算。因为w，b在创建时`requires_grad = True`在梯度下降算法中，PyTorch会自动计算这些梯度，在进行模型参数更新的时候，我们不再需要继续保留之前计算的梯度，因为我们只关心使用当前梯度进行参数更新。为了减少内存消耗和计算开销，通常会在进行参数更新时清零之前计算的梯度，以防止梯度信息累积。也就是sgd函数中的`param.grad.zero_()`

## 完整代码以及结果

```python
from matplotlib import pyplot as plt
import random
import torch as tc
import numpy as np
# y = wx + b
num_inputs = 2
num_examples = 1000
true_w = [2,-3.4]
true_b = 4.2
features = tc.randn((num_examples,num_inputs))
labels = true_w[0] * features[:,0] + true_w[1] * features[:,1] + true_b
labels += tc.normal(mean=0, std=0.01,size=labels.shape)

plt.scatter(features[:,0].numpy(),labels.numpy(),s=1)
plt.xlabel("Feature 1")
plt.ylabel("Labels")
plt.title("Scatter Plot")
plt.scatter(features[:,1].numpy(),labels.numpy(),edgecolors='r',s=1)
plt.legend

def data_iter(batch_size,features,labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0,num_examples,batch_size):
        batch_indices = tc.tensor(indices [i : min(i+batch_size,num_examples)])
        yield features[batch_indices],labels[batch_indices]




def square_loss(y_hat,y):
    return (y_hat - y.reshape(y_hat.shape))**2 / 2

def linreg(X,w,b):
    return tc.matmul(X, w) + b

def sgd(params,lr,batch_size):
    """小批量梯度下降"""
    with tc.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()



batch_size = 10
w = tc.normal(0, 0.01, size=(2,1), requires_grad=True)
b = tc.zeros(1, requires_grad=True)

num_epoch = 5
lr = 0.03
net = linreg
loss = square_loss
ind = 0
for epoch in range(num_epoch):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)
        l.sum().backward()
        sgd([w, b], lr, batch_size)
    with tc.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')



x_val = np.arange(features[:, 1].min(), features[:, 1].max(),0.1)

w_val = w[1].detach().numpy().item()
b_val = b.detach().numpy()
y_val = w_val * x_val + b_val
print(y_val)

plt.plot(x_val, y_val)
plt.legend
plt.show()
```

代码中需要注意的是torch的生成自定义标准分布随机数

`torch.randn` 函数用于生成服从标准正态分布的随机数，而不允许直接设置标准差。如果你想生成服从其他正态分布的随机数，可以使用 `torch.normal` 函数。

`torch.normal(mean=0, std=0.01,size=labels.shape)` 均值0 ，标准差0.01，形状为labels的形状(1，1000)

`torch.randn(size=(2,1))`生成服从正态分布的形状为(2，1)的随机数 

用`normal`必须指定`mean std size`的参数，用`randn`不能修改`mean std`

结果如下，其预测的是$w_2$的权重以及偏移$b$

<img src="file:///C:/Users/ASUS/AppData/Roaming/marktext/images/2024-01-26-20-06-57-image.png" title="" alt="" width="635">

# 2.Softmax回归


