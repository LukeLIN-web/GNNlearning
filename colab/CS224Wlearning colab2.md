CS224W学习

##### 环境

python3.9.7 , PyTorch has version 1.10.2

```python
# 激活环境后
conda install ogb
PackagesNotFoundError: The following packages are not available from current channels
#在anaconda 的官网尝试
conda install -c conda-forge ogb
#可以了
```

有两种类, 一个是datasets, 有一些常用数据集 一个是data, 可以把图变成tensor. 

运行, 确实是600个图在ENZYMES数据集
ENZYMES dataset has 6 classes
ENZYMES dataset has 3 features

Graph with index 100 has label Data(edge_index=[2, 176], x=[45, 3], y=[1])

Graph with index 200 has 53 edges

OGB是一个benchmark 数据集, 

The ogbn-arxiv dataset has 1 graph
Data(x=[169343, 128], node_year=[169343, 1], y=[169343, 1], adj_t=[169343, 169343, nnz=1166243])

`nn.Xxx` 需要先实例化并传入参数，然后以函数调用的方式调用实例化的对象并传入输入数据。

**`nn.Xxx`继承于`nn.Module`， 能够很好的与`nn.Sequential`结合使用， 而`nn.functional.xxx`无法与`nn.Sequential`结合使用。**

**`nn.Xxx`不需要你自己定义和管理weight；而`nn.functional.xxx`需要你自己定义weight，每次调用的时候都需要手动传入weight, 不利于代码复用。**

PyTorch官方推荐：具有学习参数的（例如，conv2d, linear, batch_norm)采用`nn.Xxx`方式，没有学习参数的（例如，maxpool, loss func, activation func）等根据个人选择使用`nn.functional.xxx`或者`nn.Xxx`方式。但关于dropout，个人强烈推荐使用`nn.Xxx`方式，因为一般情况下只有训练阶段才进行dropout，在eval阶段都不会进行dropout。在`nn.Xxx`不能满足你的功能需求时，`nn.functional.xxx`是更佳的选择，因为`nn.functional.xxx`更加的灵活(更加接近底层），你可以在其基础上定义出自己想要的功能。

##### dropout

防止过拟合的一种方法 ,随机把一些参数设置为0 , 减少参数数量, 降低模型复杂度

##### GCN网络

forward不会写, `edge_index tensor adj_t` 有什么用?

forward一层层连起来就可以了。

示意图中间这三个点是啥？

BN是啥？

就是把一个batch中的每个tensor的第一个元素都归一化， 然后对每个tensor的第二个元素都归一化， 以此类推。 为了应对有时一个向量里的元素值太大或者太小。

BN num_features 是多少呢？ 



TypeError: append() missing 1 required positional argument: 'module'？

应该`self.convs = torch.nn.ModuleList()`， 没有括号就是一个类。



##### train函数

#####  
