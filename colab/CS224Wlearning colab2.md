CS224W学习

参考 https://github.com/hdvvip/CS224W_Winter2021/blob/main/CS224W_Colab_2.ipynb

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

PyG dataset 是一个list，存储`torch_geometric.data.Data` 对象

```python
 pyg_dataset.num_classes # 标签数量
num_features = pyg_dataset.num_features # Node的特征维数
label = pyg_dataset[idx] # 获取graph
num_edges = pyg_dataset[idx].edge_index.shape[1] // 2 
```

统计无向图的edge数量需要 //2
edge_index 是二维tensor数组，第一行是start node，第二行是end node

600个图在ENZYMES数据集
ENZYMES dataset has 6 classes
ENZYMES dataset has 3 features

Graph with index 100 has label Data(edge_index=[2, 176], x=[45, 3], y=[1])

Graph with index 200 has 53 edges

#### OGB

一个benchmark 数据集, 大量的开源Graph数据集，使用OGB Data Loader 可以自动下载，处理，划分数据集，还可以用OGB Evaluator进行校验

OGB也支持PyG dataset 和 Data

The ogbn-arxiv dataset has 1 graph
Data(x=[169343, 128], node_year=[169343, 1], y=[169343, 1], adj_t=[169343, 169343, nnz=1166243])



##### nn和 nn.Functional的区别?

`nn.Xxx` 需要先实例化并传入参数，然后以函数调用的方式调用实例化的对象并传入输入数据。

**`nn.Xxx`继承于`nn.Module`， 能够很好的与`nn.Sequential`结合使用， 而`nn.functional.xxx`无法与`nn.Sequential`结合使用。**

**`nn.Xxx`不需要你自己定义和管理weight；而`nn.functional.xxx`需要你自己定义weight，每次调用的时候都需要手动传入weight, 不利于代码复用。**

PyTorch官方推荐：具有学习参数的（例如，conv2d, linear, batch_norm)采用`nn.Xxx`方式，没有学习参数的（例如，maxpool, loss func, activation func）等根据个人选择使用`nn.functional.xxx`或者`nn.Xxx`方式。但关于dropout，个人强烈推荐使用`nn.Xxx`方式，因为一般情况下只有训练阶段才进行dropout，在eval阶段都不会进行dropout。在`nn.Xxx`不能满足你的功能需求时，`nn.functional.xxx`是更佳的选择，因为`nn.functional.xxx`更加的灵活(更加接近底层），你可以在其基础上定义出自己想要的功能。

##### dropout

防止过拟合的一种方法 ,随机把一些参数设置为0 , 减少参数数量, 降低模型复杂度

### GCN网络

首先在init函数里定义好每一层的参数，并且使用torch.nn.ModuleList保存Conv和bn对象。

对于GCNConv来说，由于网络的输入，输出和隐藏层维度不同，所以需要分别定义第一层，中间层和最后一层，它们的in_channels 和 out_channels 不同。

BN层使用torch.nn.BatchNorm1d， 只需要num_features*作为输入，*也就是当前层的特征维度*（hidden_dim)*

Softmax函数使用了torch.nn.LogSoftmax，并定义dropout的概率

`data 128 torch.Size([169343, 128])`

data是一个图, 表示为一个tensor, 有16w个节点,每个节点128个feature. 

forward不会写, `edge_index tensor adj_t` 有什么用?

forward一层层连起来就可以了。

示意图中间这三个点是啥？

BN是啥？

就是把一个batch中的每个tensor的第一个元素都归一化， 然后对每个tensor的第二个元素都归一化， 以此类推。 为了应对有时一个向量里的元素值太大或者太小。

BN num_features 是多少呢？ 



TypeError: append() missing 1 required positional argument: 'module'？

应该`self.convs = torch.nn.ModuleList()`， 没有括号就是一个类。

##### train函数

` out = model(data.x, data.adj_t)`  为什么这么传? 

为什么model是传这两个? 

会自动执行forward

**因为 PyTorch 中的大部分方法都继承自 torch.nn.Module，而 torch.nn.Module 的__call__(self)函数中会返回 forward()函数 的结果，因此PyTroch中的 forward()函数等于是被嵌套在了__call__(self)函数中；因此forward()函数可以直接通过类名被调用，而不用实例化对象**

x是啥? 

x是节点的特征向量 ,y是标签 

为什么要squeeze?

 **https://pytorch.org/docs/stable/generated/torch.squeeze.html**  去掉维度为1 的维度, 在给定的维度进行一个挤压操作. 你可以把这个squeeze去掉看看会报什么错



`y = x.squeeze(1)` 等价于`y = torch.squeeze(x, 1)` 

   x = self.convs(x) 为什么会有NotImplementedError?

