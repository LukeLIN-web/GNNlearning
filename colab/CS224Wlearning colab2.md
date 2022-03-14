CS224W学习

参考 https://github.com/hdvvip/CS224W_Winter2021/blob/main/CS224W_Colab_2.ipynb

##### 环境

python3.9.7 , PyTorch has version 1.10.2

遇到了问题undefined symbol, 可以参考下面.

https://github.com/pyg-team/pytorch_geometric/issues/3484



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

防止过拟合的一种方法 ,随机把一些参数设置为0 , 减少参数数量, 降低模型复杂度.

`x = F.dropout(x, p=self.dropout, training=self.training)`

### GCN网络

首先在init函数里定义好每一层的参数，并且使用torch.nn.ModuleList保存Conv和bn对象。

对于GCNConv来说，由于网络的输入，输出和隐藏层维度不同，所以需要分别定义第一层，中间层和最后一层，它们的in_channels 和 out_channels 不同。

BN层使用torch.nn.BatchNorm1d， 只需要num_features*作为输入，*也就是当前层的特征维度*（hidden_dim)*

Softmax函数使用了torch.nn.LogSoftmax，并定义dropout的概率

`data 128 torch.Size([169343, 128])`

data是一个图, 表示为一个tensor, 有16w个节点,每个节点128个feature.  x就是二维的tensor 16w* 128 , adj_t 也是tensor 16w * 16w.

forward不会写, `edge_index tensor adj_t` 有什么用?

GCNConv需要邻接矩阵.  forward一层层连起来就可以了。

示意图中间这三个点是啥？

就是每一层都和第一层一样

BN是啥？

有时一个向量里的元素值太大或者太小。就把一个batch中的每个tensor的第一个元素都归一化， 然后对每个tensor的第二个元素都归一化， 以此类推。 

BN num_features 是多少呢？ 

是hidden dim , 为什么呢?  因为BN 连在 GCNconv的后面.

TypeError: append() missing 1 required positional argument: 'module'？

应该`self.convs = torch.nn.ModuleList()`， 没有括号就是一个类。

##### train函数

` out = model(data.x, data.adj_t)`  为什么model是传这两个? 

会自动执行forward

**因为 PyTorch 中的大部分方法都继承自 torch.nn.Module，而 torch.nn.Module 的__call__(self)函数中会返回 forward()函数 的结果，因此PyTroch中的 forward()函数等于是被嵌套在了__call__(self)函数中；因此forward()函数可以直接通过类名被调用，而不用实例化对象**

x是啥? 

x是节点的特征向量 ,y是标签 

为什么要squeeze?

 **https://pytorch.org/docs/stable/generated/torch.squeeze.html**  去掉维度为1 的维度, 在给定的维度进行一个挤压操作. 你可以把这个squeeze去掉看看会报什么错

`y = x.squeeze(1)` 等价于`y = torch.squeeze(x, 1)` 

报错: RuntimeError: 0D or 1D target tensor expected, multi-target not supported. 需要变成1维

val是训练过程中的测试集，是为了让你在边训练边看到训练的结果，及时判断学习状态。*test*就是训练模型结束后，用于评价模型结果的测试

```shell
Epoch: 01, Loss: 4.1908, Train: 25.46%, Valid: 28.79% Test: 25.75%
Epoch: 02, Loss: 2.3507, Train: 22.43%, Valid: 21.17% Test: 26.66%
Epoch: 03, Loss: 1.9348, Train: 32.51%, Valid: 26.68% Test: 31.12%
Epoch: 04, Loss: 1.7861, Train: 33.88%, Valid: 21.63% Test: 18.61%
Epoch: 05, Loss: 1.6681, Train: 35.50%, Valid: 24.05% Test: 21.02%
Epoch: 06, Loss: 1.5779, Train: 36.02%, Valid: 26.58% Test: 25.52%
Epoch: 07, Loss: 1.5019, Train: 36.38%, Valid: 30.07% Test: 31.83%
Epoch: 08, Loss: 1.4513, Train: 37.16%, Valid: 34.15% Test: 38.13%
Epoch: 09, Loss: 1.4104, Train: 38.04%, Valid: 36.17% Test: 41.44%
Epoch: 10, Loss: 1.3717, Train: 38.18%, Valid: 35.10% Test: 41.04%
Epoch: 11, Loss: 1.3339, Train: 38.92%, Valid: 34.80% Test: 40.79%
Epoch: 12, Loss: 1.3157, Train: 40.44%, Valid: 36.50% Test: 42.53%
Epoch: 13, Loss: 1.2892, Train: 42.95%, Valid: 40.94% Test: 46.42%
```

#### 图属性预测

ogbg-molhiv是个较小的分子属性预测数据集，用于图分类任务（二元分类）。有41,127个无向图，平均每个图有25.5个节点、13.75个边。任务目标是二元分类。评估指标是ROC-AUC。节点有9维特征

Task type: binary classification

加载dataset , 设置好device , 设置好split, dataset有一个属性是任务类型. 

然后加载dataset到pyG.data 的DataLoader中, 有train, 验证, 测试三种dataloader. 这时候要设置一个batch 32个graph, 训练的时候要shuffle 图的顺序. 

*torch_geometric.data.Batch* 有一个属性是batch, 标出每个节点属于哪个batch

 0 0 0 1  n-2

比如第一个0是指第一个节点属于第0个batch

首先使用AtomEncoder， 将节点属性和边属性分别映射到一个新的空间，在这个新的空间中，就可以对节点和边进行信息融合。这里只需要hidden_dim，会把输入x处理成hidden_dim的维度，不需要input_dim

然后用前面GCN的框架做Node embedding

关键用global mean pooling， 对输出的所有node_feature 做平均，得到graph的特征

```python
self.pool = global_mean_pool(hidden_dim,)
# 这个参数应该填写啥? 
其实啥也不用填, 就直接
self.pool = global_mean_pool
```

ImportError: IProgress not found. Please update jupyter and ipywidgets?

我尝试安装一下. `conda install -c conda-forge ipywidgets`

`out = model(batch)` 这个该填什么参数? 

看forward的参数设计即可. 

```python

t = t.float()    #'float32'
t = t.double()   #'float64'
```

Iteration:   0%|          | 0/1029 [00:00<?, ?it/s]

为啥不是30个epoch? 

因为是minibatch 1029个. 

为啥迭代不动? 

batch size变小了, 也还是不动. 

```python
if step % 100 ==0:
    print(loss)# 常用的输出.
    #loss 不变小. 为啥呢?
    #因为每一次都在不同的batch上训练啊,肯定开始时不稳定的,log的话可能比较小

loss = loss_fn(out[is_labeled], batch.y[is_labeled].float())

#改了一处这里, Tensor. *type_as* (tensor) → Tensor. Returns this tensor cast to the type of the given tensor`
    loss = loss_fn(out[is_labeled], batch.y[is_labeled].type_as(out))
```

一个epoch需要多少个step?



AUC-ROC（Area Under Curve）是机器学习中常用的一个分类器评价指标

可以看https://ogb.stanford.edu/docs/leader_graphprop/#ogbg-molhiv 的实现, AUC能解决两个分类的数据不平衡的问题.没有万能的指标去评价模型，必须结合业务场景，选取最适合的指标。除了ROC曲线可以计算AUC之外，PR曲线也可以有类似的AUC计算
