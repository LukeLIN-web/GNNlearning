## TGL: A General Framework for Temporal GNN Training on Billion-Scale Graphs

优点:

1. design a Temporal-CSR data structure and a parallel sampler to efficiently sample temporal neighbors to form training mini-batches. 
2.  introduce two large-scale real-world datasets 

TGNN通常通过三种方法处理邻居信息：

1. **根据时间对过去的邻居进行分组，并从组中学习序列（基于快照的TGNN）**
2. **向每个过去的邻居添加额外的时间编码**
3. **维护节点记忆（memory），该记忆总结每个节点的到目前为止的状态。**

大多数TGNN变体的统一表示——node memory、attention aggregator 和temporal sampler。

#### 问题

Sampler不能支持Online Training

## 2.1 node memory

当该节点被其它节点引用为时间邻居时，**其node memory用作补充信息，并与节点特征组合作为输入节点特征**。

为了维护每个节点的节点内存，**当事件指示出现新边时**，使用序列模型（RNN或GRU）更新相应的node memory。

## 2.3 **Temporal Sampler**

TGNN在采样时需要考虑边的时间戳。有两种主要的采样策略：

（1）均匀采样，其中过去的邻居被均匀采样

（2）最近（时间上最近）采样。

## 3TGL

#### 训练流程

1. Sample neighbors for the root nodes with times- tamps in the current mini-batch.
2. Lookup the memory and the mailbox for the supporting nodes
3. Transfer the inputs to GPU and update the memory
4. Perform message passing using the updated memory as input.
5. Compute loss with the generated temporal embeddings
6. Updatethe memory and the mailbox for next mini-batch.

### 3.1Parallel Temporal Sampler

TGL开发了一种新的数据结构用来表达时序结构的图数据，称之为T-CSR

我们只需要准备这样的一张edgelist表，并且按照edge time排序（建议数据准备期间就排好序吧，spark跑大量的edge的排序会比用源代码中的这个排序代码快得多，数据很大的话，上面的这个排序代码耗时间很长的）

algorithm 1

```
首先要确定root nodes 和timestamp.
如果l ==0 , 
如果l >0 : 那么并行二分搜索确定每个node 的snapshot.
然后, 
对于每个node, 并行sample snapshot中的k 邻居. 
```

Parallel Sampling 

evenly distributed to each thread 

需要给每个node 加细粒度锁

## 优点

1. 有大量event的时候, 比snapshot 存储的空间小. 

## 缺点

1. sample的时候会把时间排序。 没法处理实时。 不断fine tune。不是stream的。

Reference：

1. TGL https://zhuanlan.zhihu.com/p/547407225

## 代码

下载了wiki 数据集. 

```
python setup.py build_ext --inplace

python train.py --data WIKI --config ./config/TGN.yml

AttributeError: module 'dgl.function' has no attribute 'copy_src'  改成copy_u
```

