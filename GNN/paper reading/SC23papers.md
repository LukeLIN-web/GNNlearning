## BLAD

BLAD: Adaptive Load Balanced Scheduling and Operator Overlap Pipeline For Accelerating The Dynamic GNN Training.  来自 sjtu 不是ipads的.  不可复现. dockerhub和 github 都删除了.

问题:  high communication overhead  ->  把数据集分成snapshot group, 然后allocates each snapshot group to different  GPU. 

long synchronization ->

and poor resource usage.  ->  启动多个GPU,同时执行compute密集和memroy 密集的operators.  另外, 还调整 model执行顺序. 

贡献:  是对应哪个问题

假设:    哪里可以进一步提升,被攻击. 

观察challenge  :   不同snapshot 的vertice 数量差距大.   

### 摘要

调度workload 减少同步和通讯时间.  调整op执行顺序.

allocates each snapshot group to a GPU

### INTRODUCTION

Vertex partition (e.g. Aligraph, DGL, PyGT) distributes the vertices of each graph snapshot to different GPUs

snapshot partition(e.g. ESDG) distributes the snapshots to different GPUs.  hidden states are transferred with the snapshot partition

问题: high communication overhead, long synchronization, and poor resource usage.

因为 node变化, 所以不同GPU load不 balance. 

#### 方案

All groups **without data dependency** are scheduled to one GPU, and each GPU/node is assigned multiple groups.

每个gpu两个进程,训练不同的snapshot, overlap不同类型的op.

topology manager 调整 model的op 执行顺序. 

### 3Background

之前的snapshot也要提供feature. 

#### TWO-LEVEL LOAD SCHEDULING

 𝑠𝑔𝑖 represents the 𝑖-th snapshot group.

forward之后放入 queue,  为啥有的进入P2, 有的不进入? 



## DistTGL

```bash
https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/REDDIT/edge_features_e0.pt
HTTP request sent, awaiting response... 403 Forbidden
2023-12-18 12:01:25 ERROR 403: Forbidden.
# 我用这个edge feature试试 ,名字不一样:
wget -P ./DATA/WIKI https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/WIKI/edge_features.pt 
python gen_minibatch.py --data WIKI --gen_eval --minibatch_parallelism 2
# gen_minibatch.py 干了什么呢? 
好像是把neg mfg, pos mfg 都sample 出来存起来.  是否实验不公平? 
args.group = 0是不行的. 
args.group = 1 会出错. 

torchrun --nnodes=1 --nproc_per_node=2 --rdzv_id=0 --rdzv_backend=c10d train.py --data WIKI --group 1 --minibatch_parallelism 2

python gen_minibatch.py --data WIKI --gen_eval --minibatch_parallelism 1
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_id=0 --rdzv_backend=c10d train.py --data WIKI --group 1 --minibatch_parallelism 1
```

i j  k 应该是多少呢    

 𝑖 represents how many GPUs to compute each mini-batch,  i =2 

 𝑘 represents how many copies of node memory to maintain, 

and 𝑗 represents how many epochs to train in parallel for each copy of node memory.

能不能一个机器搞?

 https://github.com/amazon-science/disttgl  , optimize tgn multiple GPU training, memory-based TGNNs.  提出了三种Parallelism，搞清楚都是为什么在做什么

哪些假设 默认是对的? 

#### problem

1. figure3  staleness and information loss.  怕information leak,所以要晚一个step训练.  解决方法 - > new model  提高acc. 

2. need synchronous.  同步的开销非常大  ->  memory parallelism不用同步node memory.  

#### contribution:  怎么解决提出的问题.

1. enhances the node memory in M-TGNNs by introducing additional static node memory,   优化了 accuracy and convergence rate   (是否给每个gpu 一个static?)
2. introduces two novel parallel training strategies - epoch parallelism and memory parallelism.   还有  heuristic guidelines to determine the optimal training configurations based on the dataset and hardware characteristics.
3. adopting prefetching and pipelining techniques to minimize the mini-batch generation overhead. It serializes the memory operations on the node memory and efficiently executes them by an independent daemon process -- >  解决 complex and expensive synchronizations

### 1 介绍

batch size 增加 ,acc会减少, 为什么? 

model: **添加了additional static node memory.**   -> acc提高, 加速. 是解决哪个问题? 

System:  adopting **prefetching and pipelining** techniques to minimize the mini-batch generation overhead   ->  是解决哪个问题? 

你想你训练的时候同时在更新两件事情，一个是模型的参数，还有一个就是Node Embedding，Paper里面讲的最开始初始化为0，之所以有Info Leak的问题，也是因为我们在更新Node Embedding

当你训练结束，推理的时候我的理解用的肯定是已经训练好的Node Embedding，因为我们也会假设这个Feature包含了历史的信息，我这么理解对吗？

推理是从0开始的吗?  还是用 已经训练好的Node Embedding 继续训练?

### 2 背景

delay update node memory的原因是 防止information leak.

每个node 会有一个node memory Sv,  这node memory 是很占据内存的. 

有event , 会产生  mu, mv.  mu 的自变量有:  sv,su, time encoding 和 euv

然后update su, sv.  update函数可以是任何sequence model. 

M-TGNN并行的算法:  原先是process consecutive graph events that do not have overlapping nodes in batches by updating their node memory in parallel. 但是这个方法batch size不能太大不然肯定有overlap.    但是batch size太小又不能充分利用GPU的并行性. 所以MTGNN 大batch 处理events,  少量更新 node memory.  但是这会导致figure3 的 staleness and information loss.

###  3

While this may be true on some evolving graphs like citation graphs, it fails on the dynamic graphs where  high-frequency information is important.  为什么说fails on dynamic graphs?   

优点:  we separate the static and dynamic node memory and capture them explicitly.  DistTGL keeps the original GRU node memory on all nodes to capture the dynamic node information and **implements an additional mechanism to capture the static node information**.  效果-> 提高了acc. 

超越了tgn. 

#### 3.2

figure 7 :  i  是 第i个 mini batch的意思

mini batch, 就是简单的数据划分. 

epoch 并行: training different epochs simultaneously using only one copy of the node memory.  有3个GPU, 在3个iter 就训练同一个mini batch的3个epoch, 优点: 需要的内存少, 同时可以capture dependency. 

为什么需要 negative mini-batch?  为什么要有neg mfg?

memory parallelism: each trainer uses its own copy of the node memory to process and update the graph events within that segment.  优点: 不需要trainer之间同步node memory

### 4

#### 4.2

DistTGL only applies memory parallelism across machines,  只需要同步weight, 不需要同步 node memory. 

## 代码

self.read_status[self.rank] =1 , 没有重置为0 ? 

read_status 是在读什么东西?   似乎是卡CPU瓶颈了.  一个GPU 对应一个CPU进程。 

`node_memory.zero_()`  卡住了, 这个Size有多大？ torch.Size([9228, 100])是在cpu还是在GPU?  是在cpu. 单独测一下要多久? 一瞬间. 

是不是别的也在访问 ,  有个barrier 在等别人用.  还有什么在访问 node memory? 死锁.  需要简化场景. 怎么简化场景?  二分法注释掉无关的代码.

可能是 MailBox 也在访问? 

https://github.com/pytorch/pytorch/blob/c7e9c1510274184b41e408e6409f252bb1717085/torch/_refs/__init__.py#L6179C28-L6179C28 zero_() 的源码

 单GPU. 单minibatch， 也会卡。 for循环全部置为0 不会出错 . 

