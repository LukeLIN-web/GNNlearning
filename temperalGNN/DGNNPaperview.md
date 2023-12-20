怎么group呢? 等吗? 

在每个group 中 online serving. 

有两种,  forward inference-based (inf-based) approach and backend updatebased (upd-based) approach. 来更新dynamic graphs.

inf base, 只在 收到的时候改变图结构. aligraph.



Refer:

[图表示学习] 2 动态图(Dynamic Graph)最新研究总结（2020） - Ziyue Wu的文章 - 知乎 https://zhuanlan.zhihu.com/p/274364815

memory-based TGNNs:  DistTGL , tgn

总结一下不同的dgnn代码的对象.

| paper         | ctgn | dtgn |      |
| ------------- | ---- | ---- | ---- |
| DecoupledDGNN | √    | √    |      |
| pyg - tgn     | √    |      |      |
|               |      |      |      |

有代码

1. Aligraph , It provides a performance guarantee of sampling P99 latency in 20ms on large-scale dynamic graphs.   几百个worker, 在淘宝数据集上,  他们有很多模型, 测了模型精度.
2. https://github.com/zheng-yp/DecoupledDGNN  人大的论文,  vldb23 ,只是可用不能复现.  统一了连续 和离散dgnn.

#### 没有代码

1. Efficient Scaling of Dynamic Graph Neural Networks. SC'21  
2. SPEED: Streaming Partition and Parallel Acceleration for Temporal Interaction Graph Embedding
3. Redundancy-Free High-Performance Dynamic GNN Training with Hierarchical Pipeline Parallelism
4. Cache-based gnn system for dynamic graphs
5. STAG: Enabling Low Latency and Low Staleness of GNN-based Services with Dynamic Graphs https://arxiv.org/pdf/2309.15875.pdf
6. DynaGraph: Dynamic Graph Neural Networks at Scale
7. PiPAD: Pipelined and Parallel Dynamic GNN Training on GPUs
8. Streaming Graph Neural Networks  2018年
9. Bottleneck Analysis of Dynamic Graph Neural Network Inference on CPU and GPU . 分析了问题 

## disttgl

 https://github.com/amazon-science/disttgl  , optimize tgn multiple GPU training, memory-based TGNNs.  提出了三种Parallelism，搞清楚都是为什么在做什么

哪些假设 默认是对的? 

problem

1. figure3  staleness and information loss. - > new model 

2. need synchronous.  跨server同步的开销非常大.  -> a novel training algorithm, and an optimized system.

contribution:  怎么解决提出的问题.

1. enhances the node memory in M-TGNNs by introducing additional static node memory,   优化了 accuracy and convergence rate 
2. introduces two novel parallel training strategies - epoch parallelism and memory parallelism.  Additionally, DistTGL provides heuristic guidelines to determine the optimal training configurations based on the dataset and hardware characteristics.
3. adopting prefetching and pipelining techniques to minimize the mini-batch generation overhead. It serializes the memory operations on the node memory and efficiently executes them by an independent daemon process -- >  解决 complex and expensive synchronizations

### 1 介绍

batch size 增加 ,acc会减少, 为什么? 

model: **添加了additional static node memory.**   -> acc提高, 加速. 是解决哪个问题? 

System:  adopting **prefetching and pipelining** techniques to minimize the mini-batch generation overhead   ->  是解决哪个问题? 

### 2 背景

M-TGNN并行的算法:  原先是process consecutive graph events that do not have overlapping nodes in batches by updating their node memory in parallel. 但是这个方法batch size不能太大不然肯定有overlap.    但是batch size太小又不能充分利用GPU的并行性. 所以MTGNN 大batch 处理events,  少量更新 node memory.  但是这会导致figure3 的 staleness and information loss.

###  3

为什么说fails on dynamic graphs?   While this may be true on some evolving graphs like citation graphs, it fails on the dynamic graphs where  high-frequency information is important. 

we separate the static and dynamic node memory and capture them explicitly.  DistTGL keeps the original GRU node memory on all nodes to capture the dynamic node information and **implements an additional mechanism to capture the static node information**.

超越了tgn. 

#### 3.2

epoch 并行: training different epochs simultaneously using only one copy of the node memory.

为什么需要 negative mini-batch? 

memory parallelism: each trainer uses its own copy of the node memory to process and update the graph events within that segment. 















## BLAD

问题:  

贡献:  每个共享是对应哪个问题

假设:  

哪里可以进一步提升,被攻击. 

