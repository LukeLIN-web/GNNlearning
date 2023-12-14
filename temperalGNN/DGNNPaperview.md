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

### disttgl

 https://github.com/amazon-science/disttgl  , optimize tgn multiple GPU training, memory-based TGNNs.

problem

1. their node memory favors smaller batch sizes to capture more dependencies in graph events. Why? memory lack?这个node memory 不是内存, 是记忆. 

2. need synchronous.  跨cluster同步的开销非常大. 

方法

an enhanced TGNN model, a novel training algorithm, and an optimized system.

#### 1 介绍

batch size 增加 ,acc会减少, 为什么? 

model: 添加了additional static node memory, which improves both the accuracy and convergence rate.

System: We build an optimized system adopting prefetching and pipelining techniques to minimize the mini-batch generation overhead

并行的算法:  原先是process consecutive graph events that do not have overlapping nodes in batches by updating their node memory in parallel. 但是这个方法batch size不能太大不然肯定有overlap.  

####  3

we separate the static and dynamic node memory and capture them explicitly. DistTGL keeps the original GRU node memory on all nodes to capture the dynamic node information and implements an additional mechanism to capture the static node information.

超越了tgn. 
