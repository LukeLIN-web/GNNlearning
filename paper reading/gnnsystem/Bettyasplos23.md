Betty: Enabling Large-Scale GNN Training with Batch-Level Graph Partitioning

dongli是UCM的 ap 

#### 摘要

Betty introduces two novel techniques, redundancy-embedded graph (REG) partitioning and memory-aware partitioning, to effectively mitigate the redundancy and load imbalances issues across the partitions

## 1 简介

分析了GNN scalability bottlenecks and identify that feature vectors and their corresponding hidden maps involved in aggregation as a major source of memory consumption

阐述了Batch-Level Partitioning, 也就是micro batch的由来和困难

we formulate the batch-level partitioning problem for GNN as a multi-level bipartite graph partitioning problem, and introduce two novel techniques to partition a batch.  

1. Betty constructs a redundancy-embedded graph and transforms the redundancy reduction problem to a min-cost flow cut problem. The later can be effectively solved via a generic graph partitioning algorithm.

2) reduce the maximal memory footprint, Betty introduces a memory-aware partitioning algorithm which partitions a batch based on accurate estimation of the memory usage of GNN batch

figure3 55% GPU消耗是 input features.   单独测了每个部分的GPU. 怎么测的?

Mean aggregator , 我们是mean 吗?是的, 默认是mean. LSTM能把精度提高多少? 

### 3.3

讲了mini batch size 有问题. 

## 4 design

#### 4.1

讲了micro batch的优点, 节省GPU显存. 

#### 4.2

为什么用batch partition不用更小的batch size呢?   因为batch size大 训练效果泛化性好, 收敛性好. 

figure6, micro batch是把所有数据过了才更新梯度的, 为什么? 

#### 4.3Redundancy Reduction

1. runs OOM when some of the micro- batches have unbalanced loads that exceed the memory capacity

2. a large amount of redundant nodes.

Figure 8 , 不同分法会有不同的重复node. 

we transform the redundancy elimination problem to a min-cost flow

所以我们可以用它的分法来获得micro batch. 不过他们是DGL的.

constructing an **auxiliary graph** called Redundancy- Embedded Graph (or REG) the weight on an edge is the number of neighbors shared by the two nodes connected by the edge.   

他在用两个邻接矩阵乘了之后，就用这个新矩阵当一个新的图了

With REG, we can apply any existing graph partitioning algorithm that minimize the cut flow

转化一下,  还是用现有的算法Metis. 经典数学思维.  

如果换一些dataset和场景 可以work吗? 得做实验去看，他这个想法还是挺好的。C = A x A_T这个还是挺巧妙的，C的每个元素的值就代表了分开两个节点所要冗余的点

对于一些dense graph他这个算法可能是可以 work的，但是代码就不行了

The REG partition algorithm doesn't consider the edge weights of REG. As most of the nodes share no neighbor or just one neighbor with others in real large graph data, (most of the input graphs are sparse). Based on the results of the experiments tested on the datasets mentioned in the paper, the weight of REG edges has no possitive effect on the partition result.

#### 4.4 Reducing Maximal Memory Footprint

问题:partitioning imbalance. 取决于最大的一块micro batch.

##### 4.4.2 Source of Load Imbalance

为什么用in-degree bucketing?  not all nodes have the same in-degrees, the parallel of all nodes’ reduce functions is not trivial. 为了让batch 每个的node都同时reduce好, 不会有短板效应-- 别的node 准备好了等这个node。

如果Node的Degree特别分散怎么办呢?  由于长尾效应, 最后一个bucket的节点特别多. 

##### 4.4.3 Memory-aware Partitioning. 

in-degree不均衡的时候, betty workload不平衡. 怎么办? 

address the load imbalance problem is to further increase the number of partitions. 但是容易OOM.

估算内存消耗. 不断估算, 超出了就试试多分partition. 感觉还是很naive的想法, 有改进的空间.  idea:  in-degree 高的放在GPU上. 动态图 in-degree会变化, 是否对动态图可以优化.  degree padding. https://discuss.dgl.ai/t/internal-message-passing-implementation/494/2

## 5 IMPLEMENTATION

dgl.betty(). dgl.betty() employs dgl.adj_product_graph() to create REG and split it with dgl.metis_partition(). dgl.betty() uses the existing DGLgraph.in_edges() to get the indexes of source nodes and edges for a given destination node, which is needed for graph partition. 

Those indexes and the raw graph, after the partition, are fed to dgl.edges_subgraph() to generate subgraphs in the DGLgraph format. 

Those subgraphs are then used to generate blocks (representations of bipartite structures in DGL) by using `dgl.to_block()`. The blocks are needed for future graph operations during the GNN training.

## 6 EVALUATION

Platform. We use a system with two Xeon E5-2698 v4 CPUs (40 cores running at 2.20 GHz) and an NVIDIA Quadro RTX 6000 GPUs. We use CUDA 10.1/cuDNN 7.0 to run GNNs on NVIDIA GPUs. We use Python 3.6, Pytorch [28] v1.7.1 and DGL [40] v0.7.1 for model training.The RTX6000 we use only has 24 GB GPU memory. Though one A100 can provides 80 GB GPU memory, it still cannot meet the memory requirements for very large scale GNN training.

速度更快, 内存消耗更小

测一下GPU利用率.

为什么the data transfer time increases because of lower GPU bus utilization?  运行一下看看. 

稀疏图, redundant 少, 所以不需要betty. 

dgl的metis不支持按照edge weight来分吧.   有可能没做，或者做了发现不好，都有可能.

我估计他要是实现了带边权的metis，效果在dense graph上不一定差,可能是没时间了，或者实验后期才意识到dgl metis不能partition带边权的.  主要是我估计他也没写带边权的metis. 

## 7 conclusion

We introduce a system support (named Betty) to enable GNN training with large and complex graphs, based on the idea of micro-batches to partition the large graph and fit into GPU memory.

## 代码

https://github.com/PASAUCMerced/Betty#readme

来得早还有服务器用.

`from memory_usage import see_memory_usage, nvidia_smi_usage` 可以用来看内存情况. 

怎么改进?

