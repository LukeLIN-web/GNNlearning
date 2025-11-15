ReFresh: Reducing Memory Access from Exploiting Stable Historical Embeddings for Graph Neural Network Training

AWS上海的项目, 很多作者,   代码不开源.  不过挺良心的, 细节讲的很清楚. 手把手教你细节. 

#### 摘要

leverages a historical cache for storing and reusing GNN node embeddings instead of re-computing them through fetching raw features at every iteration

设计了corresponding cache policy,using a combination of gradient-based and staleness criteria

This paper discuss gradient magnitude criteria 

## 1Introduction

- Proposes a historical embedding cache, with a corresponding cache policy that adaptively maintains node representations (via gradient and staleness criteria)
- Implements requisite cache operations, subgraph pruning, and data loading for both single-GPU and multi-GPU hardware settings.

## 2Background

#### 2.3Existing Mini-Batch Training Overhaul

GNNlab drawback: most of the nodes have moderate visiting frequency, feature cache is unlikely to reduce memory access to them. ( in degree)

PyTorch-Direct 用UVA, 但是PCIe带宽小

那是怎么解决的呢? 

GNNAutoscale drawback: un-refreshed embeddings may gradually drift away from their authentic values -> control estimation error

## 3 Overview of ReFresh System

观察到*most of the node embeddings experience only modest change across the majority of iterations*. 大多数节点嵌入在大多数迭代过程中只经历了微小的变化。

The cache contains node embeddings recorded from previous iterations as well as some auxiliary data related to staleness and gradient magnitudes as needed to estimate embedding stability.

Cache store not only staleness but also gradient magnitudes 

we propose two cache policies (based on embedding staleness and gradient magnitudes collected during training) to detect and admit only those node embeddings

**Embedding Stability Illustration.**    他们测试了Distribution of cosine similarity between the node embeddings 

算法: 可以用一个sampled subgraph 代替原来的计算图, 然后再reuse embedding.

#### 子图

each subgraph structure itself is adaptively gen- erated based on which particular nodes happen to be in the selective historical embedding cache,

把graph sampling and graph pruning分开,  samples graphs in CPU while pruning graphs in GPU;

#### data loader

三个情景: fetching features from CPU to a single GPU, (2) fetching features from CPU to multiple GPUs, and (3) fetching features from other GPUs

| 操作或内容     | 位置 |      |
| -------------- | ---- | ---- |
| sample         | CPU  |      |
| pruning graphs | GPU  |      |
|                |      |      |
|                |      |      |
|                |      |      |

refresh的内存位置.

## 4 Historical Embedding Cache

### 4.1 cache policy

cache policy based on dynamic feedback from model training.

一个是 GAS的梯度限制 , 另一个是VR-GCN 的bound their *staleness*  

**Resulting Adaptive Cache Size**  

GAS + VR-GCN versatile paradigm 然后adaptive, 经典水文章的思路.

### 4.2 GPU Implementation

utilizes GPU threads to look up a batch of historical node embeddings in parallel.

优点: Given a batch of node IDs, each thread is in charge of fetching one node embedding and each fetching operation can be done in *O*(1). 

这里大概需要结合代码看. 讲的挺详细的.  写代码的时候可以参考. 

### 4.3 Remarks on ReFresh Convergence

吹了一通他们的收敛性很好.

## 5 Cache-Aware Subgraph Generator

子图生成分为两步:  graph sampling and graph pruning,  

pruning depends on the historical embedding cache.

**Asynchronous CPU Graph Sampling.**    multithreading instead of multiprocessing to produce subgraphs    

**GPU Graph Pruning.**   

CSR2 uses two arrays to represent row indices – the first array records the starting offset of a node’s neighbors to the column index array while the second array records its ending offset.

quickly remove neighbors.

为什么要prune呢? 不懂prune了啥.

## 6 Data Loader

负责加载feature

**One-sided Communication.** 

用UVA, 

**Multi-round Communication.**

This multi-round communication can effectively avoid congestion in all-to-all communication.

## 7 实验

ReFresh can reach this same accuracy in 25 minutes while the slowest baseline (PyG) takes more than 6 hours

 他们是多GPU吗? 共享了什么?是的, 共享了node feature. 但是embedding 有共享吗?

他们是1个hop吗? 还是多个hop?

他们测试了哪些模型? Sage，GCN，GAT

非常新, 他们用了pyg2.2.0, python3.9, A100

他们的cache空间有多大? 怎么增大cache空间? 
