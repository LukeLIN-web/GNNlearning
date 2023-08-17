MLsys2022

所有论文链接：

- **[Accelerating Training and Inference of Graph Neural Networks with Fast Sampling and Pipelining](https://proceedings.mlsys.org/paper/2022/hash/35f4a8d465e6e1edc05f3d8ab658c551-Abstract.html).**
- **[Sequential Aggregation and Rematerialization: Distributed Full-batch Training of Graph Neural Networks on Large Graphs.](https://proceedings.mlsys.org/paper/2022/hash/5fd0b37cd7dbbb00f97ba6ce92bf5add-Abstract.html)**
- **[Understanding GNN Computational Graph: A Coordinated Computation, IO, and Memory Perspective.](https://proceedings.mlsys.org/paper/2022/hash/9a1158154dfa42caddbd0694a4e9bdc8-Abstract.html)**
- **[Graphiler: Optimizing Graph Neural Networks with Message Passing Data Flow Graph.](https://proceedings.mlsys.org/paper/2022/hash/a87ff679a2f3e71d9181a67b7542122c-Abstract.html)**
- **[BNS-GCN: Efficient Full-Graph Training of Graph Convolutional Networks with Partition-Parallelism and Random Boundary Node Sampling Sampling.](https://proceedings.mlsys.org/paper/2022/hash/d1fe173d08e959397adf34b1d77e88d7-Abstract.html)**

[SALIENT](https://www.notion.so/SALIENT-bfb0eb88973249e49905d49530691a2c)

[SAR](https://www.notion.so/SAR-1d9a9617222e493b80d1cef1f01a697a)

[Understanding GNN](https://www.notion.so/Understanding-GNN-2e6b707ce12041c09853ab48b6eef2e1)

[BNS-GCN](https://www.notion.so/BNS-GCN-1f2aec2bd0f142f5a48fa53cf85d49d3)

[GRAPHILER](https://www.notion.so/GRAPHILER-1a9dc8c248814446926710baa4d5bf73)

[HYDROZOA](https://www.notion.so/HYDROZOA-67d8be7dfac241198c2c2cf78cbf8fef)





## SAR

SEQUENTIAL AGGREGATION AND REMATERIALIZATION: DISTRIBUTED FULL-BATCH TRAINING OF GRAPH NEURAL NETWORKS ON LARGE GRAPH Intel AI lab

### 摘要

1. Sequential Aggregation and Rematerialization(SAR) 在backward pass中顺序地重新构建并释放大型GNN计算图的碎片。

2. kernel fusion and attention-matrix rematerialization

### 简介

本质上是以更细的粒度来完成远程机器的通信/计算过程

问题

1.什么采样操作是最好的？

2.每台机器都有大的计算图

方法

1. 构建计算图， 在backward pass中顺序地重新构建并释放。

2. 对于GAT，避免物化（这是什么意思？不用存储）2.对于GAT，避免将昂贵的注意系数张量具体化，而是在前向和后向过程中使用融合的算子来计算它们。

SAR在DGL基础上构建

SAR运行under the hood（透明地），管理机器间的通信以及计算图的动态构建和删除。

### 相关工作

1. NeuGraph仅限于在一台主机上进行多GPU训练
2. ROC消除了单节点的限制

CAGNET分布式训练方法，最大限度地减少了节点间的通信.

对于大规模（多TB）的可扩展性问题？ 仍然没有解决。

DistGNN ，只有正向传递 "全批"。

SAR运行一个全批的前向和后向pass

方法

1. 将图划分为N个分区。分布到N台机器上。
1. 



## SALIENT

ACCELERATING TRAINING AND INFERENCE OF GRAPH NEURAL NETWORKS WITH FAST SAMPLING AND PIPELINING

### 摘要

1. 性能设计的邻域采样器，共享内存并行化策略，以及与GPU计算的批量传输管道化
2. 推理的采样，统一训练和推理。

### 简介

问题

1. 大邻居 ⇒ 计算成本，内存
2. 批量准备和传输时间 >> 损失、梯度和参数更新 

方法

1.快速邻域采样器 
2.批量准备的共享内存并行化 
3.流水线式的批量传输和计算

   采样，加强，数据移动

### 背景

### 2.1 aggregate

### 2.2 sample

### 2.3 GNN训练系统

SALIENT是基于PyTorch, PyG, mini batch. 多台机器，每台机器有多个GPU。

### 3性能特点

只有28%的时间用于GPU训练。大部分时间用于准备批处理和传输数据到GPU上。

采样使用PyTorch DataLoader和多处理方式

分片使用单个进程中的多个OpenMP线程

人们可以通过流水线和消除冗余的往返通信来实现接近最佳的数据传输率。这些优化将在第4.3节讨论。

### 4SALIENT

### 4.1快速采样

PyG中的基础算法

我们设计了一个采样MFG(message flow 图)生成的参数化实现

为了减轻采样的变异性，对每个单独的跳跃进行基准测试。

改变c++ , 使用数组而不是哈希表的集合。 

### 4.2共享内存并行批量准备

ses C++线程，它们端对端地准备batch，每个线程按顺序执行采样和切片。

串行张量切片代码→提高缓存定位，避免线程之间的争夺

直接在pin memory中进行切片.

线程通过一个无锁的输入队列动态地平衡负载，该队列包含每个小批处理的目标节点

### 4.3 数据传输管道化

增加了一个跳过断言的选项

使用单独的GPU流进行计算和数据传输，同步这些流以确保在必要的数据传输后开始训练迭代。

数据传输是怎么管道化的? 

### 8 结论和未来发展

表7包括distDGL,P3。

批量准备数据的限制因素是CPU核的数量或DRAM带宽。

## HYDROZOA

HYDROZOA: DYNAMIC HYBRID-PARALLEL DNN TRAINING ON SERVERLESS CONTAINERS

We present Hydrozoa, a system that trains DNNs on serverless containers with a hybrid-parallel architecture that flexibly combines data and model-parallelism.

### 摘要

深度神经网络（DNNs）通常在虚拟机集群上进行并行训练，以便减少训练时间。然而，这需要明确的集群管理，这很麻烦，而且经常导致昂贵的资源过度配置。在无服务器计算上训练DNN是一个有吸引力的替代方案，受到越来越多的关注。在无服务器环境中，用户不需要处理集群管理 并可以在细粒度的水平上扩展计算资源，同时只在积极使用时支付资源。 尽管有这些潜在的好处，但现有的用于DNN训练的无服务器系统是无效的，因为它们限于基于CPU的训练，并且受到昂贵的分布式通信的瓶颈影响。我们提出了Hydrozoa，该系统在无服务器容器上训练DNN，采用混合并行架构，灵活地结合了数据和模型并行性。Hydrozoa支持基于GPU的训练，并利用混合并行性和无服务器的 资源扩展，与现有的无服务器和基于虚拟机的训练系统相比，每美元的吞吐量要高出155.5倍和5.4倍。Hydrozoa还允许用户在训练过程中实施动态工人扩展策略。 我们表明，动态工人扩展提高了统计训练的效率并降低了训练成本。

### 介绍

分布式训练传统上是设计为在机器集群上运行。通常情况下，虚拟基础设施即服务（IaaS）中的虚拟机（VM）被用来比物理机方便。然而。这种基于虚拟机的方法仍然需要配置和管理一个集群，以及部署训练作业。 集群管理是一项耗时的工作，尤其是在需要为不同的培训工作调整资源的情况下。 通常情况下。 集群的分配不尽如人意，利用率低下（Delimitrou & Kozyrakis, 2014），导致在未使用的资源上造成浪费。
在无服务器计算范式中，客户在云管理的服务器上运行程序逻辑。与基于虚拟机的方法相比，用户不需要处理部署、管理和扩展自己的服务器的复杂性。 代码被自动部署，集群资源在执行结束后被取消配置。开发人员可以弹性地扩展无服务器应用程序，同时 弹性地扩展无服务器应用程序，同时在细粒度层面上分配资源，以满足资源的要求。此外，无服务器服务只对运行程序逻辑的时间收费。这与预留集群相比具有成本效益。与保留一个可能长期闲置或利用率低下的集群相比，具有成本效益。

#### i 无服务器容器

Hydrozoa中的训练任务被打包成容器，而不是函数。容器在Azure容器上无服务器地执行实例（ACI，2021。与无服务器函数一样，无服务器容器也可以在细粒度层面上进行扩展，但重要的是，它们可以以更低的成本配置。它们可以被配置为具有明显的更多的计算资源，包括GPU。无服务器容器之间可以直接通信。从而降低了通信成本。

#### (ii) 混合并行训练

Hydrozoa支持数据并行、模型并行和混合并行训练。灵活地结合数据和模型的并行性(Narayanan等人，2019；Park等人，2020）。Hydrozoa采用了一个新的规划器来优化数据和模型并行的程度，以获得两种方法的好处。该规划器包括一个分区算法，自动将DNN模型划分到在工作者之间自动划分DNN模型，以实现高效的模型并行训练。

#### (iii) 动态工作器扩展

Hydrozoa利用了无服务器计算的弹性，在训练过程中动态地调整训练过程中的工作者数量。用户可以在Hydrozoa中指定策略来控制工作者的扩展行为，以实现高度的并行性，而不牺牲模型收敛特性。

 现有的混合-并行系统(Narayanan等人，2019；Park等人，2020）部署在虚拟机集群很容易出现资源过度供应，因为任务有专门的功能，有不同的资源而虚拟机有粗粒度的要求。Hydrozoa通过为每个任务提供适当规模的可扩展无服务器资源来避免这一缺陷。动态工作者缩放（Devarakonda等人，2017；McCandlish等人，2018)已被探索为提高统计训练效率的一种手段。然而，现有的方法是低效的, 因为它们被部署在固定规模的集群上，而这些集群是超额配置以适应训练期间所需的最大数量的训练期间需要的工人数量。Hydrozoa的好处是额外的成本节约，因为它可以根据active 工人的数量动态扩展无服务器资源的动态扩展，从而节省了额外的成本。

然而，上述组件的现有实现不能简单地结合起来。比如说。 现有的MXNet和Pytorch的分布式训练逻辑假定集群的大小是固定的，训练期间动态加入和离开的无服务器计算的工人。同样地，现有的混合并行实现方式是建立在这些训练库之上的，与无服务器计算不兼容。Hydrozoa在MXNet之上实现了定制的分布式训练逻辑，无缝地集成了所有三个组件。

## GRAPHILER

GRAPHILER: OPTIMIZING GRAPH NEURAL NETWORKS WITH MESSAGE PASSING DATA FLOW GRAPH

### Abstract

1. Graphiler, a compiler stack for GNNs which achieves high performance while offering the flexibility of the UDF programming interface
2. Message Passing Data Flow Graph (MP-DFG

### Introduction

Problem

1. UDF are sub-optimal, many small tensor op → excessive func call.
2. no built-in primitives suppport hetero-GNN op.

Method

1. message passing data flow graph (MP-DFG)
2. 

Contribution

1. automatically compiles GNNs defined using UDFs into efficient execution plans
2. generalizes naturally to hetero-GNNs, allowing them to share data structures and kernels originally designed for homogeneous GNNs

Graphiler 和 MLSys 2022 见闻 - 乔枫惜的文章 - 知乎 https://zhuanlan.zhihu.com/p/570329623



## BNS-GCN

BNS-GCN: EFFICIENT FULL-GRAPH TRAINING OF GRAPH CONVOLUTIONAL NETWORKS WITH PARTITION-PARALLELISM AND RANDOM BOUNDARY NODE SAMPLING

### Abstract

dubbed BNS-GCN adopts random Boundary-Node-Sampling to enable efficient and scalable distributed GCN training. 跨机通讯主要来源于每个机器所负责节点中的边界节点（Boundary Nodes），它们会通过跨边连接到其他机器上的节点。为了规避这样的通信代价，作者提出BNS-GNN，直接按照一定的比例进行采样，放弃这些边界节点的通信，实验效果显示它不仅可以带来加速，似乎甚至还可以带来模型精度上的提升。可能泛化性更好了. 

### Introduction

和graphsage采样形成子图不同, 这是分割大图得到子图. 同时训练所有子图. 

Problem：

1. overwhelming communication volume
2. prohibitive memory requirement (啥意思？) 就是GPU显存不够.
3. imbalanced memory consumption

Method

dubbed BNS-GCN, which randomly samples features of boundary nodes at each training iteration

### Background

ROC, NeuGraph, AliGraph:  node split,        expensive CPU-GPU swaps!

CAGNET &P3: split node features and layers to enable intra-layer model parallelism

Dorylus:  pipelining computation op over CPU threads

### 3 BNS-GCN FRAMEWORK

we propose partition-parallel training of GCNs with Boundary Node Sampling, dubbed BNS-GCN

设计了一个sampling strategy 同时解决三个问题.

### Contribution

1. Identify the Underlying Cause
2. proposes BNS
3.  Validate BNS-GCN in Theory

Step 1: Sampling each boundary node with probability p

Step 2: Removing unsampled nodes

### Experiment

Datasets  : Reddit, ogbn-products, Yelp and ogbn-papers100M

Benchmarked Baselines  : ROC [MLSys’20] (swap-based) and CAGNET [SC’20] (slice-based)

Adopted Toolkits : DGL 0.7.0 and PyTorch 1.9.1

Baselines: throughput <0.7 epochs/s

Partition-based training: throughput >1.2 epochs/s

BNS-GCN: 8.9x~16.2x throughput improvement

BNS-GCN saves the memory by up to 58%

他每个GPU是什么样算一层, 要遍历所有node吗?还是获得了邻居就可以继续了? 感觉要等旁边.

看他和别的性能差距? 这是最好的baseline.看更新的文章就知道这个文章的性能. 

没有nvlink的情况下. 先估计一下.

看一下它的, send是用啥, 是同步吗? send可以receive同时吗? 

做一个同样时间的. topology aware. 

看两两之间边的数量.  节点id ,属于哪个ptr

hardware aware, 就是考虑带宽来分配流量. 考虑有的传输速度快. 跨机器, 跨越GPU. 

repo: https://github.com/GATECH-EIC/BNS-GCN  看看是不是平衡的. 

## Understanding GNN

UNDERSTANDING GNN COMPUTATIONAL GRAPH: A COORDINATED COMPUTATION, IO, AND MEMORY PERSPECTIVE

### 摘要

重复的运算计算 → 在传播前重新组织运算。

不一致的线程映射， 以顶点为中心与以边缘为中心的线程mapping schemes是不同的 → Unified thread mapping for fusion

过多的中间数据占用内存 → 中间数据重新计算

### 设计概述

本文提出了一种系统的方法来优化GNN at the inter-op level

1. 执行Apply是很昂贵的  → 首先在顶点特征上应用昂贵的Apply-
2. fuse一连串的运算符，只要它们是与图相关的内核或轻量级的Apply-
3. 重算， checkpoint

### 减少计算 

挑战：    Scatter  有重复的也有独特的计算。

找到冗余， 识别冗余， 我们提出传播-推迟运算符重组来消除这种冗余。

### 减少IO

   We aim to apply kernel fusion to eliminate further the aforementioned two-edge feature store/load and completely fuse all graph-related operators (Scatter, ApplyEdge, ReduceScatter, Aggregate). 

挑战：    Fuse operators with different thread-mapping schemes  是很难的想法：    Our key insight is that thread-mapping schemes can be decoupled from the operator type: edge-centric operator can also apply vertex-balanced mapping and vice versa。 We propose to fuse all graph-related operators with unified thread mapping eagerly。

### 减少内存

计算量小的都重新计算。 

### 实验

Baseline：  DGL和 fuseGNN 

Model:  GAT, EdgeConv, Mixture Model Network.

数据集：    Cora, Citeseer, Pubmed, and Reddit 

2090上,他们的内存也用的少. 比DGL还少. 

###  RELATED WORK 

之前是手动修改op 组合. 但是他们提出了理论分析. 

DNN的fusing , 用在 GNN 上. 解决了Inconsistent thread mapping 问题. 其实就是解决了工程问题. 

重算, 用在GNN上. A+B.

runtime优化: neighbor grouping, 均衡负载, 利用内存locality. 







reference :

MLSys 2022 参会评述随笔(Day 1) - Hsword的文章 - 知乎 https://zhuanlan.zhihu.com/p/559385274
