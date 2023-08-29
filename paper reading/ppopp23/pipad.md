PiPAD: Pipelined and Parallel Dynamic GNN Training on GPUs

buaa的三个人. 感觉非常硬核. 

## Introduction

DTDG: Discrete Time Dynamic Graphs

**sliding window** mechanism that feeds multiple continuous snapshots to the model simul- taneously,

In this paper, we focus on **DTDG-based DGNNs** and refer to the sliding window as frame for simplicity.

挑战

1. 因为要不断更新graph, 数据传输时间很长,使恶化GPU利用率
2. 有数据reuse 的可能. 

## 2 Background



### 2.2

PyGT [38] and TGL [51] are two general frame- works aiming to implement the ubiquitous support for as many DGNN models as possible .看了一下pygt才3个star没人用. 

DynaGraph  :  assumes the graph topology not evolving over time 

CacheG demands the node features remaining unchanged.

ESDG neglect the intrinsic parallelism potentials in DGNN

## 3Motivation

### 3.1Performance Bottlenecks of DGNN Training

38.7%的时间在传输数据.  -> 用cuda stream来异步传输. 

topology 变化很小.

### 3.2 Memory Access Inefficiency in GNN

一次request取的data, 并不都是有用的. 

### 3.3 Opportunities for Data Reuse and Parallelism



we first take topology overlap into consideration cooperatively and design a new graph format that can extract the overlaps efficiently to elim- inate the redundant transfer.

devise a “parallel” GNN computation pattern that can process multiple graphs simultaneously. The basic idea is to directly per- form the aggregation on the overlap topology of one snapshot group with all their feature matrices.



## 4 PiPAD

1. reducing data transfer volume via the overlap- aware data organization and inter-frame reuse
2. accelerating GNN computation with intra-frame parallelism
3. maximizing the overlap among different operations through the pipeline

### Overlap-aware Data Organization

extracting the overlap topology among adjacent snapshots and constructing a new individual adjacent matrix for it. we can avoid the needless transmission and further realize the parallel computation.

By constantly conducting overlap analysis and extraction throughout the opening training epochs, PiPAD can enable coalesced memory accesses to multiple features and alleviate bandwidth unsaturation issues.

CSR directly restricts the flexibility of overlap extraction leading to poor efficiency. 

PiPAD uses a slice-based graph representation format and overlap-aware multi-snapshot transfer method to reduce data transfer volume and alleviate bandwidth unsaturation issues. 

### 4.2Intra-frame Parallelism

performing aggregation and update operations over multiple snapshots simultaneously within a single frame. 

PiPAD implements intra-frame parallelism based on its dimension-aware parallel GNN, which can perform these operations efficiently. 

Additionally, PiPAD devises three key optimizations to further improve intra-frame parallelism: 

1. vector-memory-instruction based memory access for the large-dimension situation, can load larger amounts of data with a single memory request and multiple transactions.

2. thread-aware slice coalescing to resolve the low thread utilization issue for the small-dimension case
3. Locality-optimized weight reuse to enable efficient parallel update. 

### 4.3Pipeline Execution Framework

uses a pipeline execution framework to manage the following stages: data preprocessing, graph slicing, intra-frame parallelism, inter-frame reuse, and overlap-aware data transfer. 

integrates a runtime tuner to dynamically adjust the parallelism- and reuse-level of DGNN training for better adaptation to various datasets.

### 4.4

 reusing data across different frames and dynamically tuning two key parameters, namely the number of partitions and the size of overlap

