Charith Mendis  教授, 主要是做 ai  编译的.

问题

1.  inference 放在training 还work吗？ 论文只有inference, 没有training, 为什么?     model parameters and weights 会 change. 所以embedding就不一样. 
2. semantic-preserving什么意思? 

## 摘要



## 1 introduction

We observe and leverage redundancies in temporal embedding and time-encoding computations to considerably reduce TGAT inference run- time.

优点:  the redundancy elimination techniques we consider are not restricted to simple aggrega- tions and do not require replacing self-attention.

- Processing source and destination nodes in batches can result in redundant node embedding calculations
- Calculating embeddings for a node and timestamp may result in recalculations of the same embeddings due to exploring the same temporal neighborhoods
- For some dynamic graphs, up to 89.9% of the total embeddings generated during their lifetime can be repeated calculations
- The time-encoding operation in TGAT is frequently invoked with the same time delta values

## 2 Background

It learns a function Φ   that maps a time value to a 𝑑𝑡 -dimensional vector. This time-encoding technique allows it to capture temporal patterns of the graph. The time-encoding vector 输入 the input features of a GNN operator, thereby incorporated into the output embeddings.

temporal neighborhood :  tj >t

假设 node feature不变

## 3 Redundancies & Reuse Opportunities

### 3.1 Duplication From Batched Edges

nodes often share common neighbors and this can lead to duplicate ⟨𝑖, 𝑡 ⟩ pairs.   比如同一时刻 a->c , b->c. 那么就有两个<c,t>

### 3.2Temporally Redundant Embedding Calculations

3 most-recent neighbors  可能只变化了一个, **model parameters and weights do not change** during inference time, 所以会有相同输出.  

### 3.3 Repeated Time Encodings

 Eqs. (4, 5),   it always uses 0 for the time-encoding of 𝑧𝑖 (𝑡). Performing this computation every time is unnecessary—since **the weights for the time-encoder** are fixed at inference time

大部分 time deltas都接近0 . 

优化:  we can compute this once ahead-of-time and reuse it indefinitely.

## 4 Redundancy-Aware Optimizations

### 4.1Deduplicating Nodes

 jointly operates on the two separate arrays in order to avoid creating intermediate tensors.  hash算法, 产生unique element , 

### 4.2 Memoization of Embeddings

也是pull 和push embedding. 

the list of neighbors, their edge timestamps, edge features, and 𝐻 (𝑙 −1) as inputs.

combined into a single key value by hash

Storage Memory Location: 在CPU

### 4.3 Precomputing Time Encodings

precomputes time-encoding vectors in advance before running inference.

直接apply the Φ(·) function ,这个函数是 time-encoding function , learns a function Φ  that maps a time value to a 𝑑𝑡 -dimensional vector. This time-encoding technique allows it to capture temporal patterns of the graph.

优化了lookup 过程.

什么是 time-encoding vectors? 

## 5 实验

speedups of 4.9× on CPU and 2.9× on GPU

baseline我们可以用TGL.

## 6Related Work

之前有人precomputing a time-encoding lookup table, which is hardcoded to 128 intervals

缺点: the self-attention in TGNNs was replaced with a simplified version, thereby altering the semantics

没有优化redundancy optimizations.

#### dynaGraph 

存中间embedding, 只支持DTDG. 我们可以做CTDG

#### HAG abstraction. 

不支持复杂的自注意力机制. 

ReGNN is more tailored to hardware accelerators

## 7 future work

1. 训练加速

2. 目前是most-recent neighbor sampling strategy. 可以尝试不同的sampling 方法. 

3. 当node feature changes and deletion of edges, 怎么办? 

很多tuition 来自ref42 , FPGA.

tensorcore能用在这里吗,  可以同时计算16*16 *16 的三维的.

A100支持block is sparse. 

### 代码

https://github.com/ADAPT-uiuc/tgopt

#### 环境

` pip install torch==1.12.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu`

为什么装cpu版本. 

300行代码一个cpp文件就搞定了. 





#### dedup_src_ts



