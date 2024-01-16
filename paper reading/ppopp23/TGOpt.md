# TGOpt: Redundancy-Aware Optimizations for Temporal Graph Attention Networks

Charith Mendis  教授, 主要是做 ai  编译的. 作者uiuc mscs 直接去特斯拉工作了, 还搞了TGLite, 但是好像没中. 

问题

1.  inference 放在training 还work吗？ 论文只有inference, 没有training, 为什么?     因为训练的时候model parameters and weights 会 change. 所以embedding就不一样.  inference可以存储embedding
2. semantic-preserving什么意思? 

## 摘要

加速inference, proposes to accelerate TGNN inference by de-duplication, memorization, and pre-computation.

## 1 introduction

We observe and leverage redundancies in temporal embedding and time-encoding computations to considerably reduce TGAT inference run- time.

优点:  the redundancy elimination techniques we consider are not restricted to simple aggrega- tions and do not require replacing self-attention.

- Processing source and destination nodes in batches can result in redundant node embedding calculations
- Calculating embeddings for a node and timestamp may result in recalculations of the same embeddings due to exploring the same temporal neighborhoods
- For some dynamic graphs, up to 89.9% of the total embeddings generated during their lifetime can be repeated calculations
- The time-encoding operation in TGAT is frequently invoked with the same time delta values

## 2 Background

TGAT dataset ,  They store the edge linkages, edge features and node features respectively. https://github.com/StatsDLMathsRecomSys/Inductive-representation-learning-on-temporal-graphs

可以处理节点分类和 link prediction. 归纳推断新节点和观察到的节点的嵌入

Layer 和静态的什么区别? 

mj(t) = msg(出节点的老h ,  入节点的老h, 边的特征)  感觉也差不多? 就是要多迭代几次time.  

ri 就是一样, 把msg 用summation 等函数 来aggr, hi也是一样, 叠加NN.   h就是temporal embeddings.

是对于每个eij 都要做一次gnn操作吗? 

RandEdgeSampler, 有什么用?  就是随机找几个边作为background对比. 

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

为什么要ComputeKeys?  是根据key来找重复的.  The inputs will generally be combined into a single key value by hashing. the ComputeKeys operation can be performed in parallel across the pairs, 是可以并行的

也是pull 和push embedding. 

the list of neighbors, their edge timestamps, edge features, and 𝐻 (𝑙 −1) as inputs.

combined into a single key value by hash

Storage Memory Location: 在CPU. 所以要 move 𝐻 𝑚 to CPU device;?

We also note that each of the keys can be operated on independently, so the main loop in both CacheStore and CacheLookup can be parallelized, given that TGOpt uses a concurrent hash table implementation. We selectively paral- lelize these operations depending on the hardware.  

`    #ifndef tgopt_1t_cache_lookup`  

### 4.3 Precomputing Time Encodings

precomputes time-encoding vectors in advance before running inference.

直接apply the Φ(·) function ,这个函数是 time-encoding function , learns a function Φ  that maps a time value to a 𝑑𝑡 -dimensional vector. This time-encoding technique allows it to capture temporal patterns of the graph.

优化了lookup 过程.

什么是 time-encoding vectors? 就是把时间也编码作为一个变量. 

## 5 实验

speedups of 4.9× on CPU and 2.9× on GPU

baseline 用TGL.

table5 证明 GPU搬运 embedding很花时间, 所以 存在CPU.   figure5我成功复现了.



## 6Related Work

之前有人precomputing a time-encoding lookup table, which is hardcoded to 128 intervals

缺点: the self-attention in TGNNs was replaced with a simplified version, thereby altering the semantics

没有优化redundancy optimizations.

#### dynaGraph 

存中间embedding, 只支持DTDG.   所以做CTDG.

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

## 代码

https://github.com/ADAPT-uiuc/tgopt

#### 环境

Docker file 写的是 ` pip install torch==1.12.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu`

为什么装cpu版本?  论文说在cpu上有卓越的性能. 因为embedding位于CPU. 

300行代码一个cpp文件就搞定了. 代码量小. 

还是得装gpu,  因为GPU更快, 论文用的是cuda11.6,  Nvidia GPU (we tested on Tesla V100 16GB via an AWS instance). 

用torch11.6gpu的image

```
错误: 
tgopt_ext.cpp:1:10: fatal error: tbb/concurrent_unordered_map.h: No such file or directory
 #include <tbb/concurrent_unordered_map.h>
          ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
需要 sudo apt-get install libtbb
```

#### 数据

Model tgat, 论文其实训练了tgat的模型



train时间

| dataset    | size                                                         | cpu(s) | 1gpu(s) |
| ---------- | ------------------------------------------------------------ | ------ | ------- |
| jodie-wiki | 533M  INFO:root:num of instances: 157474.  INFO:root:num of batches: 788 | 89     | 11.3    |
| jodie-mooc | 39.5M INFO:root:num of instances: 411749.  INFO:root:num of batches: 2059 | 33     | 22      |
| snap-email | 1.6M INFO:root:num of instances: 332334.  INFO:root:num of batches: 1662 | 85     | 22      |
| snap-msg   | 337K INFO:root:num of instances: 59835. INFO:root:num of batches: 300 | 15     | 4,3.5   |



inference  old node. 

| dataset    | size                             | 1gpu(s) | 1gpu(s)  optimize |
| ---------- | -------------------------------- | ------- | ----------------- |
| jodie-wiki | 533M  , num_test_instance: 23621 | 22.6,   | 19.0              |
| jodie-mooc | 39.5M num_test_instance: 61763   | 37.3    | 34.8              |
| snap-email | 1.6M                             | 41      | 33.8              |
| snap-msg   | 337K                             | 8.4     | 8.49 (old)        |



提升没有好几倍. 

snap-msg,  old node没有加速, new node加速了40%, 为什么? 

```
./data-download.sh  snap-email jodie-mooc
python data-reformat.py -d  snap-email  snap-msg  就是把snap的文件转换为jodie 格式. 
python data-process.py -d jodie-wiki 也是数据对齐. 
python inference.py -d snap-email  --model tgat --prefix test --opt-all 
python inference.py -d snap-msg --model tgat --gpu 0
python train.py -d snap-msg --model tgat --prefix test --opt-all --gpu 0
python  e2einference.py -d snap-msg  --model tgat  --gpu 0
py-spy record -o profile.svg -- python benchmark/benchmarktgat.py .py -d jodie-wiki 

python benchmark/benchmarktgat.py -d snap-msg  --model tgat  --gpu 0

nsys profile -w true -t cuda,nvtx,cudnn,cublas --cuda-memory-usage=true --force-overwrite true -x true python benchmark/benchmarktgat.py -d snap-msg
```

论文里说30秒就infer完成了. 但是我测130s 88s ,  用了 7个CPU, vscode serever/htop要占据一个cpu.

dedup_src_ts 是什么用? 

val for new nodes 和val for old node 是啥意思?  train时见过的就是old, new就是没有train的. 

a unified framework for tgnn framework

pos_score 和  neg_score 相加为什么不为1？   因为score不是 probabilities. pos和neg是独立的. 

这个forward和contrast有什么区别? contrast有对于Background的对比. 太奇怪了, 这个forward好像没有用到, tgopt. 

缺点: 他不是end to end 的.

