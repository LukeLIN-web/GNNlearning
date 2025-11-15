#### 解决的问题

不同的tensor 大小不同,  内存占用大, 所以 gpu内存分配碎片化, 占用大量gpu 内存.  

Continuous  batching,要点就是, 不同的sequence完成的时间是差距很大, 所以有机会. 

#### 安装

需要升级pip到23.0.

SamplingParams 是什么, 有什么用? 就是管理参数

 https://github.com/vllm-project/vllm  关注block 和cache. engine

vLLM框架top down概览  - 知乎
https://zhuanlan.zhihu.com/p/645251151

怎么测的latency 和吞吐量? 

prompt是哪里输入的?   Hugging face好像会自动调取. 

## PagedAttention

all the input tokens to the LLM produce their attention key and value tensors. and these tensors are kept in GPU memory to generate next tokens. 

one can think of blocks as pages, tokens as bytes, and sequences as processes.

我们可以 node看做bytes, graph看做 processes

To ensure safe sharing, PagedAttention keeps track of the reference counts of the physical blocks and implements the *Copy-on-Write* mechanism.

## class LLMEngine

It receives requests from clients and generates texts from the LLM.

1. Tokenizer 
2. LLM
3. GPU mem space allocated for intermediate states

This class utilizes iteration-level scheduling anad efficient memory management to maximize the serving throughput.

`AsyncLLMEngine` class wraps this class for online serving.

Q

1. 是根据什么调度的?  哪些sequence group 优先级高?  running的优先级高.

先跑一遍模型, 把mem 记录下来了. 

Bach process 是pipeline的形式。第一个batch这些block可以放下KV的，但是在第一个batch没有跑完的时候，第二个batch 也开始了，所以出现内存不够。

#### add_request

 The request is added to the request pool and will be processed by the  scheduler as `engine.step()` is called. The exact scheduling policy is determined by the scheduler.

#### step

This function performs one decoding iteration of the engine. It first schedules the sequences to be executed in the next iteration and the token blocks to be swapped in/out/copy. Then, it executes the model and updates the scheduler with the model outputs. Finally, it decodes the sequences and returns the newly generated results.

### scheduler

Scheduler中包含了三个队列：waitting、running、swapped。每当新增一个SequenceGroup 时，添加至waitting队列中。

1. 当GPU空间不足, sequence group 添加新block, 换出策略: 换出sequence group to swapped.  swapped 中记录那些kvcache 暂时换出到 cpu block的 sequence group (因为block数量是固定的)
2. 当GPU空间不足, sequence group 添加新block, 重计算策略: 空间释放, 把sequence group 移动到waitting.（只有在SequenceGroup内只有一个Sequence的情况才能使用）。
3. gpu空间充足, 把sequence group  的kvcache从 cpu换回gpu. 
4. 将未生成kvcache的prompt 从 xx 调度到running中
5. 输出running 中的sequence group 组成 sequence group metadata, 并记录 is prompt 情况.  

在每次`schedule`执行时，会调度几个队列之间的SequenceGroup，维护队列间的状态，使得当前执行推理尽可能占满显存空间. 详细逻辑如上所示.

- waitting：等待计算KVCache的SequenceGroup（也就是prompt序列）

- running：执行推理的SequenceGroup，会在当前step中作为输入，一共包含两类：

- - prompt：来自waitting，未计算KVCache的SequenceGroup
  - generate token：计算过KVCache的SequenceGroup，准备生成下一个token

- swapped：KVCache暂时换出到cpu内存的SequenceGroup

#### block space manager

BlockSpaceManager的功能是管理各个SequenceGroup对应KVCache存储信息。

每个Sequence的KVCache序列会分成多个block_size长度的cache block，每个cache block的位置信息记录在BlocKspaceManager。如下图所示，BlockSpaceManager包含一个block_tables，其记录cache block到gpu显存或cpu内存物理地址的映射。 其实就是页表. 

SequenceGroup刚加入Scheduler的时候并没有分配cache block空间，第一次进入running的时候需要向BlockSpaceManager申请可用的block空间。BlockSpaceManager分配block空间是以一个SequenceGroup作为一组输入，而且默认分配空间的时候，所有SequenceGroup内的token都是一样的（即是相同的prompt），因此会为所有Sequence都指向同一片cache block区域，该区域被引用数为Sequence数量。

当需要为一个Sequence新增token时，如果空间不足, 直接增加cache block, 或者free last block.

### Worker

A worker class that executes (a partition of) the model on a GPU. Each worker is associated with a single GPU. 一个worker对应一个GPU, 执行一部分model.  The worker is responsible for maintaining the KV cache and executing the model on the GPU. In case of distributed inference, each worker is assigned a partition of the model.

worker 可以  execute_model, 

#### profile_num_available_blocks

Profile the memory usage of the model and get the maximum number of  cache blocks that can be allocated with the remaining free memory.   GPU和GPU的都会算

为什么要 get the maximum number of cache blocks ?

因为这样才能知道什么时候需要从GPU Swap到CPU，以及能支持多长的sequence. 知道了最多能支持的Block之后，比如100个，当我发现我现在已经用了80个，我就可以提前做swap out，避免用完了Memory再做，可以Overlap. 其实就是active evict

#### execute model

先 Issue cache operations. 换入换出, 然后forward. 

#### CacheEngine

initializing and managing the GPU and CPU KV  caches. It also provides methods for performing KV cache operations, such  as swapping and copying.

key是啥?  value是啥? 

提供了swap的方法.

为什么他不做training呢? 

怎么做一个最小的模型呢? 抽象很难, 还是直接做embedding 的方案吧. 因为没有一个方案能覆盖所有. 

- myEngine：是整个系统的入口，其内部组合了一个Scheduler和一组Worker。 

- - Scheduler：在每个推理步调度输入信息，其组合包含了一个BlockSpaceManager

  - - BlockSpaceManager：维护gpu显存和cpu内存的使用情况，以及embedding对应Cache的BlockTable信息。

  - Worker：在每个推理步执行推理，并返回结果。除一个graphsage模型外，其另一个核心组件是CacheEngine。

  - - CacheEngine：负责执行相关gpu、cpu空间的换入、换出、拷贝等操作。

__init__

在构造myEngine时，myEngine就会调用Worker中的CacheEngine，初始化gpu、cpu空间，计算能容纳多少个block。每个block包含`block_size`个node对应的各层embedding大小。在后续的组织中都会将graph对应的KVCache分成block_size大小的cache block，以方便管理对应block的空间。

比如一个block有4个node 的embedding. 

我们也可以叫kvcache, k就是id, v就是embedding  , 对吗?

`add_graph`接口执行多次，接收多个batch。每个输入prompt构造一个SequenceGroup， 其中包含了多个重复的Sequence为后续beam search做准备(我们重复的太少了)。SequenceGroup会最终保存到Scheduler中，以进行后续的调度。

他们是怎么跑多个batch的呢? 一个worker 一次拿一个running queue中的task吗?我们怎么同时处理多个batch呢

`step`执行一个推理步。首先Scheduler会调度一组SequenceGroup和相关信息作为当前推理步的执行输入，除了输入外，还会包含当前SequenceGroup所需KVCache的换入换出信息。然后，Worker会将执行一次推理（当然会先执行CacheEngine先准备KVCache）。Worker  输出结果会再次更新到Scheduler中的SequenceGroup内，以更新其内部的状态。最后，多次`step`循环，直到所有prompt对应的SequenceGroup都生成结束。

BGL: GPU-Efficient GNN Training by Optimizing Graph Data I/O and Preprocessing

有dynamic cache.

是否有case, 多个graph同时保存在GPU?

他们的场合很有挑战, 他们会有很多 request, 不能预测多少request, 需要多少内存. 我们的场合都是静态的，内存都是可以计算出来的。(论文里说了 ,训练都是静态的)

他们是怎么换出的呢? 是LRU还是FIFO? 是否会导致单个request的latency很大? 

他们是怎么快速Detect有哪些Embedding已经在了?



