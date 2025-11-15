GNNLab: A Factored System for Sample-based GNN Training over GPUs 

一作Jianbang Yang

#### 摘要

GNNlab提出 space-sharing. 利用了全局异步queue.sampler放一个GPU, trainer 放另一个GPU, sampling的GPU可以在GPU中持续存topo。training的GPU也可以将所有可用的GPU内存用于缓存feature。这里有一个load imbalance问题. GNNlab 用异步训练来解决.  具有约束滞后性的异步并行训练.

怎么约束滞后性的?  和P3, maricus一样. 

### 1 Introduction

1. 之前发现一些vertices 用的更多. 所以把出度大的节点特征缓存到GPU ( 比如 PaGraph  为什么更快?  因为不用从内存move,   为什么用的更多 ,  因为sampled更多)   ->   缺点:  显存不够.
2.   现有的策略,  在k hop neibor 用  ->  从特殊到一般.  各种采样和数据集都可以用.

Q: 除了 k hop neibor  还有哪些采样? 

GNNlab的stale和 SANCUS有什么区别?

问题:

1.  采样时间久
2. 大量 data moving->  需要caching node feature.  
3. 边数量不平衡, 有的节点两个hop能有非常非常大的子图. 

Prior optimization features of high out-degree vertices are cached in the GPU memory, transferred topo into GPU memory, and applied GPUs to accelerate graph sampling. 

问题1: 不适用于所有dataset和采样算法 degree-based caching policy cannot cover the diversity of sampling algorithms and GNN datasets  -> 需要泛化.   

  泛化的方法:  PreSC tries out a few rounds of sampling phases and uses the average visit count as the vertex hotness metric.   先跑几轮 sampling phase,  用平均visit count来作为hotness metric. 

问题2:  采样, 训练, caching都需要费显存, 容易爆显存.  不能同时做

solution: one GPU loads either graph topological data or cached features in its memory and only conducts either graph sampling or model training.  

但是可能imbalance.  他的方法是异步. 具有约束滞后性的异步并行训练.  adaptively determine the appropriate GPU numbers for Samplers and Trainers 是怎么动态确定的? 

dynamic switching from Samplers to Trainers to avoid idle waiting on GPUs if needed.

所以可能 sampler 做training GPUmemory占用太多. 而两个trainer之间不会load balance问题. 

之前有GPU caching  ,  比如 PaGraph.    有GPU sampling . 比如NEXTDOOR 和DGL,  本文结合了两种 , 然后进行了优化. 

### 3 Analysis of Sample-based GNN Training

怎么减少GPU上的内存?

cache也要考虑sample的算法, 不能自顾自cache, 只考虑output degree.

space sharing,  有的GPU专门sampling, 有的GPU专门训练.

GNNLab prefers to allocate GPUs to Samplers. Because switching from Samplers to Trainers can be fulfilled quickly, but in contrast, load topo is slow.

### GNN lab 架构

#### 5.1 编程模型

```python
 class khop
 def sample:
  nbrs = = samples.push(minibatch)
  for i in range(n_hops)
    # select neighbors for each vertex
    nbrs = uniform_select(nbrs, sizes[i])
    samples.push(nbrs)
return samples

class GCN (...): 
def forward(self, samples, in_feats, ...):
 out_feats = in_feats
 for i in range(n_layers)
 	out_feats = layers[i](samples.pop(), out_feats) # 利用全局队列. 
 return out_feats
```

#### 5.2 混合执行

在内存中保存 global queue,  sampler 作为producer. 

```python
class Sampler(...):
	def run(self, dev, q, graph, ...):
    load(dev, graph) # load graph to GPU memory 
    khop3 = KHOP(graph, ...)   # define 3-hop sampling ...
    while (minibatch = get_minibatch())
      samples = khop3(minibatch) 
      remap(dedup(samples))  #这里做了什么? remap不是extract吗? 
      q.enque(samples) # (async) send task

class Trainer(...):
	def run(self, dev, q, features, ...):
 		model = GCN(...) # define a 3-layer GCN
  	loss_func = ...   # define a loss function
    ...
    while (samples = q.deque()) # (async) recv task 
    	in_feats = extract(features, samples)
    	loss = loss_func(model(samples, in_feats), ...) 
      loss.backward() # DDP就在这里, 和sampler无关. 
```

不同trainer之间是同步的. loss.backward()的时候会同步吗? 会的.

它的async 是gnnlab/example/samgraph/multi_gpu/async/train_graphsage.py  不用DDP

#### 5.3 Flexible Scheduling

extracting 可以隐藏在流水线中, 怎么hide的? 

schedule,先分配 sampler, 把剩下的分配给Trainer.  因为sampler切换trainer快.  而加载图topo要好几秒. 

根据training an epoch in advance. 来估计sampler和trainer数量.  那他是每个epoch都换吗? 是的

switches Samplers to Trainers after sampling all mini-batches of the current epoch.

他的方法是一直保持graph topo. 

samgraph的文件夹整理如下: 

|                                             | DDP  | async | Model | sampler2train |
| ------------------------------------------- | ---- | ----- | ----- | ------------- |
| balance_switcher/ pinsage sampler 2 trainer | x    | x     | CPU   | √             |
| balance_switcher/ pinsage  vanilla          | x    | x     | CPU   | x             |
| multigpu/ pinsage                           | √    | x     | GPU   | x             |
| multigpu/ async/graphsage                   | x    | √     | CPU   | x             |
| train_pinsage.py  vanilla                   | x    | x     | GPU   | x             |

switcher的话, graph topological data 一直装, even though it would limit the size of the feature cache for Trainer. 那cache多少比例呢? 





## 7实验

#### 7.1 Experimental Setup

TSOTA is built upon the same codebase of GNNLab, and supports both GPU- based graph sampling and feature caching. Different from GNNLab, TSOTA follows a time-sharing design, i.e., each GPU conducts both graph sampling and model training, and adopts the degree-based caching policy

sgnn/ 就是time sharing.

Paper Figure 17 shows The runtime of one epoch in GNNLab w/ and w/o dynamic switching (DS). When there is only one trainer, it can reduce by 40% time. But when there are 7 trainers, DS is negligible.

Figure 16 A comparison of the number of gradient updates using DGL, TSOTA, and GNNLab until convergence. It shows GNNlab uses **fewer epochs and more gradient updates** until convergence. Because GNNLab allocates fewer GPUs.

GSG,  papers100M ,    一个 epoch.  33s就能收敛. 是2个sampler, 6个trainer. 

我用1个sampler, 2个trainer. papers100M 可以做到 0.9s 一个epoch.他给paper 100 fanout也是 [25, 10]  ,所以采样速度快. 



## 8discussion

#### Partitioning-based approach

splits graph topological and feature data into multiple partitions and then loads them into different GPUs

方法是adopt cross-GPU memory access.,很慢.

each partition is self-reliant, 会有冗余. 数据是3- hop random neighborhood sampling on the Twitter dataset, each of eight partitions requires to include over 95% of total vertices to be self-reliant.

#### Other sampling algorithms

Such algorithms become more lightweight and lead to highly skewed workloads.

## 代码

ipad工作确实做的好, 提供了cuda 下载地址, 告诉我们怎么检查安装正确与否. 怎么安装g++. 怎么安装软件,指定版本. 每个第三方库提供了安装脚本.  提供了build脚本.  提供了 docker镜像和dockerfile.  讲了实验细节. 

#### 编译安装

build. sh挺快的,大约30秒.

```shell
docker run --ulimit memlock=-1 --rm --gpus all -v $HOST_VOLUMN:/graph-learning -it gnnlab/fgnn:v1.0 bash

apt update
apt install sudo
apt install unzip
python papers100M.py# 有56g, 晚上慢慢下载吧. 
需要 java 包. 需要 git submodule update --init --recursive 
```

数据集都是他们自己准备的

```
python dgl/train_gcn.py --dataset products --num-epoch 10 --batch-size 100
```

手动增加了edge weight,  增加了cache rank table is a vertex-id list 根据出度排序. 为了pyg 增加了uint64 .

写一个topic中想到的各个问题的分析

Q : GNNlab是用DDP的吗? 怎么保证异步也能精度一样的? 

A :  不能一样, 用staleness. 精度差不多.

Q: 是的, Fiugre 8说 trainer之间是异步的, 那为啥还能DDP?

A: 异步不用DDP, 而是用`    cpu_optimizer = optim.Adam(global_cpu_model.parameters(), lr=run_config['lr'])`   gnnlab/example/samgraph/multi_gpu/async/train_graphsage.py

在A100跑不起来. 为什么?   gnnlab/example/samgraph/multi_gpu/train_gcn.py 参考  

gnnlab/example/samgraph/balance_switcher/train_pinsage.py `sampler_stop_event`  来控制切换sampler到trainer.

把gnnlab怎么编译的迁移到空项目.  [出现问题, ImportError: dynamic module does not define module export function (PyInit_samgraph)]

Build pybind system as gnnlab. Try to use needs_recompile 

```
fgnn_env/lib/python3.8/site-packages/samgraph-3.0.0-py3.8-linux-x86_64.egg/samgraph/torch/adapter.py
```

```python
sam.notify_sampler_ready(global_barrier) // 告诉 global barrier. 
                if (not run_config['pipeline']) and (not run_config['single_gpu']):
                    sam.sample_once()
                batch_key = sam.get_next_batch()
                t1 = time.time()
                blocks, batch_input, batch_label = sam.get_dgl_blocks(
                    batch_key, num_layer)
            
            align_up_step 是什么? 
            def process_common_config(run_config): 这些函数是很有用的, 可以多借鉴. 
```

cuda代码有不同arch的

samgraph/common/engine.cc 这个engine很重要.

sample once

```cpp
void samgraph_sample_once() { Engine::Get()->RunSampleOnce(); } 
void RunArch5LoopsOnce(DistType dist_type) {
  调用RunSampleSubLoopOnce()
```

sam.kLogEpochSampleSendTime会自动记下时间

cuda_loops.cc  // void DoGPUSample(TaskPtr task)  是怎么采样的？ 

代码是怎么处理stale bounded的? 

为什么graphsage的sampler没有多个? 最多一个?  因为sample比较快. 

为什么需要start barrier?

为什么trainer也sample? 

#### gnnlab的queue

```cpp
adapter.cc中的GetGraphFeature

GraphPool::GetGraphBatch() 
class GraphPool   
void DistEngine::SampleCacheTableInit() 中有  _graph_pool = new GraphPool(RunConfig::max_copying_jobs);
```



```
    ctx = run_config['train_workers'][worker_id]
            run_config['train_workers'] = [
            sam.gpu(i) for i in range(run_config['num_train_worker'])]
```

Trainsage也不是异步的,是同步的. trainer之间同步. 我们可以先做同步.

不懂为啥要用`class SamGraphBasics(object):`   这个类下面有很多common函数.

`void samgraph_sample_once() { Engine::Get()->RunSampleOnce(); } ` 感觉应该是用`self.C_LIB_CTYPES` 实现分别绑定cuda 和cpu.  

懂了 `void Engine::Create() {` 的时候会创建engine

为啥它能在namespace里定义一个变量? Engine* Engine::_engine = nullptr; 这是属于谁的?

#### pipeline

这是什么意思? 为什么两个barrier位置不同? 

```
        if run_config['pipeline']:
            # epoch end barrier 1
            global_barrier.wait()

        tic = time.time()
        for step in range(num_step):
            sam.sample_once()
            sam.report_step(epoch, step)
        toc0 = time.time()

        if not run_config['pipeline']:
            # epoch start barrier 2
            global_barrier.wait()
```

1. 比较GNNlab 异步和同步的训练需要的epoch数量, 区别大不大?  不大
1. How much time does the GNNlab sampler wait? How much GPU memory does the sampler use?
1. GNNlab async sampler will train or not？ 不会. 
1. GNNlab How different GPU update the model?  A: DDP or update CPU model using a global lock. 

比较  multigpu/ async/graphsage 

我为什么不用global lock 而是用all reduce ? 因为global lock 更快, 可以异步写入.  allreduce要同步很久. 

每个iteration 转换到trainer 肯定不行,  太慢了,  那就每个epoch转换.

不是同一个iteration的话, 你的trainer找不到对应的label.  要存储起来. 

1 papers100M, Where are topology and features located?  When switching to trainer, Which data will move back to the CPU? 

papers100M, 出度大的节点特征缓存到GPU feature. PA有6.4GB topo, 可以全部装下. 

switcher的话, graph topological data 一直装, even though it would limit the size of the feature cache for Trainer. 那cache多少比例呢? 需要人为保证cache + topo不会OOM 
