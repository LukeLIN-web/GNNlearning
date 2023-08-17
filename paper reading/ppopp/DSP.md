DSP: Efficient GNN Training with Multiple GPUs

Cuhk James Chen组 和亚马逊

DSP adopts a tailored data layout to utilize the fast NVLink connections among the GPUs

For efficient graph sampling with multiple GPUs, we introduce a collective sampling primitive (CSP)

design a producer-consumer-based pipeline,

The speedup of DSP can be up to 26x and is over 2x in most cases.

Quiver and DGL-UVA cannot fully utilize the GPUs as they execute the kernels for different tasks sequentially.

新layout是什么? 

## Introduction

解决的问题: the amount of read data is larger than requested 

figure1 , CSP为啥比ideal还低?  paper里面解释了，因为有一些数据已经Cache了

CSP generally expresses different graph sampling schemes and is efficient by using fast NVLinks and pushing tasks to data.

新CSP是什么? 

## 3 DSP

用METIS

DSP partitions the graph topology into patches and stores one patch on each GPU. 这个其实我也发现了,可以充分利用locality.  idea果然还是要自己想. 

For the node feature vectors, we cache as many hot nodes as possible in GPU memory. 

DSP uses large in-degrees to select hot nodes by default and is compatible with other criteria.

和quiver不同, DSP uses a partitioned cache, where each GPU caches different feature vectors. 需要的时候它会用all-to-all NVlink来传输. 

### 3.2 Training Procedure

同步用BSP, bulk synchronous parallel, an iteration should observe all model updates made in its preceding iterations.

为啥loader能在GPU上? GPU能发起传输吗? 可以的, CUDA UVA 允许用户创建超出 GPU内存大小的数据，同时利用 GPU 内核进行快速计算。将整个图结构及其特征存储在UVA后，可通过GPU内核快速提取子图特征.

不仅hot可以nvlink比pcie更快,更重要的是 hot和cold 可以并行. 

nvlink用NCCL, nvshmem 库可能更快, 但是可能有些GPU server没有nvlink用不了.  (感觉这是个很扯淡的原因)

多机的时候, 复制hot feature.  分割cold feature.

## 4 **Collective Sampling Primitive**

sample 分为三步.仔细看figure5

For shuffle, each frontier node is transferred to the GPU holding its adjacency list; 

for sample, each GPU locally samples the required number of neighbors for the frontier nodes it receives;  

reshuffle, the sampled neighbors of each frontier node 𝑣 is transferred back to the GPU requesting neighbor samples for node 𝑣.

为什么都要传输? 不会很慢吗?   

每个stage都有同步, 异步反而很慢. 为什么? the communication and sampling tasks of a single GPU are small. 是什么意思? 

CSP 是同步的, synchronization barrier at the end of each stage.

we assign each seed node to the GPU holding its adjacency list and manage a well-connected graph patch with one GPU, which makes many accesses to the adjacency lists local on each GPU.  nvlink 充分利用, 减少PCIe传输.

### 4.2

统一了layer-wise和 node-wise.

## 5 Pipeline for GNN Training

为了提高GPU利用率, 用pipeline. 

the communication kernels of the sampler 只需要很少的线程就fully utilize the NVLink bandwidth.

他们发现setting the queue capacity limit to 2 is sufficient for overlapping the tasks.

#### 防止死锁

需要一个GPU作为leader, 指定order.

On each GPU, a worker registers its id in a pending set

the queue to manage its ready communication kernels and start them in their submission order.

DSP 做了三个stage pipeline. 

我们可以看看他们的pipeline怎么写, 然后改进.   他们的GPU 使用率非常高. figure6 , 为什么?他们怎么达到这么高的. 他们做的很general, 不是专门的优化.



## 6 implement

The graph patch on each GPU is stored using the compressed sparse row (CSR) format

Each GPU holds an adjacency position list and a feature position list

## 7实验

数据集: Papers,1亿节点,   Friendster 66M个节点

We do not compare with GNNLab [42] as it requires the graph topology to fit in one GPU and conducts asynchronous training for efficiency.

 we train a 3-layer Graph- SAGE model with a hidden size of 256 for all layers and a batch size of 1024 and conduct neighbor-wise sampling with a fan-out of [15, 10, 5] by default

DSP快速收敛 因为has a shorter mini-batch time.

## 代码

https://zenodo.org/record/7463498#.ZAXVhuxBxjs

github上没有这篇文章的代码.

```
docker pull zhouqihui/dsp-ppopp-ae:latest
docker run --rm -it --runtime=nvidia --ipc=host --network=host zhouqihui/dsp-ppopp-ae:latest /bin/bash
```



### 数据集

The first is, Why DSP_AE-master/pyg/friendster.py has "assert False", it is correct?

```
  def process(self):
	assert False
```

The second is, I am using zhouqihui/dsp-ppopp-ae:latest `conda activate pyg`. But it doesn't have installed Pytorch. 

用DSP的 mps版本.  在不同model上看看差距.

他只有graphsage. 



问题

`FileNotFoundError: [Errno 2] No such file or directory: '/data/ds/distdgl/ogb-product1/ogb-product.json'`



#### dgl ds

他们写了 ParallelNodeDataLoader

用torch.cuda.stream 来提高并行度. 

```
class SubtensorLoader(Thread):
	self.out_pc_queue = MPMCQueue(1, loader_number, 1)

```



