

## 三个阶段

论文的图是  Sample t+2 batch, gather t+1 batch, train t batch. 

代码就pipeline了两个,

 sampling 是用 num_workers=args.num_workers , overlap sampling and GNN training 

#### mps

默认是没有mps, mps Zero-Copy Kernel 10% 效果最好.

```
python train_sampling_pytorch_direct.py --mps 10,90
python train_sampling_pytorch_direct.py --mps 10,90 --gpu -1
```

为什么要用MPS?   不用MPS的话GPU是分时共享的。用了MPS的话所有实例都会在一个context里。少了context switching也可以提高性能。

MPS 实验中, 可以显著减少smact和smocc, 为什么可以减少occ呢? 可能是融合了. 还可以减少5-10%的GPU memory使用. 

#### 代码

为什么要thread_wrapped_func新开一个线程去做?

### gather

共享GPU 的tensor, in_feat1 和in_feat2 有什么区别?   flag有什么用?  flag应该是控制gather 前一个 batch, 另一个feat2 gather后一个batch. 

```
producer:
in_feat2 = th.zeros(feat_dimension, device=device)
q.put((in_feat1, in_feat2, in_label1, in_label2, finish))
然后th.index_select(train_nfeat, 0, idxf1[0:idxf1_len].to(device=device), out=in_feat1[0:idxf1_len]) 写入到GPU tensor
consumer:
batch_inputs = in_feat1[0:input_nodes_n] 
```

他的index_select 会直接写入GPU tensor.   

### sampling

是CPUsample.

https://docs.dgl.ai/en/0.8.x/guide/minibatch-gpu-sampling.html



为啥epoch 时间长， 但是Speed (samples/sec) 68444.9213 数值 更大?因为wait时间久.

#### 实验

他们测了 ogbn-papers100M





#### 原理

谷歌coreML fundation. MLIR贡献的. 很多phd. 

应该编译0.6.1的dgl吗? 应该可以用最新的. 但是1.0 被废弃了. 

如果dgl 会转成Num array.好处就没了. 

有一些没测试, 写回的时候可能有问题. 写回host端.

FPGA 的老师:  UCI  是吴昆的师兄 uci sitao huang.   UIUC的 chen deming 老师.   gatech cong hao ucsb yuan xie, 存储  ucsd steven swanson

是用CPU采样吗? 应该是的.

node index 传给GPU, GPU 有base address和offset. 

cuda driver 要知道地址, 可以用pin memory. 

默认的行为是unif virtual memory, GPU访问, page fault, cuda driver page migration, 移动页.

torch的修改是吴昆实现的. tensor api是他设计的. 

工作量太大, 设计自动化的论文好发, DAC这些会议, 六页就完成. 

UIUC 申请,  浙大有优势.  UIUC有很多ap.  有个学生 liushaowei 做cv的. compiler毕业都去sde, 要么硬件, 要么应用, 中间的软件系统几乎没有岗位,  alex aiken 做legate legend,singe,warp specialization.

工业界问题是现实的, 基础设施也完善. 美国谷歌可以干TPU. 中国就干不了.   每个组的资源非常重要. 课题也是重要的项目. ap是被低估的,相当于投资. 招收的学生多. connection很重要. 

他老板在英伟达提供项目, 顾问, BAM, GPU访问ssd.    有一些queue, 驱动暴露到用户态.传给GPU. 不招生了. 

一作很懂PCIe, 他会build linux kernel, 很擅长cmake. 建议确认是不是相关.

 128 byte 一次 效率高, 叫做zero copy , 

地址pin了, 指针传输过去, 

host端不用新建tensor,

cuda copy要创建临时空间, 

GNN 中性能好, 比DMA快, 访问和计算要pipeline. 传统的dnn性能不明显,因为计算时间长.

consumer和producer之间通讯还用了CUDA interprocess communication (IPC) APIs. This specific GPU pointer-sharing procedure is implemented in the PyTorch Queue class, and we utilize it for our application

load_subtensor 代码并没有用到

## 论文

#### Large Graph Convolutional Network Training with GPU-Oriented Data Communication Architecture

PVLDB，  2021。 要看看引用它的文章。

#### 优点

1. GPU threads directly access sparse features in host memory through zero-copy accesses without much CPU help
2. automatic data access address alignment to maximize PCIe packet efficiency
3. asynchronous zero-copy access and kernel execution to overlap data transfer with training fully.

#### 问题

PCIe的block transfer， 不适合我们提取sparse data传输。

用CPU来把sparse变成dense，写到buffer， 然后GPU DMA读取。要消耗CPU资源。

#### 方法

CPU pass indices to GPU， GPU zero-copy, read feature tensor。

### zero-copy

1. automatic aligned PCIe memory read request.
2. async zero copy kernel op.  进程隔离+cuda MPS 多进程服务。可以隔离出来一部分资源做zero copy， 剩下的做训练， 用MPS做隔离。 set default GPU active thread to X, 启动main program， zero-copy gathering process 也就是producer， 然后set default GPU active thread to 100-X （看一下代码怎么写的）， 启动training process。  估计X  = 20%用来zero copy， 实际上10%的时候mini batch 训练最快。

### 挑战

GPU op不能直接DMA host mem。

自定义了unified tensor。affinity可以换成GPU同时不data movment。

extension .to(”unified”) 方法， only a portion of the functionality is migrated to this newer PyTorch version 1.8.0nightly. unified这个关键字也没有用在之后的dgl或pytorch.

第五代PCIe， CXL支持远程内存访问RMA，

cudamallochost不支持inter-process sharing。 所以需要 shared memory-》 cudahostregister() + cudahostgetdevicepointer.

bottleneck是host memory bandwidth。

feature dimension越大，加速越明显。

可以用DMA来快速CPU和GPU同时访问tensor。 可以一边加载一边train。

DGL since 0.7 has been supporting GPU-based neighborhood sampling,所以他用的大概是CPU采样。

### 3.3

pingpong的优点, The ping pong buffers are statically located for the entire training process and therefore the pointer sharing needs to be done only once at the beginning of the producer process. 

The goal of using ping pong buffers is to remove the usage of locking mechanisms between two different processes sharing data and to allow them to start working for the next minibatch immediately after finish- ing their current works.

 In our design, each process needs to be synchronized just once per minibatch. 是的, 三个也不好, 需要太多同步. 

#### 3.4多进程

代码没有开源. 

之前dgl是 increasing the number of sampler-consumer pairs and assigning one GPU to each pair.

allocate a shareable memory space and then call CUDA APIs to allow GPUs access to the space.

We utilize the cudaHostRegister() API 

### DGL1.1

感觉1.1就很多东西都没有, 和pytorch direct不一样. 

https://discuss.dgl.ai/t/using-unifiedtensor-in-dgl-1-0/3544

1. 加上dgl-1.1.0+cu113

AttributeError: module 'dgl' has no attribute 'as_heterograph'

删掉这一行, 显示没有unified https://github.com/K-Wu/pytorch-direct/issues/4#issuecomment-1598489158

thread_wrapped_func(run),  th.cuda.set_device(device)会失败而且不报错. 

去掉thread_wrapped_func会显示

```
    th.cuda.set_device(device)
RuntimeError: CUDA driver initialization failed, you might not have a CUDA gpu.
```

是mp.Process的原因,用ctx 就可以了. 

```
 train_nfeat = dgl.utils.pin_memory_inplace(train_nfeat)
直接换成pin_memory_inplace会出问题,
TypeError: index_select() received an invalid combination of arguments - got (NDArray, int, Tensor, out=Tensor), but expected one of:
 * (Tensor input, int dim, Tensor index, *, Tensor out)
 * (Tensor input, name dim, Tensor index, *, Tensor out)
```

我们应该怎么进程间通讯呢? 

2023年2月推出1.0 ,  感觉要看看之前版本的dgl是怎么用的. dgl0.9 有**UnifiedTensor** , python3.7不支持dgl0.6.1. 这个项目的dgl是到 Apr 11 2021. 然而Jul 22, 2021开始有 GPU neighborsampling

https://github.com/dmlc/dgl/commit/905c0aa578bca6f51ac2ff453c17e579d5a1b0fb

gather_pinned_tensor_rows的是会创建新tensor,不一样.

```
    in_feat1 =  dgl.utils.gather_pinned_tensor_rows(train_nfeat,  idxf1[0:idxf1_len])
  File "/opt/conda/lib/python3.7/site-packages/dgl/utils/pin_memory.py", line 55, in gather_pinned_tensor_rows
    _CAPI_DGLIndexSelectCPUFromGPU(F.to_dgl_nd(tensor), F.to_dgl_nd(rows))
  File "/opt/conda/lib/python3.7/site-packages/dgl/backend/__init__.py", line 142, in to_dgl_nd
    return zerocopy_to_dgl_ndarray(data)
  File "/opt/conda/lib/python3.7/site-packages/dgl/backend/pytorch/tensor.py", line 434, in zerocopy_to_dgl_ndarray
    return nd.from_dlpack(dlpack.to_dlpack(data.contiguous()))
AttributeError: 'NDArray' object has no attribute 'contiguous'
```

how to use dgl.ndarray? 怎么取出tensor.

就是把ndarray `pin_memory_ ()`, 要查一下 pin_memory_() 的用法

```
self._data_nd = pin_memory_inplace(self.data)
```

可以先参考https://github.com/dmlc/dgl/blob/40dcc7152d930a8761b84d9ed2041a32fa06972c/examples/pytorch/graphsage/node_classification.py#L116 

看看uva是怎么用的

参考 https://github.com/dmlc/dgl/pull/3616  这个是pin memory引入, 我们可以用pin memory.This pins dgl.graph to the page-locked memory for GPU zero-copy access.

Note: These leverages `cudaHostRegister` to pin an existing host memory, which is visible by all the CUDA context and can be shared between processes. It's for zero-copy access, e.g., UVA-based GPU sampling

但是他没有gpu采样之后 gather feature 的. 现有的例子是 gpu采样的时候gather图topo的.  注意dgl0.8的实现和pytorch direct也不太一样. 

是不是不需要传label, 因为label很小. label传的话快7% 左右.  可能只是测量误差, 几乎没快.  而且开发代码要多写很多. 

`idxf1_len.fill_(len(input_nodes))`  为什么不直接赋值? 因为不需要q put, 统一起来

`batch_feats = th.index_select(train_nfeat, 0, input_nodes.to(device=device), out=in_feat1[0:idxf1_len])` 比 to device 时间为1/3. 

train要保证顺序, 不保证顺序的话就可以直接allreduce.

#### 多GPU

  参考 https://github.com/K-Wu/pytorch-direct_dgl/issues/1

