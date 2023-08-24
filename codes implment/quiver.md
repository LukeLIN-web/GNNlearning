Quiver https://torch-quiver.readthedocs.io/en/latest/Introduction_cn/

- 图采样**latency critical problem**，高性能的采样核心在于通过大量并行掩盖访问延迟。
- 特征提取**bandwidth critical problem**，高性能的特征提取在于优化聚合带宽。

在拥有NVLink的情况下，获得**多卡超线性加速收益**。

quiver用共享内存来存储数据.  docker 容器要 shm够大来容纳数据集.

1. 为什么全放gpu也没quiver快呢? 因为quiver实现了自己的feature collection和gpu sampling
2. 怎么选取hot feature的? 统计了连接数高的.

看代码 , 先看readme, 文档, 再看文件结构, 看函数名, 看注释. 

`python3 examples/multi_gpu/pyg/reddit/dist_sampling_ogb_reddit_quiver.py`

def run中的 torch.cuda.set_device(rank)去掉的话就 训练时间变得很长, 第一张卡显存爆炸. 为什么?

可能是后面调用了tensor.cuda()这种操作,  如果没有的话就会默认到GPU0上去操作.

`quiver_sampler.sample(seeds)` 中的一些cuda操作.

用pybind11来绑定cpp代码.

class Topo: 可以检查是否有连接 , torch_qv.can_device_access_peer

```
   quiver_sampler = quiver.pyg., 传入到  mp.spawn
```

每个GPU都有sample.

### 怎么GPU采样

```python
    quiver_sampler = quiver.pyg.GraphSageSampler(
        csr_topo, sizes=[25, 10], device=0, mode='GPU')
for seeds in train_loader:
            n_id, batch_size, adjs = quiver_sampler.sample(seeds)
        
def sample(self, input_nodes):
   for size in self.sizes:
            out, cnt = self.sample_layer(nodes, size)
 #    layer调用self.quiver.sample_neighbor(0, n_id, size)
#    sample_neighbor是cuda代码 , 调用sample_kernel.
```

一开始spwan的时候是fake device, 后来self.lazy_init_quiver()之后会变成0 1 2 .就是加载到对应的GPU显存上. 

采样的速度**S**ampled **E**dges **P**er **S**econd, **SEPS**) 是怎么测的? 

需要cuda 同步

```python
        torch.cuda.synchronize()
        sample_start = time.time()
        _, _, adjs = quiver_sampler.sample(seeds)
        torch.cuda.synchronize()
        sample_time += time.time() - sample_start
```

第一个epoch的总时间是必须舍弃的, 非常大!

晚上可以跑`examples/multi_gpu/pyg/ogb-products/dist_sampling_ogb_products_pyg.py`  ,要跑很久. 

## 数据搬运timeline

```python
quiver_sampler = quiver.pyg.GraphSageSampler(csr_topo, sizes=[25, 10], device=0, mode='GPU')
会调用self.quiver = qv.device_quiver_from_csr_array(self.csr_topo.indptr,
                                                       self.csr_topo.indices,
                                                       edge_id, device,
                                                       self.mode != "UVA") 把csr topo 搬运到GPU上. 
model = model.to(device) 
x.from_cpu_tensor(data.x)  会调用self.feature_order self.csr_topo.feature_order.to(self.rank)
会新建 ShardTensor 实例, ShardTensor 实例get item的时候会nodes = nodes.to(self.current_device)
取feature的话是feature = self.shard_tensor[nodes], 没看到to device是哪一行?

sample的时候, nodes = input_nodes.to(self.device) , 会把node id to GPU,如果topology在CPU, 他会把第二层sample出来的 node id 再to GPU.
x[n_id], __getitem__的时候会先node_idx = node_idx.to(self.rank), 然后调用 ShardTensor 实例get item.
```

| 操作或内容  | 位置                                    |      |
| ----------- | --------------------------------------- | ---- |
| sample      | GPU                                     |      |
| feature     | hot的在GPU中.  cpu part 在shared memory |      |
| train index | 内存                                    |      |
| topology    | shared memory                           |      |
|             |                                         |      |

quiver的内存位置. TOPO在哪里, feature在哪里,每个算法都要研究, 能说出来. 

### feature

首先, quiver.CSRTopo会把整个图edge index转成csr 结构.

reindex, 把 节点也变成csr结构对应的.

切成两份, 一份是cpu部分一份是cache的部分. 

怎么to到GPU的?

怎么选择hot的? 统计节点多 的. 

`pytest tests/python/cuda/test_partition_feature.py::test_random_partition_with_hot_replicate`

他不是init就to的, 是第一次访问的时候会to GPU.

在461行

```
        if csr_topo is not None:
            self.feature_order = csr_topo.feature_order.to(self.rank)
```

`x.from_cpu_tensor(feature)` 不包括在 `torch.cuda.memory_allocated()`  也不包括在reserved中, 为什么?因为 init的时候不会to rank,   To rank 是在加载的时候. 其实也可以直接用watch gpustat. 

### Communication

lazy_from_ipc_handle 有什么用?

可以参考 child_sendrecv_proc 来通信 . 写了NcclComm

a wrapper of the communicator in the NCCL library. It supports both p2p communication (e.g. send/recv) and collective communication (e.g. allreduce).  It is initialized by NCCL ID, which is provided by get Nccl Id. After this ID is synced across all workers in the cluster and every worker knows the cluster topology, then an NCCL communicator can be constructed with them.

#### API

```cpp
    void send(torch::Tensor tensor, int dst)
      void recv(torch::Tensor tensor, int src)
          void allreduce(torch::Tensor tensor)
     void exchange
```

在 torch-quiver/srcs/python/quiver/feature.py 用了他们的exchange模块.

可以看一下exchange模块是怎么一个个发送的.  有没有帮助实现scatter?

可以模仿这个写scatter操作. 

#### **p2p_clique_replicate**

#### `quiver.init_p2p([0, 1])`

## 论文

https://arxiv.org/pdf/2305.10863.pdf

1. Key insight: the same batch size but a large variance in latency 
2. Choose CPU sampling or GPU sampling based on the workload.
3. Find the relationship between latency and computation workload. Use the offline information to guide the runtime decision 
4. Network topology-aware and hot-aware node feature placement

inference, 存所有的inference后的label. 推理的优化.

### 摘要

1. 计算PSGS. probabilistic sampled graph size. 并行度高的放GPUinfer. 不过开源的代码里没有看到psgs.
2. feature, 可以看代码文档

图变化了, 怎么处理? 

#### Zero-copy

quiver也有.

不要greedy cache, 10个里5个在, latency 一样, 避免9个不在需要两次PCIe.

inference可能经常查询一些out degree低的.



#### 6Evaluation

user latency threshold 是 30 ms.



## serving

prepare data, 会 generate_neighbour_num. 产生 `25_10_neighbour_num_False.npy`

reddit_serving.py开了很多进程.

```
    os.sched_setaffinity(0, [2*(cpu_offset+rank)])
OSError: [Errno 22] Invalid argument
```

方法 :  注释掉

为什么需要多个stream_input_queue_list? 因为有多个进程在读取

怎么准备不同Batch之间复用? 给有重复的输入. 

```python
    def auto_despatch(self, idx):
    
	    neighbour_num = np.load(self.neighbour_path)
                tmp_sum = np.take(neighbour_num, item).sum()
                
                if tmp_sum > self.threshold:
                    gpu_batched_queue.put(item)
                else:
                    cpu_batched_queue.put(item)
```

他有8个cpu sampler, 2个gpu sampler., 分别处理, 怎么判断所有进程都结束了呢? 

```python
    start_time = time.time()
    while True:
        result = result_queue.get()
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("Elapsed time:", elapsed_time, "seconds")
```

怎么disable CPU? 

request_mode = 'Fixed', request_mode='CPU' 和 request_mode = 'Random' 有什么区别?   cpu就是fix在cpu上, 

exp_id = 'fixed_depatch'  和 exp_id = 'auto_depatch' 有什么区别?  没区别  就是个string.

exp_id ='fixed_depatch' , sample_mode = 'GPU'有重复的时候就会报错eof , 为什么?   因为访问了同一个点

it's because an `id` in the training list (format `imgname 0 id 0`) is equal to or larger than the param `last_fc_size`.  因为没有从0 开始

The `id` should start from 0 and not exceed `last_fc_size-1`.

为什么有的给gpu，反而更慢了. 可能因为这个重复的节点 degree小. 选高degree的，不要随机一个，重复. 测试quiver 

quiver load feature 要多久？ 占多少时间。  `edge_index, _, size = adj.to(device)  和 x = x_all[n_id].to(device)` 大概占用53%  ,占用的多, 就可以用pipeline的方法. 

```
/opt/conda/conda-bld/pytorch_1670525552843/work/aten/src/ATen/native/cuda/ScatterGatherKernel.cu:144: operator(): block: [0,0,0], thread: [15,0,0] Assertion `idx_dim >= 0 && idx_dim < index_size && "index out of bounds"` failed.
```

他开多个gpu loop, 是怎么划分任务的?   A:  不断从queue中取任务. 

 threshold是1670  , 测试degree ,  一个batch 48个点,  48个点的neighbor num sum = 576, threshold是1670. 所以判断他在CPU里快. 实验也说明了这一点. 

看不同重复率的时候, 跳过, 用的时间. 

速度差不多

 Batch size 调大点看看? 

可能因为neighbor num 总量没有超过GPU带宽的上限, 所以速度差不多

sampling的时间应该比较大. 

-1, -1 不能产生neighbor num .`python prepare_data.py ` 会自动退出. 



### 安装问题

为啥没在gpu上跑. 因为没有共享gpu `--gpus all \ `

先 apt install nvidia-cuda toolkit. 然后 安装英伟达的docker cuda toolkit

下载 ngc  22.04.  torch1.12.0 但是这个有两个版本的torch, 会导致pyg出错. 

https://github.com/pyg-team/pytorch_geometric/tree/master/examples/multi_gpu

https://github.com/pyg-team/pytorch_geometric/blob/master/examples/quiver/README.md

Pip 安装失败runtimeError:    CUDA was not found on the system, please set the CUDA_HOME or the CUDA_PATH  environment variable or add NVCC to your system PATH. The extension compilation will fail.   这是一个假的cudatoolkit, 没有nvcc. 不能编译. 

```bash
1. 安装python3.8 
2. 安装conda  
apt update
apt install python3.8
bash Miniconda3-latest-Linux-x86_64.sh
3. 安装cuda11.3  
 conda install -c conda-forge cudatoolkit=11.3.1 # 这一步有问题. 
4. torch
安装pyg  安装quiver.
```

运行distributed sampling 会显示. 

```
/opt/conda/lib/python3.7/multiprocessing/semaphore_tracker.py:144: UserWarning: semaphore_tracker: There appear to be 12 leaked semaphores to clean up at shutdown
  len(cache))
Traceback (most recent call last):
  File "distributed_sampling.py", line 123, in <module>
    mp.spawn(run, args=(world_size, dataset), nprocs=world_size, join=True)
  File "/opt/conda/lib/python3.7/site-packages/torch/multiprocessing/spawn.py", line 240, in spawn
    return start_processes(fn, args, nprocs, join, daemon, start_method='spawn')
  File "/opt/conda/lib/python3.7/site-packages/torch/multiprocessing/spawn.py", line 198, in start_processes
    while not context.join():
  File "/opt/conda/lib/python3.7/site-packages/torch/multiprocessing/spawn.py", line 146, in join
    signal_name=name
torch.multiprocessing.spawn.ProcessExitedException: process 0 terminated with signal SIGBUS
```

#### zenotan/torch_quiver

```
用这个镜像 pip install torch_quiver
python3 examples/pyg/reddit_quiver.py
AttributeError: Can't get attribute 'DataEdgeAttr' on <module 'torch_geometric.data.data' from '/opt/conda/lib/python3.7/site-packages/torch_geometric/data/data.py'>
感觉pyg装坏了.https://github.com/pyg-team/pytorch_geometric/discussions/5862
```

重新下载reddit数据集. 又hang住了. `quiver_sampler = quiver.pyg.GraphSageSampler(csr_topo, sizes=[25, 10], device=0, mode='GPU') # Quiver ` 

```
qv.device_quiver_from_csr_array(self.csr_topo.indptr,self.csr_topo.indices,
                                                       edge_id, device,
                                                       self.mode != "UVA") 这里hang.
运行torch-quiver/srcs/cpp/src/quiver/cuda/quiver_sample.cu的new_quiver_from_csr_array
```

和tryquiver1一样的问题.

````
尝试pip uninstall torch-quiver
sh ./install.sh
````

还是在device_quiver_from_csr_array hang .

在V100. compute_70 mcnode36试一下这个镜像 , 成功了!!!!

```
python3 examples/multi_gpu/pyg/reddit/dist_sampling_ogb_reddit_quiver.py
```

尝试卸载torch,torchvision,安装torch.

```
这样会OSError: /opt/conda/lib/python3.7/site-packages/torch_sparse/_convert_cuda.so: undefined symbol: _ZNK2at6Tensor5zero_Ev 
原因是：不要通过 anaconda-navigator 安装 pytorch！即使您只选择 pytorch，它也会同时安装 cpu 和 gpu 版本。您可以从以下位置下载特定版本的 pytorch：https://download.pytorch.org/whl/torch_stable.html 
安装pip install torch-1.10.0+cu102-cp37-cp37m-linux_x86_64.whl
```

![](https://img-blog.csdnimg.cn/20201227144536788.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQwMzI5Mjcy,size_16,color_FFFFFF,t_70#pic_center)

H100也是可以编译的. 

##### 环境

python3.7 , cuda10.2 , torch 1.9.0,torch-geometric        2.0.2,torch-quiver           0.1.1

#### tryquiver1

https://github.com/quiver-team/torch-quiver/issues/133

用这个镜像 ,torch-quiver   0.1.0 .

`python3 examples/pyg/reddit_quiver.py`  AttributeError: module 'torch_quiver' has no attribute 'device_quiver_from_csr_array'. 这个是内部cpp出了问题. https://github.com/quiver-team/torch-quiver/issues/133   卸载重装torch-quiver 0.1.1.  运行上面的, hang住不动了. 

#### gnnlab

尝试用gnnlab的镜像来运行.

https://github.com/quiver-team/torch-quiver/issues/134

先尝试pip install torch_quiver:     subprocess.CalledProcessError: Command '['ninja', '-v']' returned non-zero exit status 1.   是cuda10.1 不够新.   nvcc fatal   : Unsupported gpu architecture 'compute_80' .您需要安装 cuda 11.0 工具包来编译这些示例。cuda 11.0 支持 compute_80（安培拱 GPU）。 

在V100. compute_70 mcnode36. 

```python
docker pull gnnlab/fgnn:v1.0
pip install torch_quiver
 undefined symbol: ncclRecv nccl版本不对.
```

#### torch1.10

尝试自己安装.  成功了.

```dockerfile
FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel
# Install PyG.
RUN CPATH=/usr/local/cuda/include:$CPATH && \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH && \
    DYLD_LIBRARY_PATH=/usr/local/cuda/lib:$DYLD_LIBRARY_PATH
RUN pip install scipy==1.5.0
RUN pip install --no-index torch-scatter -f https://data.pyg.org/whl/torch-1.10.0+cu113.html && \
    pip install --no-index torch-sparse -f https://data.pyg.org/whl/torch-1.10.0+cu113.html && \
    pip install torch-geometric
RUN pip install -v .
# Set the default command to python3.
CMD ["python3"]
```

BI=0 -gencode=arch=compute_70,code=compute_70 -gencode=arch=compute_70,code=sm_70 -std=c++14

model 是不是 whole graph.

https://pybind11.readthedocs.io/en/stable/

#### 可编辑模式

尝试可编辑模式安装出错

```
/opt/conda/lib/python3.8/distutils/extension.py:131: UserWarning: Unknown Extension options: 'with_cuda'
      warnings.warn(msg)
    error: ("Can't get a consistent path to setup script from installation directory", '/root/share', '/root/share/torch-quiver')
```

https://stackoverflow.com/a/9491859/14923589 

先尝试 增加一个 pyproject.toml , 出现问题:   ERROR: Could not find a version that satisfies the requirement setuptools>=40.8.0 (from versions: none)
  ERROR: No matching distribution found for setuptools>=40.8.0

我setuptools  == 58.0.4, 升级到65, 还是显示  ERROR: No matching distribution found for setuptools>=40.8.0.

现在普通安装都失败了. 然后去掉 pyproject.toml  可以普通安装. 

就是路径前面多了一个`./` , 去掉就好了.

可以试试把quiver改成 python 绑定 来熟悉https://pybind11.readthedocs.io/en/stable/compiling.html

为什么有的地方是import torch_quiver , 有的地方是quiver

为什么 quiver make 没有用? 

因为make是在setup.py中调用的.重新执行`pip install --no-index -U -e .` 

还是不对, pip -e只是把build的文件复制到srcs/python中, 没有重新make.

Cmake 3.10.2不支持TARGET_LINK_DIRECTORIES, minimum to 3.13 at this point.

pip -e 还会生成 build文件夹torch-quiver/build/temp.linux-x86_64-3.8  和lib.linux-x86_64-3.8/

## 编译速度改进

原版pip install 需要 4分39s 太慢了. 

安装太久了,  能不能把make和 pip 安装解耦? 不行,因为必须从下到上. 

Python 的queue底层也是cpp, 所以没必要自己实现.

 





