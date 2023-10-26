GNNAutoScale: Scalable and Expressive Graph Neural Networks via Historical Embeddings —— 论文阅读笔记 - Blue Stragglers的文章 - 知乎 https://zhuanlan.zhihu.com/p/434251933

#### 摘要

把任意message passing的GNN拓展到大图.  利用之前iteration的历史embedding.

他是怎么保持expressive的?

提供历史embedding的误差估计

#### 优点

GAS 的优势：

- GAS 在全体数据中更新。每一轮迭代中，每条边都被计算且仅被计算一次，进而得到历史嵌入，从而保证了线性复杂度。GPU内存占用小. 
- GAS 的推理时间复杂度是固定的。
- GAS 易于实现。
- GAS 有理论保证。

## 1 介绍

GAS prunes the GNN computation graph so that only nodes inside the current mini-batch and their direct 1-hop neighbors are retained







## 2. Scalable GNNs via Historical Embeddings

目前理解是比如 v1 v2是minibatch,  v3 v4是 1hop, 那么把 v2 push到RAM或者hard drive storage. 下一层给v3 pull.  

 v2 push了之后可以给之后用吗？ 可是v2也不是out of mini batch nodes 啊. 好像懂了, 他是别的batch的 out of mini batch nodes.

先 origin graph, 利用metis, 分成40个partition, 得到index, 比如batch size =10, 那就取10个partition的index, 进行一次hop, 获得所有邻居,  得到一个subdata.

```python
class ScalableGNN(torch.nn.Module):
self.histories = torch.nn.ModuleList([
            History(num_nodes, hidden_channels, device)
            for _ in range(num_layers - 1)
        ])
```

给每个node都维护一个embedding. 

问题

1. 为什么不是 v2给下一层的 v1 pull?因为batch内不pull ,
2. Figure1 不用v5 -v8 参与计算吗? 

和GNNlab是正交的, 所以可以结合GNNlab.

### 代码

他是在.

```
def
forward(self, x, edge_index, *args):
x = self.conv1(x, edge_index).relu()
x = self.push_and_pull(self.histories[0], x, *args)# 加一句
x = self.conv2(x, edge_index)
return x
```

V100torch1.11cu113cmake3.25  报错 RuntimeError: Not compiled with METIS support  

试一下pyg2.2-torch1.11-cu113, 也显示 `RuntimeError: Not compiled with METIS support` 看来最高就是cu111

用 V100torch1.9cu102   就可以了

但是这个版本没有AttributeError: module 'torch' has no attribute 'isin' 



have implemented our non-blocking transfer scheme with custom C++/CUDA code to avoid Python’s global interpreter lock. 看看是怎么传输的. 

1. 异步 copy histories from out of mini batch nodes to pinned memory 
2. 异步传输 pinned memory  buffer to GPU 
3. 同步 cuda stream before GPU access.

#### large benchmark 

速度就是figure4, Our concurrent memory transfer reduces I/O overhead caused by histories by a wide margin. 什么意思? 看不懂这个实验是怎么测试性能的.

ValueError: Cannot load file containing pickled data when allow_pickle=False

python main.py model=pna dataset=reddit root=/tmp/datasets device=0 log_every=1 . omegaconf.errors.ConfigKeyError: Key 'reddit' is not in struct

https://github.com/rusty1s/pyg_autoscale/issues/12

i can run on the datasets of ppi,arxiv,products.But for the reddit,flicker,yelp,there are some problems

依然没有测时间, 不知道figure4 , Table4和5 是用什么代码跑的. 

metis 是啥? 

`class History(torch.nn.Module):`

largebenchmark   \- model: pna_jk   \- dataset: reddit 出错

```
Traceback (most recent call last):
  File "main.py", line 70, in main
    data, in_channels, out_channels = get_data(conf.root, dataset_name)
  File "/root/share/pyg_autoscale/torch_geometric_autoscale/data.py", line 113, in get_data
    return get_planetoid(root, name)
  File "/root/share/pyg_autoscale/torch_geometric_autoscale/data.py", line 17, in get_planetoid
    return dataset[0], dataset.num_features, dataset.num_classes
  File "/opt/conda/lib/python3.7/site-packages/torch_geometric/data/dataset.py", line 240, in __getitem__
    data = data if self.transform is None else self.transform(data)
  File "/opt/conda/lib/python3.7/site-packages/torch_geometric/transforms/compose.py", line 24, in __call__
    data = transform(data)
  File "/opt/conda/lib/python3.7/site-packages/torch_geometric/transforms/to_sparse_tensor.py", line 70, in __call__
    is_sorted=True, trust_data=True)
TypeError: __init__() got an unexpected keyword argument 'trust_data'

Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.
```

Hisotry会把设备放在device上. 

为什么要pre parition?  

为什么要permute? 根据索引来排列adj 

为什么  metis?为了找到索引

要最小化 inter-connectivity between sampled mini-batches , which minimizes history access, and increases closeness and reduces staleness in return.

也是用graph clustering techniques, *e.g.*, METIS (Karypis & Kumar, 1998; Dhillon et al., 2007), to achieve this goal. 可以参考betty的新实现. 

好处

the number of neighbors out- side of B is heavily reduced, and pushing information to the histories now leads to contiguous memory transfers. 为什么? 

len(cluster) 2708
perm tensor([2459, 2439, 2440,  ..., 2552,  104, 1591])

Ptr :  每个cluster 到哪里

perm tensor([2459, 2439, 2440,  ..., 2552,  104, 1591]) 所有node id

ptr [29,52,...] 40

4个gpu , 4 *4 matrix 显示有多少边. 

cluster 是一个一维tensor ,perm也是一维tensor, 就是index. 

算法` torch.ops.torch_sparse.partition`是来源于https://github.com/rusty1s/pytorch_sparse/blob/3bf43eb09a68efb684d5a09e33a2e2114dd81689/csrc/metis.cpp

为什么 `ptr = torch.ops.torch_sparse.ind2ptr(cluster, num_parts)`?

https://github.com/rusty1s/pytorch_sparse/blob/eabdc5846a2080db87d5d5955773f41dec8d6c7a/csrc/cpu/convert_cpu.cpp  看不懂 ind2ptr是在干嘛,好像一种稀疏变换, 但是不知道ptr是啥. [2, 3, 7] -> [0, 0, 0, 1, 2, 2, 2, ...]

为什么ptr length是41呢?   懂了, ptr就是一个指针向量. 就是每个的位置 .41个点有40个parts.

`data = permute(data, perm, log=True)` 就是按索引改adj.

```
class ScalableGNN pool_size 和buffer size 有什么区别? 
        pool_size (int, optional): The number of pinned CPU buffers for pulling
            histories and transfering them to GPU.
            Needs to be set in order to make use of asynchronous memory
            transfers. (default: :obj:`None`)
        buffer_size (int, optional): The size of pinned CPU buffers, i.e. the
            maximum number of out-of-mini-batch nodes pulled at once.
            Needs to be set in order to make use of asynchronous memory
            transfers. (default: :obj:`None`)
```

他会存多久之前的embedding ?   一个epoch之前的, 还是所有的, 还是一个ieration之前的?  `class History(torch.nn.Module):` 

forward的时候会调用`self.push_and_pull(hist, x, *args)`

调用

```
            history.push(x[:batch_size], n_id[:batch_size], offset, count)
            h = history.pull(n_id[batch_size:])
```

感觉是每个iteration都会迭代. 

cora数据集, 一个feature 5732byte大. 

```
  File "/root/share/pyg_autoscale/torch_geometric_autoscale/__init__.py", line 9, in <module>
    torch.ops.load_library(importlib.machinery.PathFinder().find_spec(
  File "/opt/conda/lib/python3.8/site-packages/torch/_ops.py", line 220, in load_library
    ctypes.CDLL(path)
  File "/opt/conda/lib/python3.8/ctypes/__init__.py", line 373, in __init__
    self._handle = _dlopen(self._name, mode)
OSError: libcudart.so.10.2: cannot open shared object file: No such file or directory
```

怎么把gnnautoscale分成多个nano batch?

为什么我pip 安装的时候没有编译成cu113?

好像懂了, 应该先把build删除. 是的

她存了所有node 的embedding, 但是图大的时候不可能?  可以的吧, CPU内存够大就行. 

是的, 它的都是全图比较.节点数量不超过5w,WIKI-CS 1w多, AmazonComputerDataset 1w节点

但是也有ogbn- products. yelp,

装不下的时候, 需要别的优化. 

它不算1 hop neighbor， 每次都是用之前算的embedding来pull。目前算出来的node embedding， 会成为别的node的 neighbor embedding.甚至没有embedding的时候， 它就pull 一个空向量， 计算图裁剪了非常多. 



print看看  gas  loader 循环了几次，40/10 =4 次吗？ 是的,就是4次. 

torch.Size([1021, 1433])
torch.Size([934, 1433])
torch.Size([966, 1433])
torch.Size([906, 1433])

 1/4的图训练,  拿1hop的每一层old embedding,  更新每一层,batch内的embedding 然后1/4的图训练.

他会先做一个inference 充满这个embedding.不然都是空的.

注意history是不同层都不同的,  一次forward是把每一层的mini batch 都push新的.

```
            h = history.pull(n_id[batch_size:]) #把mini batch外的都pull下来, 给下一个conv 用.
            return torch.cat([x[:batch_size], h], dim=0) 
```

gas是这样,但是cat了之后梯度就没了. 我不知道他是怎么传递梯度给下面的.

gas 是to cpu来重新创建.  

GNNAutoScale的pull和push是不需要梯度的, 
