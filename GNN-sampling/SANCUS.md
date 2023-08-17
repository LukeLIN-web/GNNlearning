VLDB 2022 上的论文“面向大规模图神经网络的陈旧性感知通信回避的去中心化全图训练框架  Best Regular Research Paper

SANCUS: Staleness-Aware Communication-Avoiding Full-Graph Decentralized Training in Large-Scale Graph Neural Networks

其实想法也很简单, 就是 

adjs (A)和embedding (H) 划分到各个GPU. 每个GPU保存完整model和完整权重矩阵(W).

GPU一层训练后,检查embedding stale性, 如果active( 什么样是active 的? ), 就广播到所有节点 并且缓存. 如果stale, 那就重复使用history embedding. 反向传播的时候梯度也是一样. 

更新模型的时候要all reduce 梯度.

提出(尝试)了三个衡量stale方法

方法1 .  epoch, 就是GNNlab的bounded staleness

方法2 . 如果neighbor的 embedding 已经是n个epoch 之前的 , 向其他设备广播最新embedding

方法3 :embedding变化超过阈值时广播. 

每个GPU1个进程

 感觉都是各种近似. 理论推导好难.

他们居然有四个大学的合作, 不知道怎么合作的. 

如果我们就是micro batch少传 embedding的话,感觉和他们的一模一样. 



Sync at what point?



### 代码:

- CAGNET 的完整重构。 这是一个之前的gnn分布式训练库.
- 分布式实用程序，例如日志、计时器等。
- 节点特征缓存训练。
- 磁盘上的分区图缓存。
- 更多数据集。支持来自 pyg、dgl、ogb 的大多数大图。
- 训练仅依赖于 pytorch。
- 分布式 GAT 训练。
- 支持最新的 pytorch 版本。
- 支持 CSR 图。
- 支持半精度训练。

prepare data 运行了20分钟还没完成. 

Flick process的时候error cannot reshape array of size 3932144 into shape (89250,500) .`pyg_dataset: torch_geometric.data.Dataset = dataset_sources[pyg_name](root=os.path.join(pyg_root, pyg_name))` 

reddit的 fail  CUDA error: no CUDA-capable device is detected CUDA kernel errors might be asynchronously reported at some other API call,so the stacktrace below might be incorrect.

prepare data 卡住. 

到底什么时候barrier呢? 

```python
from torch.cuda.amp import autocast
from dist_utils import DistEnv
import torch.distributed as dist
#autocast的上下文后，上面列出来的那些CUDA ops 会把tensor的dtype转换为半精度浮点型，从而在不损失训练精度的情况下加快运算

def use_cache(tag, src):
    F_L1 = tag == 'ForwardL1' and g_bcast_counter[tag][src]>0 # if there is enough gpu mem
    F_L2 = tag == 'ForwardL2' and (g_bcast_counter[tag][src]>50 and g_epoch_counter[tag]%2==0)
    use = g_cache_enabled[tag] and (F_L1 or F_L2)
    if use:
        assert(src in g_cache[tag])
    return use
  # 用global的 dict dict来保存. 
                    dist.broadcast(feature_bcast, src=src)
                    g_bcast_counter[tag][src] += 1
                    if g_cache_enabled[tag]:
                        g_cache[tag][src] = feature_bcast.clone()
```

是先计算，然后再判断，还是先判断，然后再计算嵌入？

先判断, 再计算嵌入.

W 权重矩阵的大小是F*F (embedding weight)



需要修改layer

```python
 model = CachedGCN(g, env, hidden_dim=16)
 class CachedGCN(nn.Module):
    def forward(self, features):
        hidden_features = F.relu(DistGCNLayer.apply( features, self.weight1, self.g.adj_parts, 'L1'))
        
class DistGCNLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, weight, adj_parts, tag):
        ctx.save_for_backward(features, weight)
        ctx.tag = tag
        z_local = cached_broadcast(adj_parts, features, 'Forward'+tag)
        return z_local
```

class DistEnv: 是很不错的分布式自定义类模板. 



#### 6 related work

大部分是用PS架构. 比如NeuGraph,单机多卡. 

RoC 提出了动态分区, 在线回归, 

PaGraph.
