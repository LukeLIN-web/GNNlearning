TEMPORAL GRAPH NETWORKS FOR DEEP LEARNING ON DYNAMIC GRAPHS 

 TemporalDataLoader是根据时间顺序排列的吗? 

为什么要Sample negative destination nodes?

TGNs：动态图中高效通用的时序图卷积网络框架 - dreamhomes的文章 - 知乎 https://zhuanlan.zhihu.com/p/360758453

源码: https://github.com/twitter-research/tgn

memory里面存了什么? 

Mem 是一个可学习的内存更新函数，例如 LSTM 或者 GRU 等.  

#### 训练流程

按时间t遍历整个dataset 所有event.

`z, last_update = memory(n_id)`  forward调用_get_updated_memory

_get_updated_memory会做下面三件事: 

1.  之前的节点特征通过MLP生成msg ,代码中是_compute_msg,这里会把msg store中的东西取出来计算. 
2. 用最近message aggregate多个时间的msg
3.  gru （）节点特征 z。   这里用到了self.memory存的. 

 Gnn 得到 node embedding。 `z = gnn(z, last_update, edge_index, data.t[e_id].to(device), data.msg[e_id].to(device))`

-> 计算edge probabilities ,  - >   算出loss后

`memory.update_state(src, pos_dst, t, msg) ` 会调用`_update_memory` 和`_update_msg_store`

` _update_memory`也会调用` _get_updated_memory`,  这两个n_id是一样的吗? 

不一样,   last update 得到的是1hop的n_id.

update state 用的`n_id = torch.cat([src, dst]).unique() ` 没有neg_dst 也没有1hop.  是更新target node, 但是我觉得这是可以重用的!    `self._update_memory(n_id) # 5.73% ` 不过这个占比不是很大.  但是可以作为contribution! 可以用来吹.

_update_msg_store  ,  会用for loop把一堆tensor存在msg store中. 

msg store用dict 存 tensor的引用, tensor都在GPU上, 但是, 占用的GPU mem还是很少. 可能是图不够大. 

msg_store存的是什么?  存了event.  为了存最近的msg eij(t) raw_msg

self.memory 存的是什么? 是之前的所有节点的信息

