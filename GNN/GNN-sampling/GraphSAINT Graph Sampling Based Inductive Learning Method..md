发表在ICLR2020上的论文GraphSAINT: GRAPH SAMPLING BASED INDUCTIVE LEARNING METHOD》，它也是基于抽样的图神经网络框架。GraphSAINT引入一种基于抽样子图的图神经网络模型，每个minibatch上的计算在抽样子图上进行，不会出现”邻居爆炸“现象，同时作者给出的[抽样方法](https://www.zhihu.com/search?q=抽样方法&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"article"%2C"sourceId"%3A"394747362"})在理论上有无偏性和最小方差。此外该模型把抽样和GNN解耦开来，可以基于论文提出的抽样方法，使用其他GNN模型如GAT、JK-net等。

## **1.动机**

目前的大部分图神经网络模型主要集中在解决相对较小的图上浅层模型，如GraphSAGE、VRGCN、ASGCN、FastGCN等。在大图上训练深度模型仍然需要更快的方法。

影响图神经网络训练的更深的一个主要问题是”邻居爆炸"问题。GraphSAGE、VRGCN、PinSage需要限制邻居采样数量到很小的水平；FastGCN和ASGCN进一步把邻居膨胀系数限制到了1，但同时也遇到了规模化、精度和计算复杂度方面的挑战。

ClusterGCN通过先对图进行聚类，然后在小图上进行图神经网络训练的方式，加速了训练过程，不会遇到“邻居爆炸”问题，但引入了偏差。

## **2. 模型**

GraphSAINT先经过抽样器从G中抽出子图，然后在子图上构建GCN。

其中有两个关键的东西是SAMPLE抽样器和两个[正则化系数](https://www.zhihu.com/search?q=正则化系数&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"article"%2C"sourceId"%3A"394747362"})和。它们一起保证了无偏性和方差最小性。

### **2.1 无偏性**

### **2.2 方差最小抽样**

### **2.3 抽样方法**

作者给出了三个抽样器：

### nodesampler 

思路很简单，对每个节点的度进行标准化，**即每个节点被采样到的概率=每个节点的度/图上所有节点的度之和；**

### **edgesampler**

思路也比较直观，对每个每个edge被采样到的概率定义为

edge的source和target node的degree的倒数之和/所有edge的上述定义下的计算结果之和

然后使用edges对应的source和target nodes 构建subgraph

### rw和mrw sampler

就是deepwalk里用到的采样方式，deepwalk有随机采样和带权采样两种，随机采样就是neibour完全随机选择，带权采样则是考虑了edge的weight之后做sampling，简单来说就是和当前节点之间edge weights越大的邻节点被采样到的概率越大。

和cluster-gcn中做non-overlap static 的community detection的方式不同，graphsaint所做的事情可以理解为overlap dynamic的community detection，每个batch下得到的subgraph（或者说一个community）是动态产生的，并且我们如果最终把所有产生的communities 进行合并可以发现得到的是一个overlap的community detection结果。

直觉上，这种动态重叠的社区发现的方法更合理，因为实际上是从更多的视角上去对graph做subgraph的抽取，可以类比为ensemble中有放回和不放回的bootstrap的策略的差异。

## **4.结论**

GraphSAINT是一种基于抽样子图的图神经网络训练方法。通过对minibatch上的激活输出和loss的估计偏差和方差的分析，提出了正则化和抽样方法来提高训练的效果。因为每个batch的训练都是在子图上完成的，克服了“邻居爆炸”的问题。与目前的SOTA模型对比，不仅在精度上有提升，在速度上也很快。