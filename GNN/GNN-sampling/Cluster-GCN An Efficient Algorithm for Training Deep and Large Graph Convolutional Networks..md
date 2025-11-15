一种收敛更快、内存需求更少、适用于big graph、基于mini-batch的的*GCN*模型

### 相关文献

#### mini batch SGD 每次epoch花费时间长的原因

来源：[Inductive Representation Learning on Large Graphs](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1706.02216)，即**GraphSAGE**

在GraphSAGE中，在第L层layer的一个节点上计算loss需要在第L-1层layer中寻找该节点的[邻居结点](https://www.zhihu.com/search?q=邻居结点&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"article"%2C"sourceId"%3A"359439472"})的embedding，然而，寻找第L-1层layer的邻居节点的embedding又需要在第L-2层layer中去寻求该节点的邻居的邻居的embedding，如此反复，造成了“邻居规模爆炸问题”，因此，其时间复杂度与GCN的深度成指数增长关系。

由于mini-batch SGD使用的邻居采样策略是随机采样，加上在graph非常大且稀疏时，每个节点的“embedding utilization”是非常小的。

#### VR-GCN所需内存大的原因

来源：[Stochastic Training of Graph Convolutional Networks with Variance Reduction](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1710.10568)

VR-GCN虽然提出了一种能够减少邻居节点采样数量的技术，但是它需要在内存中存储所有节点的intermediate embeddings，因此需要 ![[公式]](https://www.zhihu.com/equation?tex=O%28NFL%29) 的内存空间。当面对节点庞大的graph时，这无疑是高昂的代价。

### motivation

（1）使用mini-batch SGD批量训练模型。由于每次反向传播仅基于小批次的梯度来更新模型的参数，所以能够减少内存需求。并且，因为采用了小批次训练的方式，在每一次epoch中能够使用不同批次的数据进行训练，即一次epoch内能执行多次反向传播来update模型参数，加快了收敛整体loss的收敛速度。

（2）设计一种新的采样方法来构造mini-batch，而不是使用GraphSAGE的随机采用方法，从而最大化**embedding utilization**的值。

### 解决方案

想要最大化embedding utilization就意味着我们需要设计一个batch，使得这个subgraph内部的edge尽可能的多. 本文将基于graph的聚类方法来构造每一个batch。graph clustering algorithms的文章来源：[A fast and high quality multilevel scheme for partitioning irregular graphs](https://link.zhihu.com/?target=https%3A//www.researchgate.net/profile/Vipin-Kumar-54/publication/242479489_Kumar_V_A_Fast_and_High_Quality_Multilevel_Scheme_for_Partitioning_Irregular_Graphs_SIAM_Journal_on_Scientific_Computing_201_359-392/links/0c96052b67d84c5d25000000/Kumar-V-A-Fast-and-High-Quality-Multilevel-Scheme-for-Partitioning-Irregular-Graphs-SIAM-Journal-on-Scientific-Computing-201-359-392.pdf)  

### 理论描述

本文使用graph clustering algorithms来进行graph的切分，使得切分得到的每个cluster的内部节点之间的关联关系数量远比不同cluster之间的节点的关联关系数量多（即within clusters links远比between-cluster links多）。这样做的**好处**有;

- **提高embedding utilization**：因为embedding utilization的值就等同于每个batch的within clusters links数量。值得的一提的是，直观来说，每个节点的[邻居节点](https://www.zhihu.com/search?q=邻居节点&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"article"%2C"sourceId"%3A"359439472"})通常也和该节点处在同一个cluster中，随着[layer层数](https://www.zhihu.com/search?q=layer层数&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"article"%2C"sourceId"%3A"359439472"})的加深，可以清楚地发现该方法能够有效避免“邻居数量爆炸”的问题。它的邻居数量始终是有限的。
- 减少误差

### 理论改进

虽然上述的Cluster-GCN已经能够降低内存需求和减少计算时间，但是，不得不承认这个理论至此还是不完善的。原因在本文中已经你被指出：

- 由于丢失了between-cluster links，模型的性能（准确率）也许会受到影响；
- 由于graph clustering algorithms趋向于将相似的节点划分到一个cluster中，如果直接使用一个cluster作为一个batch，在使用SGD进行更新时会导致full gradient的偏向估计。具体来说就是：每个cluster内部的标签分布都趋向于一些特定的标签。

根据以上两个问题，本文提出了一种stochastic multiple clustering approach（随机多重聚类方法），该方法既能保留between-cluster links又能够解决cluster内部标签的“趋向问题”。

具体做法如下:

把图分成 p个clusters, 然后, 一个batch B 不再只是包含一个cluster中的节点，而是随机挑选q个 clusters, 并将它们中的所有节点囊括进同一个batch，然后这些clusters 的links 也被添加回邻接矩阵, 本文通过在Reddit数据集上实验，给出了使用单一cluster作为一个batch和使用多个clusters作为一个batch的准确率收敛图. 证明方法准确率高. 

#### 加深GCN层数

本文还提出了训练“层数更深的GCN”存在的问题，并给出了解决方案。

在文章[Semi-Supervised Classification with Graph Convolutional Networks](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1609.02907.pdf)（original GCN）中，作者指出单纯增加GCN的层数是没有用的，而本文的作者认为，这可能是他们使用的数据集太小，导致过拟合的原因。

本文作者观察到深度GCN模型的优化之所以变得困难，是因为它有可能阻碍来自前几层的信息的传递。在original GCN的设置中，每个节点需要聚合它上一层的邻居的信息，而这个策略在更深层的GCN中是不适用的，因为层数的加深会导致“邻居数量爆炸”的问题。为了解决这个问题，本文指出“距离更近的邻居应该比距离更远的邻居产生的影响更大才对”。因此，针对GCN的聚合方式，本文提出了新的方式：

