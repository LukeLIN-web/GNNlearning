## 作者的访谈.

https://www.youtube.com/watch?v=lnvI8stkPOU

motivation : unified  framwork for his own research . as general as possible.

core difference between gnn vs cnn, rnn?

gnn可以翻译成 cnn, rnn,

gnn , 要考虑别的节点的feature.  cnn, filter范围内也会考虑.    不过gnn没有一个fix structure .

image synthesis from textual input as gnn

gnn可以作为cnn和rnn结合, (但是gnn和rnn有什么共性呢? 不太懂. )

gnn怎么做图像增强?

把bounding box看作node,  可以连接然后infer relations of them. 可以通过站位分析出人和bike 的关系

gnn可以做信号滤波吗?

gnn can model classic NN we already have. 如果有fully connected graph, 就可以当作 transformer.  可以计算nodewise or pairwise attention score to each node.

### gnn的计算复杂度如何?

主要为了sparse data, (dense data 全连接图表现不佳. )

对于 irregular structured data , 只能gnn.

他也写了一篇gnn autoscale.

计算复杂度主要和边有关.

message 复杂度是多少?

aggregate 复杂度是多少?

空间复杂度, 主要取决于你怎么定义你的gnn operator. 和node数量无关. 可能和edge数量有关

一些operator不用materialize message.

一开始是full batch, 同时optimize 所有node.

### 怎么minibatch?

optimize the model parameters only based on these current node in our mini batch

每个node 会和  k-hop 邻居相关, k是gnn的layer数.   gnn 很deep的时候, 性能会很差,而且可能需要aggregate from nodes in your complete graph.  这样就没有mini batch的意义了. 

解决的方法是

1. neighborhood sampling.  sample 5 or 10 neighbor per iteration.  k-hop
2. subgraph sampling.  a  set of nodes highly connected to each other,  just perform message passing 在this set of mini batch nodes  

100k 小规模graph,比如分子, 怎么把小图接成大图? 如果大小不一样怎么办?  有一个叫batch的object,去index node feature是属于哪个batch, 第4个node属于第2个batch.

1 million。可以单个gpu 不过需要一些scalablity technology 优化显存,    ≤ 1 hour

100 million, 需要多个gpu.

这也和你gnn layer层数还有 feature 维度有关.

### 数据集

point cloud可以用在自动驾驶.

推荐系统可能也有潜力.  传统的推荐方法可能没有考虑各种复杂的关系.

pyg的优点: 不用考虑数据是怎么 feature encode的.  不过在工业界这是主要问题.  一些feature 可能irrelevant.  有的passing messag就会自动生成, 而你作为输入就可能重复了. 有很多方式encode, 很难确定哪种最好.

可以用MLP测试你的feature encoder.

 按时间划分数据集, 2018 年作为train, 2019年作为valid



PyG 都是tensor的概念, 没有graph forward pass. 

graph 这个object. 

为啥都用 pyg, 不用dgl呢?

少了灵活度, graph为中心,  和torch兼容不好. 处理一些low level的时候, dgl 就不方便了.

pyg 社区比较好, 而且是最早的, contributor多, 和斯坦福合作,  dgl就是亚马逊推广. meta就不会用, 公司有竞争.  











