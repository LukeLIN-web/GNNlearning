

###   lec1

•节点分类: 预测一个节点的属性. 例如 给用户和商品分类 ,AlphaFold

•边预测: 预测是不是有边. 例如 知识图谱补全, 好友关系分析,推荐系统

•图分类:  例如判断分子是哪种分子

•聚类(clustering): 判断节点是否可以构成子图 例如 社交圈判断

•图生成: 寻找新药物

•图演化: 物理模拟 



### lec2 传统特征

#### node level

##### centrality

 Node centrality 𝑐𝑣 takes the node importance in a graph into account

Eigenvector centrality: the sum of the centrality of neighbouring nodes:

 Betweenness centrality: ▪ A node is important if it lies on many shortest paths between other nodes. 最短路经过它的数量

 Closeness centrality: ▪ A node is important if it has small shortest path lengths to all other nodes  别的节点到你的距离加起来最短. 

 Clustering coefficient counts #(triangles) that a node touches.三角形越多 这个系数越大.

GDV counts #(graphlets) that a node touches.  Graphlet degree vector的意义在与它提供了对于一个节点的本地网络拓扑的度量，这样可以比较两个节点的GDV来度量它们的相似度。由于Graphlet的数量随着节点的增加可以很快变得非常大，所以一般会选择2-5个节点的Graphlet来标识一个节点的GDV。

获得节点feature的方法有:

▪ Importance-based features: 

1 Node degree 2  Different node centrality measures 

▪ Structure-based features: 

1 Node degree 2 Clustering coefficient 3 Graphlet count vector

#### link-level

▪ Distance-based feature 

▪ local/global neighborhood overlap

#### graph level

Kernel methods are widely-used for traditional ML for graph-level prediction.

有两种kernel :  Graphlet Kernel [1]  和 Weisfeiler-Lehman Kernel [2]

##### Graphlet Kernel

目标: Design graph feature vector 𝜙  

核心思想: Count the number of different graphlets in a graph.

问题: if 𝐺 and 𝐺 ′ have different sizes, that will greatly skew偏度(等于distort) the value. 

 解决方法: normalize each feature vector

缺点: Counting graphlets is expensive! 如果枚举的话是 NP hard的,  所以我们需要一个更有效率的graph kernel

##### Weisfeiler-Lehman Kernel

目标: 设计高效的graph feature vector 𝜙  

核心思想: Use neighbourhood structure to iteratively enrich node vocabulary.

Color refinement 算法, 一开始每个node一个颜色, 后面迭代refine. After 𝐾 steps of color refinement 就可以 summarizes the structure of 𝐾-hop 邻居

 In total, time complexity is linear in #(edges).  和边的数量成线性关系

Graph is represented as Bag-of-colors

### lec3 node embed

从一个节点出发, 随机行走 

最简单的想法是固定步长, unbiased random walks starting from each node

目标:  Embed nodes with similar network neighborhoods close in the feature space.

#### node2vec 算法

 1) Compute random walk probabilities 

 2) Simulate 𝑟 random walks of length 𝑙 starting from each node 𝑢 

 3) Optimize the node2vec objective 用随机梯度下降

线性时间复杂度  上面三步可以独立并行

PPT里还有一些其他的random walk 算法 

核心思想: Embed nodes so that distances in embedding space 就反映出节点的相似度.

没有一种method 可以在所有情况下都更好. 必须根据具体应用来选择node 相似度的算法

#### embedded 整个图

方法1  把所有节点的embedding 求和或者求平均,simple but effective

方法2  引入一个virtual node

方法3  匿名walk embedding

▪ Idea 1: Sample the anon. walks and represent the graph as fraction of times each anon walk occurs. ▪ Idea 2: Learn graph embedding together with anonymous walk embeddings.

### lec4 pagerank

In this lecture, we investigate graph analysis and learning from a matrix perspective. 

 Treating a graph as a matrix的好处

▪ 确定节点的importance 通过 random walk (PageRank) 

▪ 获得节点的 embeddings via matrix factorization因式分解 (MF) 

▪ View other node embeddings (e.g. Node2Vec) as MF 

Random walk, matrix factorization and node embeddings 都是紧密相关的! 

我们讨论下面几个 Link Analysis approaches 来计算graph中节点的重要性: 

1.  PageRank 
2. Personalized PageRank (PPR) 
3. Random Walk with Restarts

#### page rank

利用link 结构来measure 节点的重要性

rank vector r 是随机邻接矩阵的一个eigenvector: We can now efficiently solve for r! ▪ The method is called Power iteration

开始每个重要性都是1/节点个数,  然后一步步迭代, 矩阵乘法. 大约50次迭代就足够 估计出有限解. 

两个问题 

1  有的页是dead end的. 这是一个问题, matrix 不是列随机的.

解决方法: 随机瞬移到其他所有页面. 实现, 就是调整邻接矩阵.   

2 spider traps , 所有的out link都在一个group里, 最终这个trap会吸收所有的importance . 比如最简单的就是一个自环节点.  这导致score不是我们想要的. 

解决方法: 引入概率, 每一步, beta 概率 随机选一个link然后follow a link , 1-beta 概率 瞬移到随机一页.  beta通常 0.8-0.9. 

Personalized PageRank: 瞬移到一些指定的节点而不是所有节点

#### Random Walk with Restarts

推荐系统

节点的proximity怎么比较?

Personalized PageRank: ▪ Ranks proximity of nodes to the teleport nodes S

Proximity on graphs: ▪ Q: What is most related item to Ite Q? ▪ Random Walks with Restarts ▪ Teleport back to the starting node: 𝑺 = {Q}

##### 随机行走

Idea  每个节点一样重要 ▪ Importance gets evenly split among all edges and pushed to the neighbors: 

给出查询的节点, 可以模拟一次随机行走: 

▪ Make a step 随机到一个邻居并且记录下来(visit count) 

▪ alpha的概率, restart the walk at one of the QUERY_NODES 

▪ 有最多次 visit count的节点就有最大的proximity to the QUERY_NODES

方法的优点:  Because the “similarity” considers: 

▪ Multiple connections ▪ Multiple paths ▪ Direct and indirect connections ▪ Degree of the node

#### 因式分解

DeepWalk and node2vec have a more complex node similarity definition based on random walks 

 DeepWalk is equivalent to matrix factorization of the following complex matrix expression:

通过矩阵因式分解和随机行走， 可以有node embedding  比如 deepwalk或 node2vec 

缺点： 

1. 不能获得不在训练集中的node embedding。 需要重新计算所有node embedding
2. 不能发现结构的similarity，会有截然不同的embedding
3. 不能充分利用node edge graph的features

解决方法：  GNN， deep representation learning

### lec5 节点分类和标签传播

Main question today: 网络有一些标签, 怎么assign 标签给其他的node?

一种是lec3提到的node embedding , 今天讨论另一种框架: 消息传递

因为网络中有correlation.

我们会讨论三个技术   relational 分类, iterative 分类和 belief propagation.

homophily: 节点之间会相似, 相似爱好的人会有更多联系. 

节点的联系, 又会影响节点的属性. 

#### 半监督二分类

 三种方法  relational分类, iterative 分类, 和 correct and smooth

#####  relational分类

 把邻居节点的label加权求和. 

难点:  不收敛, 模型没有利用节点的feature信息=> 所以需要iterative 分类 

##### iterative 分类 

训练两个分类器, base 分类器根据attributes 预测.  

relational 分类器根据邻居节点的label和attributes  预测label

可以提高 collective 分类

##### correct and smooth

2021年9月SOTA的分类方法. 

分为三步

1. 训练base 预测器 (可以很简单, 比如就MLP多层感知器)
2. 用base预测器来预测所有节点的soft label(每个类 的可能性)
3. 利用图的结构 修改2中的预测 . 分为correct和smooth.  correct的原理是: 一般base预测的错误和图的边是正相关的, 所以一个节点的error会传播到其他error. 

### lec6 GNN1 模型

shallow encoder的缺点:

1. 需要空间, 没有共享参数, 每个node有自己的embedding
2. 不能面对新的node, 输入的数据包含测试集的数据,训练过程能够看到这些数据,所以是transductive直推式学习 不是inductive
3. 不能结合node 的feature

用GNN 多层, 可以解决分类, 预测link, 关系探索, 网络相似性等问题. 

#### 深度学习基础

常用的分类loss 是cross entropy CE交叉熵 

学习率 LR, 是一个超参数, 控制gradient step的大小

当 validation set没有变化的时候,我们就停止训练. 

每个点都计算太慢, 所以用随机梯度下降, 每一步选不同的minibatch. epoch 就是整个dataset 全部过了一遍. 

SGD是full gradient 的无偏估计. 

##### back propagation 

链式法则, 获得梯度. 

前向传播, 

从输入开始乘法, 慢慢得到最后的loss

##### 引入非线性

ReLU,  rectified linear unit, 就是 max(x,0)

sigmoid , 就是 1/(1+e^-x)  值域在 0到1 

#### 图深度学习

##### naive approache

直接把临接矩阵feed into DNN, 问题是参数多, graph大小不同就不适用.

利用CNN ? 图是非常多变的, 没有固定的滑动窗口来卷积. 

所以提出了新的网络: 

#### 图卷积网络GCN

每个节点根据邻居, 定义了一个计算图 

##### matrix formulation

Many aggregations 可以通过稀疏矩阵的运算加速.

有讲到degree一样的sparse matrix multiplication吗? 好像没有

不是所有GNN都可以表示成矩阵形式, 比如aggregation 函数特别复杂的时候. 

The same aggregation parameters 可以被所有节点分享. 而且模型可以泛化到没见过的节点. 

**平均法**

每个节点, 收集来自邻居传递的信息, 然后汇总后更新自己.

每个节点和它邻居都是相似的, 那么每个节点就等于邻居节点的平均值.

**加权平均法**

每个节点和邻居的关系强度是不同的, 考虑到边的权重关系, 只需要将邻接矩阵 A 变为有权图

**添加自回环**

前面提到的平均法, 加权平均法都忽略了自身节点的特征, 故在更新自身节点时, 一般会添加个自环,把自身特征和邻居特征结合来更新节点

**归一化**

不同节点, 其边的数量和权重幅值都不一样, 比如有的节点特别多边, 这导致多边(或边权重很大)的节点在聚合后的特征值远远大于少边(边权重小)的节点. 所以需要在节点在更新自身前, 对邻居传来的信息(包括自环信息)进行归一化来消除这问题

#### GNN CNN和transformer

CNN可以看作一种特殊的GNN, 邻居的大小是固定的, ordering 也是固定的. 

CNN不是等价交换permutatino的, 改变像素的顺序会有不同的输出. 

##### transformer

NLP很有用, 序列处理的问题上是最受欢迎的一种模型. 

transformer 也可以看作一种特殊的GNN, 是在一个全连接的word图上. 

### lec7 GNN2 Design Space 

用pyG 解决一个实际问题. 发布一个blog post. 

##### 什么是好的博客?

一步步解释 图学习技术, 假设你的读者熟悉ML, 但是不熟悉 PyG

可视化, gif > image > Text , 可视化越多越好

code snippets要有, 

链接到colab 来复现结果. 

应用在哪里?

推荐系统, 分子分类, 论文分类, 知识图谱, 产品购买分类, 蛋白质结构

在哪里可以找到模型? OGB leaderboard 和top ML 会议:  KDD, ICLR, ICML , neurlPS WWW, WSDM 

deep graph encoder , 就是GCN , 然后activation function, 然后 正则化, 比如dropout,  然后GCN, 

every node base on neighbor确定了一个计算图, 可以传播信息来计算node feature .  aggregate 信息. 

GNN Layer = Message + Aggregation   一层GNN layer , 就是把一些vectors 压缩成一个vector . 

message 可以用一个线性层, aggregation 可以用max, mean或者sum 

怎么用代码写出这些呢? colab3 的难点就在这里. 

##### message 计算

每个node 计算出一个message. 

例如, 一个linear layer, 把node feature 乘上一个权重矩阵. 

##### aggregation

从邻居那里聚合message. `sum( [ message for message in neighbor ] )`

例如,  sum,  max, min 就是aggregation. 

##### 考虑自己

除了邻居还要考虑自己的信息, 这时message 用另一种函数计算, aggregation, 用concatenation串联 或者summation

##### 非线性

 activation , 可以 add expressiveness, 一般用ReLU或  Sigmoid .  加到message或aggregation都可以. 

#### 经典的GNN 层

##### GCN

message:  乘上一个权重矩阵然后根据入度正则化

aggregation:  用sum, 然后activation

##### GraphSAGE

有两个贡献，一是提出了更多的聚合方法（mean/lstm/pooling），二是对邻居信息进行多跳抽样

message 用AGG计算, aggregation分两个stage ,先agg 邻居,再agg自己. 

`权重矩阵 串联( 自己,(聚合邻居))`

L2正则化, 可以在每一次apply, 作用是有一致的scales, 有时候可以提高performance. 

##### Graph Attention Networks

GCN 和 GraphSAGE中, 每个邻居是一样重要的, 权重系数和图的结构属性正相关(具体的来说, 和节点的度正相关)

GAT中,  attention关注重要的信息, 计算一个attention coefficient,  normalize 后得到最终的权重. 加权求和.

multi-head是在channel上进行切分分别计算最后拼接/平均

优点: 计算高效, 计算 attention coefficient可以并行, 存储更高效, 参数固定不再和图大小增长, 

我们可以考虑很多现代的DL技术, 比如 batch normalization来稳定训练, dropout来防止过拟合, attention/gateing来控制一个消息的重要性

##### batch normalization

dropout 随机把一些神经元置为0 

GNN中 , linear layer  message function 应用dropout 

#### 层次相连问题

##### 太多GNN层问题

 over smoothing, 所有embedding 收敛到同一个值.  为什么会这样?

堆叠太多层- >  receptive 域高度重叠, 也就是大家hop过来都是这些节点 -> node embedding 高度相似 

##### 解决方法

1.不要加太多层, 分析必要的receptive field. 

那怎么具有足够的表达能力呢?  在每层内修改. 

2. 增加不传递message 的layer 

一个GNN 不一定全是GNN layer  , 可以在前面后面加一些MLP layer .  实际中很有用. 

当encoding很重要的时候,  比如node表示images/texts 时, 就加pre processing layers. 

reasoning/transformation 重要的时候,  比如graph classification, knowledge graphs,  加Post-processing layers

如果必须需要很多层呢?

那就add skip connection: increase the impact of earlier layers on the final node embeddings, 在GNN中增加shortcut

这样可以有更多可能性, 就有混合 浅GNN 和深GNN . 

###  lec8 augmentation 

为什么要增强图?

1. features , 缺乏features 
2. 图结构, 可能太sparse 导致消息传递太没效率或者太dense导致消息传递开销太大, 太large 导致显存不够.  

所以, 输入的图往往不是最适合的计算图

#### 增强的方法

##### 特征增强

可能没有node feature

标准的方法是

1. 给nodes分配常量值. 
2. 给node分配unique ID, one hot vectors, 这个方法更expressive, 但是不能泛化到unseen node, 计算的开销也更大, 可以用在小图和transductive图(没有新的node).  

可能GNN 学不到一些特定的结构, 比如环的长度 

学不到就直接加进去, 比如 Node degree § Clustering coefficient § PageRank § Centrality

##### 结构增强

**太sparse  ->  Add virtual nodes / edges**

怎么加边?  Connect 2-hop neighbors via virtual edges  例子是Bipartite graphs

加一个虚拟node, 超级源点. 这样所有节点都距离为2 这样可以大大加快稀疏矩阵中的消息传递. 

**太dense  -> Sample neighbors when doing message passing**

随机采样一些节点的邻居, 有一些邻居不计算了, 来减少计算量. 在实践中也work的很好

**太large  -> 采样子图**

#### 用GNN预测

3个level , 节点, edge, graph. 

global pooling 会丢失结构信息.  -> 解决方法, 分层aggregate 所有的node embedding 

实践中可以用 diffpool :  利用两个独立的GNN, 分层计算.  联合训练 GNN1和GNN2

regression就是label有连续的值, 分类是label离散的. 

分类可以用CE 交叉熵作为损失函数, 

regression就用 mean squared error MSE  , 也叫L2 loss 

##### 数据集分割

Fixed split: 只切分一次

图的数据集切分很特殊, 会影响message passing- > 影响节点的embedding

解决方法:

1.  transductive setting, 输入图可以被 所有splits 看到, 只分割节点的label. 在训练时用整个图计算embedding,然后用部分label训练.  评估时也是, 整个图计算embedding, 然后 用部分label evaluate.
2.  Inductive setting, 不同split之间的边切掉, 一个split就是一个独立的图, 在训练时用单个split的图计算embedding,然后用这个split的label训练. validation也是. 

在训练, 验证和测试set中都能看到整个图. 叫做transductive setting.  我们只分割标签. transductive setting 只能用于 node/edge 预测任务

或者 inductive setting, 把边也拆开. 获得多个独立的图. 用独立的图各自计算嵌入.  inductive可以用于node/edge/graph 任务. 比如 图分类, 因为必须test没见过的图.  用整个图计算嵌入, 用几个节点 train, 然后 另外几个节点evaluate.  这种是不能处理图预测任务的. 

##### 预测edge

Link Predict

这是无监督/自监督任务, 要自己创建label.  分割边两次

step1 : 边分为两种

inductive : 把边分成两种类型, message edge 和supervision edges . supervision edge不输入GNN 

 message edges(用来消息传递)和supervision edges(用来计算) supervision edges 作为model 做出预测的label, 不会被fed into GNN. 

step2: edge split,  

transductive : 是默认的设置, 把边分成四种, Training message edges ,Training supervision edges, Validation edges ,Test edges.  这是非常trick和复杂的, 不过PyG, DeepSNAP和GraphGym 有很完善的库

DeepSNAP provides core modules for 预测 pipeline 

GraphGym 进一步实现了整个 pipeline to facilitate GNN design

#### 出现问题了怎么解决

general tips:

1. 数据处理很重要, 因为node attributes 有些很大可能几千几万, 需要用正则化 .attribute是属性吧，比如degree，中心性这些.
2. 优化器 ,ADAM 是相对比较robust的
3. 激活函数, ReLU 比较好,§ Other alternatives: LeakyReLU, SWISH, rational activation,  输出层不要用激活函数
4. 每一层包含bias term
5. embedding 维度, 32 , 64 , 128 比较好

##### 怎么debug

loss/accuracy 不收敛: 

1. 检查pipeline, 比如有没有zero grad 
2. 调整超参数
3. 检查权重参数初始化

过拟合: 调试loss func, 尝试可视化

### lec9 GNN理论

这些GNN 模型,  GCN, GAT, GraphSAGE, design space.  

expressive power (ability to distinguish different graph structures) 有什么区别?

怎么设计一个最expressive 的GNN model?

GCN (mean-pool) [Kipf and Welling ICLR 2017]  Element-wise mean pooling + Linear + ReLU non-linearity

GraphSAGE (max-pool) [Hamilton et al. NeurIPS 2017]MLP + element-wise max-pooling

GNN是怎么捕捉 local neighborhood structures 的?

关键是计算图

不同的local neighborhoods  决定了不同的计算图

单射injective .  当*a* ≠ *b*时，*f*(*a*) 一定≠ *f*(*b*)

最有expressive 的GNN 应该map subtrees to the node embeddings injectively. 

如果每一步neighbor aggregation都是单射的, 那么GNN 就可以完全分辨不同的子树结构. 

#### 设计最强大的GNN

如上所述, 最重要的就是injective aggregation function. 

GCN mean , GraphSAGE max 都不是单射的.  所以不够powerful

Graph Isomorphism Network (GIN) [Xu et al. ICLR 2019]   

Apply an MLP, element-wise sum, followed by another MLP. 是单射的, is THE most expressive GNN in the class of message-passing GNNs!  

GIN 用 NN to model 单射哈希函数

GIN 和 WL graph kernel 的expressive 差不多, 

### lec10 Knowledge Graph Embeddings

#### 异质图

每个边的类型可能不同

##### Relational GCNs 

使用不同的NN 权重来表示不同的relation 类型. 



##### Knowledge Graphs 





##### Embeddings for KG Completion



### lec16 advanced topic

