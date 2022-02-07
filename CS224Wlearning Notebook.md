### lec2 传统特征

#### node level

##### centrality

 Node centrality 𝑐𝑣 takes the node importance in a graph into account

Eigenvector centrality: the sum of the centrality of neighbouring nodes:

 Betweenness centrality: ▪ A node is important if it lies on many shortest paths between other nodes. 最短路经过它的数量

 Closeness centrality: ▪ A node is important if it has small shortest path lengths to all other nodes  别的节点到你的距离加起来最短. 

 Clustering coefficient counts #(triangles) that a node touches.三角形越多 这个系数越大.

GDV counts #(graphlets) that a node touches.  Graphlet degree vector的意义在与它提供了对于一个节点的本地网络拓扑的度量，这样可以比较两个节点的GDV来度量它们的相似度。由于Graphlet的数量随着节点的增加可以很快变得非常大，所以一般会选择2-5个节点的Graphlet来标识一个节点的GDV。

We have introduced different ways to obtain node features. 

 They can be categorized as: 

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

Key idea: Count the number of different graphlets in a graph.

Problem: if 𝐺 and 𝐺 ′ have different sizes, that will greatly skew偏度(等于distort) the value. 

 Solution: normalize each feature vector

Limitations: Counting graphlets is expensive! 如果枚举的话是 NP hard的,  所以我们需要一个更有效率的graph kernel

##### Weisfeiler-Lehman Kernel

目标: 设计高效的graph feature vector 𝜙  

Idea: Use neighbourhood structure to iteratively enrich node vocabulary.

Color refinement 算法, 一开始每个node一个颜色, 后面迭代refine. After 𝐾 steps of color refinement 就可以 summarizes the structure of 𝐾-hop 邻居

 In total, time complexity is linear in #(edges).  和边的数量成线性关系

Graph is represented as Bag-of-colors

### lec3 node embed

从一个节点出发, 随机行走 

 What strategies should we use to run these random walks? 

▪ Simplest idea: Just run fixed-length, unbiased random walks starting from each node

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

 Random walk, matrix factorization and node embeddings 都是紧密相关的! 

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

Proximity on graphs: ▪ Q: What is most related item to Item Q? ▪ Random Walks with Restarts ▪ Teleport back to the starting node: 𝑺 = {Q}

##### 随机行走

Idea  每个节点一样重要 ▪ Importance gets evenly split among all edges and pushed to the neighbors: 

给出查询的节点, 可以模拟一次随机行走: 

▪ Make a step 随机到一个邻居并且记录下来(visit count) 

▪ alpha的概率, restart the walk at one of the QUERY_NODES 

▪ 有最多次 visit count的节点就有最大的proximity to the QUERY_NODES

方法的优点:  Because the “similarity” considers: 

▪ Multiple connections ▪ Multiple paths ▪ Direct and indirect connections ▪ Degree of the node

#### 因式分解

DeepWalk and node2vec have a more complex node similarity definition based on random walks  DeepWalk is equivalent to matrix factorization of the following complex matrix expression:

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



#### 图卷积网络



#### GNN CNN和transformer

