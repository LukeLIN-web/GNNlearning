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

