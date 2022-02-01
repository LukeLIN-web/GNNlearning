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

