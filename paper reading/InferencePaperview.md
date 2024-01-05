https://jeongseob.github.io/readings_mlsys.html

### SHEPHERD: Serving DNNs in the Wild

nsdi 2023 , 一作张弘. 延迟要求: 50-500ms

NSDI 2023有哪些值得关注的文章？ - 孙挺Sunt的回答 - 知乎
https://www.zhihu.com/question/543376768/answer/3119939466

clock work  osdi 2020 ,   online global policy. 

nexus , 周期性的, per-stream policy.

calculat CV for each group streaming, 100-1000个stream, 就会比较稳定而且可以预测到达pattern.

怎么group呢? 等吗? 

在每个group 中 online serving. 

有两种,  forward inference-based (inf-based) approach and backend updatebased (upd-based) approach. 来更新dynamic graphs.

inf base, 只在收到的时候改变图结构. aligraph.

https://github.com/zheng-yp/DecoupledDGNN

没有代码:

1. Efficient Scaling of Dynamic Graph Neural Networks. SC'21

2. SPEED: Streaming Partition and Parallel Acceleration for Temporal Interaction Graph Embedding

3. Redundancy-Free High-Performance Dynamic GNN Training with Hierarchical Pipeline Parallelism

4. Cache-based gnn system for dynamic graphs
5. STAG: Enabling Low Latency and Low Staleness of GNN-based Services with Dynamic Graphs https://arxiv.org/pdf/2309.15875.pdf
6. DynaGraph: Dynamic Graph Neural Networks at Scale
7. Approximate Caching for Efficiently Serving Diffusion Models 
8.  https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9644829  PCGraph. 也是加速gnn inference.

survey : https://arxiv.org/pdf/2306.14052.pdf   . A Survey on Graph Neural Network Acceleration: Algorithms, Systems, and Customized Hardware

#### workload调查

大部分都是从硬件加速的角度来的.  都是静态到达, 不是慢慢到达, 没有从进入pattern来考虑.  

1. FlowGNN: A Dataflow Architecture for Real-Time Workload-Agnostic Graph Neural Network Inference.  gatech,  是用message ,embedding, FPGA来加速 feature加载. 角度不同.    用的是普通的dataset.
2. 很多都是用Channel Pruning .  accelerate GNN inference by pruning the dimensions in each layer with negligible accuracy loss. viktor@USC的工作.  还有裁剪 feature .
3. GNNIE: 数据集就是普通的, 不是动态到达的. 
4. Aligraph , It provides a performance guarantee of sampling P99 latency in 20ms on large-scale dynamic graphs.   几百个worker, 在淘宝数据集上,  他们有很多模型, 测了模型精度.
5. Automatic Generation of High-Performance Inference Kernels for Graph Neural Networks on Multi-Core Systems 是编译器优化
6. Bottleneck Analysis of Dynamic Graph Neural Network Inference on CPU and GPU . 分析了问题
7. **DGI: Easy and Efficient Inference for GNNs** 也就是静态的数据集. 他提出了快速翻译代码到layer wise.和我们是正交的. 
8. SERVING GRAPH COMPRESSION FOR GRAPH NEURAL NETWORKS 也是静态的数据集. 他们证明acc loss不大. 

#### GNN serving in a cluster 

好像没有讨论过

quiver有 cluster, 三个server,  their scalability becomes limited by these network bottlenecks.

quiver latency 就是测 sample +  to device + forward的时间. thoughtput  就是batch size / 最后的end time- 第一个end time.

### serving+速的方法





