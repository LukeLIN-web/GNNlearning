https://jeongseob.github.io/readings_mlsys.html

Flash-LLM: Enabling Cost-Effective and Highly-Efficient Large Generative Model Inference with Unstructured Sparsity  阿里巴巴和悉尼大学的VLDB24

四个矩阵乘法, 用unstructured weight pruning来降低内存消耗.

先把稀疏的变成dense的. For each iteration, each thread block loads 𝐴𝑇𝑖𝑙𝑒 (shape [𝑀, 𝐾]) in sparse format and 𝐵𝑇𝑖𝑙𝑒 (shape [𝐾, 𝑁]) in dense format from global memory. 𝐴𝑇𝑖𝑙𝑒 is then transformed to dense format with our efficient *Sparse-to-Dense Transformation* strategy.   Finally, each thread block consumes the dense data in shared memory and generates the output tile through tensor core computations.

---

 ICML'23 BPipe: Memory-Balanced Pipeline Parallelism for Training Large Language Models  在GPU之间传输 activation. 

FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU  , 可以搜索 存储和访问tensor的方式.  **GPU端仅仅进行一个Transformer layer的计算，一旦计算完成就对KVcache、激活、weight权重参数进行checkpoint，也是流水化overlapping的将数据转移到CPU DRAM和磁盘**   推理延迟已经拉长到3.3个小时了（这也限制它的使用场景，仅适合离线批量计算场景）对细节的持续思考（比如进一步量化压缩改进CPU-GPU访存带宽、设计自动方法寻找最优的优化参数）来改进系统

GPipe 可以解决单卡显存不足的问题。 

PipeDream 通过快速反向传播, 节省显存, 缺点是需要维护多个版本的模型参数, 不适合参数多的LLM模型. 

Megatron-LM的第二篇论文, 给device 1分配 层1\2\9\10, 而不是1-4层, 降低bubble 率. [*Memory*-Efficient *Pipeline*-Parallel DNN Training](https://zhuanlan.zhihu.com/p/650744349)  

INFaaS: Automated Model-less Inference Serving  ,Stanford  , 动态地选择不同属性、大小的模型model-level autoscaling，利用[VM-level horizontal autoscaling]. 

[FasterTransformer](https://link.zhihu.com/?target=https%3A//github.com/NVIDIA/FasterTransformer) by NVIDIA  层优化：融合进单一kernel；activation cache；重用每一层activate/output的内存buffer；tensor并行和pipeline并行、通信优化；MatMul底层实现方式自动调整；

STI: Turbocharge NLP Inference at the Edge via Elastic Pipelining, ASPLOS 23, UVA  fleix xiaozhu lin.   模型分片。STI 将模型参数作为独立可调的分片进行管理，并分析它们对准确性的重要性。其次，使用预加载缓冲区进行弹性管道规划。STI 实例化 IO/计算管道，并使用一个小缓冲区进行预加载分片，以引导执行，而不会在早期阶段停滞;它根据分片对资源弹性执行的重要性明智地选择、调整和组装分片，从而最大限度地提高推理准确性。   we implement the decompression in separate 200 SLOC of C code using OpenMP

Serving DNNs like Clockwork: Performance Predictability from the Bottom Up, osdi 2020,   online global policy. 

 MnnFast: a fast and scalable system architecture for memory-augmented neural networks 为了减少内存带宽消耗，我们提出了一种新的基于列的流式算法，该算法最大限度地减少了数据溢出的大小，并隐藏了大部分片外内存访问开销。其次，为了降低高昂的计算开销，我们提出了一种零跳跃优化来绕过大量的输出计算。最后，为了消除缓存争用，我们提出了一个专门用于高效缓存嵌入矩阵的嵌入缓存 在FPGA上. 

有两种,  forward inference-based (inf-based) approach and backend update based (upd-based) approach. 来更新dynamic graphs.

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

#### ink stream 

InkStream: Real-time GNN Inference on Streaming Graphs via Incremental Update . 

 https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9644829  也是加速gnn inference.  缓存切分verices。 选择最优的inference 算法

动态图的inference有一些paper 比如inkstream ，实际应用和Motivation更强

https://arxiv.org/pdf/2309.11071.pdf

问题: 改了之后再fetch neighbor 内存不够.

方法: 可以Incremental Update。 怎么incremental update？  只fetch 受影响的node。  快了300x， 感觉能中顶会。

tgnn benchmark。

#### 介绍

figure1 说明subgraph construction 占据了50%.

figure3 说明受影响的只有1%. 但是只算 affected area 也要几秒.

https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9644829  PCGraph. 也是加速gnn inference.

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

如果临时输出较大就可以算子融合。 对于dense layer, 可以堆叠batch 处理.

