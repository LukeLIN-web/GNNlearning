## Poster

**AutoScratch: ML-Optimized GPU Cache Management**

nvidia

propose a Machine Learning (ML) based framework called AutoScratch to automatically discover and optimize the GPU's L2 residency settings

develop two versions of AutoScratch, AutoScratch-RL harnessing reinforcement learning (RL) and AutoScratch-EA leveraging a state-of-the-art evolutionary algorithm (EA

integrate AutoScratch with NVIDIA's TensorRT framework to fully automate the optimization pipeline for arbitrary DL inference applications

**RevBiFPN: The Fully Reversible Bidirectional Feature Pyramid Network**

Activation 内存占用巨大. 为什么? 

在训练一个多层神经网络时，在前向传播中，每一层的中间结果都要被存下来用于计算反向传播的梯度。这些中间结果，又被叫做「激活值」（activation)，实际上占据了大部分的内存消耗.

Existing reversible methods, however, do not apply to multi-scale feature fusion and are therefore not applicable to a large class of networks. Bidirectional multi-scale feature fusion promotes local and global coherence and has become a de facto design principle for networks targeting spatially sensitive tasks e.g. HRNet and EfficientDet. When paired with high-resolution inputs, these networks achieve state-of-the-art results across various computer vision tasks, but training them requires substantial accelerator memory for saving large, multi-resolution activations. These memory requirements cap network size and limit progress. Operating across resolution scales, the RevSilo alleviates these issues. Stacking RevSilos, we create RevBiFPN, a fully reversible bidirectional feature pyramid network.

https://mlsys.org/media/PosterPDFs/MLSys%202023/2474.png?t=1684458867.827531

看不懂. 好像就是优化算法. 

**Practical Edge Kernels for Integer-Only Vision Transformers Under Post-training Quantization**

梯度裁剪

## 论文

PIPEFISHER: EFFICIENT TRAINING OF LARGE LANGUAGE MODELS USING PIPELINING AND FISHER INFORMATION MATRICES

用K-FAC 而不是普通梯度下降, 没太看懂. 

### salient++

[Communication-Efficient Graph Neural Networks with Probabilistic Neighborhood Expansion Analysis and Caching](https://mlsys.org/virtual/2023/poster/2469)

就是SALIENT 继续改代码. 

calculate the probability that any vertex will be present in the sampled L-hop expanded

解决了SALIENT has the drawback of replicating the entire data set on each machine.

 看 figure3 , 和quiver区别就是 , 不同的GPU存储的比例不同. 

https://github.com/MITIBMxGraph/SALIENT_plusplus_artifact

#### 实验

用的是cluster,16个machine. 16个A10G.

papers100M , 8机. 3s 一个epoch.

Sufficient space for datasets: ogbn-arxiv (1.0 GB needed); ogbn-products (4.0 GB needed); ogbn-papers100M (200.0 GB needed)
        [Info] Insufficient space for datasets: MAG240 (600.0 GB needed)

Partitioning  for larger datasets like ogbn-papers100M it can take several hours and substantial memory to complete successfully. A machine with 512GB of RAM should be able to partition the larger datasets without running out of memory.

我设置了 2个node

`python download_datasets_fast.py --simulation_downloads_only --skip_confirmation` 下载非常慢, 

他自己写了cpp的 sampler. 

安装了matplotlib  , prettytable和nvtx

### HYPERGEF: A FRAMEWORK ENABLING EFFICIENT FUSION FOR HYPERGRAPH NEURAL NETWORK ON GPUS

ucsd Jishen Zhao老师的

edge-split partition scheme to achieve higher efficiency and better workload balancing

 introduces a shared memoryaware grouping scheme to reduce writing conflicts

Edge-split Partition :有的边承担了太多顶点, 就要拆分 ,

为什么有的边承担了这么多顶点?  超图, 一个边有不止两个点. hyperedges can be connected to multiple vertices 

Thus, the GNN computation on hypergraphs can be processed with two message passing steps, *i.e.,* messages from vertices to hy- peredges and hyperedges to vertices

figure2 的c是什么意思? partition是啥意思? 

挑战:  fusion 难, 因为 dynamic shape and sparse data attributes.

#### 4.3 Shared Memory Aware Grouping

挑战: 目前方法a coalesced row caching technique that is restricted to the vertex-balanced partition. 或者 under the condition of GNN’s neighbor-group partition
