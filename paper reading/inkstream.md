#### ink stream 

InkStream: Real-time GNN Inference on Streaming Graphs via Incremental Update . 

动态图的inference有一些paper 比如inkstream ，实际应用和Motivation更强

https://arxiv.org/pdf/2309.11071.pdf

问题:  只有minor updates in graph也需要取所有neighbor, GPU内存装不下, fetch all neighbor  from CPU 很费时. 

方法: 只fetch受影响的节点. 

#### 介绍

figure1 说明subgraph construction 占据了50%.

figure3 说明受影响的只有1%. 但是只算 affected area 也要几秒.

#### 4 实验设置

学习  T-gcn: A sampling based streaming graph neural network system with hybrid architecture

随机选时间戳. 





GraphTensor: Comprehensive GNN-Acceleration Framework for Efficient Parallel Processing of Massive Datasets
