作者：孙挺Sunt
链接：https://zhuanlan.zhihu.com/p/664765417
来源：知乎
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

[Transparent GPU Sharing in Container Clouds for Deep Learning Workloads](https://link.zhihu.com/?target=https%3A//www.usenix.org/conference/nsdi23/presentation/wu) 【[TGS](https://www.zhihu.com/search?q=TGS&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"article"%2C"sourceId"%3A"664765417"})】

 Bingyang Wu and Zili Zhang, *Peking University;* Zhihao Bai, *Johns Hopkins University;* Xuanzhe Liu and Xin Jin, *Peking University* 来自北大金鑫组。

用[container cloud](https://www.zhihu.com/search?q=container cloud&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"article"%2C"sourceId"%3A"664765417"})训练DL模型时，静态地绑定GPU和container虽然性能隔离性好，但会导致GPU的利用率低下。比如，微软GPU集群的GPU利用率平均值只有52%，阿里GPU集群的GPU利用率中位数只有10%不到。对于opportunistic training jobs，也就是那些训练时间要求不那么严格的任务，GPU是可以共享的。这种共享可以在应用层面实现，比如肖文聪他们的OSDI '20 AntMan——不过由于改了DL框架所以用户被迫只能使用特定版本的DL框架；也可以在OS层面实现，比如Nvidia的[Multi-Process Service](https://www.zhihu.com/search?q=Multi-Process Service&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"article"%2C"sourceId"%3A"664765417"})，但它要求用户设置资源限制并且不支持GPU memory oversubscription。所以，本文提出了TGS，这是container和GPU之间薄薄的一层，它截获了container发往GPU的[system call](https://www.zhihu.com/search?q=system call&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"article"%2C"sourceId"%3A"664765417"})来做调度，从而解决这些限制。

## [ARK: GPU-driven Code Execution for Distributed Deep Learning](https://link.zhihu.com/?target=https%3A//www.usenix.org/conference/nsdi23/presentation/hwang)

 Changho Hwang, *KAIST, [Microsoft Research](https://www.zhihu.com/search?q=Microsoft Research&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"article"%2C"sourceId"%3A"664765417"});* KyoungSoo Park, *KAIST;* Ran Shu, Xinyuan Qu, Peng Cheng, and Yongqiang Xiong, *Microsoft Research*

分布式训练中，GPU间要频繁传递小消息，开销很大。目前有两种方案来做这个通信，一种是让CPU来主导这一通信，另一种是由CPU引发的GPU DMA engine。为了进一步减少这一开销，本文提出由GPU引发的[GPU DMA engine](https://www.zhihu.com/search?q=GPU DMA engine&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"article"%2C"sourceId"%3A"664765417"})，将[All-Reduce](https://www.zhihu.com/search?q=All-Reduce&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"article"%2C"sourceId"%3A"664765417"})的吞吐量提升了0.8倍。

## [BGL: GPU-Efficient GNN Training by Optimizing Graph Data I/O and Preprocessing](https://link.zhihu.com/?target=https%3A//www.usenix.org/conference/nsdi23/presentation/liu-tianfeng)

 Tianfeng Liu, *Tsinghua University, Zhongguancun Laboratory, ByteDance;* Yangrui Chen, *The University of Hong Kong, ByteDance;* Dan Li, *Tsinghua University, Zhongguancun Laboratory;* Chuan Wu, *The University of Hong Kong;* Yibo Zhu, Jun He, and Yanghua Peng, *ByteDance;* Hongzheng Chen, *ByteDance, Cornell University;* Hongzhi Chen and Chuanxiong Guo, *ByteDance*

字节的[GNN训练库](https://www.zhihu.com/search?q=GNN训练库&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"article"%2C"sourceId"%3A"664765417"})，做的优化包括：codesign caching policy和sampling order以加速[feature retrieving](https://www.zhihu.com/search?q=feature retrieving&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"article"%2C"sourceId"%3A"664765417"})、优化graph partition算法以减小subgraph sampling时跨partition的通信开销、优化了预处理时的资源分配以减少竞争。

来自[字节跳动](https://www.zhihu.com/search?q=字节跳动&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"article"%2C"sourceId"%3A"664765417"})郭传雄组（现在创业去了）、港大吴川组、清华[李丹](https://www.zhihu.com/search?q=李丹&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"article"%2C"sourceId"%3A"664765417"})组。

## [Zeus: Understanding and Optimizing GPU Energy Consumption of DNN Training](https://link.zhihu.com/?target=https%3A//www.usenix.org/conference/nsdi23/presentation/you)

 Jie You, Jae-Won Chung, and [Mosharaf Chowdhury](https://www.zhihu.com/search?q=Mosharaf Chowdhury&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"article"%2C"sourceId"%3A"664765417"}), *University of Michigan*

发现能耗和[性能优化](https://www.zhihu.com/search?q=性能优化&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"article"%2C"sourceId"%3A"664765417"})之间有[非线性tradeoff](https://www.zhihu.com/search?q=非线性tradeoff&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"article"%2C"sourceId"%3A"664765417"})，然后提出了一个[optimizer](https://www.zhihu.com/search?q=optimizer&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"article"%2C"sourceId"%3A"664765417"})来自动做这个tradeoff，它会在线profiling以调整batch size和GPU的功率限制。

作者：孙挺Sunt
链接：https://zhuanlan.zhihu.com/p/664765417
来源：知乎
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。



## [Bamboo: Making Preemptible Instances Resilient for Affordable Training of Large DNNs](https://link.zhihu.com/?target=https%3A//www.usenix.org/conference/nsdi23/presentation/thorpe)

 John Thorpe, Pengzhan Zhao, Jonathan Eyolfson, and Yifan Qiao, *UCLA;* Zhihao Jia, *CMU;* Minjia Zhang, *Microsoft Research;* Ravi Netravali, *Princeton University;* Guoqing Harry Xu, *UCLA*

提出Bamboo，利用[云服务](https://www.zhihu.com/search?q=云服务&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"article"%2C"sourceId"%3A"664765417"})里廉价的spot instance来做训练，从而降低训练成本。Spot instance会动不动就挂掉，失败率远高于普通场景，所以不同于传统的[checkpointing机制](https://www.zhihu.com/search?q=checkpointing机制&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"article"%2C"sourceId"%3A"664765417"})，Bamboo从[RAID](https://www.zhihu.com/search?q=RAID&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"article"%2C"sourceId"%3A"664765417"})中借鉴了[redundant computation](https://www.zhihu.com/search?q=redundant computation&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"article"%2C"sourceId"%3A"664765417"})思想，然后利用pipeline parallelization中的bubble来做这些redundant computation。Bamboo是构建在[DeepSpeed](https://www.zhihu.com/search?q=DeepSpeed&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"article"%2C"sourceId"%3A"664765417"})之上，最后发现性价比高出[on-demand训练系统](https://www.zhihu.com/search?q=on-demand训练系统&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"article"%2C"sourceId"%3A"664765417"})两三倍。

跟[ETH Torsten Hoefler](https://www.zhihu.com/search?q=ETH Torsten Hoefler&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"article"%2C"sourceId"%3A"664765417"})组今年发的MLSys '23 [PipeFisher](https://www.zhihu.com/search?q=PipeFisher&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"article"%2C"sourceId"%3A"664765417"})一样，都利用了pipeline parallelization中的bubble来做一下extra work，从而变废为宝。

来自UCLA[徐国庆](https://www.zhihu.com/search?q=徐国庆&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"article"%2C"sourceId"%3A"664765417"})组。

## [Shockwave: Fair and Efficient Cluster Scheduling for Dynamic Adaptation in Machine Learning](https://link.zhihu.com/?target=https%3A//www.usenix.org/conference/nsdi23/presentation/zheng)

 Pengfei Zheng and Rui Pan, *University of Wisconsin-Madison;* Tarannum Khan, *The University of Texas at Austin;* Shivaram Venkataraman, *University of Wisconsin-Madison;* Aditya Akella, *The University of Texas at Austin*

在dynamic adaptation（比如训练时动态改变batch size）时，怎么调度才能同时取得fairness和throughput。 动态调整模型结构（例如，彩票假设）或超参数（例如，批量大小）可以在不牺牲准确性的情况下显着加快训练速度。但是，现有的 ML 集群调度程序并非旨在处理动态适应。我们表明，在动态适应下，当训练吞吐量随时间变化时，现有方案无法提供公平性并降低系统效率。我们设计了 Shockwave，这是一个基于两个关键思想构建的未来规划调度器。首先，Shockwave将经典市场理论从静态设置扩展到动态设置，以协同优化效率和公平性。其次，Shockwave 利用随机动态规划来处理动态变化。我们为 Shockwave 构建了一个系统，并通过跟踪驱动的模拟和集群实验来验证其性能。结果表明，对于具有动态适应的ML作业痕迹，与现有的公平调度方案相比，Shockwave提高了1.3×公平性，公平性提高了2×。

来自UW-Madison Shivaram Venkataraman组和UT Aditya Akella组。







SHEPHERD: Serving DNNs in the Wild , nsdi 2023 , 一作张弘. 延迟要求: 50-500ms 

nexus , 周期性的, per-stream policy.

calculat CV for each group streaming, 100-1000个stream, 就会比较稳定而且可以预测到达pattern.

怎么group呢? 等吗?   在每个group 中 online serving. 
