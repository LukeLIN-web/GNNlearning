

#### powerinfer2



和1 差不多， 

他们是大矩阵NPU prefill， 小矩阵CPU decode。  取个名字叫 small neuron clusters。 针对不同的LLM权重类型设计了不同的cache 策略。

也是经典pipeline。 

 offline planner运行前分析使用率， 我猜也是经验模型

分段缓存会针对不同的权重类型（如注意力权重、预测器权重、前馈网络权重）采取不同的缓存策略，以提高缓存命中率，减少不必要的磁盘 I/O。缓存还会使用 LRU 替换算法动态更新每个神经元的实际冷热情况来保证缓存中存放的都是最热的神经元。此外，PowerInfer-2.0 还针对手机 UFS 4.0 存储的性能特点，设计了专门的模型存储格式，以提高读取性能。



团队收集了包括网页、代码和数学数据集在内的多样化继续训练语料. 是不是原来精度不够， 加入了更好的数据集就效果更好。



#### powerinfer1

llama.cpp 怎么实现GPU的？应该就是改成了cuda 

[powerinfer-20231219.pdf (sjtu.edu.cn)](https://ipads.se.sjtu.edu.cn/_media/publications/powerinfer-20231219.pdf)

[SJTU-IPADS/PowerInfer: High-speed Large Language Model Serving on PCs with Consumer-grade GPUs (github.com)](https://github.com/SJTU-IPADS/PowerInfer)



 LLM 推理过程中的高度局部性特征，即神经元激活呈现[幂律分布。这意味着某些频繁激活的“[热神经元在各种输入中都会被激活,而其他冷神经元”则因输入不同而变化。

将“热神经元”预先加载到 GPU 中以便快速访问，而“冷神经元”的计算则在 CPU 上进行.

数据流转其实不简单， 花的时间很多。 

对神经元影响度的定量标准，用来区分神经元 hot/cold 程度，进而决定如何分配到具体设备上。这里论文引入了一个整数线性规划的**最优求解问题**来解决这个划分问题。可以理解为具体的划分比例是被搜索出来的。所以我认为这里的比例应该是针对特定的 GPU，**卡着 GPU 能装得下的最大值去做的，其他的放 CPU**。同时再兼顾网络数据传输压力等等。

也是高度定制化，完全不通用。 



reference：

1. https://www.zhihu.com/question/636760404/answer/3350412621
2. 

引用了dejavu， 

发现 注意力头中超过80%的部分会在实验中变得不活跃，而 MLP 层上的平均稀疏性则可以发现，超过95%的MLP参数都可以不参与推理。

作者使用了一个额外的神经网络分类器作为近邻搜索器，实时预测每个层的上下文稀疏性，即哪些 Attention head 和 MLP 参数在给定输入下是活跃的，就只加载这些参数进行计算。

工程实现上大头的还是内存管理，这几篇论文都是这样的，不过这篇没怎么讲。