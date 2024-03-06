17年  iclr [NEURAL ARCHITECTURE SEARCH WITH REINFORCEMENT LEARNING](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1611.01578.pdf) 

[Learning Transferable Architectures for Scalable Image Recognition](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1707.07012.pdf)[CVPR'18] 

[Regularized Evolution for Image Classifier Architecture Search](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1802.01548)[AAAI'19]，搜索空间沿用上面的NASNet Search Space，只是将优化算法换成了regularized evolution

[Progressive Neural Architecture Search](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1712.00559)[ECCV'18]，主要目标就是做加速。将搜索空间减小并且用[预测函数](https://www.zhihu.com/search?q=预测函数&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A"922467317"})来预测网络精度

上面四个都太慢了.

weight sharing加速validation的[Efficient Neural Architecture Search via Parameter Sharing](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1802.03268)[ICML'18] 大概思路就是将NAS的采样过程变成在一个DAG里面采样子图的过程，采样到的子图里面的操作(卷积等)是weight sharing. ENAS , 搜个cell然后按照之前的方式堆起来，在搜索cell的时候只需要**一张卡跑半天**就能跑完.

用可微分的方式进行搜索的[DARTS: Differentiable Architecture Search](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1806.09055)[ICLR'19]  开源了代码，然后代码写得很漂亮改起来也比较容易. DARTS选择某个操作的时候并不是像之前按照RNN的输出概率进行采样，而是把所有操作都按照权重加起来，权重就在计算图里面了，直接通过梯度下降来优化权重, 保留权重最大的操作就是搜到的结构。  搜索一次一张1080ti跑一天即可.

但并不一定适合实际部署(比如DARTS能在很小的参数量下达到很高的精度，但是实际forward其实很慢)

[EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1905.11946)[ICML'19]  搜索了宽度/深度/分辨率之间的相对缩放因子，将他们按照一定的比例放大, 简单且有效。

[ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1812.00332)[ICLR'19]， mit 韩松, 魔改 MobileNetV2, 想要某个数据集在某硬件上的结果就直接在对应的数据集和硬件上进行搜索，而不是像之前的工作先在cifar10搜然后transfer到ImageNet等大数据集上，并且在目标函数上加入硬件latency的损失项作为多目标NAS，直接面向部署的硬件。

谷歌 [MnasNet: Platform-Aware Neural Architecture Search for Mobile](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1807.11626)[CVPR'19]用来魔改MobileNetV2里面的layer数/卷积操作/加[attention](https://www.zhihu.com/search?q=attention&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A"922467317"})等等，然后用之前同样的RNN+policy gradient的方式来优化

Once-for-All: Train One Network and Specialize it for Efficient Deployment  ICLR2020 直接train一个大网络，需要哪种网络直接去里面取就可以，将训练过程和搜索的过程解耦开来，相当于只需要train一次。 比较神奇的地方在于训完了之后不需要fine-tune就能取得很好的结果，和progressive shrinking的训练方式相关。如果再fine-tune精度会更高.

在现在，更多的架构调整其实解决两个问题，更小（从MHA到[MQA](https://www.zhihu.com/search?q=MQA&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A3329996852})到GQA/local attention），更长（[位置编码](https://www.zhihu.com/search?q=位置编码&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A3329996852})）。
当然也有很多基于架构解决其他问题的工作，解决[hallucination](https://www.zhihu.com/search?q=hallucination&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A3329996852})（个人觉得不够本质），白盒[transformer](https://www.zhihu.com/search?q=transformer&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A3329996852})……

#### Predicting Inference Latency of Neural Architectures on Mobile Devices

预测 latency很难.   于是提出了predictors.

方法

1. 提取 计算图中的op.
2. for mobile GPUs, we deduce (without deploying to the mobile device) the actual kernels executed after kernel fusion and kernel selection. 
3. 用ML models to predict inference latency of each operation from its parameters (e.g., input shape and number of channels 
