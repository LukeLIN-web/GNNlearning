

#### 背景

tvm 只有固定pattern, 不够广.

多面体 缺乏算子信息. 错过fuse机会. 

#### 方法

developing a classification of both individual operators and their combinations

DNNFusion includes 1) a novel mathematical-property based graph rewriting framework (为什么可以reduce?) to reduce evaluation costs and facilitate subsequent operator fusion, 2) an integrated  fusion plan generation that leverages the high-level analysis and accurate light-weight profiling, and 3) additional optimizations during fusion code generation. 

分类 : One-to-One, Reorganize, Shuffle, One-to-Many, and Many-to-Many

就对各种情况进行分析.  提出 Extended Computational Graph 作为IR.

先用tvm和mnn产生计算图, 然后加入他们的infomation 产生ecg.

#### Mathematical-Property-Based Graph Rewriting

看图二好像就是结合律, 分配律, 交换律

#### Light-Weight Profile-Driven Fusion Plan Exploration

认为 One-to-One mapping fuse 潜力最大.  作为seed op. 找前后的op.

在patDNN上写的. 

别的框架好像很多都不支持.

## Speed Is All You Need

: On-Device Acceleration of Large Diffusion Models via
GPU-Aware Optimizations

Specialized Kernels: Group Norm and GELU

#### Enhancing Attention Module Efficiency

是把 matmul V和softmax fuse到一起. 看图2.

