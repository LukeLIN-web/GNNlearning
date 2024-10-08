## 可复现的

### Minuet: Accelerating 3D Sparse Convolutions on GPUs

稀疏卷积 (SC) 广泛应用于本质上高度稀疏的 3D 点云网络。现有的 SC 发动机存在三个缺点。首先，它们依靠哈希表来提取必要的通用矩阵乘法 (GEMM) 运算，从而导致通过昂贵的 GPU 全局内存进行不规则的数据访问。其次，它们采用单个固定的图块大小来处理多个输入/输出特征通道，这并不总是能够在各个层、输入数据集和 GPU 架构上提供最佳性能。第三，它们填充零数据，并将多个 GEMM 操作分组为遵循权重偏移排序的单个内核启动。这种分组方案会导致许多冗余数据访问和对无用（零）数据的计算。

为此，我们提出了 Minuet，这是一种专为现代 GPU 量身定制的新型内存高效 SC 引擎。Minuet 引入了 (i) 分段排序双遍历二分搜索算法，该算法在提取卷积所需的 GEMM 操作时充分利用 GPU 片上内存层次结构的所有级别，(ii) 一种轻量级自动调整方案，可在以下情况下配置图块大小：处理多个通道，从而使 SC 执行适应每一层、数据集和 GPU 架构的特征，以及 (iii) 填充高效的 GEMM 分组方法，可减少填充成本和冗余（零）数据访问和计算。我们的评估表明，Minuet 的平均性能明显优于之前最先进的 SC 发动机1.74×1.74 ×（取决于2.22×2.22 ×）用于端到端点云网络执行。我们新颖的分段排序双遍历二分搜索算法通过以下方式实现了卓越的加速15.8×15.8 ×平均（最多26.8×26.8 ×）超过了之前最先进的作品。

### SplitFT：通过远程内存记录实现分类数据中心的容错

我们推出 SPLITFT，这是一种新的容错方法，适用于分类数据中心中以存储为中心的应用程序。SPLITFT 使用新颖的拆分架构，其中大写操作直接在底层分解文件系统上执行，而小写操作则在计算层内进行容错。分体式架构使应用程序能够在不影响性能的情况下获得强大的耐用性保证。SPLITFT 使用一种称为近计算日志 (NCL) 的新抽象，以快速、廉价且透明的方式使小写入具有容错能力。我们将三个 POSIX 应用程序（RocksDB、Redis 和 SQLite）移植到 SPLITFT，并证明与可能丢失数据的弱版本应用程序相比，它们提供了强有力的保证；SPLITFT 应用程序这样做，同时近似弱版本的性能（在 YCSB 下仅产生 0.1%-10% 的开销）。与强版本相比，SPLITFT 显着提高了性能（在写入繁重的工作负载下大约提高了 2.5 倍到 27 倍）。

### Orion：针对 ML 应用程序的干扰感知、细粒度 GPU 共享

https://github.com/eth-easl/orion

GPU 对于最大化深度神经网络 (DNN) 应用的每瓦吞吐量至关重要。然而，即使使用大批量大小并消除任何输入数据处理或通信停顿，DNN 应用程序通常也无法充分利用昂贵的 GPU 资源。根本原因是 DNN 工作负载由许多依赖于数据的运算符组成，每个运算符执行 10 到 1000 个 𝜇，计算和内存要求截然不同。 ( gnn 并没有这么多运算符, 主要是在数据加载的过程中. )

虽然操作员可能会使 GPU 计算单元或内存带宽饱和，但它通常会使其他 GPU 资源闲置。尽管 GPU 共享技术很流行，但当前的方法还不够细粒度或干扰感知，无法最大限度地提高 GPU 利用率，同时最大限度地减少 DNN 应用所需的 10 𝜇 粒度的干扰。我们提出了 Orion，这是一个软件系统，可以透明地拦截来自共享 GPU 的多个客户端的 GPU 内核启动。Orion 以各个算子的粒度安排 GPU 上的工作，并通过考虑每个算子的计算和内存需求来最大限度地减少干扰。我们将 Orion 集成到 PyTorch 中，并在各种 DNN 工作负载搭配用例中展示其优势。Orion 为高优先级推理作业保持低尾部延迟，同时配置尽力而为的推理作业，将每个 GPU 请求吞吐量提高高达 7.3 倍。Orion 在搭配 DNN 训练时还保持较低的推理延迟，与为每个工作负载使用专用 GPU 相比，可节省高达 1.49 倍的训练成本。与最先进的基线相比，Orion 显着改善了尾部延迟。

这是一种 GPU 调度器，用于拦截来自共享 GPU 的多个 DNN 作业的操作员。Orion 根据操作员的大小、计算与内存要求以及作业优先级来安排操作员，以便可以并置具有互补资源需求的操作员。

Orion 为高优先级 DNN 作业保持高性能（低尾部延迟），同时通过并置尽力而为的作业来提高 GPU 利用率。Orion 将运行 DNN 工作负载的成本降低了 1.49 倍，与为每个作业专用一个 GPU 相比。与之前的 GPU 共享技术相比，Orion 还改善了尾部延迟。

#### 介绍

They believe MPS  and GPU Stream 是不够 interference-aware.  

干扰是什么呢?  就是把两个memory-intensive的 BN2d kernels 放在一个SM中. 

3.2

大部分GPU, preempt后不能抢占, 所以是怎么调度的? 就是在torch下一层,  hardware scheduler 上一层.

how they put in same SM?  you can allocate it by yourself?  MPS.  set a queue? 

###  5 orion

orion是一个a dynamically linked lib that controls GPU operations submitted by an application framework (e.g., PyTorch)

#### 5.1

```python
1 def run_scheduler ( client_q_hp高优先级 , client_q_be 最大努力 ) :
2 be_duration = 0 , be_submitted = Event ()
3 hp_task_running = False
4 while True :
5 	op_hp = client_q_hp . pop ()
6 	op_be = client_q_be . peek () # 先不删除. 
7 	if ( op_hp != None ) :
8 		launch_kernel ( op_hp , stream_hp ) # 有高选高
9 		hp_task_running = True
10 	if ( op_be != None ) :
11 		schedule = schedule_be ( op_hp , op_be )
12 if ( be_duration > DUR_THRESHOLD ) :
13 		if ( be_submitted . finished () ) :
14 	be_duration = 0
15 else :
16 schedule = False
17 if ( schedule ) :
18 client_q_be.pop ()
19 launch_kernel ( op_be , stream_be )
20 be_duration += op_be . duration
21 be_submitted . record ( stream_be )
22
23 def schedule_be ( op_hp , op_be ) :
24 		schedule = False
25 if (! hp_task_running ) :
26 		schedule = True
27 else if ( op_be . sm_needed < SM_THRESHOLD 
28 			and have_different_profiles ( op_hp , op_be ) ) :
29 schedule = True
30 return schedule
```

阈值是怎么确定的?

SM_THRESHOLD  默认是总SM数量,  如果high-priority job是training, SM_THRESHOLD 可以很大. 二分法调整 SM_THRESHOLD. 为了防止高优先级 job starve. 

 3000 lines of C++/CUDA code.



### DynaPipe：通过动态管道优化多任务训练

多任务模型训练已被采用来学习单个深度神经网络模型（通常是大型语言模型）来处理多个任务（例如，问答和文本摘要）。由于不同任务的上下文不同，多任务学习通常接收长度差异很大的输入序列。通常采用填充（到相同的序列长度）或打包（将短示例放入相同长度的长序列）来准备用于模型训练的输入样本，但这在空间或计算效率上不高。本文提出了一种动态微批处理方法来解决序列长度变化并实现高效的多任务模型训练。我们提倡使用可变长度微批次对大型模型进行管道并行训练，每个微批次可能包含不同数量的样本。我们使用基于动态编程的方法优化微批量构建，并通过动态管道和通信调度处理微批量执行时间变化，从而实现高效的管道训练。对 FLANv2 数据集的广泛评估表明，与基于打包的基线相比，训练 T5 时的训练吞吐量提高了 4.39 倍，训练 GPT 提高了 3.25 倍。

提供了aws服务器

