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

fuse, 或许可以用 dynamic shape fusion 试试. 

flashattention 不是从循环程序分析角度能得到的, 注意到了memory level的, online softmax不是dag能描述的, 依赖关系太复杂. dnn 计算复杂了 调度空间太大了. 

https://llm.mlc.ai/docs/get_started/quick_start  tvm运行llama3.

## fuse

https://ai.lefebvre-sarrut.eu/2023/07/20/deep-dive-into-kernel-fusion-accelerating-inference-in-llama-v2/

1. 实数虚数分开表示.
2. 在同一地址上执行两次连续的加载操作时，第二次操作可能会从 GPU 缓存中获取数据, 所以不会花费双重 DDR 加载.
3.  rms weight 乘法的同时 统计rms  所有元素平方的均值
4. feedforward, 一次loop 加载两个矩阵乘法. 

#### Rotary Embeddings

函数 precompute_freqs_cis 生成一个范围，根据此范围计算频率值，然后计算这些频率的余弦值和正弦值，这些值作为两个单独的张量返回。

函数apply_rotary_emb_pytorch将输入张量 x分割成实数和虚数部分，执行类似于复数乘法的计算，最后将实数和虚数部分合并回原始张量。

就是避免用torch polar, 不用创建复数 不用 torch.complex128保存.  就是分开实部和虚部. 

为什么这么做? 像Triton这样的低级工具缺乏对它们的直接支持.triton 没有complex128格式.https://github.com/triton-lang/triton/blob/f21009031e94f4f4da2d01641d7f20f8b8e2d70b/python/triton/language/core.py#L324  最多uint64, fp64 

llamacpp是fp32, 没有量化. 实部和虚部已经分开了.



#### RMSNorm 

第一个循环读取数据并计算 RMSNorm 统计信息。

第二个循环使用上一步中计算的rstd 修改读取的数据。

当在同一地址上执行两次连续的加载操作时，第二次操作可能会从 GPU 缓存中获取数据，这意味着在同一个 Triton 程序中执行两次传递不会花费双重 DDR 加载。

我们还在加载和存储张量时使用掩码。在处理可能包含张量边界之外的地址的固定大小块时，这些掩码对于避免非法内存访问至关重要。

为什么需要 rms weight 乘法?

llamacpp 也是两个循环,  但是没有 out = x_hat * rms_w ,  是放在单独算子做吗?

#### 融合

RMSNorm 成本更高的任务是数据加载，分两个不同的阶段执行。紧接着，during matrix multiplication from output of RMSNorm and model weight (to perform the projection)，iterate 输入张量。这种情况为计算 RMSNorm 所需的统计信息提供了机会。

由于转换需要将输入除以特定值，因此我们可以在tile 矩阵乘法后执行此操作。从本质上讲，这意味着我们可以合并这两个操作：在归一化之前在张量上执行矩阵乘法，然后归一化其（tile）输出。

矩阵乘法后，我们可以将输出与旋转嵌入合并。

在此优化过程中，旋转嵌入的使用被认为是可选的，因为它仅用于 Q 和 K 张量，而不用于 V 张量（RBE_EPILOGUE bool 参数）。

内核的初始指令侧重于确定开始数据读取过程的精确位置。然后，该循环执行三个主要操作：投影（即矩阵乘法）、与 RMSNorm 权重的乘法，以及方便地计算 RMSNorm 的统计数据，这得益于正在进行的数据读取。RMSNorm 的实现最终是通过以下操作完成的：`accumulator = accumulator * a_norm[:, None]`

之后做rbe,  Here, it's crucial to note the synchronization barrier.   当在共享地址上同时发生读/写操作时，必须确保单个 warp 中的所有线程同步完成。否则，这可能会导致与并发相关的复杂调试问题。

值得注意的是一个重要的权衡：RMSNorm 和矩阵乘法的融合导致的计算量超过了严格必要的计算量。具体来说，对于每个tile（张量的一部分），该过程会重新计算整个输入行的统计数据，然后对tile进行归一化。这种方法主要基于这样一个事实，即在小批量的上下文中（可能与全局内存 I/O 和 CPU 开销得到有效管理和摊销的大批量无关），我们受到内存带宽的限制。因此，这种额外的计算被认为是可以接受的。不过，对于较大的批次，比较融合和未融合的 RMSNorm 操作之间的性能配置文件肯定是一个好主意。

速度提升不高的原因主要是因为，在 Triton 中为投影执行的矩阵乘法并不比在 PyTorch 中快得多。这主要是因为将权重从全局存储器移动到片上存储器的过程非常耗时，无论实施何种优化策略，情况仍然如此。尽管如此，速度提高 1.5 倍是一个重大改进。

#### Feed Forward

一次loop 加载两个矩阵乘法

silu 运算由两个元素运算组成，也可以与第一个矩阵乘法的输出合并，从而实现整体更精简的运算。

将 CUDA 时间减少 30%。然而，第二个优势在于减少了内核启动数量和其他可能的开销。这些因素对总墙时间有很大贡献，特别是在计算相对有限的情况下，例如在推理中，当批量大小设置为 1 时更是如此，就像这里一样。

 Flash Attention 确实对perplexity有轻微影响。在此项目中，Flash Attention 仅在 FP8 场景中使用。这可能归因于我们的 FA 内核实现中的一个错误。我们已经实现了原版 Triton 和我们自己针对 Llama 的自定义版本，两者都导致了同样的问题。或者，它可能根本不是一个错误，而是一个灾难性取消catastrophic cancelling的例子(就是两个近似值相减误差 比实际误差大非常多倍)。我们没有对此进行深入调查，因为我们的主要目的是测试 FP8 格式及其对速度的影响。



