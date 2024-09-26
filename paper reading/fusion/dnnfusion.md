https://arxiv.org/pdf/2108.13342 一篇是fusion 

https://arxiv.org/pdf/2304.11267 一篇是另外给你参考

#### 背景

tvm 只有固定pattern, 不够广.

多面体 缺乏算子信息. 错过fuse机会. 

#### 方法

developing a classification of both individual operators and their combinations

DNNFusion includes 1) a novel mathematical-property based graph rewriting framework (为什么可以reduce?) to reduce evaluation costs and facilitate subsequent operator fusion, 2) an integrated  fusion plan generation that leverages the high-level analysis and accurate light-weight profiling, and 3) additional optimizations during fusion code generation. 

分类 : One-to-One, Reorganize, Shuffle, One-to-Many, and Many-to-Many

各种情况进行分析.  提出 Extended Computational Graph 作为IR.

先用tvm和mnn产生计算图, 然后加入他们的infomation 产生ecg.

#### Mathematical-Property-Based Graph Rewriting

看图二就是结合律, 分配律, 交换律

#### Light-Weight Profile-Driven Fusion Plan Exploration

认为 One-to-One mapping fuse 潜力最大.  作为seed op. 找前后的op.

在patDNN上写的. 

## Speed Is All You Need

: On-Device Acceleration of Large Diffusion Models via
GPU-Aware Optimizations

Specialized Kernels: Group Norm and GELU

#### Enhancing Attention Module Efficiency

是把 matmul V和softmax fuse到一起. 看图2.

flashattention 不是从循环程序分析角度能得到的, 注意到了memory level的, online softmax不是dag能描述的, 依赖关系太复杂. dnn 计算复杂了 调度空间太大了. 

https://llm.mlc.ai/docs/get_started/quick_start  tvm运行llama3.

## fuse

https://ai.lefebvre-sarrut.eu/2023/07/20/deep-dive-into-kernel-fusion-accelerating-inference-in-llama-v2/

但是没想到啥创新,  先做试试.  也得继续看论文.    

1. 实数虚数分开表示. llama cpp已经做了. 
2. RMSNorm , 在同一地址上执行两次连续的加载操作时，第二次操作可能会从 GPU 缓存中获取数据, 所以不会花费双重 DDR 加载.  llama cpp已经做了.   
3.  rms weight 乘法的同时 统计rms  所有元素平方的均值.  llama cpp 是分开乘的. 
4.   一个大kernel, 融合rms norm 和rbe . 一个循环同时乘linear 和rms的weight.  llama cpp 没做.    
5. feedforward, 一次loop 加载两个矩阵乘法. 

#### Rotary Embeddings

函数 precompute_freqs_cis 生成一个范围，根据此范围计算频率值，计算这些频率的余弦值和正弦值，这些值作为两个单独的张量返回。

函数apply_rotary_emb_pytorch将输入张量 x分割成实数和虚数部分，执行类似于复数乘法的计算，最后将实数和虚数部分合并回原始张量。

就是避免用torch polar, 不用创建复数 不用 torch.complex128保存, 分开实部和虚部. 

为什么这么做? 像Triton这样的低级工具缺乏对它们的直接支持.triton 没有complex128格式.https://github.com/triton-lang/triton/blob/f21009031e94f4f4da2d01641d7f20f8b8e2d70b/python/triton/language/core.py#L324  最多uint64, fp64 

llamacpp是fp32, 没有量化. 实部和虚部已经分开了.

#### RMSNorm 

第一个循环读取数据并计算 RMSNorm 统计信息。

```
for 
 var += x^2
 
var = sum(x^2)
var = 求 var 平均值
rstd = sqrt (var)

for :
	xhat = x * rstd
```

第二个循环使用上一步中计算的rstd 修改读取的数据。

当在同一地址上执行两次连续的加载操作时，第二次操作可能会从 GPU 缓存中获取数据，这意味着在同一个 Triton 程序中执行两次传递不会花费双重 DDR 加载。

我们还在加载和存储张量时使用掩码。在处理可能包含张量边界之外的地址的固定大小块时，这些掩码对于避免非法内存访问至关重要。

为什么需要 rms weight 乘法?    gi是否就是weight.https://mltalks.medium.com/rmsnorm%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-bfae83f6d464

是的, 就是rmsnorm就有一个weight. 初始化为1. layernorm也有weight.

llamacpp 也是两个循环,  但是没有 out = x_hat * rms_w ,  是放在单独算子做. 

```
2024-08-26 20:24:46.680 llama-cli[18076:6315274] ,f32, RMS_NORM , ffn_inp-0 , (4096; 2;1; 1),  , (0; 0;0; 0),0.004888
2024-08-26 20:24:46.680 llama-cli[18076:6315274] ,f32, MUL , norm-0 , (4096; 2;1; 1),  blk.0.ffn_norm.weight, (4096; 1;1; 1),0.007033

2024-08-26 20:24:46.680 llama-cli[18076:6315274] ,f32, RMS_NORM , l_out-0 , (4096; 2;1; 1),  , (0; 0;0; 0),0.007033
2024-08-26 20:24:46.680 llama-cli[18076:6315274] ,f32, MUL , norm-1 , (4096; 2;1; 1),  blk.1.attn_norm.weight, (4096; 1;1; 1),0.006080
```





#### 融合

RMSNorm 成本更高的任务是数据加载，分两个不同的阶段执行。紧接着，during matrix multiplication from output of RMSNorm and model weight (to perform the projection)，iterate 输入张量。这种情况为计算 RMSNorm 所需的统计信息提供了机会。

由于转换需要将输入除以特定值，因此我们可以在tile 矩阵乘法后执行此操作。从本质上讲，这意味着我们可以合并这两个操作：在归一化之前在张量上执行矩阵乘法，然后归一化其（tile）输出。

矩阵乘法后，我们可以将输出与旋转嵌入rbe_triton合并。

在此优化过程中，旋转嵌入的使用被认为是可选的，因为它仅用于 Q 和 K 张量，而不用于 V 张量（RBE_EPILOGUE bool 参数）。

该循环执行三个主要操作：projection（即矩阵乘法）、与 RMSNorm 权重的乘法，以及方便地计算 RMSNorm 的统计数据.  RMSNorm 的实现最终是通过以下操作完成的：`accumulator = accumulator * a_norm[:, None]`

之后做rbe,  Here, it's crucial to note the synchronization barrier.   当在共享地址上同时发生读/写操作时，必须确保单个 warp 中的所有线程同步完成。否则，这可能会导致与并发相关的复杂调试问题。

值得注意的是一个重要的权衡：RMSNorm 和矩阵乘法的融合导致的计算量超过了严格必要的计算量。具体来说，对于每个tile（张量的一部分），该过程会重新计算整个输入行的统计数据，然后对tile进行归一化。主要基于这样一个事实，即在小batch size，受到内存带宽的限制。因此，这种额外的计算被认为是可以接受的。

速度提升不高的原因主要是因为，在 Triton 中矩阵乘法并不比在 PyTorch 中快得多。这主要是因为将权重从全局存储器移动到片上存储器的过程非常耗时，无论实施何种优化策略，情况仍然如此。尽管如此，速度提高 1.5 倍是一个重大改进。

#### Feed Forward

一次loop 加载两个矩阵乘法

silu 运算由两个元素运算组成，也可以与第一个矩阵乘法的输出合并，从而实现整体更精简的运算。

将 CUDA 时间减少 30%。然而，第二个优势在于减少了内核启动数量和其他可能的开销。这些因素对总wall time有很大贡献，特别是在计算相对有限的情况下，例如在推理中，当批量大小设置为 1 时更是如此，就像这里一样。

 Flash Attention 确实对perplexity有轻微影响。在此项目中，Flash Attention 仅在 FP8 场景中使用。这可能归因于我们的 FA 内核实现中的一个错误。我们已经实现了原版 Triton 和我们自己针对 Llama 的自定义版本，两者都导致了同样的问题。或者，它可能根本不是一个错误，而是一个灾难性取消catastrophic cancelling的例子(就是两个近似值相减误差 比实际误差大非常多倍)。我们没有对此进行深入调查，因为我们的主要目的是测试 FP8 格式及其对速度的影响。

参考 https://github.com/JonasGeiping/linear_cross_entropy_loss



torch compile 有fuse优化吗, 有 .FX Passes：修改计算图，替换掉一些低效的操作。比如帮你帮[attention](https://zhida.zhihu.com/search?q=attention&zhida_source=entity&is_preview=1)换成flash attention。
修改计算图本质上就是在改用户代码。如果用户不犯傻，那这些pass根本就没用。

AOTAutograd：自动决定中间结果是要保存下来还是重算。
完全可以通过手写torch.autograd.Function来实现。

FSDP优化：自动调整计算和通信的顺序，尽量使得计算时间掩盖通信时间。
本质上还是在改用户代码。不过都用上多卡训练了，这也该是基操了吧。

Inductor：op fusion。
这个值得单独拿出来讲。

tensorrt也是compile一遍. 



https://pytorch.org/blog/cuda-free-inference-for-llms/?hss_channel=tw-776585502606721024

#### 一些idea

1. 连续矩阵乘的融合收益还是要看具体场景，虽然能减少中间矩阵的搬移，但也会影响并行度以及对旁侧矩阵的复用，最好是中间矩阵足够大、旁侧矩阵足够小才有比较好的收益，主流场景中-基本只有各类 attention、cnn 的前几层、一些小型 mlp 比较符合这个要求. 

2. 为什么没有自动生成任意算子fusion kernel的工作？ - 尘伊光的回答 - 知乎
   https://www.zhihu.com/question/666742071/answer/3621843363

3. 效果最好的场景就是连续多个的elementwise ops。通过算子融合可以显著减轻memory bound。 cuda capture+cuda graph算cuda层动态融合.,算子融合是框架静态图的特长，torch compile  已经融合的不错了.  torch静态图的难点是构图，Dynamo在原理上我猜已经完备了，至于如果对一个包含各种op的图做融合算法是需要时间积累的,  另外推理引擎也是专业做op融合。cuda层做融合就确实没怎么听过.

    elementwise 代表的layernorm 操作其融合会显著提高 SRAM 的访存效率，并减少layout的重置。可以先做. 

   是呀一般layernorm 都会fuse 可以参考AMD CK gemm_in_pipeline 写法

4. https://arxiv.org/abs/2008.13006   运行时统计数据, 根据数据来做块量化和块稀疏, 然后生成合适的算子来执行, 整个算力会有质的飞跃. 块量化同样精度下需要的bits更少, 块稀疏可以跳过不必要的计算.我之所以笃定能做这个事, 是因为 "矩阵乘法"的复杂度是n3, 而数据是n2, 这意味着对数据的统计占比不大. 对这个有兴趣的可以找我, 理论简单, 意义重大, 属于高性价比的内容了. 块量化这个肯定是有意义的, 这个在之前的channel量化就验证了. 块稀疏这个也就人做了, 它论文里说效果还行  https://github.com/galois-stack/galois/blob/preview/galois/ir/ir.hpp

attn是绝对的memory bound（不算qkvo proj),  大部分文章指的是gemm因为bs太小 memory bound了. attn我觉得不需要考虑tc了，除非说你某些量化的计算需要tc的支持.

https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/XQA-kernel.md  目前tensorrt-llm是在做的, decoding阶段用tensor core加速.

GQA/MQA就不一样了 这种情况下kv cache是可以被多个query head reuse的. 

# Chimera:

 An Analytical Optimizing Framework for Effective Compute-intensive Operators Fusion

优化框架，可以有效地提高计算密集型算子链在不同硬件加速器上的locality.

在 Chimera 中，每个计算密集型算子都由一系列计算块组成。为op 链生成高效的融合内核，需要对block间和 block内 intra进行优化. 对于块间优化，Chimera minimizing the data movement volume among blocks using an analytical model.来确定优化的块执行顺序。对于块内优化，Chimera 使用统一的可替换微内核对不同的加速器应用特定于硬件的优化。最后，Chimera 为计算密集型 op链生成融合内核。

#### 难点

1. 很难确定计算密集型算子链的计算执行顺序。链中的每个算子都可以分解成一系列的计算块，这些计算块的不同执行顺序会导致块之间的数据移动量不同，因此性能也会发生巨大变化.  因为它们缺乏精确的性能模型来评估不同排序选择的算子的数据移动量. 
2. 使用特定于硬件的功能优化每个模块内的计算具有挑战性。缺乏一种统一的方法来为不同的硬件加速器生成可扩展和灵活的微内核.

#### solution

Chimera 列举了不同的区块执行顺序，并分析估计了区块之间的输入/输出数据移动量。之后，Chimera 选择提供最小数据移动量的执行顺序，以实现最佳数据局部性。

#### Inter-block Optimization

Chimera 的输入是机器学习中的计算 DAG（由领域特定语言描述）。首先将 DAG 中的每个算子分解为一系列计算块, 然后用分析模型,  选择 执行顺序. 



矩阵 *B* 不会被重用，因为当我们沿维度 L 遍历块时，会访问矩阵 *B* 的不同数据块.

矩阵 *D* 和 *E* 始终沿 *k* 维度重复使用，因为 *k* 是第一个 GEMM 的私有对象，它不会迭代第二个 GEMM 的计算.

评估 bert, MLP mixer, vit 





halide 比tvm  在 sdp 平台上表现更好.   

stable diffusion  的 gemm 有vit这么多吗?    https://github.com/hahnyuan/LLM-Viewer 可以测op, roofline model.

group wise在硬件上写起来太困难了. 













