## flashattention

什么是flash attention

block size:  M/4d,  M是啥？ 是SRAM的大小。 

Bc： M/4d  Br：   M/4d 就是K和V分别放几个

为什么是4d？ 一个float32是4byte.

xi - m 很小的时候会丢失fp16精度吗? 可能就是关注最大的数. 接近0的不重要. 

llama.cpp 原版没flashattention 是safe softmax吗? 

oi-1需要要开空间去存吗?

tiling版本, K和V 一次是b行吗? 还是b个block? 

softmax是沿着哪个维度算的？ 



FA是为了训练， 因为要不断计算N平方的复杂度。 推理的benefit很小。 写回的attention map 也只有 1x d（sequence length 还是cache length？ ） 为什么？  因为Q 没有cache。所以应该到不了2.4x加速

dropout ， inference 不用， training才有。 把10%的元素设为0

看源码的话， 看 Triton 官网上的版本，  cutlass里面的实现就是和flash attention v2一模一样 稍微都点区别的是cutlass里面的softmax分块用的是最后一个分块的值作为最大值，并不是整个的作为最大值，还有就多用了一个b2b的gemm的优化方法，其余都和常规的gemm一样。

 用的mma 没有针对一个线程的。就是一个thread group或者一个warp一起做一个mma

#### online softmax

为了数值稳定性（因为指数增长太快，数值会过大甚至溢出），会减去最大值：这样带来的代价就是要对𝑥多遍历1次。

fuse有把Vfuse进去吗， 还是要softmax整行算完才 算V？ 

要计算几个值， 每行的最大值mij,   计算  Pij ， 每行的Pij加起来  l ij 。

torch的需要transpose，flash attention的不需要，这样写结果是对的

怎么做到backward 不存储中间attn 矩阵?  存储前向传递的归一化因子. 

https://github.com/facebookresearch/xformers 提供了各种attention的实现. 

【论文分享：从Online Softmax到FlashAttention-2】 https://www.bilibili.com/video/BV1aa4y1r7Fb/?share_source=copy_web&vd_source=bb7496f78e4d303270b7c97ae8f69402

根据roofline model, d小于256的时候, 大部分情况下是memory bound的. 

N是啥. seq length. 

因为SRAM是 HBM速度的十多倍.  所以我们要尽量让数据在SRAM中多次计算.

我们先看如何把softmax 给fuse. 这个算法叫online softmax.

首先, 为了防止梯度爆炸,  我们要减去一行中的最大值. 

所以呢，我们需要遍历三次，第一次找到最大值， 第二次计算到总和，    第三次再计算e除以总和

x>=11, e^x exceed fp16 范围 (2^16).   所以要求一行中的最大值.

需要三次softmax. online softmax节省到两次.  

为什么递推 就可以取代mn呢? 保证最后一个结果dN = dN'相等就行. 

提出di' , 和di-1'.  di' 的计算, 就不需要知道mn, 所以就可以和第一次遍历合并. 只要最后保证dN是对的就行.  i= N的时候, dN' = dN.

attention, 用了online softmax 两次遍历. 

之后flash attention发现, 我们不用算mN. 就直接算mi. 同样道理, i= N的时候保证o'n = on.  所以也是一样道理, 改变oi的中间结果, 保证on对就行.  就可以消去ai. 只需要一次loop.

就可以一次遍历!

#### tiling

在GPU上, 需要tiling.

A100 20MB  sram, 108SM x 192KB per SM .

#### 伪代码

4d可能是fp32的原因.  可能是为了sram里完全放的下Kj,Vj,Qi,Oi这四个块，不至于内循环重复读取Qi和Oi的时候Kj和Vj被换出去. 

v2就在外面了. V1 q在里面. 

Lij就是 上面的dij 全局EXP求和项.   为什么要写回li和mi to HBM?  Li是用来backward 计算的. forward没用到.  softmax 的更新要用 l_all 和 m_max，因为过程中记录的局部变量只有这 2 个。l(x(1)) 和 e^m(x1) 在运算完首块之后就没了。

diag 就是把vector 变成矩阵.  **给定一个一维张量，它会返回一个以该一维张量为对角线元素的方阵**.  具体还要看看代码. 怎么实现的. 

前提就是d^2 / M << 1, 才能加速.  sram size M大概 100KB, d 一般是64或128.

在分块计算softmax的时候，能否在最后一次整体地更新一次全局softmax，而不是经过每个分块都要当前全局更新一下？ 不过这个代价就是对每个分块都要一直maintain这两个局部scalar，SRAM的存储压力会变大. 所以就有了flashattention2

## flashattention2

flashattention v1论文typo太多了，有些思路也被v2推翻了

贡献

1. 减少non-matmul FLOPs并尽可能多地执行matmul FLOPs非常重要。利用gemm 专用单元, 如nvidia的tensorcore, 苹果是否也有?  有 AMX  . 使用 Metal Performance Shaders （MPS） 框架
2. 提出了在序列长度维度上并行化。该方法在输入序列很长（此时batch size通常很小）的情况下增加了GPU利用率。即使对于单个head，也在不同的thread block之间进行并行计算。
3. 在一个attention计算块内，将工作分配在一个thread block的不同warp上，以减少通信和共享内存读/写。



为了减少non-matmul FLOPs，本文在FlashAttention基础上做了两点改进. 

1. 计算局部attn, 先不考虑分母.
2. 不需要rescale, 得弥补之前局部max值，最后一步再调整. 

#### 跳过计算

由于FlashAttention和FlashAttention-2已经通过块操作来实现，对于所有列索引都大于行索引的块（大约占总块数的一半），我们可以跳过该块的计算 .   之前计算了然后扔掉, 现在就不需要计算了! 

不需要对那些行索引严格小于列索引的块应用casual mask。这意味着对于每一行，我们只需要对1个块应用casual mask。

#### Parallelism

fa1的优化:

一个thread block 处理一个attn head. 

但是 在处理长序列输入时，由于内存限制，通常会减小batch size和head数量，这样并行不够, 。因此，FlashAttention-2还在序列长度这一维度上进行并行化.   达到A100上108个SM的并行度. 

为此, 需要splitting Q in outer loop.

#### warp通信减少

一个thread block中的warps,  需要用smem communicate. 

一个warp中的thread 可以用fast shuffle inst. or cooperate 来perform matmul.

**Forward pass.** FlashAttention算法有两个循环，K,V在外循环j，Q,O在内循环i。FlashAttention-2将Q移到了外循环i，K,V移到了内循环j，**由于改进了算法使得warps之间不再需要相互通信去处理**Qi，所以外循环可以放在不同的thread block上。



为啥Q在外面 KV在内 就可以共享K和V?

因为代码上把Q 分成4个warp, 每个warp都可以访问K和V.

fa1 需要跨warp同步 QxK的结果,  fa2就不用.

应该是i = 1, warp1 可以一份QK给多个线程共享. 不同线程可以share QK.

可以讲一讲代码.  看看bilibli有没有讲代码的视频. 

讲讲怎么并行. 

S就是QK的中间矩阵. 

li 是

Output one 𝑂C at a time, so that only one rescaling is needed after the inner loop ends, reducing non-matmul calculations.

rescaling 啥意思?

  https://zhuanlan.zhihu.com/p/645376942

O 写入放在外层循环. 内层完全不需要写入Oi到 HBM. v1中，o需要不断的从hbm读写，但是，v2中，kv也要反复从hbm里面读到sram在这一点上，算打个平手。

https://github.com/tspeterkim/flash-attention-minimal/blob/main/flash.cu  refer  【[手写flash attention v1 & v2] baseline的基础实现】 https://www.bilibili.com/video/BV1zM4m1S7gg/?share_source=copy_web&vd_source=bb7496f78e4d303270b7c97ae8f69402

## FlashDecoding++  

## fa3. 

用了h100的MTA. flash attention 3和2相比感觉没啥新意. sm90 hgemm版本的attention在cudnn里早就有了. fp8的用triton也早就实现过了. 也只是有些shape下快，不少shape还更慢呢. 都是只挑自己的长处. 

待从源码中学习下 V 的 [in-kernel transpose 另一个亮点就是 [fp8 quantization error](https://www.zhihu.com/search?q=fp8 quantization error&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A3559908421}) 的控制.



怎么确认

Layernorm, 先算平均值,再算方差, 3遍. 数组大的时候, 然后更新一遍值. 

fa 算子应该几十ms , 调用kernel前后需要 torch.cuda.synchronize()