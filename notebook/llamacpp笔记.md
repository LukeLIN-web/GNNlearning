

## 安装

`make -j` 一分钟。 

模型下载,  就是hugging face 下载， 然后转换模型。 

```
python convert-hf-to-gguf.py Path_To_Qwen

./llama-cli --hf-repo huggingface的模型 -GGUF --hf-file 对应的.gguf -p "The meaning to life and the universe is"
```

也有安卓, termux

### 结构

ggml真复杂, cpp 推理很难, 要写非常多的代码 infra. 

 llama.cpp 大概是：a算子的运行，结果给b算子，运行时候去逐个算子执行，不是初始化的时候把DAG就先构建出来

ggml_cgraph 是负责结构的， ggml_object 是负责存储的，ggml_cplan 是负责多线程执行的，ggml_context 应该是维护 object 级别关系的。

ggml类项目都先构图的 

在llama_decode_internal 里，先build graph 再 graph compute；每一次推理都需要重新构建一次计算图，再计算



llama.cpp 需要自己要从头建图，以完成数据流转、串联算子、调度运算

他们把不同模型比如 build_baichuan 都塞在llama.cpp文件里面， 

ggml 格式模型是 fp16 的.

正常llama hidden sizes 是4096.



#### docs

在 https://github.com/ggerganov/llama.cpp/blob/master/examples/main/README.md

一些微调模型通过缩放 RoPE 扩展了上下文长度。例如，如果原始预训练模型的上下文长度（最大序列长度）为 4096 （4k），而微调模型的上下文长度为 32k。这是比例因子 8，应该通过将上述 `--ctx-size` 设置为 32768 （32k） 和 `--rope-scale` 8 来工作。

为什么rope scale 可以拓展上下文长度? 



metal 代码不能打印变量. 

#### common/sampling.cpp

每一个 token 的输出的时候都要 sampling 一次





llama_synchronize 

n queued tokens是什么? 

```cpp
    if (ctx->n_queued_tokens == 1) {
        ctx->t_eval_us += ggml_time_us() - ctx->t_compute_start_us;
        ctx->n_eval++;
    } else if (ctx->n_queued_tokens > 1) {
        ctx->t_p_eval_us += ggml_time_us() - ctx->t_compute_start_us;
        ctx->n_p_eval += ctx->n_queued_tokens;
    }

n_queued_tokens是什么? 
  把 n_queued_tokens 150个全部加上.在哪里加? 
number of tokens processed in batches at the beginning
prompt processing到底干了啥? 
```





## 量化

./examples/quantize/quantize.cpp

ggml 的 fp16 模型转换为 int8、int4 等格式。其实牛逼的 llama.cpp 支持了很多种量化方法，并且还贴心的在定义中给出了每个量化方法的 ppl 变化，如下图所示，例如 Q8_0 只增加了 0.0004 的 ppl，相当于就没啥变化，相当强。并且，llama.cpp 自己还提供[测试 ppl 的脚本方法](https://link.zhihu.com/?target=https%3A//github.com/ggerganov/llama.cpp%3Ftab%3Dreadme-ov-file%23perplexity-measuring-model-quality)。

ppl是啥, 就是不确定度. 

4bit 来存32bit. 

对称量化 , 对称量化则是非对称量化的一种特殊形式，其限制了将零点映射为0



### Legacy quants

#### Q4 0

32个weight 一个block, 每个block 内一次量化, 

每个block 求max. 

-max 到max 切开 2^4份,  Q4 的方法在实际实现时，是复用了 Q8 的存储空间存放了两个 int4 ，一个放在低四位上一个放在**位移 4 位**后的高四位上，这就是 INT 量化方法比 FP 量化的优势所在。这是因为具体硬件中没有单独的数据结构来存放 Q4 的数据.

因为值域范围太小，15.5 以上的值就没有表示方法了，所以代码中做一个简单的截取，这在 Q8 中没有做，这种截取本质的问题我认为不是丢弃了outlier ，而是只压缩了贴近 16 的一小部分数值。

#### Q4 1

Q4_1 和 Q4_0 的差异在于， Q4_0只有 scale，是对称 quant，没有减掉 minus. 运算快.  

Q4 1 , bias可以防止偏态分布导致的量化空间浪费.  缩放系数d 更大, 保留更多信息. 

会多存一个M,   takes a block of 32 weights and gives each block a scaling factor 'd' and takes the minimum of the weights 'm' 因此量化权重“q”的最终权重是 q * d + m，并且采用相对较小的块大小使它们更有可能都在合理的量化范围内。值得注意的是，d 和 m 可以在不牺牲太多空间的情况下更准确地存储，因为开销除以 32。

Q4_k更进一步，取了 8 个块的“超级块”，并对其应用了另一个比例因子“d_s”和最小“m_s”，因此最终权重为 （q * d + m） * d_s + m_s，附加因子存储为 6 位而不是 4 位。  https://news.ycombinator.com/item?id=36577898  



一个block 是16个字节吗? 

- `GGML_TYPE_Q4_K` - "type-1" 4-bit quantization in super-blocks containing 8 blocks, each block having 32 weights. Scales and mins are quantized with 6 bits. This ends up using `4.5` bpw.

https://github.com/ggerganov/llama.cpp/pull/1684

https://github.com/ggerganov/llama.cpp/discussions/1121

K 后缀代表 [K-quants](https://link.zhihu.com/?target=https%3A//github.com/ggerganov/llama.cpp/pull/1684) 方法，再后边 S、M、L、XS 等等代表尺寸.

#### 量化分片

如果数据尺度差好几个数量级, 小尺度可能归零, 所以要考虑fp8 等非线性量化.  同样的原因, 一般不用per tensor, 全矩阵一个scal factor

per channel, 就是一列或者一行共享一个scal factor. 一般 AxB,  A按行, 有M个行factor,  B 按列, 有N个列factor,  就是沿着inner dim 量化, 

LLM中. A是activation, 一行是一个token的embedding, 所以也叫per token量化. 

进一步分片, groupwise. 

但是tensorRT-LLM 会各种高级操作预处理, interleave col, permute row, add bias, 充分利用硬件速度. 

#### 预处理memorylayout

cuda用ldg 128 从HBM 到SMEM . 混合精度的时候, B要并行加载两列.  预处理就要把两列交织在一起.   就是 两个第一个列 然后两个第二列, interleave col , 也是一种memory layout吧. 

SMEM ldmatrix.x1 到 寄存器. 为warp中每个线程加载 连续4个bytes. 需要和 mma.m16n8k16 指令需要的数据排布对对齐. 所以也需要重排. 

mma 处理fp16的输入, 

int8转fp16 有特殊的算法. 具体看视频.

如果没有pack好,   编译器 unroll 会失效.  因为用了过多的寄存器,  必须从最后一层开始.  最后一个for loop太大了就不能unroll. `#pragma clang loop unroll(full)`





### reference

1. 【NVIDIA AI 加速精讲堂-TensorRT-LLM量化原理、实现与优化】 https://www.bilibili.com/video/BV1GE4m1R7Sa/?share_source=copy_web&vd_source=bb7496f78e4d303270b7c97ae8f69402
2. 笔记：Llama.cpp 代码浅析（四）：量化那些事 - 刀刀宁的文章 - 知乎
   https://zhuanlan.zhihu.com/p/672983861

## metal

metal的第一个commit是ecb217db4fcfa3880300ad08531a5fb6bb14.

#### 结构



包装了metal buffer, context,

```objective-c
struct ggml_metal_context * ggml_metal_init(void) {
read the source from "ggml-metal.metal" into a string and use newLibraryWithSource
//  load kernels 
  #define GGML_METAL_ADD_KERNEL(name)这个宏会为每个计算核（kernel）执行以下步骤：
使用给定的名称从库中创建一个新函数，并将其保存到 ctx->function_##name 。
使用该函数创建一个新的计算管线状态，并将其保存到 ctx->pipeline_##name 。
  
  ggml_metal_graph_compute{
  for (int i = 0; i < gf->n_nodes; ++i) {
  根据每个op 进行set. 比如 
    [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
    
  }
     [command_buffer commit];
    [command_buffer waitUntilCompleted];
}
  
  .metal的文件不是直接调用的. 通过string读入, 用 newLibraryWithSource 加载到 ctx->library.
  .metal的函数命名就是 kernel_  +  .m里的名字, mul_mat_q4_0_f32
```

https://github.com/ggerganov/llama.cpp/pull/1642/commits/b23fe8c9c78d066461c81447566844ecf22a4a8e



没有测试, 都不知道怎么单独运行算子. 

将缓冲区里的指令提交到指令队列中。在完成这些后，Metal API 会将这些指令发送至 GPU 以进行计算。

可以同时存在多个命令队列，每个队列可以独立地调度和提交命令缓冲区。在单个命令队列中，命令缓冲区按提交顺序执行；在多个命令队列中，不同队列中的命令缓冲区可以并行执行。

参考 https://github.com/ggerganov/llama.cpp/pull/1860

- Use multiple command buffers, 
- Enqueue them at start
- Encode the commands in parallel on separate threads
- Each thread commits its buffer
- Wait for the last one to finish



该 `dispatch_apply` 函数是 Apple C 语言 Grand Central Dispatch （GCD） 的一部分。它用于并行执行循环，将循环的迭代分布在可用线程之间。这可以显著加快可并行化的操作速度。

n_cb是什么?  就是  gf->n_threads;  command buffer数量.  

为什么要iter多次呢? 

```objective-c
dispatch_apply(n_cb, ctx->d_queue, ^(size_t iter) {
  for (int i = node_start; i < node_end; ++i) {

   // 每个node插入一个handler. 时间波动很大, 为什么? 
                CFAbsoluteTime startCommitTime = CFAbsoluteTimeGetCurrent(); //这个是分配的时间, 不是开始的时间.

            [command_buffer addCompletedHandler:^(id<MTLCommandBuffer>  command_buffer) {
                               CFTimeInterval start = command_buffer.GPUStartTime;
                CFTimeInterval end = command_buffer.GPUEndTime;//CFTimeInterval精度是秒. 精度太低了. 
            }];
            // end profiling
  }
}
    //  因为每个handler是并行的, 不是串行的 ? 也不是, 但是startCommitTime 也是每个node独立的. 
               %.9f 才能保留到纳秒, .f 只能到微秒. 
               CFAbsoluteTimeGetCurrent()单位秒，保留到纳秒
               CFTimeInterval , 单位是秒, 保留到纳秒.  
               command_buffer.GPUStartTime 精确到纳秒.
```

可能用的是第一个token的缘故.

都用最后一个token. 

metal文档有object c和swift 两种语言

靠buffer没法得到运行时间.  但是不靠buffer我们不知道什么时候完成. 



intel cpu上, pytorch 不支持fp16, apple 底层arm是支持的. 



#### MTLComputeCommandEncoder

encoder 是负责 setbuffer, setbytes 

创建一个 command buffer，用它创建一个 `MTLComputeCommandEncoder` 对象. `MTLComputeCommandEncoder` 对象用于编码计算指令、设置入参、执行计算程序. 设定资源对象（缓存，纹理，采样器 state，线程组内存）

```cpp
id <MTLFunction> func = [library newFunctionWithName:@"filter_main"]; //绑定函数
id <MTLComputePipelineState> filterState
              = [device newComputePipelineStateWithFunction:func error:&errors]; //MTLFunction 对象被用来创建一个叫做 filterState 的 MTLComputePipelineState 对象。
[computeCE setComputePipelineState:filterState]; // 绑定encoder

为 state 对象设置资源 (MTLBuffer, MTLTexture,MTLSamplerState) ，这些资源中包含了待处理数据或是被 state 对象返回的数据. 同时还要设置这些资源的参数索引表
```

`endEncoding` 方法结束 Encoder 编码过程。最后 `MTLCommandBuffer` 对象的 `commit` 方法被调用，buffer真正提交到 command queue。

并行计算程序按照 Encoder 被推入 command buffer 的次序执行.  前一个 Encoder 产生的数据可以被下一个 Encoder 使用。





#### handler

 addScheduledHandler,    Registers a completion handler the GPU device calls immediately after it **schedules** the command buffer to run on the GPU. 

gpu 可以 identifies command buffer’s dependencies, 然后schedule command buffer和tasks. 然后 sets the command buffer’s status to [`MTLCommandBufferStatus.scheduled`](https://developer.apple.com/documentation/metal/mtlcommandbufferstatus/scheduled) and calls your scheduled completion handler. 

addCompletedHandler, Registers a completion handler the GPU device calls immediately after the GPU **finishes** running the commands in the command buffer. 

这两个都是 completion handler



#### 测时间

说法

1 The best way to measure these things is to use on device performance monitor counters  , 是Xcode? 

2 可以用events, MTLSharedEvent. `MTLSharedEvent` 是 Metal 中的同步原语，可用于协调 CPU 和 GPU 之间的工作，或不同 GPU 命令缓冲区之间的工作。虽然它主要不是为精确的计时测量而设计的，但它仍然可以通过使用其信号和等待功能来测量 GPU 任务的持续时间。 这个和scheduled time 功能一样. 

3  MTLCounterSampleBuffer,    在 Metal 内核中直接存取时间戳并不常见。通常，时间戳记录在 CPU 端通过 `MTLCounterSampleBuffer` 实现。

```objective-c
id<MTLCounterSet> timestampCounterSet = [device counterSets][0]; // Assume the first counter set is the timestamp counter set.
id<MTLCounterSampleBuffer> sampleBuffer = [device newCounterSampleBufferWithDescriptor:[MTLCounterSampleBufferDescriptor descriptor] error:&error];

[computeEncoder sampleCountersInBuffer:sampleBuffer atSampleIndex:0 withBarrier:NO];
```

但是llamacpp目前写法是dispatch_apply   全部command buffer 结束了后一个个command buffer 查看状态, 没法这么插入. 

4  在 Metal 内核中直接存取时间戳并不常见

```
uint64_t startTime = clock::get_timestamp();
```

metal代码里不能直接print. 



sd.cpp 用Ggml graph print 可以打印时间. 





## llama原理

计算大部分都是fp16的.

重点的代码在llama文件夹下面的generation.py和model.py。

ColumnParallelLinear

apply_rotary_emb 是啥 就是加上position embedding. xk 加上位置编码之后就放入 kvcache

ROPE算子就是加 position embedding

位置编码还有  ALiBi can support long inference context length. 百川模型. 

采样是可以天然并行执行的.

#### prefill 阶段

会计算用户所有的输入，并生成对应的 KV 缓存.

也叫 prompt processing 

####  decoding 

每一个 decoding 过程服务器都会生成一个字符，并将其放入到 KV 缓存当中

dim = 768 /12 

乘法 activation 在前面还是权重在前面? 权重在前面, WX.  权重是4096 * 4096.

 LLAMA_CUBLAS已经废弃了. now it is LLAMA_CUDA

输入的embedding 每个128 计算, 不是dim=4096直接计算. 

```
ub是physical maximum batch size 很可能指的是 “micro-batch”（微批次）。
b是batch size   , logical maximum batch size
c是  --ctx-size N",           "size of the prompt context
npp PP (Prompt Processing):
ntg  TG (Text Generation)
npl   prompt length
S t/s: Total speed (tokens per second)，总速度
```

#### alibi

在query和key做[矩阵点乘](https://www.zhihu.com/search?q=矩阵点乘&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"article"%2C"sourceId"%3A"657161287"})的基础上，加上一个常数负值，比如距离当前位置前1位为-1， 前两位为-2，这些常数要乘上 权重 m. **优势:**

- 减少了需要训练的Embedding，加快训练速度
- 较原位置编码，具有更好的长度外推性

```cpp
int64_t ne[GGML_MAX_DIMS]; // number of elements
//这就是tensor的shape,不过它的存放方式是倒着的
比如[batch_size,multi-head,seq_len,head-dim] 
他的存储方式是ne[0]=head-dim,ne[1]=seq_len,ne[2]=multi-head,ne[3]=batch_size
```

1. sample , promote eval和eval分别是啥? 我们应该看哪个数据? 

sample 就是decode出来的长度,  都是eval time+1,  时间很短可以忽略不计.  吕程说是 长度为2的矩阵把网络过了一遍. 可能是预分配空间

 promote eval 是prefill 加上生成第一个token的速度,tokens数包括start符号,  算attention的时间少, 会快一点. 

eval就是decode, cache的时间长append, 会越来越慢. 用户的体验. 

total time, tokens数不包括start符号

chat 和普通的区别? chat weight不一样.   text是续写. chat要考虑提示词.  

第一次 prompt eval  用 gemm kernel.   prompt eval 是并行的吗 ?   

后面的decode  用 gemv kernel . 因为 只有一个tokens. 

