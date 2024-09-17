

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

#### rope

都是用yarn吗? 还是就是普通的?  rope_neox和rope_norm 都是yarn了  , YaRN（另一种 RoPE 扩展方法），这是一种计算效率高的方法，用于扩展此类模型的上下文窗口，与以前的方法相比，需要的令牌少 10 倍，训练步骤少 2.5 倍,上下文长度可达 128k。

  就是 **GPT-NeoX style** RoPE风格. meta的llama 是**GPT-J**风格. 

GGML_METAL_KERNEL_TYPE_ROPE_NORM_F32 , 是直接和norm合并吗? 我不知道为啥这个叫norm,  没有norm操作. 



#### docs

在 https://github.com/ggerganov/llama.cpp/blob/master/examples/main/README.md

一些微调模型通过缩放 RoPE 扩展了上下文长度。例如，如果原始预训练模型的上下文长度（最大序列长度）为 4096 （4k），而微调模型的上下文长度为 32k。这是比例因子 8，应该通过将上述 `--ctx-size` 设置为 32768 （32k） 和 `--rope-scale` 8 来工作。

为什么rope scale 可以拓展上下文长度? 

https://www.omrimallis.com/posts/understanding-how-llm-inference-works-with-llama-cpp/

metal 代码不能打印变量. 

llama.cpp重构之后kv cache管理怎么这么复杂.一堆策略，看的晕死.

#### kvcache

llama.cpp的kv cache update这里为什么只有k的更新没有见到v的更新.  k与v的idx应该是相同的. 你看后面有个结构体叫kv_cell

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

https://github.com/ggerganov/ggml/blob/21f9e5c426b105841c2e346d8f1aafec398edf15/src/ggml-quants.c#L1515 可以参考 



### Legacy quants

#### Q4 0

32个weight 一个block, 每个block 内一次量化, 

每个block 求max. 

-max 到max 切开 2^4份,  Q4 的方法在实际实现时，是复用了 Q8 的存储空间存放了两个 int4 ，一个放在低四位上一个放在**位移 4 位**后的高四位上，这就是 INT 量化方法比 FP 量化的优势所在。这是因为具体硬件中没有单独的数据结构来存放 Q4 的数据.

因为值域范围太小，15.5 以上的值就没有表示方法了，所以代码中做一个简单的截取，这在 Q8 中没有做，这种截取本质的问题我认为不是丢弃了outlier ，而是只压缩了贴近 16 的一小部分数值。

参考 https://github.com/ggerganov/llama.cpp/discussions/1121#discussioncomment-10361167

一个block是几个字节?前两个字节包含 FP16 的比例因子,  18个字节吗? 

#### Q4 1

Q4_1 和 Q4_0 的差异在于， Q4_0只有 scale，是对称 quant，没有减掉 minus. 运算快.  

Q4 1 , bias可以防止偏态分布导致的量化空间浪费.  缩放系数d 更大, 保留更多信息. 

会多存一个M,   takes a block of 32 weights and gives each block a scaling factor 'd' and takes the minimum of the weights 'm' 因此量化权重“q”的最终权重是 q * d + m，并且采用相对较小的块大小使它们更有可能都在合理的量化范围内。值得注意的是，d 和 m 可以在不牺牲太多空间的情况下更准确地存储，因为开销除以 32。

Q4_k更进一步，取了 8 个块的“超级块”，并对其应用了另一个比例因子“d_s”和最小“m_s”，因此最终权重为 （q * d + m） * d_s + m_s，附加因子存储为 6 位而不是 4 位。  https://news.ycombinator.com/item?id=36577898  



- `GGML_TYPE_Q4_K` - "type-1" 4-bit quantization in super-blocks containing 8 blocks, each block having 32 weights. Scales and mins are quantized with 6 bits. This ends up using `4.5` bpw.

https://github.com/ggerganov/llama.cpp/pull/1684

https://github.com/ggerganov/llama.cpp/discussions/1121

K 后缀代表 [K-quants](https://link.zhihu.com/?target=https%3A//github.com/ggerganov/llama.cpp/pull/1684) 方法，再后边 S、M、L、XS 等等代表尺寸.



#### 代码

```cpp
template [[host_name("kernel_mul_mm_q8_0_f32")]]    kernel mat_mm_t kernel_mul_mm<block_q8_0,    2,     dequantize_q8_0>;

block_q =block_q8_0  , nl  = 2 , dequantize_func = dequantize_q8_0

device const block_q * x = (device const block_q *)(src0 + (r0 * BLOCK_SIZE_M + thread_row) * nb01 + offset0) + offset1;

each block_q contains 16*nl weights.  32个weight. 
  
  GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_MUL_MM_Q8_0_F16_F16,   //这个是.m文件中的描述符
mul_mm_q8_0_f16_f16,   //这个是.metal文件中的host_name
ctx->support_simdgroup_mm);// 参看函数的定义
```

llm_build_inp_embd是干啥的? 

是怎么调用kernel_mul_mm_q8_0_f32的?

就是用op, 然后op建图. 

模仿  GML_METAL_KERNEL_TYPE_MUL_MM_Q4_0_F16_F16

f16_f16_f16,  强调c=a*b都是f16,不然他默认c是f32

我们需要输出16

prompt eval和 eval的fa不是一个. 写上判断, 不开也跑f16.

#### 量化分片

如果数据尺度差好几个数量级, 小尺度可能归零, 所以要考虑fp8 等非线性量化.  同样的原因, 一般不用per tensor, 全矩阵一个scal factor

per channel, 就是一列或者一行共享一个scal factor. 一般 AxB,  A按行, 有M个行factor,  B 按列, 有N个列factor,  就是沿着inner dim 量化, 

LLM中. A是activation, 一行是一个token的embedding, 所以也叫per token量化. 

进一步分片, groupwise. 

但是tensorRT-LLM 会各种高级操作预处理, interleave col, permute row, add bias, 充分利用硬件速度. 

#### 预处理memorylayout

cuda用ldg 128 从HBM 到SMEM . 混合精度的时候, B要并行加载两列.  预处理就要把两列交织在一起.   两个放第一个列 .  两个 放第二列, interleave col , 也是一种memory layout吧. 

SMEM ldmatrix.x1 到 寄存器. 为warp中每个线程加载 连续4个bytes. 需要和 mma.m16n8k16 指令需要的数据排布对对齐. 所以也需要重排. 

mma 处理fp16的输入, 

int8转fp16 有特殊的算法. 具体看视频.

如果没有pack好,   编译器 unroll 会失效.  因为用了过多的寄存器,  必须从最后一层开始.  最后一个for loop太大了就不能unroll. `#pragma clang loop unroll(full)`

### reference

1. 【NVIDIA AI 加速精讲堂-TensorRT-LLM量化原理、实现与优化】 https://www.bilibili.com/video/BV1GE4m1R7Sa/?share_source=copy_web&vd_source=bb7496f78e4d303270b7c97ae8f69402
2. 笔记：Llama.cpp 代码浅析（四）：量化那些事 - 刀刀宁的文章 - 知乎
   https://zhuanlan.zhihu.com/p/672983861

https://github.com/pytorch/ao?tab=readme-ov-file#inference

市面上挺多都是基于trt做的,ppq也是 .   现在钦定的换成了torchao, torch的量化都凉了,开发成本过高.  而且这个weightonly的量化也不会加速  torchao的dynamic quantization可以得到int8/fp8 tensor core的助力.  但是不支持cuda .    自己基于torchinductor做的backend，和trt 10.3的性能差距在2.5%以内. 

## metal

数据类型 : https://github.com/alexiscn/metal-shading-language-specification/blob/master/ch02.md

metal的第一个commit是ecb217db4fcfa3880300ad08531a5fb6bb14.

#### 编译

```cmake
    if (GGML_METAL_SHADER_DEBUG)
        #   xcrun -sdk macosx metal    -fno-fast-math -c ggml-metal.metal -o ggml-metal.air
        #   xcrun -sdk macosx metallib                   ggml-metal.air   -o default.metallib
        #       ref: https://github.com/ggerganov/whisper.cpp/issues/1720
               add_custom_command(
            OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/default.metallib
            COMMAND xcrun -sdk macosx metal    ${XC_FLAGS} -c ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/ggml-metal.metal -o ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/ggml-metal.air
            )
           xcrun -sdk macosx metal -c MyLibrary.metal -o MyLibrary.air
           xcrun -sdk macosx metallib MyLibrary.air -o MyLibrary.metallib
```
https://juejin.cn/post/7029658159832629285

https://developer.apple.com/documentation/metal/shader_libraries/metal_libraries/building_a_shader_library_by_precompiling_source_files?language=objc





I've downloaded Xcode. Then I moved all *special* tools to CommandLineTools. After that I threw Xcode into trash. For now everything seems to work fine.

I wonder whether I could `cp /Applications/Xcode.app/Contents/Developer ` to `/Library/Developer/CommandLineTools` , Xcode hack and open every markdown, really bad Xcode. 

移动到哪里呢?  /Library/Developer/CommandLineTools/Library/Developer/ 还是sdk文件夹. 



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

ggml assert不对会直接报错吗. 停吗? 

#### fa

threadgroup就是 cuda的 block.

simdgroup 是 cuda 的 warps.  

但是他是怎么切分矩阵的? 要看出来.  

两个unroll嵌套，我记得会提示展开失败吧. 两个unroll嵌套, 会展开4x4次吗? 会很长?



simdgroup要手动指定, warp是GPU自动指定的吗? 是的.  如果没有32个并行的怎么办? CUDA仍然会将这些线程分配到`warp`中，但这些`warp`会部分填充，剩余线程被称为“空闲线程”，不会实际执行指令。

为什么他要有1x4和8x4两个kernel? 



fa不需要读取weight.  所以和Q8 无关. 

mm 矩阵乘法有模版.  mv没有模版. getrows有模板. 



prompt和 eval 不是同一个kernel. 

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

#### tokenizer

Tokenizer也是在庞大的预训练语料上训练出来的，只不过由于计算需求相对训练模型少很多。 

 常用的Tokenization方法包括Byte Pair Encoding（BPE)，Uniform Language Model（ULM），WordPiece等等方法。而SentencePiece中主要是用的就是BPE和ULM，并且LLAMA2 选择的是SentencePiece中的BPE方法。

Llama-3将tokenizer由sentencepiece换成了tiktoken，这与GPT4 保持一致. Vocabulary-size  扩大到128K,  这是模型能识别的所有不同token的数量, ，使用了GQA.  Context Window Size = 8K也就是 max_seq_len.

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



LLM_ARCH_NOMIC_BERT, LLM_ARCH_BERT等用的是gqa

一样的超参数处理.   kv用的是gqa

gqa是什么? 不是所有 Q 头共享一组 KV，而是**分组一定头数 Q 共享一组 KV**，比如两组 Q 共享一组 KV. 

输入前处理在cpu, 后处理有的也在cpu. 

文字质量, 可以用数据集测.  

q40 和q80, mul_mv是不同的kernel. 

对比调用kernel 的差异. 

Q4 完全不行. 对于小模型,  量化到 10-11位是开始有影响的, 12位不会有影响.

https://www.53ai.com/news/qianyanjishu/1120.html

为什么8B 了 比7B 更大了?   **intermediate_size：11008->14336。**只是FFN时的中间维度变了，计算范式不变。参数量**增大**：`32*4096*(14336-11008)*3*2/1024/1024` Byte **(2496MB)**     

训练脚本在hugging face上 https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct/blob/main/config.json,  

tokenizer已经处理了, 是cpu处理的吗?  不知道该怎么确认. 

只有linear 有weight.

inf, 为什么?  累加需要用f32, 不然会f16 爆范围. 

128的fa 没有注释, 256的fa 注释了, 为什么? 

 T-Mac,   优点,非常快,  缺点, 要生成代码, 工作量很大. 

LUT  , 把每种向量和矩阵组合记住,  

为什么GPU的share memory不够快? 

把可能的排列都记下来. 保存矩阵乘法的结果. 

