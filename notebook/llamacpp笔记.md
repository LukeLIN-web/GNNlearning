

## 安装

`make -j` 一分钟。 

模型下载,  就是hugging face 下载， 然后转换模型。 

```
python convert-hf-to-gguf.py Path_To_Qwen

./llama-cli --hf-repo huggingface的模型 -GGUF --hf-file 对应的.gguf -p "The meaning to life and the universe is"
```

也有安卓, termux

### 结构

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



#### 量化

./examples/quantize/quantize.cpp

ggml 的 fp16 模型转换为 int8、int4 等格式。其实牛逼的 llama.cpp 支持了很多种量化方法，并且还贴心的在定义中给出了每个量化方法的 ppl 变化，如下图所示，例如 Q8_0 只增加了 0.0004 的 ppl，相当于就没啥变化，相当强。并且，llama.cpp 自己还提供[测试 ppl 的脚本方法](https://link.zhihu.com/?target=https%3A//github.com/ggerganov/llama.cpp%3Ftab%3Dreadme-ov-file%23perplexity-measuring-model-quality)。

ppl是啥

K 后缀代表 [K-quants](https://link.zhihu.com/?target=https%3A//github.com/ggerganov/llama.cpp/pull/1684) 方法，再后边 S、M、L、XS 等等代表尺寸.

笔记：Llama.cpp 代码浅析（四）：量化那些事 - 刀刀宁的文章 - 知乎
https://zhuanlan.zhihu.com/p/672983861

## metal

为了根据指定的数据执行流水线，我们需要创建一个指令缓冲区**（command buffer）**来写入指令，并将缓冲区里的指令提交到指令队列中。在完成这些后，Metal API 会将这些指令发送至 GPU 以进行计算。

在 Metal API 中，可以在GPU上运行的代码被称为 **Shader** 



参考 https://github.com/ggerganov/llama.cpp/pull/1860

- Use multiple command buffers
- Enqueue them at start
- Encode the commands in parallel on separate threads
- Each thread commits its buffer
- Wait for the last one to finish



该 `dispatch_apply` 函数是 Apple C 语言 Grand Central Dispatch （GCD） 的一部分。它用于并行执行循环，将循环的迭代分布在可用线程之间。这可以显著加快可并行化的操作速度。

n_cb是什么? 

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

`endEncoding` 方法结束 Encoder 编码过程。最后 `MTLCommandBuffer` 对象的 `commit` 方法被调用，使得计算指令被尽快执行。

并行计算程序按照 Encoder 被推入 command buffer 的次序执行. 

并行计算着色程序执行完毕。这意味着前一个 Encoder 产生的数据可以被下一个 Encoder 使用。



#### handler

 addScheduledHandler(_:)  Registers a completion handler the GPU device calls immediately after it **schedules** the command buffer to run on the GPU.

gpu 可以 identifies command buffer’s dependencies, 然后schedule command buffer和tasks. 然后 sets the command buffer’s status to [`MTLCommandBufferStatus.scheduled`](https://developer.apple.com/documentation/metal/mtlcommandbufferstatus/scheduled) and calls your scheduled completion handler. 

addCompletedHandler Registers a completion handler the GPU device calls immediately after the GPU **finishes** running the commands in the command buffer. 

这两个都是 completion handler







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

 LLAMA_CUBLAS已经废弃了. now it is  LLAMA_CUDA



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







