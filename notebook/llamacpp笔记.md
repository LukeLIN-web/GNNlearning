

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



#### llama原理

重点的代码在llama文件夹下面的generation.py和model.py。

ColumnParallelLinear

apply_rotary_emb 是啥 就是加上position embedding.

```
       values = values.transpose(1, 2) # (bs, n_local_heads, cache_len + seqlen, head_dim)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        # values会越来越大？  怎么越来越大呢？ 是拿之前所有的还是新加的？ 
```

采样是可以天然并行执行的.

#### prefill 阶段

会计算用户所有的输入，并生成对应的 KV 缓存.

也叫 prompt processing 

####  decoding 

每一个 decoding 过程服务器都会生成一个字符，并将其放入到 KV 缓存当中

dim = 768 /12 

乘法 activation 在前面还是权重在前面? 权重在前面, WX.  权重是4096 * 4096.

 LLAMA_CUBLAS已经废弃了. now it is  LLAMA_CUDA



```
ub是physical maximum batch size 很可能指的是 “micro-batch”（微批次）。
b是batch size   , logical maximum batch size
c是  --ctx-size N",           "size of the prompt context
npp PP (Prompt Processing):
ntg  TG (Text Generation)
npl   prompt length
S t/s: Total speed (tokens per second)，总速度

```



```
int64_t ne[GGML_MAX_DIMS]; // number of elements
//这就是tensor的shape,不过它的存放方式是倒着的
+                                   //比如[batch_size,multi-head,seq_len,head-dim] 
+        //他的存储方式是ne[0]=head-dim,ne[1]=seq_len,ne[2]=multi-head,ne[3]=batch_size
```

