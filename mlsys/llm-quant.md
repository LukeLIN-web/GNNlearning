## 量化

./examples/quantize/quantize.cpp

llama.cpp 支持了很多种量化方法，并且还贴心的在定义中给出了每个量化方法的 ppl 变化，如下图所示，例如 Q8_0 只增加了 0.0004 的 ppl，相当于就没啥变化，相当强。并且，llama.cpp 自己还提供[测试 ppl 的脚本方法](https://link.zhihu.com/?target=https%3A//github.com/ggerganov/llama.cpp%3Ftab%3Dreadme-ov-file%23perplexity-measuring-model-quality)。

对称量化则是非对称量化的一种特殊形式，其限制了将零点映射为0

https://github.com/ggerganov/ggml/blob/21f9e5c426b105841c2e346d8f1aafec398edf15/src/ggml-quants.c#L1515 可以参考 

溢出就取最大, 超过127的都变成127. 

一般 weight 是per channel, activations  不能per channel, 因为硬件不支持,  是per tensor.

#### group 量化

 32个数字一个scale.  一个channel用一个scale 精度会掉太多,但是 paper都是这样做的, ppl不会掉, 但是实际上用起来非常差.  weight only 8bit  实际能用的都没有.  

group wise  weight 和 activation一起,是做不了的.   会出现所有scale 互相乘的组合. 

 很多论文完全没有测精度, 很多速度非常快 , 实际上 模型的精度根本不能用.  

architecture 优化模型速度的论文都不work.  llama说不出人话 , 但是 准确率指标还是很高.

weight 4 bit的精度也很差. 大模型不行, resnet, yolo, bert  可以. 

activation  channel wise? 

英伟达推4x4, 4x8. group wise. 

英伟达中间可以转浮点数, activation用浮点数 8bit就行.  

8 和16 bit 都做. 会增加不少. 做8bit.  中间全是定点数. 

E2e 量化, 最难就是32 怎么定点回到 8, 直接切 scale只能取1/2的倍数, 精度损伤巨大. 必须重新train.  必须要有浮点运算的东西.   也可以找人支持一下16bit. 

NPU 准确率是失控状态.  用定点数 凑出浮点数, 他们甚至没有浮点数算子.  都是假定定点数. 

必须倒回CPU 做 32bit 转8bit . 不能直接切, 要转浮点数, 重新量化8bit , 输入conv2.

activation  8bit, 结果肯定不行.   中间结果 16bit 可能可以做. 但是 硬件平台只能做到这个地步. 

 高通  int x 浮点数,  要先int 转浮点数.  mac可能 compiler会自动处理. 

 16 x16可以放32bit, 但是定点数不能放32bit , 会超出. 

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

#### 量化分片

如果数据尺度差好几个数量级, 小尺度可能归零, 所以要考虑fp8 等非线性量化.  同样的原因, 一般不用per tensor, 全矩阵一个scal factor

per channel, 就是一列或者一行共享一个scal factor. 一般 AxB,  A按行, 有M个行factor,  B 按列, 有N个列factor,  就是沿着inner dim 量化, 

LLM中. A是activation, 一行是一个token的embedding, 所以也叫per token量化. 

进一步分片, groupwise. 

但是tensorRT-LLM 会各种高级操作预处理, interleave col, permute row, add bias, 充分利用硬件速度. 

市面上挺多都是基于[TensorRT](https://developer.nvidia.com/tensorrt) 做的.   现在钦定的换成了torchao, torch的量化都凉了,开发成本过高.  而且这个weightonly的量化也不会加速,  torchao的dynamic quantization可以得到int8/fp8 tensor core的助力.  但是不支持cuda. 自己基于torch inductor做的backend，和trt 10.3的性能差距在2.5%以内. 

https://github.com/pytorch/ao?tab=readme-ov-file#inference

#### AffineQuant

该方法通过左乘仿射变换矩阵到线性层的权重，并右乘激活的逆矩阵，优化仿射变换矩阵，以减少量化误差。这种方法不仅能够提高模型在低比特量化配置下的表现

#### 动态量化和静态量化

动态量化 （DQ） 主要针对激活，支持从高精度格式（如 bf16）到较低精度格式（如 int8）的动态量化。Int8 动态量化在 [SAM](https://github.com/pytorch-labs/segment-anything-fast) 等计算受限模型上效果最佳，而 batchsize=1 的 Llama 往往受内存限制，因此性能相当低。

动态量化的关键思想是，我们将根据运行时观察到的数据范围动态确定激活的比例因子。这可确保对比例因子进行“调整”，以便保留有关每个观测数据集的尽可能多的信号。缺点是计算量大, 计算过程中计算比例因子. 

静态量化, 先用代表性数据运行模型, 得到大致可能的范围, 作为scale. 一般静态比较多. 

f1就是第一层 量化前的float. bias处理有技巧, llama 所有层都没有bias.

非e2e 量化, 算了int乘法, 然后scale乘 Sw1 Sa1, 再scale 除以 Sa2, 一般会fusion 这两步.

E2e,就是一个int全部算完,

加起来是32bit, 先变成 16bit, 还需要除Sd1, 向右移动, 再把低位取出来.   精度的误差主要来源于 Sd1 只能取2的指数. 

有指数的, 比如 silu不能e2e量化, e^-x, 就是e^sx  != s e^x,没法对两个x找到同一的s.   

#### fp8

nv的FP8推理的时候是直接FP8* FP8=FP16/FP32.  全部累加结束之后，再类型转换到FP8. FP8运算你最后要调的指令只能是fp8* fp8=bf16/fp16/fp32/tf32. 如果你在mma指令之前就去解量化，那你要调用的就只能是fp16的mma指令,

但是fp8_e4m3这种数制表示范围很大，fp8 e4m3的表示范围是 [-448, 448],  他基本可以不使用scale. 在量化和解量化过程中，你都可以不使用scale，他的精度也是足够的，fp16->fp8的quant, 以及fp8->fp16的dequant直接类型转换就行.  即使你在此基础上加入了per channel, per row 的 scale，他们大部分情况下对精度也几乎没有任何影响，这是因为fp8量化对scale的选取很不敏感.  你只需要强制类型转换就行. 

矩阵乘中，你调用的指令是 fp8*fp8=fp16/bf16/fp32/tf32.其输出结果的类型则由矩阵乘法算子的后处理逻辑决定.如果你使用的是cublst里面的fp8矩阵乘，它的输入输出应该都是fp8. 也就是cublst里面的矩阵乘，会在矩阵后处理的时候直接把结果强制类型转换到fp8. mma是输出高精度结果， 根据下一层算子 再决定是否要转成fp8

#### quanto

很方便的ptq. 

https://github.com/huggingface/optimum-quanto/blob/main/examples/vision/StableDiffusion/quantize_StableDiffusion.py https://huggingface.co/blog/quanto-diffusers  

一般来说大部分qat也就十几个 epoch 微调一下. 

## torch 的量化框架

建议量化新用户先试用 PyTorch 2 Export Quantization，如果效果不佳，用户可以尝试 Eager 模式量化。

https://github.com/OscarSavolainen/Quantization-Tutorials/tree/main 学习, https://youtu.be/hlcPz7wuf3M?si=lQou_ZgOkHaG4z6K

### eager mode

需要单元测试.

quantstubs 有量化parameter, dequantstubs是stateless的.

两个量化的tensor 加减乘除,concat, relu,  都需要重新量化! 

要用 `torch.nn.quantized.FloatFunctional`  来替换加法, 但是没有除法, 

https://pytorch.org/docs/stable/generated/torch.ao.quantization.fuse_modules.fuse_modules.html 可以根据名字fuse,  缺点是换个名字就会fail silently!    还需要prefix不然会和上一层的同名conv混起来. 要非常细心去找名字.

fuse 也有 fuse_modules_qat

##### Qconfig

指定对称还是affine, 初始qparam, 用ptq还是qat.

参数的具体解释可以参考https://pytorch.org/blog/quantization-in-practice/

```
'fbgemm' 是x86, arm用qnnpack

for i in modules_to_list:
   print(i)
```

qat可能要用 https://github.com/pytorch/pytorch/blob/main/torch/ao/quantization/_learnable_fake_quantize.py 

ptq 步骤

- 手动加入quantstubs 和  FloatFunctional
- fused modules (optional but highly recommended)
- attach qconfig to desired layers
- torch.ao.quantization.prepare_qat

Convert model之前要放到cpu.

https://youtu.be/jNZ1rkIfwsM?si=N8bLU85eKxOFYV0Q 实践. 

```
fuse_modules 会fuse conv和batch norm, bn会变成identity,  fuse_modules_qat会保持 batch norm, 

FakeQuantize 是ptq 专用的, 不是learnable的.
symmetric快很多. 
```

### fx graph mode

dag 优化

only for networks that are [symbolic traceable](https://pytorch.org/docs/stable/fx.html#torch.fx.symbolic_trace)

什么样的是symbolic traceable的呢? 

有分支就不行, 不过可以 fx.symbolic_trace(f, concrete_args={'b': False}) 来跟踪. 

sd1.5的 unet 就有太多if .  是否导致fx行不通? 怎么办? 

那为什么会有  Proxy object 

训练用fx也是有的. 还是应该试一下unet. 

Pytorch FX量化实践：对基于transformer模型进行QAT，附踩坑指南 - 守夜人的文章 - 知乎
https://zhuanlan.zhihu.com/p/600142009

_LearnableFakeQuantize 需要init scale和 zero point , 可以传播梯度. 

为什么需要 act statci qconfig?

```
fx_model.graph.print_tabular()打印出来看看
import ipdb
ipdb.set_trace() # 可交互的pdb, 非常有用, 

flatten, relu , 没有weight tensor  所以是FakeQuantize
activation_post_process_30 == activation_post_process_31, 因为他们共享量化参数, fx 智能做了很多优化. 
avgpool , quan error很小

prepare_qat_fx 会设置所有的qconfig , 加入quantstubs 和  FloatFunctional, fused modules, 

prepare_qat_fx之后, 还是有batch norm 的weight, 所以需要Fusing Bn in ConvBnReLU into ConvReLU   https://github.com/OscarSavolainen/Quantization-Tutorials/blob/main/Resnet-FX-Graph-Mode-Quant/main.py 

qat
插入了learn qconfig之后直接训练就可以了. 
```



#### export

也是graph base


他说他们已经在一个预训练模型上进行了QAT  , and that neede 200 hundred or so gpus. chatgpt是错的.  QAT 需要的计算量 比original full training还多.  先普通训练，再qat微调