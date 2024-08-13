## 安装

```bash
#conda的方法
conda create -n condaexample python=3.11 #enter later python version if needed
conda activate condaexample 
# Full list at https://anaconda.org/nvidia/cuda-toolkit
conda install -c "nvidia/label/cuda-12.1.1" cuda-toolkit

下载torch cuda版本并不能拥有nvcc
```

### 内存模型

![](https://github.com/dmlc/web-data/raw/main/tvm/tutorial/gpu_memory_hierarchy.png)

![](https://assets-global.website-files.com/61dda201f29b7efc52c5fbaf/6501bc80f7c8699c8511c0fc_memory-hierarchy-in-gpus.png)

constant memory 是只读的。  scratchpad 也叫 shared memory. shared和local都是scratchpad.

每个SM里有专属的L1cache, shared memory和constant memory.

多个SM共享L2 cache. A100  L1 cache 128 KB ,  L2 cache , 2 MB to 6 MB. 

GPU 有48KB的l1 cache ,可以32KB scratchpad, 16KB cache, 编译器也可以决定划分成32KB的cache.  cache.大小是会根据编译器变化的？那怎么确定呢

cudaMalloc 是在哪里分配内存? 是在gpu dram.

#### share memory

DDR复制到share memory和share memory写回DDR 都是需要手动操作的. 

以`__device__ __shared__`为关键词声明的变量会被分配至SM上的shared memory， 可以由block内的全部线程所共享，生命周期也随着block的结束而结束。

refer : https://courses.grainger.illinois.edu/cs484/sp2020/24_gpgpus.pdf

A100 最大48KB

### Programming Model

从大到小

硬件: GPU -> SM ->  warp-> cuda core

CUDA:  Grid-> block -> thread 

#### **Grid**

- **Grid** 是一个由多个**线程块（blocks）**组成的二维结构。
- 当主机CPU上的CUDA程序调用一个**kernel**时，会启动一个**grid**。
- 每个**线程block**都是**grid**的一部分。

grid_size是用来描述Grid的三个维度的大小。例如，如果一个Grid在每个维度上都有10个 block，则其grid_size为(10, 10, 10)。 如果只写一个的话, 那就是默认block.x 

` <<<dimGrid, dimBlock>>>`  

#### block

block size最大可以取1024.

 一个SM可能有多个block.  On current GPUs, a thread block may contain up to 1024 threads. 一个*CUDA core*可以执行一个thread，

一个SM的*CUDA core*会分成几个*warp* ,由*warp* scheduler负责调度. 但是太小的矩阵几个warp就够了.  

在NVIDIA A100 GPU中，包含6912个CUDA核心.  一个SM有64个cuda core和4个tensorcore, 108 个SM.

单个block中的所有thread将在同一个SM中执行.

我们在 GPU 可以对各个block执行parallelization，对于block内部的thread可以执行vectorization.

一个block中一般用128或者256个thread. 

"block_size"是每个线程块的大小，通常也是一个由三个整数值组成的元组，用来描述线程块在每个维度上的大小。例如，如果一个线程块在每个维度上都有256个线程，则其block_size为(256, 1, 1)。

#### warp

一个warp是一组连续的线程，通常包含32个线程。这些线程在执行时会以SIMD（单指令多数据）的方式并行执行相同的指令，称为"同步线程束执行"。

在硬件层面上，`warp` 的调度是由SM来负责的。

单个block里的thread总数 <= 1024.

Warp tile是在算法设计中，将问题分割成小块，每个块的大小等于一个warp的大小。这种设计可以带来一些优势:

1. **数据并行性**：将问题分解成warp tile可以更好地利用这种并行性，每个warp中的线程可以同时处理一个tile中的不同数据。
2. **访存效率**：通过让每个warp共享一个内存请求，可以减少访存的总次数。因为warp中的线程通常访问的是连续的内存地址，所以在访存时可以利用缓存的局部性。
3. **线程同步**：warp内的线程可以非常高效地进行同步，因为它们执行相同的指令。这使得在warp内进行同步操作时，不需要额外的开销。

只有一个warp是最小并行单位. 不同的warp直接会切换来隐藏延迟. 

https://www.kdocs.cn/l/caT5nb73SO1Z?f=201&share_style=h5_card   可以看figure7. SM的结构.  

a100, 是四个sub partition, 每个partition能放一个warp. warp是有状态的,可以说真正一个clock执行的就4个, 但是准备用来切换的warp是很多的.

For a block whose size is not a multiple of 32, the last warp will be padded with inactive threads to fill up the 32 thread positions.

对于if else,  GPU可能会做两次pass. The cost of divergence是 execution resources that are consumed by the inactive threads in each pass. 不过 From the Volta architecture onwards, the passes may be executed concurrently. 叫做 independent thread scheduling.

就是当一个 warp 中需要进行 global memory access 这类型的耗时操作时，这个 warp 就会被换下，执行其他的 warp； 和 CPU 中的逻辑类似；  因为有这个调度, 所以 A100 GPU, an SM has 64 cores but can have up to 2048 threads assigned to it at the same time.

#### cpp是怎么编译的

```bash
用find找到缺失的库.  是在~/miniconda3/envs/condaexample/lib/ 里面
ls /lib/ 里面有所有的cpp标准库. 
cat /etc/ld.so.conf 里面存了ld的配置文件. 
$cat /etc/ld.so.conf.d/libc.conf 
# libc default configuration
/usr/local/lib
$cat /etc/ld.so.conf.d/x86_64-linux-gnu.conf 
# Multiarch support
/usr/local/lib/x86_64-linux-gnu
/lib/x86_64-linux-gnu
/usr/lib/x86_64-linux-gnu
$ file a.out
可以看解释器是啥
a.out: ELF 64-bit LSB pie executable, x86-64, version 1 (SYSV), dynamically linked, interpreter /lib64/ld-linux-x86-64.so.2, BuildID[sha1]=0f09da1a7357e829dcb685e2b114275921de1b44, for GNU/Linux 3.2.0, not stripped
就是$./a.out其实等价于 $ /lib64/ld-linux-x86-64.so.2 a.out
 $file /lib64/ld-linux-x86-64.so.2 可以看到他连着啥. 
/lib64/ld-linux-x86-64.so.2: symbolic link to /lib/x86_64-linux-gnu/ld-linux-x86-64.so.2
$ldd ./a.out  就可以看缺了哪些库.

$LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/miniconda3/envs/condaexample/lib ldd ./a.out  就可以知道新的库链接关系
nvcc gemmwmma.cu -o a.out -lcublas -lcurand -arch=sm_80  # 是不行的,-I 目录：添加头文件搜索路径。 -L 目录：添加库文件搜索路径。 -l 添加需要搜索的库文件, 可以man ld 搜索--library.


nvcc gemmwmma.cu -o a.out  
-L只是告诉编译器去哪儿找so，看里面的符号. 没法告诉ld这件事. 你想hardcode so的路径,也有别的办法. 
为什么他需要的是libcurand.so.10 不是12?  因为nvcc是cuda12.1
```



```
error while loading shared libraries: libcublas.so.11: cannot open shared object file: No such file or directory
undefined reference to `cublasGemmEx'
nvcc gemmwmma.cu -o a.out --gpu-architecture=compute_80 -lcublas -lcurand 
```



 可能和编译器搜索路径有关系, 也可能和动态链接库搜索地址和库的软链接有关系.

试试这个. 

```

还是不行, ./a.out: error while loading shared libraries: libcublas.so.12: cannot open shared object file: No such file or directory 




LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/linj/miniconda3/envs/condaexample/lib 就可以了.
普通的$LD_LIBRARY_PATH不全. 
LD_LIBRARY_PATH 中的动态链接库拥有被调度的更高的优先级,有同名也会用它. 
```

#### PTX

 CUDA instruction set architecture, called *PTX* ,Parallel Thread eXecution

然后再编译成设备相关机器码SASS. [PTX](https://link.zhihu.com/?target=https%3A//docs.nvidia.com/cuda/parallel-thread-execution/index.html)在NVIDIA官网有官方的文档，但是SASS没有，只有一些零散的非官方资料。 

the PTX cooperative thread array(CTA) is conceptually and functionally the same as a **block** in CUDA or a **workgroup** in OpenCL.

```
__global__ void vector_add(float *out, float *a, float *b, int n) {
    int index = threadIdx.x; 
    int stride = blockDim.x; 

    for(int i = index; i < n; i += stride){
        out[i] = a[i] + b[i];
    }
}
    vector_add<<<1,256>>>(d_out, d_a, d_b, N);
//一次就有256个线程同时进行,index = threadIdx.x=0 到index = 255 . blockDim.x = 每个block中的线程数 = 256

int tid = blockIdx.x * blockDim.x + threadIdx.x;
```



https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html

Similar to how threads in a thread block are guaranteed to be co-scheduled on a streaming multiprocessor, thread blocks in a cluster are also guaranteed to be co-scheduled on a GPU Processing Cluster (GPC) in the GPU.

#### tensorcore cuda

A100上的tc 可以fp16输入也可以fp32输入. 一代tc甚至只支持fp16xfp16+fp32=fp32. 

https://github.com/Bruce-Lee-LY/cuda_hgemm/blob/10a8a8451f0dcd162b3790045cd7597cb48b8beb/src/wmma/wmma_naive.cu#L17

WMMA需要按照每个warp处理 一个WMMA_M * WMMA_N大小的tile的思路来构建，因为Tensor Core的计算层级是warp级别，计算的矩阵元素也是二维的。接下来，与CUDA Core naive的处理思路一致，首先确定当前warp处理矩阵C的[tile坐标](https://www.zhihu.com/search?q=tile坐标&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"article"%2C"sourceId"%3A"620766588"})，声明计算tilie所需的fragment，再以WMMA_K为步长遍历K并直接从global memory中加载所需A、B矩阵tile到fragment参与计算，最后将计算结果从fragment直接写回矩阵C。

https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma-description

fragment就是小的 ,16x 16.  k就是整个, k1是右边,k2是左边. 

  `wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);` 就是matrix multiplication and accumulation.

为什么a 是col major， 不是b应该col major吗？

`__half` 是16bit浮点数. 

wmma::fragment里面的WMMA_M,WMMA_N,WMMA_K必须是常数，不能是变量

```bash
nvcc wmma.cu -o wmma -arch=sm_80 # a100
nvcc gemmwmma.cu -o a.out --gpu-architecture=compute_80 -lcublas -lcurand
nvcc gemmwmma.cu -o a.out -lcublas -lcurand -arch=sm_80 
```

https://github.com/NVIDIA-developer-blog/code-samples/blob/master/posts/tensor-cores/simpleTensorCoreGEMM.cu

编译失败, gemmwmma.cu(88): error: name followed by "::" must be a class or namespace name 因为没有指定 `-arch=sm_80`   

#### bank conflict

shared memory, 连续的内存是分摊到每个bank的同一层中. 当同一个 warp 中的不同线程访问一个 bank 中的不同的地址时（访问同一个地址则会发生广播），就会发生 bank conflict

cuda怎么生成随机int?

https://on-demand.gputechconf.com/gtc/2018/presentation/s81006-volta-architecture-and-performance-optimization.pdf  66页讲的很清楚. 

友好的方式: 每个线程访问 32bit数据, 每个线程并没有与bank一一对应，但每个线程都会对应一个唯一的bank，也不会产生bank冲突。

访问步长(stride)为2，线性访问方式，造成了线程0与线程16都访问到了bank 0，线程1与线程17都访问到了bank 2...，于是就造成了2路的bank冲突。

当一个warp中的所有线程访问一个bank中的**同一个字(word)地址**时，就会向所有的线程广播这个word，这种情况并不会发生bank冲突。

我知道A的行做转置，是因为拿数据按照列读数据了.

 `A[Bm][Bk]`,每次从Bm里面拿Tm数据，相邻两个线程相差Bk * Tm * sizeof(float) ,  32 bank是128字节，所以 如果Bk * Tm凑够了32倍数， 他就会conflict.  就是同一时间， t1访问第一个bank， t2访问第64个bank， 就conflict了

按列, 每次从Bk中拿Tk数据.  T1 访问8个bank. t2访问 8个bank. ?

#### 时间测试

cpu 测大部分都是launch kernel的时间了.

launch kernel大概需要150us -250us.  GPU 底层机制分析：kernel launch 开销 - lychee的文章 - 知乎
https://zhuanlan.zhihu.com/p/544492099

时间太短的话, 计时函数分辨率有限, 所以可以让他算10遍算总时间. 

#### 带宽测量

之前用 nvprof, 现在用nsight, 

100 run平均

数据量要大,  10mb数据pcie 4.0x16还不到1ms, cpu基本上只能测到噪声. 

gpu内存带宽, 理论性能是：`5120/8*1512*1e6*2/1e12` TBps, 大概1.9T左右 那个2的系数是因为ddr

#### debug

#### vectorize

从global memory读取数据可以使用lgd.128指令，一次读4个float32的数据，从share memory 读取数据，可以用lgs.128. 首先需要从循环中把可以vectorize的shape手动拆出来，再进行向量化.

内存带宽,  24年H100玩法是TMA + WGMMA. 等B100又有新套路了, 他们B100从23年就开始做新指令的性能优化了 在Hopper上只有WGMMA才能达到最高的性能. 

ldmatrix都已经落后一个版本了. 纠结LDS是2021年以前的玩法.

#### TMA

张量内存加速器 (TMA) 单元，它可以在全局内存和共享内存之间非常有效地传输大块数据。 TMA 还支持集群中线程块之间的异步复制。还有一个新的[异步biarrier](https://www.zhihu.com/search?q=异步biarrier&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"article"%2C"sourceId"%3A"488340235"})用于进行原子数据移动和同步。

synchronous Copy+：TMA.  之所以称为“Asynchronous Copy+”，是因为A100中已经提供了从主存不经过L2直接到SMEM的异步copy能力。但是H100将其增强了，变成了一个“DMA引擎”。这个东西有些像Google TPUv4中提到的“[Four-dimensional tensor](https://www.zhihu.com/search?q=Four-dimensional tensor&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"article"%2C"sourceId"%3A"486224812"}) DMA“，不过Nvidia增加了一个维度，最大支持5维。

GEMM  被解决完了, 只有其他任务可能还有memory conflict.

#### wgmma

wgmma指令，完成异步mma计算

## cutlass

cutlass的tile size 是固定的,就三四种, 所以tile数量是固定的, 但是block数量太少就没法overlap了, 因为一个block 必须先算然后comm,  所以要进一步tile, 4个block 来计算4 个replicas, 然后all gather. 

社区许多高效的 CUDA 算子是通过 CUTLASS 3.x 实现的。在 NV 的知名的开源软件下搜寻，如 [TensorRT-LLM](https://www.zhihu.com/search?q=TensorRT-LLM&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"article"%2C"sourceId"%3A"689829403"}), Megatron-LM, 都不难发现 [CUTLASS 3.x](https://www.zhihu.com/search?q=CUTLASS 3.x&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"article"%2C"sourceId"%3A"689829403"}) 的影子。除此之外，[大模型训练](https://www.zhihu.com/search?q=大模型训练&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"article"%2C"sourceId"%3A"689829403"})几乎必备的 [Flash Attention(FMHA) 2.0](https://link.zhihu.com/?target=https%3A//crfm.stanford.edu/2023/07/17/flash2.html) 也采用了 CUTLASS 3.x 的特性。

CUTLASS 3.x 进行了大范围的重构，引入了新的 GEMM 编程模型。其中最主要的模块包括 [CuTe](https://link.zhihu.com/?target=https%3A//github.com/NVIDIA/cutlass/blob/main/media/docs/cute/00_quickstart.md)， TMA， [Warp-Specialization](https://link.zhihu.com/?target=https%3A//github.com/NVIDIA/cutlass/blob/main/media/docs/efficient_gemm.md%23warp-specialization)，

 GPU 上面存在的两个异构硬件 TMA 和 TensorCore。Pipeline 的引入基本上宣告了 SIMT 编程的让位。



## reference

1. https://github.com/NVIDIA/cuda-samples 讲解了各个api的例子. 
2. https://github.com/DefTruth/CUDA-Learn-Notes  中文讲解各种例子. 
3. https://www.zhihu.com/question/26570985

4. 强烈建议你看一下massively 那本书,  前6章就好,可以当工具书查,  看完之后看一下b站关于ncu和nsys的分析kernel的视频 https://www.bilibili.com/video/BV13w411o7cu/?vd_source=c8fbfaa03f04095bf6cd95630d210cc5
5. [施工中] 在Hopper GPU上实现CuBLAS 90%性能的GEMM - 郑思泽的文章 - 知乎
   https://zhuanlan.zhihu.com/p/695589046
