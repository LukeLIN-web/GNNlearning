## 安装

```bash
#conda的方法
conda create -n condaexample python=3.11 #enter later python version if needed
conda activate condaexample 
# Full list at https://anaconda.org/nvidia/cuda-toolkit
conda install -c "nvidia/label/cuda-12.1.1" cuda-toolkit

下载torch cuda版本并不能拥有nvcc
```

#### 内存模型

![](https://github.com/dmlc/web-data/raw/main/tvm/tutorial/gpu_memory_hierarchy.png)

constant memory 是只读的。  scratchpad 也叫 shared memory.

每个SM里有专属的L1cache, shared memory和constant memory

多个SM共享L2 cache. A100  L1 cache 128 KB ,  L2 cache , 2 MB to 6 MB. 

GPU 有48KB的l1 cache ,可以32KB scratchpad, 16KB cache, 编译器也可以决定划分成32KB的cache.  cache.大小是会根据编译器变化的？那怎么确定呢

cudaMalloc 是在哪里分配内存? 是在gpu dram.

以`__device__ __shared__`为关键词声明的变量会被分配至SM上的shared memory， 可以由block内的全部线程所共享，生命周期也随着block的结束而结束。

refer : https://courses.grainger.illinois.edu/cs484/sp2020/24_gpgpus.pdf

#### Programming Model

一个grid可以有多个thread block. On current GPUs, a thread block may contain up to 1024 threads.

一个*CUDA core*可以执行一个thread，一个SM的*CUDA core*会分成几个*warp* ,由*warp* scheduler负责调度. GPU有几万个cuda core, 但是太小的矩阵几个warp就够了.  

单个block中的所有thread将在同一个SM中执行.

我们在 GPU 可以对各个block执行parallelization，对于block内部的thread可以执行vectorization

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

mma.h

https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma-description

fragment就是小的 ,16x 16.

k就是整个, k1是右边,k2是左边. 

  `wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);` 就是matrix multiplication and accumulation.

手写一个.

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

#### bank冲突

shared memory, 连续的内存是分摊到每个bank的同一层中. 当同一个 warp 中的不同线程访问一个 bank 中的不同的地址时（访问同一个地址则会发生广播），就会发生 bank 冲突.

cuda怎么生成随机int?

#### 时间测试

cpu 测大部分都是launch kernel的时间了.

launch kernel大概需要150us -250us.  GPU 底层机制分析：kernel launch 开销 - lychee的文章 - 知乎
https://zhuanlan.zhihu.com/p/544492099

时间太短的话, 计时函数分辨率有限, 所以可以让他算10遍算总时间. 

#### debug

## reference

https://github.com/NVIDIA/cuda-samples 讲解了各个api的例子. 

https://github.com/DefTruth/CUDA-Learn-Notes  中文讲解各种例子. 

https://www.zhihu.com/question/26570985/answer/3247401363
https://www.zhihu.com/question/26570985/answer/3465784970

