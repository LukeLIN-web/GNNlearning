## 安装



error while loading shared libraries: libcublas.so.11: cannot open shared object file: No such file or directory

```
pip install nvidia-cublas-cu11

undefined reference to `cublasGemmEx'

nvcc gemmwmma.cu -o a.out --gpu-architecture=compute_80 -lcublas -lcurand 
```

问题是没有cublas.so还是能编译通过.但是会找不到.so

-L 用pip下载的so也不行. 

为什么他需要的是libcurand.so.10 不是11?



https://github.com/NVIDIA/cuda-samples 讲解了各个api的例子. 





 CUDA instruction set architecture, called *PTX* ,Parallel Thread eXecution

然后再编译成设备相关机器码SASS. [PTX](https://link.zhihu.com/?target=https%3A//docs.nvidia.com/cuda/parallel-thread-execution/index.html)在NVIDIA官网有官方的文档，但是SASS没有，只有一些零散的非官方资料。



```
__global__ void vector_add(float *out, float *a, float *b, int n) {
    int index = threadIdx.x; 
    int stride = blockDim.x; 

    for(int i = index; i < n; i += stride){
        out[i] = a[i] + b[i];
    }
}
    vector_add<<<1,256>>>(d_out, d_a, d_b, N);
//一次就有256个同时进行,index = threadIdx.x=0 到index = 255 . blockDim.x = block的大小 = 256

int tid = blockIdx.x * blockDim.x + threadIdx.x;
```



https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html

Similar to how threads in a thread block are guaranteed to be co-scheduled on a streaming multiprocessor, thread blocks in a cluster are also guaranteed to be co-scheduled on a GPU Processing Cluster (GPC) in the GPU.







####  tensorcore cuda

tensorcore cuda怎么写? 

https://github.com/Bruce-Lee-LY/cuda_hgemm/blob/10a8a8451f0dcd162b3790045cd7597cb48b8beb/src/wmma/wmma_naive.cu#L17

Nvidia Tensor Core-WMMA API编程入门 - 木子知的文章 - 知乎
https://zhuanlan.zhihu.com/p/620766588

WMMA需要按照每个warp处理一个矩阵C的WMMA_M * WMMA_N大小的tile的思路来构建，因为Tensor Core的计算层级是warp级别，计算的矩阵元素也是二维的。接下来，与CUDA Core naive的处理思路一致，首先确定当前warp处理矩阵C的[tile坐标](https://www.zhihu.com/search?q=tile坐标&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"article"%2C"sourceId"%3A"620766588"})，声明计算tilie所需的fragment，再以WMMA_K为步长遍历K并直接从global memory中加载所需A、B矩阵tile到fragment参与计算，最后将计算结果从fragment直接写回矩阵C。

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
```

https://github.com/NVIDIA-developer-blog/code-samples/blob/master/posts/tensor-cores/simpleTensorCoreGEMM.cu



编译失败, gemmwmma.cu(88): error: name followed by "::" must be a class or namespace name 因为没有指定 `-arch=sm_80`



