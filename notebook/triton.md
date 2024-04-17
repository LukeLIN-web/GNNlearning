triton报错还是挺友好的.

#### autotune

triton.autotune 是调多少次?啥时候结束?   就是这个函数，第1次被调用时，除了编译本身，也会有一个auto tune的过程 25.00%

AttributeError: 'Constant' object has no attribute 'lineno'



矩阵乘法, accumulator 是创建一个矩阵存结果. 

addvector, 在BLOCK_SIZE很小的时候triton性能很差,为什么?  因为blocksize 太小没法充分利用. 

A100, 40 MB L2 cache ,  SRAM就是cache, L1数据高速缓存和共享内存的总容量，在A100中为**192 KB / SM**

#### triton.Config 

 num_stages是啥  **num_stages** –  software-pipelining loops. Mostly useful for matrix multiplication workloads on SM80+ GPUs.   也就是A100之后的GPU可以用上pipeline, A100是sm80.  H100是sm90. 

triton怎么写递归呢?  可以matmul_kernel 中调用leak. 但是不能调用自己. 





#### 语法

```python
grid = lambda META: (triton.cdiv(n, META['BLOCK_SIZE_N']) * triton.cdiv(n, META['BLOCK_SIZE_N']), ) # meta是哪里来的? 
懂了, 就是triton.autotune里的
```

