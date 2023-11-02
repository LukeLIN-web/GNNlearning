







## UGache

https://dl.acm.org/doi/pdf/10.1145/3600006.3613169

UGache: A Unified GPU Cache for Embedding-based Deep Learning





#### 代码

注意docker 需要 degrade network 到3.1, 因为3.2 需要python3.9.





## Gsampler

https://github.com/gsampler9/gSampler



不能控制sample id. 

### 代码

`_bench_gm` 可以测action时间, 应该有采样时间? 





为什么`num_batches = int((batch_size + small_batch_size - 1) / small_batch_size)` 需要两个batch size? 







#### ladies算法

forward GPU 利用率这么低, 为什么? 
