#### Paella: Low-latency Model Serving with Software-defined GPU Scheduling

upenn Vincent Liu

#### 摘要

co-designing the model compiler, local clients, and the scheduler to bypass the built-in GPU scheduler and enable software control of kernel execution order. 

#### related work

模型推理框架: Clipper [19], Cocktail [29], and INFaaS [61], for instance, adaptively select model variants and introduce related optimizations in autoscaling, caching, and batching of their execution

Paella attempts to address deficiencies in the GPU hardware scheduler, which requires a co-design of CUDA kernels and model serving

Clockwork是一个GPU一个模型, Paella targets a different point in the design space, opting to maximize GPU occupancy for high throughput and minimize JCT given that constraint.  Paella is the first to leverage thread-block instrumentation to softwareize the scheduling decisions of GPUs.



## UGache

https://dl.acm.org/doi/pdf/10.1145/3600006.3613169

UGache: A Unified GPU Cache for Embedding-based Deep Learning

ugache不能处理在过程中改变index

怎么改成inference呢? 在每个epoch 改变index 的访问可以吗?



它是啥时候solve生成cache policy的? 

怎么看cache policy呢? 

Coordination between Extractor and Solver is achieved through a simple per-GPU hashtable.

Refresher collects statistics and periodically re-evaluates Solver’s model with the new hot- ness.  累积一段时间调整会导致性能差吗?

solver调用filler 来fill cache .

它是怎么判断哪些是hot的? 是在后台运行的, 代码中是无缝的.



#### 代码

注意docker 需要 degrade network 到3.1, 因为3.2 需要python3.9.

看了半天也没看懂他啥时候re-evaluates Solver’s model

product , 2449029 个节点

为什么 表示边的indices: torch.Size([123718152]) 是一维数组?  因为是csc 

- `self.indptr`：该数组可能保存表示图中每个节点的边的开始和结束的索引。最后一个值可能是边的总数或可以添加新边的端点。
- `self.indices`：该数组可能包含由边连接的节点的索引。索引指的是该数组中列出每条边的连接节点的位置。

```cpp
coll_torch_record  // warp
void coll_cache_record(int replica_id, uint32_t* key, size_t num_keys) {
  _freq_recorder[replica_id]->Record(key, num_keys);
  max_num_keys = std::max(max_num_keys, num_keys);
}
维护一个
class FreqRecorder 
void FreqRecorder::Record(const KeyT* input, size_t num_inputs){
  就是记录freq.
}

coll_torch_init_t{
  torch_eager_gpu_mem_allocator //

coll_cache_init()
}
  
coll_cache_init(){
CollCache::build_v2  
}


CollCache::build_v2{
  初始化solver的参数
    solve_impl_master
    然后cache_ctx->build   
}

CacheContext::build(){
分配一些gpu 内存, 

}

CollCache::solve_impl_master{
  _nid_to_block = Tensor::CreateShm
        _block_placement = solver->block_placement;
    _block_access_advise = solver->block_access_from;
    _block_density = solver->block_density_tensor;
}

CollCache::solve_impl_slave(){
//重载了六个同名的函数. 
  他solve就是修改了一下CollCache中的_block_access_advise
    _block_access_advise = Tensor::OpenShm
}

Tensor::OpenShm{
 void* data = cpu::MmapCPUDevice::MapFd(MMAP(MMAP_RW_DEVICE), nbytes, fd); 
}

cpu::MmapCPUDevice::MapFd{
  
  void* ptr = mmap(nullptr, nbytes, prot, MAP_SHARED | MAP_LOCKED, fd, 0);
  //是否就是说明是让host驱动的.
}


::torch::Tensor coll_torch_create_emb_shm //好像没干啥
  coll_torch_lookup_key_t_val_ret 调用了common::coll_cache_lookup 
    coll_cache_lookup 把每个steps[replica_id]++; 算出偏移然后寻址_coll_cache->lookup(replica_id, key, num_keys, output, stream, step_key);
  CollCache是 global cache manager. one per process
  CollCache::lookup, 就是提取_session_list[replica_id]->ExtractFeat

    
TensorPtr Tensor::CopyToExternal{
    Context working_ctx = Priority(source->Ctx(), ctx);
    Device::Get(working_ctx)->CopyDataFromTo(source->_data, 0, tensor->_data, 0,nbytes, source->_ctx, tensor->_ctx, stream);//copy的话 cpu gpu都有， 看是什么working ctx
  Device::Get(working_ctx)->StreamSync(working_ctx, stream);//StreamSync 应该是gpu 做
  }
定义了一个enum DeviceType { kMMAP = 0, kCPU = 1, kGPU = 2, kGPU_UM = 3}; 会比较Context 优先级, 好像是哪个优先级大就用哪个设备发起传输.  为啥kmmap 比kcpu还小? 
    
    
```



他是怎么输出求解器的解的呢?  怎么输出placement策略

用的是Unified Memory吗? 数据搬运是 gpu发起的还是cpu发起的? 

改一下代码, 重新运行

```
bash /tmp/setup_docker.sh
bash /tmp/setup_docker.gnn.sh

```

安装挺慢的, 大概需要20分钟-30分钟. 



## Gsampler

https://github.com/gsampler9/gSampler



不能控制sample id. 

### 代码

`_bench_gm` 可以测action时间, 应该有采样时间? 





为什么`num_batches = int((batch_size + small_batch_size - 1) / small_batch_size)` 需要两个batch size? 







#### ladies算法

forward GPU 利用率这么低, 为什么? 
