### tensor

`edge_index = edge_index[:, edge_mask]` 就是取第二维度根据mask来取值. https://stackoverflow.com/questions/59344751/is-there-any-diffrence-between-index-select-and-tensorsequence-in-pytorch查看. 

注意tensor(0) 和tensor(0) 会被set认为是不同的. 

torch.allclose 来测试两个tensor是否一样

Model, 会调用 torch/nn/modules/module.py(1128)_call_impl() , forward_call(*input, **kwargs) 调用forward函数.

to是一个语法糖, 一种是变成其他类型, 比如 int16, 一种是torank用法, 就调用cuda()方法

#### 利用torch cpp库

https://pytorch.org/tutorials/advanced/cpp_frontend.html

怎么安装 

```
CMake Error at CMakeLists.txt:68 (find_package):
  By not providing "FindTorch.cmake" in CMAKE_MODULE_PATH this project has
  asked CMake to find a package configuration file provided by "Torch", but
  CMake did not find one.

  Could not find a package configuration file provided by "Torch" with any of
  the following names:

    TorchConfig.cmake
    torch-config.cmake

  Add the installation prefix of "Torch" to CMAKE_PREFIX_PATH or set
  "Torch_DIR" to a directory containing one of the above files.  If "Torch"
  provides a separate development package or SDK, be sure it has been
  installed.
```

https://stackoverflow.com/questions/53819318/torch-cmake-error-at-cmakelists-txt10-find-package-in-c  说 may also try to set variable `CMAKE_PREFIX_PATH` to `/home/afshin/libtorch`

https://github.com/pytorch/pytorch/issues/12449 

In order for CMake to know *where* to find these files, we must set the `CMAKE_PREFIX_PATH` when invoking `cmake`

```
cmake -DCMAKE_PREFIX_PATH=/root/share/learningpybind/libtorch ..
cmake --build . --config Release -- -j 32
```

### 性能调优指南

等所有代码写完了, 需要性能的时候再调性能.  

https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html

`DistributedDataParallel`提供 [no_sync()](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel.no_sync) 上下文管理器，它禁用特定迭代的梯度全部减少。 `no_sync()`应该应用于`N-1`梯度累积的第一次迭代，最后一次迭代应该按照默认执行并执行所需的梯度all-reduce。

### 损失函数

    return torch._C._nn.cross_entropy_loss(input, target, weight, _Reduction.get_enum(reduction), ignore_index, label_smoothing)

RuntimeError: 0D or 1D target tensor expected, multi-target not supported

`cross_entropy`

out torch.Size([1024, 47]) target torch.Size([1024, 1])

要squeeze掉一维,   it's safer to use `squeeze(1)` to specifically remove the extra dimension of size 1 along dimension 1, which is the case for your target tensor.

### 内存管理

`reserved`内存包含`allocated`缓存内存。

存储在缓存中的内存，无需重新分配即可重复使用。

根据正在使用的操作/库，会有一些额外的开销，例如（cuDNN、cuBLAS 等内核将使用内存）。在许多情况下，预留内存可能很大（大约数百 MiB）。

`torch.cuda.empty_cache()`：此函数释放 PyTorch 在当前 GPU 上分配的所有未使用内存。这对于释放训练或推理期间不再需要的内存很有用。

You might consider profiling your code using a GPU profiler, such as NVIDIA's `nvprof` or PyTorch's `torch.autograd.profiler`, to identify the specific parts of the code that are consuming the most memory.

cached是老api,已经被改成了reserved. 

torch.cuda.reset_peak_memory_stats() 会把之前的不算, 只记录后面分配的吗? 

torch.cuda.max_memory_allocated() 

```
torch.empty(size=1,embedding_dim,device=device,pin_memory=pin_memory)# size =2或3 可能内存空间是一样的, 因为内存是一页页分配的. torch.empty(100,也是占据空间的.
```

GPU0to GPU 0会创建新tensor吗? 不会.  做实验确认了.id, 修改变量. 

### 梯度保存

creates a new tensor based on an existing tensor, the new tensor will have the **same gradient functio**n as the original tensor. This means that any gradients computed with respect to the new tensor will be propagated back to the original tensor.

x.detach()会创建一个副本.

梯度是根据model构建还是我的conv? forward的时候生成计算图, 根据我裁剪过的计算图. 

`@torch.no_grad()` 用了之后, 传入的参数还有参数, 函数里的操作产生的tensor都不会有梯度.

tensor.clone. 还有梯度 .

`loss.backward(retain_graph=True)` 合理吗? 不行, 会OOM.

self.emb[n_id] = x.to(self.emb.device) # to了之后还在计算图上吗? detach()会离开计算图.  原来那个还在计算图上, 副本也会在计算图上. 

## 怎么测试

1. 为什么forward之后没有被计算的embedding不全为0 ?   因为weight和bias不是0.

