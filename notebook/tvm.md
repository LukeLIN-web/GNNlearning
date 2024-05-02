## 介绍

tvm 前身halide, 是mit 图形学的教授组, 发明 split这一套抽象.谷歌在用halide. 毕竟tvm是前llm时代的东西,性能和好用很多时候只能二选一。tvm之前用的是compile到小设备上, 比如音箱。llm不能运行在这些小设备上.  tvm缺点是 debug 极其复杂, 多年积累学习成本高的离谱. 

### 替代方案

ansor也是一个 dnn 自动编译器. ion和zhuodan yang他们做的. 

cutlass 是英伟达新官方cpp 模板库, gemm比tvm快很多. 

AITemplate  可以把DNN转换为 into CUDA/ HIP (AMD GPU) C++ code.

triton在GPU上写的快, 但是不能用在别的设备上. 

## 安装

#### llvm

```bash
git clone --depth 1 https://github.com/llvm/llvm-project.git
llvm-as --version
cd llvm-project
mkdir build
cd build 
cmake ../llvm -G Ninja -DCMAKE_INSTALL_PREFIX=/你的路径/llvm-project/build -DBUILD_SHARED_LIBS=on -DCMAKE_BUILD_TYPE="Debug" -DLLVM_ENABLE_PROJECTS=clang 
cmake --build . # 此命令在底层运行Ninja，因为我们在配置步骤中告诉了CMake生成Ninja文件。
```

#### CPU版

```bash
git clone --recursive https://github.com/apache/tvm

brew install miniforge # macbook需要
可以通过conda 直接建依赖
# Create a conda environment with the dependencies specified by the yaml
conda env create --file conda/build-environment.yaml#  mbp半个多小时.服务器15-20分钟. 
dependencies:
  - python=3.7 # or 3.8. See https://github.com/apache/tvm/issues/8577 for more details on >= 3.9 您可以使用 3.9，但 TVM 的某些部分可能无法正常工作（例如 hybridscript）。
  
# Activate the created environment
conda activate tvm-build

mkdir build
cp cmake/config.cmake build
# 默认没有打开llvm编译开关, 把llvm和 metal/cuda 设置为ON.

cd build
cmake .. -G Ninja
ninja

conda install conda-forge::ninja不行, pip可以. 
# then set python part so we can use in python, refer https://tvm.apache.org/docs/install/from_source.html#tvm-package
```

#### with cuda 

```bash
成功的方法:
conda创建一个cuda环境,手动安装tvm和llvm.成功了. 

下面都是失败的方法:
1. conda activate tvm-build 然后 conda install -c "nvidia/label/cuda-12.1.1" cuda-toolkit 会找不到, conda 总是啥也找不到. 可以试试用mamba.
 一开始build就要指定所有, 否则之后install都是冲突.
 试试直接在conda/build-environment.yaml 加上- cuda-toolkit试试, 
Found conflicts! Looking for incompatible packages.
2. 
conda create -n cu116tvm python=3.10
conda install nvidia/label/cuda-11.6.0::cuda
pip install apache-tvm-cu116 -f https://tlcpack.ai/wheels 不行, conda太慢了. 
3. 
cuda环境直接装tvm的话, 不用llvm行不行? 不行, Warning: Cannot parse Arm(R)-based target features without LLVM support. Segmentation fault (core dumped)
还是要装llvm.  因为main module 还是llvm写的.
config.cmake 有set(USE_LLVM OFF)
他会被llvm.cmake用到, if(NOT ${USE_LLVM} MATCHES ${IS_FALSE_PATTERN})
  find_llvm(${USE_LLVM})
4. 
$docker pull tlcpack/ci-gpu:20240105-165030-51bdaec6# 会显示没有tvm
```

## TIR语法

先看 https://www.cnblogs.com/whiteBear/p/16756035.html

https://tvm.apache.org/docs/how_to/optimize_operators/opt_gemm.html

Vectorization 速度没有变快.  `C_1[cse_var_1:cse_var_1 + 64] `   是因为产生的代码 被编译器自动优化了 相当于做了矢量优化 .

学习 https://tvm.hyper.ai/docs/how_to/te_schedules/primitive/

T.grid  的意思:

```cpp
  // for i_0 in T.serial(2):
  //     for i_1 in T.serial(2):
  //          for i_2 in T.serial(2):
  //            |
  //            |
  //            V 
  // for i_0, i_1, i_2 in T.grid(2, 2, 2):
```

也可用 `nparts` 来拆分 axis，它拆分 axis 的方式与 `factor` 相反。  factor是 inner loop 32, nparts是outer part 32. 

怎么在te compute中计算一些中间变量? 定义一个函数即可. 注意参数要对齐数量

```python
def c1_compute(io, jo): # 注意参数要对齐数量
	tmp = io +1 # 计算一些中间变量
	return (
    )
  c1 = te.compute((n // block, n // block,), c1_compute)
```

即使split之后, s[C].op.axis 依旧不会显示拆开的, 还是原来的.

#### tile

参考

1. 循环优化之循环分块（loop tiling） https://zhuanlan.zhihu.com/p/292539074  和
   https://zhuanlan.zhihu.com/p/403163009
2. https://aijishu.com/a/1060000000381408
3. 推荐看 https://halide-lang.org/  这是tvm的前身.

#### reorder

TVMError: Operate on iter var T.iter_var(j_outer, None, "DataPar", "")that has already been split. 

因为 你下面的reorder里面有mi & ni ,  但是下面它们已经被拆成nio mio了。

```
mo,mi = s[C].split(C.op.axis[0], factor=bn)
no,ni = s[C].split(C.op.axis[1], factor=bn)
ko,ki = s[C].split(s[C].op.reduce_axis[0], factor=bn)
mio,mii = s[C].split(mi, factor=bn2)
nio,nii = s[C].split(ni, factor=bn2)
kio, kii = s[C].split(ki, factor=bn2)
# s[C].reorder(mo, no,ko, mi, ni, ki)
s[C].reorder(mo,no,ko, mio,nio,kio, mii, nii,kii)
print(tvm.lower(s, [A, B, C], simple_mode=True))
func = tvm.build(s, [A, B, C], target=target, name="mmult")
```

C.op.axis[0] 和 C.op.axis[1] 区别是啥? 好像 一个是行, 一个是列

 `s[packedB].vectorize(littleN)` 没啥用.去掉反而更快. 因为 llvm会帮你做规整的vectorization.

packing之后好像可以快两倍。 

```python
s[C].vectorize(nii)  #可以更快
s[packedB].parallel(bigN) # 没啥用
s[c1].reorder(c1.op.axis[0], c1.op.axis[1],s[c1].op.reduce_axis[0], c1.op.axis[2], c1.op.axis[3]) # reduce axis也是可以reorder的
```

https://tvm.hyper.ai/docs/0.10.0/how_to/te_schedules/compute_reduce/

虽然一次 batch 操作会有多个输出，但它们只能一起调度。 就是说调度一个的时候`s[B0].compute_at(s[C], C.op.axis[0])` 另一个也会做同样的调度.

一个cm1是对的， 但是tuple的时候会报错InternalError: Check failed: (ReduceEqual(reduce, reduce_)) is false: The Reduce inputs of ComputeOp should have the same attribute except value_index 为啥呢。

最笨的办法。你去ReduceEqual里面。把它的每一个condition打出来。 看区别是啥。

#### compute at

```python
C.op.axis 是[T.iter_var(i, T.Range(0, m), "DataPar", "")]  # 真是抽象啊. 不知道多个axis是啥样的. 
s[B].compute_at(s[C], C.op.axis[0]) # 可以在一个循环做多个事情.  实际上是把B的计算移动到C的第一个循环, axis可以理解为循环
s[B].compute_inline() # 可以省掉变量B. 可读性变差, 代码行数变少. 
s[B].compute_root() #类似于compute at的逆操作, 提回到root.
```

compute at 其实是一个对齐的过程. 比如 `s[ConvF].compute_at(s[Conv], oc) `  ConvF在 compute at之后, 就和Conv一样的轴了, 每个轴都相同. 

` (n // block, n // block, block, block),`  

```python
cm1 (n // block, n // block, n // block, block // 2, block // 2),
c1  (n // block, n // block, block, block)
c11  (n // block, n // block, n // block, block // 2, block // 2),
s[c11].compute_at(s[c1], c1.op.axis[1]) # 就是把前两个对齐
s[cm1].compute_at(s[c11], c11.op.axis[4]) # 前4个都对齐. 
```

InternalError: Check failed: (!out_dom_map->count(this->reduce_axis[i])) is false:  是为啥?  之前算了te.sum, 后面不能直接加起来? 不是,  是因为 `k = te.reduce_axis((0, recur), "k")` ， 一个reduce_axis 同时传入多个te.compute就容易不满足他的assumption check. 同一个句柄认为这两个op不一样. 所以要新建多个reduce axis,可以在函数里新建.

可以把每个中间变量打印出来.

你要去看每一个schedule api的语义 .  知道每个api能干什么

怎么reorder reduce轴?  一样的. 

preserve_unit_loops是什么? 

- fused loop will retain any unit loops (loops with a single iteration) from the original loops.
- Essentially, it ensures that if any of the original loops had a unit iteration (meaning it runs only once), that property is preserved in the fused loop.

#### reduce 

```python
B = te.compute((n,), lambda i: te.sum(A[i, k], axis=k), name="B")
就是对于每一行i,都做一个lambda, 就是做一个te.sum(A[i, k], axis=k). 根据k来规约.
他会自动初始化, 做 B[i] = B[i] + A[i][k];
axis的是reduction的, lambda后面的是data parallel的.

C.op.axis[0], C.op.axis[1] 为什么没有op.axis[2]
A_1 = T.Buffer((1048576,), data=A.data) # loop的buffer 会先展平. 
   ko, ki = s[C].split(kaxis, factor=kfactor) #因为有k, 所以需要split
   ko, ki = s[C].split(s[C].op.reduce_axis[0], factor=kfactor)  #  op.reduce_axis[0]是什么? 
   B = np.sum(A, axis=1) 好像是把第一维消除的意思. 
  reduce_axis是一个列表,  axis 也是一个列表. 
  reduce_axis是用在B.op, lambda i: te.sum(A[i, k], axis=k)里,  axis是split的时候用. 
```

## GPU

### METAL

Cooperative Fetching 好像没啥用,  Memory Hierarchy cache read write 也没啥用, 这些去掉反而更快了,  为什么? 因为meta内存模型和cuda不同

https://github.com/octoml/Apple-M1-BERT

https://youtu.be/Jrn2RrwgHAI?si=4-vRAIqCQSSYinxH

### A100

local 内存，它充当由一小群线程共享的快速暂存器。每个人对这个暂存器内存都有不同的名称。Intel称其为SLM（共享本地内存），Nvidia称其为Shared Memory，AMD称其为LDS（本地数据共享）。Apple 称其为 Tile Memory。为了简单起见，我们将使用 OpenCL 术语，并将其称为本地内存。

https://leiblog.wang/tir-effcient-gemm/ 跟着这个写.  https://zhuanlan.zhihu.com/p/560729749

#### cache read/write

那cache_read(Apad, "shared", 和s.cache_write(B, "local")  这个local又是什么呢?

local是  local registers.

`AA = s.cache_read(A, "shared", [B])`）  会先将A的数据load到shared memory中，然后计算B。在这里，我们需要引入一个stage的概念，一个op对应一个stage，也就是通过cache_read会新增一个stage。

CPU 不要cache read, 只要cache write. cache read没啥用, 因为cpu没有share memory

CPU  512 bit, 每次取32bit, 可以用cache read 来处理这种情况,但是一般编译器都处理的很好了. 所以cache read没啥用. 

cache write就是计算矩阵乘法C是16 x16的时候cache locality不好, 就开一个 flatten的 C' 1x256, cache write 回memory

 多个SM 共享share memory, 会有barrier sync一下,  thread级别就是每个SM自己管自己.

share memory 都要bind thread和block.

对cache read不懂, 先做最简单的 A= B, 先fetch到local再fetch到 share. tvm的例子不好, 因为卷积太复杂了. 

share好像就是要配合local, 只有share 就效果很差. 

#### bind

`thread_x = te.thread_axis("threadIdx.x")`  thread数量是tvm自动选择的.

- 绑定轴到线程轴会影响每个线程使用的寄存器数量。
- 如果绑定轴增加了寄存器压力（例如由于更多的局部变量或复杂的计算），可能会影响性能。

绑定最里面的yi最快, 因为内存访问是连续的.如果 `xi` 和 `yi` 的范围较大，绑定线程束到这些坐标轴可能导致性能下降。这是因为线程束的数量是有限的，如果范围较大，那么每个线程束需要处理更多的迭代次数，从而导致较长的执行时间。

#### Virtual Thread

是什么? 陈天奇说是we create inner-most serial loops to simulate concurrent execution of the threads. Because vthread executes in the same thread, the vthread lowering will perform optimization to detect sharable computation among different vthread and only compute once.

Such compound effect is useful to create shared stridded access patterns such as those in gemm. 目的是为了缓解bank conflict.

但是还是看不太懂.    也有教授说不用管它.就没啥用.  

#### 卷积

https://tvm.apache.org/docs/how_to/optimize_operators/opt_conv_cuda.html 

batch = 1 很快, batch = 128 会很慢, 可能是不能并行. 

thread_axis最多有几个? 还是最多xyz. 那我有超过3个维度怎么办? 就只能控制其中3个维度.

https://sandeep06011991.github.io/papers/2021-3-10-TVM-Scheduling/

`block = s[B].fuse(x,y)` 在cpu上好像不能加速.

gpu可以用vectorize, 但是只有2. 也可以用unroll。

没装llvm会报错,  tvm/src/target/parsers/aprofile.cc:118: Warning: Cannot parse Arm(R)-based target features without LLVM support.  Segmentation fault (core dumped) `func = tvm.build(s, [A, W, B], "cuda")` 

```python
 # cpu 会get到llvm module, 这个module 直接就是它的host.  
 fadd.get_source())
```

Print GPU代码,  imported module才是cuda kernel 的module, 对这个module get source可以拿到cuda 的source代码.

```python
func = tvm.build(s, [A, B, C], target="cuda")
dev_module = func.imported_modules[0]
print(dev_module.get_source())
```

一般是外loop bind "blockIdx.x", 内loop bind "threadIdx.x"

`thread_x = te.thread_axis((0, num_thread), "threadIdx.x")` 中num_thread是  warp的数量吗? 不是,就是并行度. 类似于split.  因为一个warp 有32个thread, 所以  threadIdx.x 最好是32的整数.

可以每个阶段都`write_code(str(tvm.lower(s, [A, B, C], simple_mode=True)), "progress/2.split_i.cu")` 保存一下看差距

#### 报错

1. InternalError: Check failed: (match) is false: T.iter_var(blockIdx_y, None, "ThreadIndex", "blockIdx.y") domain already inferred, cannot prove their extents are the same 1024 vs 4

意思是说你bind了这个blockIdx.y  两次, 但是thread 数量不一致,  一个是1024一次是4。 解决方法:  轴要对齐, 在第几个也要一样, 然后 thread.y 或者blockIdx.y 数量要一样.

2. InternalError: Check failed: iv->iter_type == kDataPar (2 vs. 0) : Can only relayout with in data parallel dimensions

不知道

3. TVMError: Assert fail: T.tvm_struct_get(A, 0, 10, "int32") == 2, Argument mmult.A.device_type has an unsatisfied constraint: 2 == T.tvm_struct_get(A, 0, 10, "int32")

意思是没有to device, 报错不告诉你要to cuda. 从tvm.nd.array(a)改成了 tvm.nd.array(a,dev)   就对了.

4. 尝试bind reduce轴 ki, `s[C].bind(ki, thread_z)`  ptxas error   : Entry function 'default_function_kernel_1' uses too much shared data (0x10000 bytes, 0xc000 max)

超出共享内存限制.  0xc000 就是 49152 就是 shared memory, 在每一个SM里面最大 48K. 

5. TVMError: not implemented .`print(tvm.lower(s, [A,W, B], simple_mode=True))`  

其实是没有GPU. 报错不友好.

6 TVMError: CUDALaunch Error: CUDA_ERROR_INVALID_VALUE
 grid=(2048,1,1),  block=(2048,1,1)

因为单个block里的thread总数 <= 1024.

7. InternalError: Check failed: (found_attach || stage_attach.size() == 0) is false: Invalid Schedule, cannot find the producer compute(A.shared, body=[A[ax0, ax1]], axis=[T.iter_var(ax0, T.Range(0, 4096), "DataPar", ""), T.iter_var(ax1, T.Range(0, 4096), "DataPar", "")], reduce_axis=[], tag=, attrs={}) along the loop nest specified by compute_at of consumer compute(A.shared.local, body=[A.shared[ax0, ax1]], axis=[T.iter_var(ax0, T.Range(0, 4096), "DataPar", ""), T.iter_var(ax1, T.Range(0, 4096), "DataPar", "")], reduce_axis=[], tag=, attrs={})

loopnest, 循环嵌套, 一般来说一个算子就是一个loopnest, 比如C = te.compute((m, n) 就是两层循环的loopnest.

8. InternalError: Check failed: (found) is false: Cannot find the axis T.iter_var(jj_outer_inner, None, "DataPar", "") in parent's leaf_iter_vars parent=stag

因为 `s[AL].compute_at(s[C], ty)` 和`s[AL].compute_at(s[CC], ty)` 是不一样的, C有ty 这个axis, 但是CC没有.

9. s[CC].tensorize(ty, intrin_wmma_gemm()) 显示TVMError: Operate on iter var T.iter_var(jj_outer_inner, None, "DataPar", "")that is not part of the schedule

就是没有这个轴, 得把CC split开. 

#### tensorcore

https://daobook.github.io/tvm/docs/how_to/optimize_operators/opt_conv_tensorcore.html 运行了一下, conv2d with tensor core: 1.191321 ms  如何使用TensorCores优化卷积 - 吴建明wujianming的文https://zhuanlan.zhihu.com/p/338608677

我发现github就有 gemm的代码. https://github.com/apache/tvm/blob/6252fa5802c94df522306519da94b874b3a45eda/python/tvm/topi/cuda/batch_matmul_tensorcore.py#L283    

可以参考 batch_matmul_tensorcore.py

```python
block_row_warps =4 # 每个块包含 2x4 个 warps
block_col_warps = 2 
warp_row_tiles = 2 # 每个 warp 调用 4x2 TensorCore 指令, 一个输出是16x16,所以一个warp输出是64x32, 一个block输出刚好128x128.由于共享内存空间的限制，我们一次只加载 2 个块（2x128x128 tile）。
warp_col_tiles = 4
chunk = 2
所有 TensorCore 指令都是 warp 级指令，这意味着 warp 中的所有 32 个线程都应该同时执行此指令.
使 threadIdx.x 范围32 是解决此问题的最简单方法之一。 就是 warp_size = 32, threadIdx.x =32  
warp_size=32, 是为了让 share memory也是32x32 . 方便读取, 解决bank conflict.
to, ti = s[AS].split(t, factor=warp_size)
s[AS].bind(ti, thread_x)

然后，我们可以将 threadIdx.x 绑定到任何循环，但那些直接或间接包含 TensorCore 内部函数的循环除外.
我们唯一应该做的就是确保 warp 中的所有线程都可以同时调用 TensorCore。
```

其实就是nc拆成3个轴, block_i, nc ,nci. 他的命名一直叫nc容易让人误会.  block_i 就是 block数量, nc最后是warps的数量, nci是tensorcore 也就是tile的数量. oc 同理. block_j是oc block数量, oc是 output channel warp数量, oci是tile数量.  把oc和nc bind 上thread.

`s[ConvF].compute_at(s[Conv], oc)`, 为什么share memory bind oc warp数量而不是nc warps数量呢? 

Tensorcore, we need to use a special instruction to  Write back from register to global memory (or shared).

把filter的大小改成1x1然后ic oc(还是height weight? )就是改成你的矩阵的大小。你就可以用conv做mm。反正gemm是conv的一个特殊形式。那这是多大的矩阵? batch size个矩阵? 

#### TIR gemm

复现https://leiblog.wang/tir-effcient-gemm/ 

tir有 two primitive `compute_at` and `reverse_compute_at` while `te` mixes two primtives into one `compute_at`

`reverse_compute_at` 是 **TVM** 中的一个调度操作，用于将一个消费者块（consumer block）移动到特定循环下，并重新生成由该块引发的循环，以便消费者块消耗的缓冲区区域可以覆盖给定循环下其生产者块产生的区域

- `preserve_unit_loops`：是否保留范围为1的微不足道的循环。

- `index = -1` 表示插入到最后一个可能的插入点；
- `index = -2` 表示插入到第一个可能的插入点；

没看懂和普通compute at有啥区别.

`block` in tir but `stage` in te; `lazy mode` in te and `interactive mode` in tir. 交互模式要求在每个步骤中进行验证检查，这意味着操作顺序很重要。但是，TE 使用lazy模式。我们可以改变原语的顺序。 

### refer

1.  tvm算子优化schedule（二）--GPU篇 - https://zhuanlan.zhihu.com/p/403370698

## topi

https://tvm.hyper.ai/docs/tutorial/TOPI

```python
C = topi.sum(A, axis=1) 
topi.broadcast_add,   topi.broadcast_mul
topi.nn.softmax(tarray)
可将 topi.nn.conv2d 和 topi.nn.relu 融合在一起。
```

## unroll

减少分支预测失败，如果循环体内语句没有数据相关，增加了并发执行的机会，也有利于指令流水线的调度

TVM 怎么写递归或者循环呢? 没法写, 只能手工用compute op unroll. 

`te.const(0, "int8")` 但是为什么还是出现了float32?

## Tensorize

可以内联函数.利用硬件的某个指令。

```
error: expected type
define i32 @gemv_update(ptr noundef %0, ptr noundef %1, ptr noundef %2, i32 noundef %3, i32 noundef %4, i32 noundef %5) #0 {
```

现在是否有更好的tensorize的方法? 

Tensorize 其实是reduce的 vectorize, 一次vectorize 两个循环.

vnni在tvm标准库中应该有, 去找,在哪里?

tensorize 会把B broadcast , tensorize会找出A B C , 然后plugin进去. 

## debug

tvm/src/driver/ driver.cc里面有

`IRModule mod = ScheduleToModule(std::move(sch), args, name, binds, global_var_supply);`

加一个  `LOG(INFO) << mod;`就可以看到数组一维化之前的二维数组. 就知道访问内存box 情况.   



