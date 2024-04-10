#### 未来替代

ansor也是一个 dnn 自动编译器. ion和zhuodan yang他们做的. 

tvm 前身halide, 是mit 图形学的教授组, 发明 split这一套抽象.谷歌在用halide. 毕竟tvm是前llm时代的东西,性能和好用很多时候只能二选一。tvm之前用的是compile到那种小设备上的,比如音箱。llm不能塞这些小设备. 

cutlass 是英伟达新官方cpp 模板库, gemm比tvm快很多. 

### 安装

```bash
git clone --recursive https://github.com/apache/tvm

brew install miniforge # macbook需要
可以通过conda 直接建依赖
# Create a conda environment with the dependencies specified by the yaml
conda env create --file conda/build-environment.yaml# m1 半个多小时.服务器15-20分钟. 
# Activate the created environment
conda activate tvm-build

mkdir build
cp cmake/config.cmake build
# 默认没有打开llvm编译开关., 把llvm和 metal 设置为ON.

cd build
cmake .. -G Ninja
ninja

conda install conda-forge::ninja不行, pip就可以. 
# then set python part so we can use in python, refer https://tvm.apache.org/docs/install/from_source.html#tvm-package
```



## TIR语法

tvm很难debug, 肉眼看tir 非常困难.

https://tvm.apache.org/docs/how_to/optimize_operators/opt_gemm.html

Vectorization  速度没有变快.  `C_1[cse_var_1:cse_var_1 + 64] `   是因为产生的代码 被编译器自动优化了 相当于做了矢量优化 .

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

在 GPU 上，全局内存的工作方式类似于 CPU 内存。有constant 的内存，它是只读的。还有local 内存，它充当由一小群线程共享的快速暂存器。每个人对这个暂存器内存都有不同的名称。Intel称其为SLM（共享本地内存），Nvidia称其为Shared Memory，AMD称其为LDS（本地数据共享）。Apple 称其为 Tile Memory。为了简单起见，我们将使用 OpenCL 术语，并将其称为本地内存。



报错很不友好, TVMError: not implemented .`print(tvm.lower(s, [A,W, B], simple_mode=True))`  不会告诉你没有GPU. 

https://sandeep06011991.github.io/papers/2021-3-10-TVM-Scheduling/

`block = s[B].fuse(x,y)` 在cpu上好像不能加速.`AA = s.cache_read(A,"shared",[B])` 用了反而更慢了. 

不要cache read, 只要cache write.

GPU 有48KB ,可以32KB scratch pad, 16KB cache, 编译器也可以决定划分成32KB的cache. 

CPU  512 bit, 每次取32bit, 可以用cache read 来处理这种情况,但是一般编译器都处理的很好了. 所以cache read没啥用. 

cache write就是计算矩阵乘法C是16 x16的时候cache locality不好, 就开一个 flatten的 C' 1x256, cache write 回C矩阵. 

### 卷积

https://tvm.apache.org/docs/how_to/optimize_operators/opt_conv_cuda.html 

batch  =1 很快, batch = 128 会很慢, 可能是不能并行. 

cache read没啥用, 因为cpu没有share memory

Cooperative Fetching 好像没啥用,  Memory Hierarchy cache read write 也没啥用, 这些去掉反而更快了,  为什么?

如果不bind block的话, 会非常慢. 

thread_axis最多有几个? 无数个? 还是最多xyz. 那我有超过3个维度怎么办?是否就不能用GPU加速? 









### METAL

苹果 能的。你理论上tvm把target改成metal就行.

https://github.com/octoml/Apple-M1-BERT

https://youtu.be/Jrn2RrwgHAI?si=4-vRAIqCQSSYinxH

```
tvm._ffi.base.TVMError: Traceback (most recent call last):
  Did you forget to bind?
    Variable `C` is directly accessed by host memory (it is not contained in a thread environment or in the function arguments.
```

因为没有生成计划.  







## tile

tile技术, 太复杂了 , 看不懂了.  

参考

1. 循环优化之循环分块（loop tiling） https://zhuanlan.zhihu.com/p/292539074  和
   https://zhuanlan.zhihu.com/p/403163009
2. https://aijishu.com/a/1060000000381408
3. 推荐看 https://halide-lang.org/ 

#### reorder

为什么会TVMError: Operate on iter var T.iter_var(j_outer, None, "DataPar", "")that has already been split

因为 你下面的reorder里面有mi & ni ,  但是下面它们已经被拆成nio mio了。

tvm写了一下, 64 效果最好. 写一个 32 x32 , 然后 128 x128, 写了 . 效果不好. 因为没有reorder

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

compute at 其实是一个对齐的过程.

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

怎么reorder reduce轴? 

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
