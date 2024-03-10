### 安装

```bash
git clone --recursive https://github.com/apache/tvm tvm

brew install miniforge
conda init
conda create --name tvm python=3.8
conda activate tvm

或者可以通过conda 直接建依赖
# Create a conda environment with the dependencies specified by the yaml
conda env create --file conda/build-environment.yaml# 大概半个多小时.
# Activate the created environment
conda activate tvm-build

mkdir build
cp cmake/config.cmake build

cd build
cmake .. -G Ninja
ninja

默认没有打开llvm编译开关.

# RuntimeError: Distributed package doesn't have NCCL built in

https://download.llamameta.net/*?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoicDIwaWMwNGRkbGVkdDVmMWN1dG5pcm54IiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZG93bmxvYWQubGxhbWFtZXRhLm5ldFwvKiIsIkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTcwODk1ODQxNn19fV19&Signature=F9S62rlAoXtU4woDuX5-jwyQINILB-Jx2m0aly3sdu7DO2T-RQm9G3WR-OpYIXoUQzR213zCskvXCegN3mclhpqccWCHkhxgtcKrBcHZIsxS9PdB7Ynpx-bnRoqAswOKSt9np3wuQsSewfPVBuw0Xvdz9OMgIZpd57vnIlnTlo-57PkrWDBA9KasZFoQzSnrpvmZU0e2bq5mXyn6gv0YUZpxoNqYBgMBV9xIzaFge6wck%7EGKM2FfRkWJpWQ2e6ocncmOtZQofIOksXXTkB9FvYqQI3Y0%7E2m2NZHprwnYkzjpLU6kUcrBsWVrIE9OQp2Dpn69PWw9yb9OSZgSdZ0vmQ__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=1083881392818407

```

https://tvm.apache.org/docs/how_to/optimize_operators/opt_gemm.html

Vectorization  速度没有变快.  `C_1[cse_var_1:cse_var_1 + 64] `   是因为第一种优化方法[blocking]产生的代码 被编译器自动优化了 相当于做了矢量优化 

## TIR

tvm很难debug, 肉眼看tir 非常困难.

学习 https://tvm.hyper.ai/docs/how_to/te_schedules/primitive/

T.grid  

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

```python
C.op.axis[0], C.op.axis[1] 为什么没有op.axis[2]?
A_1 = T.Buffer((1048576,), data=A.data) # loop的buffer 会先展平. 
   ko, ki = s[C].split(kaxis, factor=kfactor) #因为有k, 所以需要split
   ko, ki = s[C].split(s[C].op.reduce_axis[0], factor=kfactor)  #  op.reduce_axis[0]是什么? 
   B = np.sum(A, axis=1) 好像是把第一维消除的意思. 
  reduce_axis是一个列表,  axis 也是一个列表. 
  reduce_axis是用在B.op, lambda i: te.sum(A[i, k], axis=k)里,  axis是split的时候用. 
```

## tile技术

tile技术, 太复杂了 , 看不懂了.  

参考

1. 循环优化之循环分块（loop tiling） https://zhuanlan.zhihu.com/p/292539074

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



 `s[packedB].vectorize(littleN)` 没啥用.去掉反而更快. 因为 llvm会帮你做规整的vectorization.



packing之后好像可以快两倍。 

```python
# packedB = te.compute(
#     (N / bn, K, bn), lambda bigN, k, littleN: B[k, bigN * bn + littleN], name="packedB"
# )

# C = te.compute(
#     (M, N),
#     lambda m, n: te.sum(A[m, k] * packedB[n // bn, k, tvm.tir.indexmod(n, bn)], axis=k),
#     name="C",
# )
s[C].vectorize(nii)  #可以更快
s[packedB].parallel(bigN) # 没啥用
```

https://tvm.hyper.ai/docs/0.10.0/how_to/te_schedules/compute_reduce/

虽然一次 batch 操作会有多个输出，但它们只能一起调度。 就是说调度一个的时候`s[B0].compute_at(s[C], C.op.axis[0])` 另一个也会做同样的调度.

一个cm1是对的， 但是tuple的时候会报错InternalError: Check failed: (ReduceEqual(reduce, reduce_)) is false: The Reduce inputs of ComputeOp should have the same attribute except value_index 为啥呢。

最笨的办法。你去ReduceEqual里面。把它的每一个condition打出来。 看区别是啥。

func = tvm.build(s, [A, B, out], target=target)

InternalError: Check failed: (!out_dom_map->count(this->reduce_axis[i])) is false:  是为啥? 

之前算了te.sum, 后面不能直接加起来? 不是,  是因为 `k = te.reduce_axis((0, recur), "k")` ， 一个k 同时传入多个te.compute就容易不满足他的assumption check. 同一个句柄认为这两个op不一样, 

```python
C.op.axis 是[T.iter_var(i, T.Range(0, m), "DataPar", "")]  # 真是抽象啊. 不知道多个axis是啥样的. 
# axis可以理解为循环
s[B].compute_at(s[C], C.op.axis[0]) # 实际上是把B的计算移动到C的第一个循环
```

https://sandeep06011991.github.io/papers/2021-3-10-TVM-Scheduling/

`block = s[B].fuse(x,y)` 在cpu上好像不能加速.`AA = s.cache_read(A,"shared",[B])` 用了反而更慢了. 

不要cache read, 只要cache write.

GPU 有48KB ,可以32KB scratch pad, 16KB cache, 编译器也可以决定划分成32KB的cache. 

CPU  512 bit, 每次取32bit, 可以用cache read 来处理这种情况,但是一般编译器都处理的很好了. 所以cache read没啥用. 

cache write就是计算矩阵乘法C是16 x16的时候cache locality不好, 就开一个 flatten的 C' 1x256, cache write 回C矩阵. 



