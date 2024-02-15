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


```

https://tvm.apache.org/docs/how_to/optimize_operators/opt_gemm.html

一开始是Numpy running time: 0.015463
Baseline: 1.414840

32 blocking 稳定是0.15,快了10倍. 16 blocking 是0.10,更快了, 64也是  0.10

Vectorization  速度没有变快.  `C_1[cse_var_1:cse_var_1 + 64] `   是因为第一种优化方法[blocking]产生的代码 被编译器自动优化了 相当于做了矢量优化 

T.Broadcast 是啥?  



## TIR

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



lambda不能写这么复杂的函数.  tvm写function.怎么写? 



