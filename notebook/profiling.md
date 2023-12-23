pyg2.1.0不能用torch_profile, 不过它就包装了一行, 自己写也可以. 

https://github.com/mrshenli/ptd_benchmark

https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html 

注意: 在利用profiler进行cuda性能测试时，profiler的开销不可忽略

pytorch效率分析 - 咸鱼的文章 - 知乎 https://zhuanlan.zhihu.com/p/382121335

1. with timeit(avg_time_divisor=args.num_epochs) 和  with torch_profile():有什么区别? 这两个分别有什么作用? 
2. profile_file怎么保存?  
3. 测不同batchsize.  分布式的benchmark怎么保证所有进程都结束了? 方法1. 要pkill.  Destory group. 

Aten::stack 耗时最长,是在什么地方的? 

https://github.com/pytorch/pytorch/tree/master/benchmarks/distributed/ddp/compare

profiler可以用: Python如何快速定位最慢的代码？ - pyinstrument   知乎 https://www.zhihu.com/question/485980118/answer/2113308987

计算 Python 代码的内存和模型显存消耗的小技巧 - deephub的文章 - 知乎 https://zhuanlan.zhihu.com/p/446903067

##### Py-spy

https://github.com/benfred/py-spy  只能测cpu time

启动之前需要[`--cap-add SYS_PTRACE`](https://docs.docker.com/engine/security/seccomp/)

#### Torch bench

是timeit的包装, 它会做cuda同步的操作. https://pytorch.org/tutorials/recipes/recipes/benchmark.html 

### 显存检查

https://github.com/Stonesjtu/pytorch_memlab

https://github.com/nicolargo/nvidia-ml-py3 



## Nsight

nsight system是总体的, nsight computing对cuda更深入分析. *nsys* 之前的名称叫做*nvprof*.

英伟达 https://developer.nvidia.com/nsight-systems

### 安装

怎么用本地nsight gui, 看远程docker container中的profile结果 ?

`ssh -X` 即可

如果处理文件, 就不用ssh 连接远程容器, 如果直接运行,就需要ssh 连接远程容器

https://docs.nvidia.com/nsight-systems/UserGuide/index.html#linux-launch-processes 

1. 他没告诉我应该profile文件还是直接启动进程? 

要改container的sshd_config，因为22端口可能被host占用

`nvprof` is a legacy tool and will not be receiving new features. 

在 iter开始和结束的位置打一个标签,`TORCH.CUDA.NVTX.RANGE_PUSH `  

```bash
nsys profile -w true -t cuda,nvtx,cudnn,cublas --capture-range=cudaProfilerApi --force-overwrite true -x true -O 512 python train.py --data WIKI --config ./config/TGN.yml
nsys profile -w true -t cuda,nvtx,cudnn,cublas --capture-range=cudaProfilerApi --force-overwrite true -x true -o ugache python dgl_sample.py  --data WIKI --config ./config/TGN.yml 
```

nsys-rep 可以在remote server 可视化吗? 

Generated: 没有生成文件是为啥? 



#### Nsight gui 怎么在容器中启动?

可以本地下载gui , 然后ssh连接



refer:

1. PyTorch训练加速的量化分析 - 风车车车的文章 - 知乎 https://zhuanlan.zhihu.com/p/416942523
2. https://dev-discuss.pytorch.org/t/using-nsight-systems-to-profile-gpu-workload/59
