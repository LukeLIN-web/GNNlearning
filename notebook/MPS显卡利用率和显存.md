### 原理

一个core有多个active warps来issue instructions. the physical core won’t be stalled when some of the warps are waiting to complete their memory requests. 

#### MPS

https://www.zhihu.com/question/307643863

```bash
man nvidia-cuda-mps-control # Describes usage of this utility.
nvidia-cuda-mps-control -d # Start daemon in background process.
ps -ef | grep mps # See if the MPS daemon is running.
echo quit | nvidia-cuda-mps-control # Shut the daemon down. 我遇到退出会卡死
nvidia-cuda-mps-control -f # Start deamon in foreground
```

命令怎么用参考:  https://docs.nvidia.com/deploy/mps/index.html#topic_5_1_1

EXCLUSIVE_PROCESS怎么用? 



## GPU-Utilization 

### SM使用率

`nvidia-smi dmon`也可以看, 和dcgm有什么区别? 

 SMACT， DCGM 工具抓取。如果 SMACT 利用率为 0.3 (30%)，那么混 3 个同类任务大概率吞吐可以达到之前 3 倍。 但是我们不是同类任务. 

按照https://developer.nvidia.com/dcgm#Downloads ,  我还发现我安装了20.04的版本不是18.04但是好像没关系 

```bash
dcgmi stats -g 1  -s pid
dcgmi stats -x pid
dcgmi stats -j  pid
dcgmi dmon -e 1002,1003,1005 -c 100 -d 1000 -i 0 # -i 选择某个GPU
```

这个-e, field id是什么意思?  就是一些指标

SM Activity是时间占比, SM Occupancy是空间占比. 

https://docs.nvidia.com/datacenter/dcgm/2.0/dcgm-user-guide/feature-overview.html#profiling-metrics

Memory BW Utilization The ratio of cycles the device memory interface is actively sending or receiving data.

The PCIe bandwidth is measured in bytes per second, so 864034668 means 864MB/s PCIe bandwidth

dcgmi group -d 0
Error: Cannot destroy group 0. Return: The Group is not found.

https://stackoverflow.com/questions/40937894/nvidia-smi-volatile-gpu-utilization-explanation

是 CUDA kernel activity.  For a given period, it reports the percentage of time one or more cuda kernel(s) were active (i.e., running).

Using `nvidia-smi --help-query-gpu` 

```
"utilization.gpu"                                                                         
Percent of time over the past sample period during which one or more kernels was executing on the GPU.    
The sample period may be between 1 second and 1/6 second depending on the product.       
"utilization.memory"                                                                     
Percent of time over the past sample period during which global (device) memory was being read or written.
The sample period may be between 1 second and 1/6 second depending on the product.      
```



```
RuntimeError: CUDA out of memory. Tried to allocate 318.00 MiB (GPU 0; 15.78 GiB total capacity; 11.61 GiB already allocated; 255.62 MiB free; 11.65 GiB reserved in total by PyTorch) 为什么这样会爆显存呢? 
```



#### GPU虚拟化

ampere 有 MIG，但是不知道对性能影响怎么样. MIG 可以切GPC和 memory. 按GPC graphics processing clusters分. 
