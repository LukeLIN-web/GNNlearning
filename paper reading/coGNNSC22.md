## 摘要

organizes the tasks in a queue and estimates the memory consumption of each task based on cost functions at operator basis. In addition, the CoGNN implements scheduling policies to generate task groups, which are iteratively submitted for execution.



定义: PMC  : peak memory 



## 3 motivation

1. input输入对训练影响很大

2. GPU利用率低
3. 预先估计最大memory 非常重要 .否则会很慢. 





### inference的挑战

inference要实时调度.

## 4 cognn 设计

有一个内存profiler, 估计内存消耗. 

他们也没法准确估计, 所以用了一个阈值. 



### scheduling policies

#### base

尽可能多package

#### BMC policy

从小到大 根据PMC 排序.  左边一个, 右边一个. 

为什么这样设计? 







## 代码

https://github.com/guessmewho233/CoGNN_info_for_SC22/tree/master/environment

#### base policy的实现

用hash name取出.

joblist存了什么? 

