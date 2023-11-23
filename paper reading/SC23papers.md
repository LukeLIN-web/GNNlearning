BLAD: Adaptive Load Balanced Scheduling and Operator Overlap Pipeline For Accelerating The Dynamic GNN Training

sjtu 不是ipads的.  不可复现. dockerhub和 github 都删除了.

### 摘要

调度workload 减少同步和通讯时间.  调整op 执行顺序.

allocates each snapshot group to a GPU

### INTRODUCTION

Vertex partition (e.g. Aligraph, DGL, PyGT) distributes the vertices of each graph snapshot to different GPUs

snapshot partition(e.g. ESDG) distributes the snapshots to different GPUs.  hidden states are transferred with the snapshot partition

问题: high communication overhead, the long synchronization, and the poor resource usage.

因为 node变化, 所以不同GPU load不 balance. 

#### 方案

All groups **without data dependency** are scheduled to one GPU, and each GPU/node is assigned multiple groups.

每个gpu两个进程,训练不同的snapshot, overlap不同类型的op.

topology manager 调整 model的op 执行顺序. 



### 3Background

之前的snapshot也要提供feature. 



### TWO-LEVEL LOAD SCHEDULING

 𝑠𝑔𝑖 represents the 𝑖-th snapshot group.

forward之后放入 queue,  为啥有的进入P2, 有的不进入? 

