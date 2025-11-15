scatter

首先util 定义了 broadcast。 src变成和other一样大。 index变成和src一样大

dim == 1， 那么就数据填充。 

然后scatter add调用了torch自带的,  TORCH.TENSOR.SCATTER_.

torch tensor 的scatter和gather： scatter是gather的逆操作。 gather好像比较简单， 就是根据index来填入out 。 out的每个元素不止和input有关还和index 有关。 

  scatter_是按index来填入自己。  scatter add也是。

pyg class Aggregation(torch.nn.Module):会用这个方法 



![](https://raw.githubusercontent.com/rusty1s/pytorch_scatter/master/docs/source/_figures/add.svg?sanitize=true)

expand https://pytorch.org/docs/stable/generated/torch.Tensor.expand.html。 可以变长，直接复制元素（要么为1要么就一样长），   -1就是不变。 

numpy中 size是元素个数， shape是大小。 

tensor的shape是一个属性， size是一个方法。



260 is not a hard course to pass, wouldn't take you too much time 

240 is solid and involves tough coding。

 STAT210 They call it a free A course

I've been told Knowledge Rep is kinda abstract, a challenging course.



接下来一周， 看一下别人是怎么文献总结的，然后我要先阅读一下pyG的文档， 看看scatter和sparse的实现。

https://arxiv.org/abs/1803.08601

Design Principles for Sparse Matrix Multiplication on the GPU
