## MessagePassing

#### propagate

    参数
    edge_index (Tensor or SparseTensor) - 一个torch.LongTensor或torch_sparse.SparseTensor，它定义了底层图的连接性/消息传递流。edge_index持有形状为[N, M]的一般（稀疏）分配矩阵的索引。如果edge_index是torch.LongTensor类型，它的形状必须定义为[2, num_messages]，其中来自edge_index[0]的节点的消息被发送到edge_index[1]的节点（如果flow="source_to_target"）。如果edge_index是torch_sparse.SparseTensor的类型，它的稀疏指数（row, col）应该与row=edge_index[1]和col=edge_index[0]有关。这两种格式的主要区别在于，我们需要将转置的稀疏邻接矩阵输入到propagate()中。
    size（元组，可选）--在edge_index是LongTensor的情况下，分配矩阵的大小（N，M）。如果设置为None，大小将被自动推断并假定为二次。如果edge_index是一个Torch_sparse.SparseTensor，这个参数将被忽略。(默认：无)
    **kwargs - 任何额外的数据，这些数据需要用来构建和聚合消息，以及更新节点嵌入。

propagate 首先 message产生,然后执行agg. 

​       sageconv调用forward ,  `out = self.propagate(edge_index, x=x, size=size)`

为什么propagate要 collect? 

collect是干嘛的? 

```python
def __collect__(self, args, edge_index, size, kwargs):返回一个dict . 包含了edge index i, j.就是可以对sparsetensor或tensor都产生一些metadata.
```

那么, index, src是什么东西呢?  

https://github.com/pyg-team/pytorch_geometric/discussions/4759  

lift 是干嘛的? 

```python
def __lift__(self, src, edge_index, dim):
根据index选出 src node 中的source node 和target node indices, This is needed to compute a message per edge, which we then later aggregate according to destination node indices. 
```

decomposed_layers是什么? 

比如有 28个边, message之后, out = 一个28 行的矩阵. 

#### aggre

out = self.aggregate(out, **aggr_kwargs)   之后,  有一个10行1列的矩阵.  但是为啥别的node也更新了呢? 明明minibatch呀. 

