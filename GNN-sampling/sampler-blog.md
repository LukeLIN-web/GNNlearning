Juyi Lin

We can learn a function that generates embeddings by sampling and aggregating features from a node’s local neighborhood(Hamilton et al. 2017). Today I would like to talk about a part of the PyG sampler. 

 Let’s start with the following code:

```python
from torch_geometric.loader import NeighborSampler
...
train_loader = NeighborSampler(data.edge_index, node_idx=train_idx,
                                   sizes=[25, 10], batch_size=1024,
                                   shuffle=True, num_workers=0)
        for batch_size, n_id, adjs in train_loader:
            optimizer.zero_grad()
            out = model(x[n_id], adjs) 
            loss = F.nll_loss(out, y[n_id[:batch_size]])
            loss.backward()
            optimizer.step()
```

It's really similar with normal PyTorch training. The most important is NeighborSampler. We need to take it apart. What’s going on in there?

`class NeighborSampler(torch.utils.data.DataLoader)` have a method`def sample(self,batch) ` When was the sampling done? 

```python
    def __init__(self, edge_index: Union[Tensor, SparseTensor],
                 sizes: List[int], node_idx: Optional[Tensor] = None,
                 num_nodes: Optional[int] = None, return_e_id: bool = True,
                 transform: Callable = None, **kwargs):
        ...
        super().__init__(
            node_idx.view(-1).tolist(), collate_fn=self.sample, **kwargs)
    def sample(self, batch):
    		...
        for size in self.sizes:
            adj_t, n_id = self.adj_t.sample_adj(n_id, size, replace=False)
        ...
        return out
```

You can find `super(). __init__(node_idx.view(-1).tolist(), collate_fn=self.sample, **kwargs)`  The collate_fn is overridden.

In PyTorch DataLoader, you can find `collate_fn` :

```python
class DataLoader(object):
    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0, collate_fn=default_collate,
                 pin_memory=False, drop_last=False, timeout=0,
                 worker_init_fn=None)
	...
    def __next__(self):
        if self.num_workers == 0:  
            indices = next(self.sample_iter) # note: torch sampler is selecting index, it is totally different from what we mention PyG sample.
            batch = self.collate_fn([self.dataset[i] for i in indices]) # Dataset
            ...
            return batch
```

What `collate_fn` does is merge the data of a batch.  For example, in CNN the default `collate_fn` is to merge img and label into imgs and labels respectively. In general purpose, the DataLoader starts by picking random idx and provides `collate_fn` so that different types of data(imgs, labels, nodes, and so on) can become batch. 

In our case, the NeighborSampler starts by picking some nodes index as minibatch, and the PyG `sample` function, as `collate_fn`, does a very important job. The underlying c++ code is called to sample some neighboring nodes and process them to adjs. Then return it.

#### sample neighbor

NeighborSampler inits and create `adj_t` SparseTensor. You can refer to [https://github.com/rusty1s/pytorch_sparse](http://link.zhihu.com/?target=https%3A//github.com/rusty1s/pytorch_sparse) to see the detail. 

sample will call `pytorch_sparse/csrc/sparse.h sample_adj` api, which call `sample_adj_cpu` api.

If you want to know how the sampler does, you have to get familiar with Compressed Sparse Row（CSR） storage. https://en.wikipedia.org/wiki/Sparse_matrix. It is very useful to refer to a CSR matrix while looking at codes.

```cpp
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
sample_adj_cpu(torch::Tensor rowptr, torch::Tensor col, torch::Tensor idx,
               int64_t num_neighbors, bool replace) {
......
  else if (replace) { // Sample with replacement ===============================
 // iterate num_neighbors
    for (int64_t i = 0; i < idx.numel(); i++) { // iterate input nodes
      n = idx_data[i];
      row_start = rowptr_data[n], row_end = rowptr_data[n + 1];
      row_count = row_end - row_start; // node out degree
      if (row_count > 0) {
        for (int64_t j = 0; j < num_neighbors; j++) { //  find num_neighbors nodes
          e = row_start + uniform_randint(row_count);// col, replace so it can be repeated.
          c = col_data[e]; // find corresponding col
          if (n_id_map.count(c) == 0) {
            n_id_map[c] = n_ids.size();
            n_ids.push_back(c);
          }
          cols[i].push_back(std::make_tuple(n_id_map[c], e));
        }
      }
      out_rowptr_data[i + 1] = out_rowptr_data[i] + cols[i].size();// output matrix also CSR format
    }
 } else { // Sample without replacement via Robert Floyd algorithm ============
    for (int64_t i = 0; i < idx.numel(); i++) { // iterate input nodes
      n = idx_data[i];
      row_start = rowptr_data[n], row_end = rowptr_data[n + 1];
      row_count = row_end - row_start;// this node outdegree
      std::unordered_set<int64_t> perm;
      if (row_count <= num_neighbors) {
        for (int64_t j = 0; j < row_count; j++)
          perm.insert(j);
      } else { // See: https://www.nowherenearithaca.com/2013/05/
               //      robert-floyds-tiny-and-beautiful.html
        for (int64_t j = row_count - num_neighbors; j < row_count; j++) {
          if (!perm.insert(uniform_randint(j)).second)
            perm.insert(j);
        }
      } // find num_neighbors not repeated nodes
      for (const int64_t &p : perm) {
        e = row_start + p;
        c = col_data[e];
        if (n_id_map.count(c) == 0) {
          n_id_map[c] = n_ids.size();
          n_ids.push_back(c);
        }
        cols[i].push_back(std::make_tuple(n_id_map[c], e));
      }
      out_rowptr_data[i + 1] = out_rowptr_data[i] + cols[i].size();// output matrix also CSR format
    }
  }
}
```

So far, we have known how the naive PyG sampler does. However, it leads to complicated GNN designs that are tightly coupled to a specific sampler. The tuple `(batch_size, n_id, adjs)` is not very elegant. Thus, PyG improves it to NeighborLoader. NeighborLoader can obtain a cleaner GNN design. 

```python
    def collate_fn(self, index: Union[List[int], Tensor]) -> Any:
        out = self.neighbor_sampler(index)
        if self.filter_per_worker:
            # We execute `filter_fn` in the worker process.
            out = self.filter_fn(out)
        return out
```

It is easier to read. we wrapped some properties(like batch size) into the `Data` class. And we could support HeteroData and further functions. 

And then, As we need to support more and more features and data, our interfaces are becoming more and more complex. we improve the output to a dedicated class, it is still in progress:

```python
class SamplerOutput:
    node: torch.Tensor
    row: torch.Tensor
    col: torch.Tensor
    edge: torch.Tensor
    batch: Optional[torch.Tensor] = None
    # TODO(manan): refine this further; it does not currently define a proper
    # API for the expected output of a sampler.
    metadata: Optional[Any] = None
```

In conclusion, no matter how complex the functionality is, it is always helpful to understand the underlying principles.  The point of this blog is how PyG combines graph data with the basic functionality provided by PyTorch.

Thanks, **Ivaylo Bahtchevanov** for giving me this chance to write this blog.
