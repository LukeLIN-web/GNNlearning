

### 安装

https://data.dgl.ai/wheels/repo.html

`pip install https://data.dgl.ai/wheels/dgl_cu113-0.7.1-cp38-cp38-manylinux1_x86_64.whl`

#### GAT

有mini batch的方法.可以在https://ogb.stanford.edu/docs/leader_nodeprop/上找代码. 

https://github.com/devnkong/FLAG/blob/main/ogb/nodeproppred/products/gat_ns.py 这个repo有很多sampler的模型. 

#### Dgl 0.7

sampled_coo = aten::COOTranspose(aten::COORowWiseSampling(
aten::COOTranspose(hg->GetCOOMatrix(etype)),nodes_ntype,
fanouts[etype], prob_or_mask[etype], replace)); aten库的cpp代码运行在GPU上

DGLContext ctx = aten::GetContextOf(nodes); 应该是包含了设备信息

0.6.1 就没有设备信息ctx.

#### dgl1.1

UVA采样, 就是GPUs perform the sampling on the graph pinned in CPU memory via zero-copy access

pytorch1.12的`tensor.copy_` 必须要一样大 

RuntimeError: The size of tensor a (250000) must match the size of tensor b (106353) at non-singleton dimension 0

### neighborsampler

prefetch

set_dst_lazy_features
