### tensor

https://blog.csdn.net/Thera_qing/article/details/95361598

`edge_index = edge_index[:, edge_mask]` 就是取第二维度根据mask来取值. https://stackoverflow.com/questions/59344751/is-there-any-diffrence-between-index-select-and-tensorsequence-in-pytorch查看. 

注意tensor(0) 和tensor(0) 会被set认为是不同的. 



torch.allclose 来测试两个tensor是否一样
