# GNNlearning

solutions for Stanford University course CS224W: Machine Learning with Graphs Fall 2021 colabs

Course materials can be found here [CS224W: Machine Learning with Graphs Materials](http://web.stanford.edu/class/cs224w/)

Lectures can be found here [CS224W: Machine Learning with Graphs Video Lectures](https://youtube.com/playlist?list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn)

我的笔记 ：斯坦福cs224w图机器学习课程lec1-8笔记  https://zhuanlan.zhihu.com/p/479505721

Visualiztion:  https://distill.pub/2021/gnn-intro/ . 

http://web.stanford.edu/class/cs224w/  
colab答案https://github.com/hdvvip/CS224W_Winter2021 

https://sands.kaust.edu.sa/classes/CS294E/F21/schedule.html

PyG: https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html

GNN Sampling Papers:
1. Inductive Representation Learning on Large Graphs.
2. Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks.
3. GraphSAINT: Graph Sampling Based Inductive Learning Method.


There are three general types of prediction tasks on graphs: graph-level, node-level, and edge-level. 预测点, 边的性质. 比如一个分子表示成图, 就可以知道她的味道, 有几个环可能和性质有关. 边的预测, 可以判断图像物体之间的关系. 

CRS  compress row storage

对row_ptr的补充：
[0, 3, 4, 6]中，
0：矩阵第一行数11在values的索引是0.
3:　矩阵第二行的第一个数值19 在values中的索引是3.
４：矩阵第三行的第一个数值23 在values中的索引是4.
6：显然矩阵没有第四行。6表示当矩阵有虚拟的第四行时，6就是这虚拟行的第一个值在values里的索引。因此，6也是values的元素数量。

行数 = length(row_ptr) -1
列数 = max(col_index)



