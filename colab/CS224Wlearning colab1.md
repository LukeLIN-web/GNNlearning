CS224W学习

https://www.zhihu.com/people/zepengzhang/posts 知乎大佬一起学习

输入一个图, 每个node 有一个feature vector, 输出一个embedding, 也就是一个对下游任务有用的向量. 

#### colab0 

2022年1月28日开始学习, 安装环境.

networkx安装失败 ValueError: Check_hostname Requires Server_HostName.尝试关闭代理服务器. 成功

果然自己跑一次很多细节才会注意到. 

conda 不行, 不懂了, 加了环境变量还是不行.  绝了, pycharm里面不行, 外面用终端就可以. 

`EnvironmentNotWritableError: The current user does not have write permissions to the target environment.`

忽然意识到不应该直接写入, 而是应该虚拟环境

累了, 感觉GUI和命令行混合太复杂了,  还是继续用GUI 

`conda info --env`

我想用pycharm来编辑，但是那个虚拟环境不好用，安装老是失败，我想自己命令行安装，这应该怎么安装？

救命, 太奇怪了, powershell就不行, cmd就可以. windows 太奇怪了. cmd 显示的也是base 环境, 但是powershell 前面就不显示. 因为powershell需要init 

先安装, 然后生成requirement.

`pip freeze > requirements.txt`

```shell
conda create --name colab0
conda activate colab0
conda install pytorch torchvision torchaudio cpuonly -c pytorch
 conda info -e # 就知道在哪里. 可能可以指定地方的,  默认就是.conda里
```

错误

[CondaHTTPError: HTTP 000 CONNECTION FAILED for url 

https://stackoverflow.com/questions/50125472/issues-with-installing-python-libraries-on-windows-condahttperror-http-000-co 把ssl verfiy关了, 而且dll也复制了.  还换了源. 成功了!

成功在本地运行了colab0!

#### colab1

指标库 https://www.cxybb.com/article/nuoline/8610722

write a full pipeline for **learning node embeddings**.

把图结构变成一个tensor, 然后训练

本地pycharm需要加上 ` import matplotlib.pyplot as plt  最后加上plt.show()`

打算把这个做了之后看看论文. 或者边做边看论文. 

##### 聚类系数

一个节点的聚类参数被称为 Local Cluster Coefficient。它的计算方法也是非常的简单粗暴。先计算所有与当前节点连接的节点中间可能构成的 link 有多少个，这个数会作为分母，然后计算实际上有多少个节点被连接上了，这个数会作为分子。最终的计算结果就是 Local Cluster Coefficient。

知道了如何计算单个节点的聚类系数，现在来看如何计算整个图的 Cluster Coefficient。

一种最简单粗暴的方法是，先计算每一个节点的 Local Cluster Coefficient，然后取平均值。

##### pagerank

https://zhuanlan.zhihu.com/p/86004363

##### **closeness中心性**

`closeness = round(nx.closeness_centrality(G)[node] / (len(nx.node_connected_component(G, node)) - 1), 2)`

接近中心性需要考量每个结点到其它结点的最短路的平均长度。也就是说，对于一个结点而言，它距离其它结点越近，那么它的中心度越高。一般来说，那种需要让尽可能多的人使用的设施，它的接近中心度一般是比较高的。

中介中心性指的是一个结点担任其它两个结点之间最短路的桥梁的次数。一个结点充当“中介”的次数越高，它的中介中心度就越大。如果要考虑标准化的问题，可以用一个结点承担最短路桥梁的次数除以所有的路径数量。

#### pytorch

```python
zeros = torch.zeros(3, 4, dtype=torch.float32) #可以指定dtype
zeros = zeros.type(torch.long) # 可以转换dtype
torch.t(x) # 可以把tensor 变成一个多行两列的tensor, Expects input to be <= 2-D tensor and transposes dimensions 0 and 1.
nn.Embedding(num_embeddings=4, embedding_dim=8) # 我们可以确定个数和维度数. 
id = torch.LongTensor([1]) # 64-bit integer (signed)	torch.LongTensor
emb_sample(id)  # 选择一个embedding
# 初始化embedding
shape = emb_sample.weight.data.shape
emb_sample.weight.data = torch.ones(shape)
torch.rand(shape) #就是 uniform distribution, in the range of  [0,1)
```

#### 采样negative边

**实现负采样函数，讨论给出的五个边是否为负边（原图不存在的边）**

```python
for i in range(epochs):        
    	optimizer.zero_grad() #
        pred = sigmoid(torch.sum(emb(train_edge)[0].mul(emb(train_edge)[1]), 1))# 计算余弦相似度 ,加了sigmoid 把他normalize了,  做了个内积, 把结果变到0到1之间
        loss = loss_fn(pred, train_label) # loss
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        #train_edge里面有些节点之间是有连边的，这些节点相邻，属于正例，有些之间不存在连边，这个节点不相邻，就是负例. 
        #这个训练就会把相邻的节点训练得更加相似, 所以就可以判断. 
```

