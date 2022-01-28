CS224W学习

输入一个图, 每个node 有一个feature vector, 输出一个embedding, 也就是一个对下游任务有用的向量. 

colab0 

2022年1月28日开始学习, 安装环境.

networkx安装失败 ValueError: Check_hostname Requires Server_HostName.

尝试关闭代理服务器. 成功

果然自己跑一次很多细节才会注意到. 

conda 不行, 不懂了, 加了环境变量还是不行.  绝了, pycharm里面不行, 外面用终端就可以. 

`EnvironmentNotWritableError: The current user does not have write permissions to the target environment.`

忽然意识到不应该直接写入, 而是应该虚拟环境

累了, 感觉GUI和命令行混合太复杂了,  还是继续用GUI, 

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



