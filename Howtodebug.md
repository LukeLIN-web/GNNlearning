#### Python

1. 在cpu上 debug.  model不要放在gpu上.  
2. pdb step也挺麻烦, 不如print exit. 
3. vscode的报错可能比终端要少. 终端执行的报错比pytest少.  vscode可以用 debug直接输入某个数值查看. 
4. 出错原因:为什么?  然后往上一层函数排查. 
5. 大图这样太难debug了.  思想就是搞个小图, 不要搞这么大的图.  最短的代码复现. 都先看最小的, 小的对了再看大的. 
6. 做一些检查确定你的假设成立, 比如list确实sorted了.
7. 用assert, 不要肉眼看, 出错了再肉眼看. 让电脑来判断,  gdb的话不用在前面看, 就直接在出错的地方看. 
8. 可以写在  utils/ 和/test   自己写了test ,  可以用pytest 自动化测试.
9. 你要知道流程里哪些数据是关键, 想好要加哪些log, 小数据试试, 要知道每个预期是啥, log出来看看是否和预期一样, 不是就往上找为啥, quit来中途退出检查每一行代码. 
10. 如果你代码强依赖事件流, 比如连接到服务器维持心跳, 但是单步调试会卡住心跳, 就不能单步调试. 

```
import IPython
IPython.embed() #类似于vscode的debug.
import ipdb
ipdb.set_trace()可交互的pdb, 非常有用, 

https://github.com/OscarSavolainen/Quantization-Tutorials/blob/main/Resnet-FX-Graph-Mode-Quant/ipdb_hook.py
```



#### Cpp

1. Segmentation fault (core dumped) ,gdb python opt_conv_cuda.py 看看. 
2. vscode 参数，debug：online values .   在debug的时候能实时显式出来所在栈帧的状态.    还有 clangd: toggle inlay hints 

#### python调用cpp断点

可以attach进程  https://zhuanlan.zhihu.com/p/106640360

从 python 进入 c++ 后端的代码，你需要编译时候配置debug 信息，并切爆链接到 debug 版本 so 文件
c++后端 调试 ： CXXFLAGS <- -g -O0
cuda 设备代码调试： CUDA_NVCC_FLAGS <- -g;-G;-Xptxas;-dlcm=ca

注意gdb打断点的路径是你编译时候source 文件的绝对路径

