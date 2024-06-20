#### dockerfile

dockerfile, 非常好用.可以看看经典仓库的dockerfile. 以后建环境的时候把过程记在dockerfile上, 因为dockfile 文件小不用下载很久. 

#### wsl 安装cuda

wsl2只是虚拟机底层，还要去应用商店下载ubuntu系列。安装完wsl的ubuntu 后，再去docker-desktop中设置，把引擎改为ubuntu。 wsl2就是一个虚拟机，但能获取硬件资源。 没有wsl2的docker是用windows的docker引擎，和nv那些基于linux的原版docker镜像不兼容的。



#### pip

几个常用的参数: 

```cmd
-U, --upgrade 
--no-index # 可以禁用pypi 
-e, --editable <path/url> #Install a project in editable mode 
```

https://packaging.python.org/en/latest/guides/distributing-packages-using-setuptools/#working-in-development-mode  怎么可编辑模式

把egg info删除了, pip list就找不到了.  pip安装的包，会以egg或者wheel文件的放在site-package里，egg本质是个zip包，内部的C++会编译成.so文件

#### Conda

conda list 不能识别pip安装的软件.  

mamba会快很多. 

https://carpentries-incubator.github.io/introduction-to-conda-for-data-scientists/  conda 从入门到入土. 

#### 安装PyG

https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pyg 官方容器

https://data.pyg.org/whl/ 可以找到对应的. 

```
python -c "import torch; print(torch.version.cuda)"
python -c "import torch; print(torch.__version__)"
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.12.0+cu113.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-1.12.1+cu113.html
pip install torch-geometric
```

卸载sparse重装. 参考 https://github.com/rusty1s/pytorch_scatter/issues/248 要卸载两次. 就是不要直接安装, 手动下载比较稳健.

#### quiver

问题

1. torch   1.12.0 , torch-geometri 2.1.0.post1,  subprocess.CalledProcessError: Command '['which', 'g++']' returned non-zero exit status 1.

换一个docker . 用官方dockerfile来安装. 

安装成功, 但是所有的`quiver_sampler = quiver.pyg.GraphSageSampler(csr_topo, sizes=[25, 10],device=0, mode='GPU')  # Quiver` 都会hang住.  因为 V100 编译的二进制机器码不能在A100上跑.  用nvcc编译CUDA需要指定架构号. 所以还是要用和别人一模一样的设备.  GPU的架构 -> 驱动 -> CUDA tool kit -> nvcc -> pytorch 

#### Nvidia driver

```shell
cat /proc/driver/nvidia/version
sudo dmesg 查看是否lib和kernel不匹配
cat /sys/module/nvidia/version
dpkg -l nvidia-*|grep ^ii ii就是安装了的
apt list | grep nvidia-container-toolkit # 查看是否安装toolkit, 有[installed]就是安装了.
```

用` sudo apt-get --purge remove "*nvidia*"`删除了所有, 

直接安装 nvidia-container-toolkit, 会把driver 自动装了. https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html 

https://askubuntu.com/questions/1436506/how-to-resolve-unmet-dependencies-error-when-upgrading-depends-nvidia-kernel-c   有这个问题. 根据这个网址的第一个回答可以显示nvidia-smi , 但是不能成功docker . 多重启几次. 

nvidia-smi 报错：Failed to initialize NVML: Driver/library version mismatch - endRuz的文章 - 知乎 https://zhuanlan.zhihu.com/p/443208000  

#### DGL

安装老的dgl

```
pip install dgl==0.9.1 默认是 cpu版本的. 
```

源码编译0.6.1

Scipy 1.11 太高了, 1.7.0差不多. Apr 8, 2021

自己安装scipy1.7

dgl 0.7需要 python3.8

### 源码安装pytorch

编译https://github.com/K-Wu/pytorch-direct

用torch 1.10.0-cuda11.3-cudnn8的镜像, 删除torch 重装. 参考https://malloc-42.github.io/intro/2021/07/25/Installing-PyTorch/

```bash
conda install numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses

export $LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
确认path有/usr/local/cuda/bin
```

