# 数据集

数据集存在 bucket .

也可以直接用rsync 传. 

网页的数据这样下载

```
sudo wget --recursive --no-parent --cut-dirs=4 --no-host-directories --reject "index*,web*" https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/

mv bridge_dataset bridge_orig
A100, 网速60MB/s, 需要35分钟. 
```

hg的数据这样下载

```
git lfs install
git clone https://huggingface.co/datasets/lerobot-raw/fractal20220817_data_raw
```

inference是和train 不是一个pool.

training有单独的pool. 

https://cloud.google.com/compute/docs/instance-groups/create-resize-requests-mig



**flex-start** 

最长一周.

也可以delete GPU.

training pool 便宜 40% . 

GKEhttps://cloud.google.com/kubernetes-engine/docs/concepts/dws

 flex-start在哪里呢? 没找到. 

It must have kubernetes?

shoudl we need queued provisioning?

好像必须用命令行来启动. 

https://sfcompute.com/

非常便宜



谷歌云, 每个region 都必须申请quota, 啥权限都没有, 最弱智的云. 





# 申请VM 

一开始换了region都申请不到一个GPU, 因为 gcp个人账号有quota要人工audit.  点 request quota adjustment, 就会被拒绝.. 得先去要quota批了才能用. 过两天就批准了.  但是还是只有一个A100,  不能多个GPU,  又申请一次. 

一个A100 40GB , 默认硬盘只有10GB 磁盘, 一个7.8GB的image都不够.  

操作系统这里可以选版本, 搜索"cuda",  默认安装cuda版本.

- `highmem` - 每个 vCPU 7 到 14 GB 内存；通常每个 vCPU 8 GB 内存。
- `megamem` - 每个 vCPU 14 到 19 GB 内存 mega内存大一点. 

很多地方申请不到, 需要换地区申请, 





# vscode连接

安装插件 cloud code . 就可以直接终端连接.

ssh好像还是要添加 key.

# 配环境

用docker, 写一个docker安装脚本, 官网是ubuntu, 可能要用debian  https://docs.docker.com/engine/install/ubuntu/

```
#!/bin/bash
# Add Docker's official GPG key:
sudo apt-get update
sudo apt-get install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/debian \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update

sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

安装 NVIDIA Container Toolkit

https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installing-the-nvidia-container-toolkit

```
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
sudo docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
```



创建docker imag. 

上传到docker hub, pull下来. 

写个dockerrun.sh,  

数据集太大了所以要mount. 不变的东西都在外面改, 因为docker 里面是root权限, docker里面改了外面就很难改. 

代码用git clone.

```
#!/bin/bash

docker run --rm -it \
  --name effvla-run \
  --gpus all \
  -v /home/linj/:/root/workspace/ \
  -v /data/:/data/ \
  1263810658/effvla:0.1 \
  bash
```



#### 机器映像

搞好了环境可以创建机器映像. 机器映像包含虚拟机的属性、元数据、权限及其所有挂接磁盘中的数据。您可以使用机器映像创建、备份或恢复虚拟机 

