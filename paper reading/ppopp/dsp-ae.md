# This is the experiment code of DSP for the AE process of PPoPP23
In this document, we provide guidelines to reproduce the main results (Table 4 and 6) in DSP. The experiments are run with 8 V100 GPUs connected with NVLinks. In the docker, the environment of DSP and baselines are settled. Users only need to download datasets and run scripts to replay the results.

## Prepare docker environment
1. Install docker on a GPU server and add NVIDIA Runtime for the docker (User guide: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/user-guide.html).

2. Download the docker from our docker hub repository using `docker pull zhouqihui/dsp-ppopp-ae:latest`

3. Run the docker with 

   ```
   docker run --rm -it --runtime=nvidia --ipc=host --network=host -e zhouqihui/dsp-ppopp-ae:latest /bin/bash
   ```


## Prepare datasets
To minimize the docker size, we provide scripts to download datasets used by DSP and baselines.

### DSP datasets
We uploaded the partitioned datasets of Products, Papers100M and Friendster to S3. The following command is used for downloading Products.
```
cd /root/projects/DSP_AE
bash dsp/download_prods.sh
```
The downloading could take long because it covers all the partition settings (1, 2, 4, 8 GPUs) of the three datasets.
### Baseline datasets

We use a script `preprocess.sh` to download all datasets and convert them into the formats required by DGL, PyG and Quiver. The processed datasets for three baseline systems are stored under "/data/dgl/", "/data/pyg/", and "/data/quiver/", respectively.

问题

1. pyg conda环境没有torch. 可能是叫quiver 环境
2. pyg的paper100M好像有问题, 会exit.

### Run sampling experiments

```
cd /root/projects/DSP_AE
bash sample.sh
```
This is a large script to reproduce Table 6, which generates results in `log/${sys}/sample`. Users can extract some of the commands to reproduce the result of a system in a specific setting.

```
# datadir=/data/ds/distdgl/${dataset}${n_gpus}/${dataset}.json
```

这一行不对, 找不到 .FileNotFoundError: [Errno 2] No such file or directory: '/data/ds/distdgl/ogb-product2/ogb-product.json' 数据存放的位置不对. 

paper100M  finish loading graph 之后要等很久. 

### Run end-to-end experiments
```
cd /root/projects/DSP_AE
bash train.sh
```
This is a large script to reproduce Table 4, which generates results in `log/${sys}/train`. Users can extract some of the commands to reproduce the result of a system in a specific setting.

