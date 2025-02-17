## Vit

Quasar-ViT: Hardware-Oriented Quantization-Aware Architecture Search for Vision Transformers

HeatViT: Hardware-Efficient Adaptive Token Pruning for Vision Transformers: 

 很容易录了HPCA, algorithm  publish 别的paper了.   FPGA是林雪的学生写的, 回国了. 动态remove patchs, sequence layer也可以remove,  可以predict, 训练了一个predicter.  优点就是prune 一层,  acc下降了, 不会传播. 没人知道 hardware怎么支持不同dimension的. 

Peeling the Onion: Hierarchical Reduction of Data Redundancy for Efficient Vision Transformer Training

一文读懂Stable Diffusion 论文原理+代码超详细解读 - 蓝色仙女的文章 - 知乎
https://zhuanlan.zhihu.com/p/640545463

Unet:  https://pic3.zhimg.com/v2-03cf776c6281ff727e157e6088dbb394_r.jpg

深入浅出完整解析Stable Diffusion XL（SDXL）核心基础知识 - Rocky Ding的文章 - 知乎 https://zhuanlan.zhihu.com/p/643420260

目录 https://www.zhihu.com/column/c_1646154470676168704

vae : https://pic3.zhimg.com/v2-a390d53cc59c0e76b0bbc86864f226ac_r.jpg

stable diffusion training, teacher是冻结的, 训练student  vae+ Unet + encoder, 两个loss 加起来. 辨别器会平衡到50% 输出达到真假难辨. 

在扩散模型中，如果定义了 3 个反向去噪的步骤（Step），**UNet 会在每个步骤中执行一次**。每一步都会将当前的带噪数据传递给 UNet，让其去噪并生成一个更接近最终输出的数据。因此，经过 3 个步骤，UNet 就会被调用 3 次。

#### sdxl turbo code

```
python scripts/txt2img.py --prompt "a professional photograph of an astronaut riding a horse" --ckpt /root/share/stablediffusion/checkpoints --config /root/share/stablediffusion/configs/stable-diffusion/v2-inference.yaml 

需要numpy 1.24 高了低了都不行.

turbo 不是ckpt, 但是sd 的pytorch 需要ckpt. 怎么办?   我写了一个试试.
ckpt被废弃了, 这两个仓库也被废弃了, 大家都在用diffusers.

https://github.com/diStyApps/Safe-and-Stable-Ckpt2Safetensors-Conversion-Tool-GUI 也不好用. 

https://github.com/Stability-AI/generative-models/blob/main/scripts/demo/turbo.py

可以用调试 unet_debugging = torch.load("unet.pth")
然后在debug里面看. 或者 print (unet.xxx.attention_head_dim)  https://huggingface.co/docs/diffusers/en/api/models/unet2d-cond 找source找到
https://github.com/huggingface/diffusers/blob/v0.30.3/src/diffusers/models/unets/unet_2d_condition.py#L71

debug也挺好的, 可以看所有东西, 不用写很多print语句. 

要么hack diffusers. 但是最大难点是代码量太大了, 不知道跳转到哪里去了,可以debug step in. 但是无关的代码太多也难研究. 
但是不要随意乱改lib,  可能以后出错了, 环境调不对就不知道哪里出错了.
我找到了optimal, 就pip install -e .就行了. 先debug找到用了哪个模型,根据名字找到 UNet2DConditionModel. 
scripts/demo/sampling.py 没有turbo, 可以显示图片. 

PYTHONPATH=. streamlit run scripts/demo/turbo.py 不显示图片, 为什么? 
要先点击load model
open_clip_pytorch_model.bin有10个G.  
OOM了. 
f16可以用吗? 怎么用? 
f16也OOM.
diffusers似乎默认用Lora 就可以. 
diffusers/src/diffusers/models/unets/unet_2d_condition.py
```

compile,太慢了. 用了7分49s. 第二次就是2分20s. 

为了防止量化引起的任何数值问题，我们以 bfloat16 格式运行所有内容。

https://pytorch.org/blog/accelerating-generative-ai-3/

torchao,FP8 precision must be used on devices with NVIDIA H100 and above

[diffusion-fast](https://github.com/huggingface/diffusion-fast) 

https://huggingface.co/blog/simple_sdxl_optimizations

fuse所有activation.

```
f16, Peak GPU memory usage: 7.47GB , time 6.42s一张图.  30 step,  4.62s. 

pipe.unet.set_default_attn_processor()
pipe.vae.set_default_attn_processor()  
没有区别, 为啥?
Time taken: 7.13 seconds 慢了 7.13-6.42=0.71 , 慢了9.96%.  sdpa可以快10%
Time taken: 4.37 seconds , 快了4.62-4.37=0.25,快了 5.72%

Peak GPU memory usage: 7.49 
```

Turbo, total params memory size = 6558.89MB (VRAM 6558.89MB, RAM 0.00MB): clip 1564.36MB(VRAM), unet 4900.07MB(VRAM), vae 94.47MB(VRAM), controlnet 0.00MB(VRAM), 

turbo就是 https://github.com/Stability-AI/generative-models 

https://learn.microsoft.com/en-us/windows/ai/directml/dml-fused-activations

## diffusion 基础

https://lilianweng.github.io/posts/2021-07-11-diffusion-models/

X0->x1->x2 -> z

#### Forward diffusion process

每次加一点高斯噪声. 

#### reverse diffusion

用Unet, 每个step

## Unet

跳过路径直接将丰富且相对更多的低级信息从 Di 转发到 Ui 。在 U-Net 架构中的前向传播期间，数据同时通过两条路径遍历：主分支和 skip 分支. 

skip分支代码在哪里?`hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)`

视野范围越来越大, 但是看不到小的了.  两个数之间 距离其实越来越大. 只能看到最big picture, 看不到细节.  所以需要大层面的resnet. 用concat. 

attention和 spatial transformer 差异是啥?  

## pruning

需要大规模数据集来重新训练这些轻量级模型.

#### SnapFusion

Text-to-Image Diffusion Model on Mobile Devices within Two Seconds nips 23

没有公开代码. 仓库只有图. 

prune和 NAS, 需要微调. 

fig2 说 UNet 中间部分 参数多, , the slowest parts of UNet are the input and output stages with the largest feature resolution, as spatial cross-attentions have quadratic computation complexity with respect to feature size (tokens).

 step distillation是怎么做的? 

input and output stages 指的是哪个stage?具有最大特征分辨率 是哪个? 

通过评估单个残差块和注意力块的重要性获得高效的 UNet 架构. 怎么评估? 

3.1 efficient Unet

跳过一部分 crossattention, resnet, 是or 的关系还是两个随机独立?

algo1

```
while not conver:
    perform robust training
    if f perform architecture evolving at this iteration:
        perform architecture evolving
        for eachblock[i, j]:
            delta_clip = evaluate_block(去掉i, j, )
            delta_latency = evaluate_block(i, j)
        if latency > latency_threshold:
            A = argmin(delta_clip/delta_latency)
        else:
            A = argmin(delta_clip/delta_latency)
这个算法不讲人话,非常难懂. 
```

3.2 image decoder

applying 50% uniform channel pruning to the original image decoder

#### 4 Step Distillation

distilling the teacher, *e.g.*, at 32 steps, to a student that runs at fewer steps, *e.g.*, 16 steps

step就是循环Unet 32次. 

不逐步进行,  因为凭经验观察到渐进式蒸馏比直接蒸馏略差. 32蒸馏到16, 16蒸馏到8 step.

16node*8个 40G A100GPU.

## stable video diffusion

量化可以节省显存, 省GPU, pruning可以吗? 可以 

34w下载, stable-video-diffusion-img2vid-xt, 9.56gb, fp16是 4.2 GB. 但是还是跑不起来. 不支持bf16.

onediff的 int8 运行不起来, 没人管. 

A100支持bf16.

加速 cogvideox 智普AI和清华的. cogvideox 有8w下载, 非常可以. 

```
   Function                                  Runtimes (s)
--------------------------------------  --------------
_compile                                      146.867
OutputGraph.call_user_compiler                132.066
create_aot_dispatcher_function                132.273
compile_fx.<locals>.fw_compiler_base          113.608
```

compile的时间非常久. 
