## Vit

Quasar-ViT: Hardware-Oriented Quantization-Aware Architecture Search for Vision Transformers

HeatViT: Hardware-Efficient Adaptive Token Pruning for Vision Transformers: 

 很容易录了HPCA, algorithm  publish 别的paper了.   FPGA是林雪的学生写的, 回国了. 动态remove patchs, sequence layer也可以remove,  可以predict, 训练了一个predicter.  优点就是prune 一层,  acc下降了, 不会传播.   为什么? 问问arman .  没人知道 hardware怎么支持不同dimension的. 

Peeling the Onion: Hierarchical Reduction of Data Redundancy for Efficient Vision Transformer Training

一文读懂Stable Diffusion 论文原理+代码超详细解读 - 蓝色仙女的文章 - 知乎
https://zhuanlan.zhihu.com/p/640545463

Unet:  https://pic3.zhimg.com/v2-03cf776c6281ff727e157e6088dbb394_r.jpg

深入浅出完整解析Stable Diffusion XL（SDXL）核心基础知识 - Rocky Ding的文章 - 知乎 https://zhuanlan.zhihu.com/p/643420260

目录 https://www.zhihu.com/column/c_1646154470676168704

vae : https://pic3.zhimg.com/v2-a390d53cc59c0e76b0bbc86864f226ac_r.jpg



stable diffusion training, teacher是冻结的, 训练student这个vae+ Unet + encoder, 两个loss 加起来. 辨别器会平衡到50% 输出达到真假难辨. 

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

vscode debug失败. attempted relative import with no known parent package
```

turbo就是 https://github.com/Stability-AI/generative-models 

## diffusion 基础

https://lilianweng.github.io/posts/2021-07-11-diffusion-models/



X0->x1->x2 -> z

#### Forward diffusion process

每次加一点高斯噪声. 



#### reverse diffusion

用Unet, 每次



## Unet

跳过路径直接将丰富且相对更多的低级信息从 Di 转发到 Ui 。在 U-Net 架构中的前向传播期间，数据同时通过两条路径遍历：主分支和 skip 分支. 



skip分支代码在哪里?`hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)`

视野范围越来越大, 但是看不到小的了.  两个数之间 距离其实越来越大. 只能看到最big picture, 看不到细节.  所以需要大层面的resnet. 用concat. 

attention和 spatial transformer 差异是啥?  



#### ResnetBlock2D





#### cross attention

Q is projected from noisy data zt, K and V are projected from text condition













## pruning

#### 挑战

challenge stems from the step-by-step denoising process required during their reverse phase, limiting parallel decoding capabilities 

方法:  减少采样步骤的数量, 通过模型修剪、蒸馏和量化等方法减少每步的模型推理开销.

需要大规模数据集来重新训练这些轻量级模型.

#### DeepCache

: Accelerating Diffusion Models for Free. CVPR'24

观察到连续步骤之间高级特征的显着时间一致性。我们发现这些高级特征甚至可以缓存，可以计算一次，然后再次检索以进行后续步骤。通过利用 U-Net 的结构特性，可以缓存高级特征，同时保持在每个降噪步骤中更新的低级特征。

是怎么缓存的? 

作者改进出了, https://github.com/VainF/Diff-Pruning

#### SnapFusion

: Text-to-Image Diffusion Model on Mobile Devices within Two Seconds nips 23

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

不要逐步进行, 凭经验观察到，渐进式蒸馏比直接蒸馏略差. 32蒸馏到16, 16蒸馏到8 step.

16node*8个 40G A100GPU.

#### structual pruning for for diffusion models.

https://arxiv.org/pdf/2305.10924

该术语 |𝜽′|0 表示参数的 L-0 范数，它计算非零行向量的数量，并 s 表示修剪模型的稀疏性。然而，由于扩散模型固有的迭代性质，训练目标（用 表示） ℒ 可以被视为相互关联的任务的组合 T ： {ℒ1,ℒ2,…,ℒT} 。每项任务都会影响并依赖于其他任务，从而带来了不同于传统修剪问题的新挑战，传统修剪问题主要集中在优化单个目标上。根据公式 [4](https://arxiv.org/html/2305.10924?_immersive_translate_auto_translate=1#S4.E4) 中定义的修剪目标，我们首先深入研究了每个损失分量 ℒt 在修剪中的单独贡献，随后提出了一种专为扩散模型修剪而设计的定制方法，即 Diff-Pruning。

https://arxiv.org/abs/2410.10812   Hybrid Autoregressive Transformer.  用传统的AR模型生成图像,  混合分词器，它将自动编码器的连续潜在因素分解为两个部分：代表大局的离散分词和代表离散分量无法表示的残余分量的连续分量。离散分量由可扩展分辨率的离散 AR 模型建模，而连续分量则使用只有 37M 参数的轻量级残差扩散模块进行学习. 

## stable video diffusion

量化可以节省显存, 省GPU, pruning可以吗? 

34w下载, stable-video-diffusion-img2vid-xt, 9.56gb, fp16是 4.2 GB. 但是还是跑不起来. 不支持bf16.

A100支持bf16.

也可以看看加速 cogvideox 智普AI和清华的. cogvideox 有8w下载, 非常可以. 

LanguageBind/Open-Sora-Plan-v1.3.0, 0下载. 没啥人用.  不管他.

```
torch._dynamo.exc.Unsupported: call_method NNModuleVariable() to [ConstantVariable(str)] {}

from user code:
   File "/usr/local/lib/python3.10/dist-packages/accelerate/hooks.py", line 717, in offload
    self.hook.init_hook(self.model)
  File "/usr/local/lib/python3.10/dist-packages/accelerate/hooks.py", line 696, in init_hook
    return module.to("cpu")

Set TORCH_LOGS="+dynamo" and TORCHDYNAMO_VERBOSE=1 for more information


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = Truez
   Function                                  Runtimes (s)
--------------------------------------  --------------
_compile                                      146.867
OutputGraph.call_user_compiler                132.066
create_aot_dispatcher_function                132.273
compile_fx.<locals>.fw_compiler_base          113.608
```

compile的时间非常久. 

为了防止量化引起的任何数值问题，我们以 bfloat16 格式运行所有内容。

https://pytorch.org/blog/accelerating-generative-ai-3/

torchao,FP8 precision must be used on devices with NVIDIA H100 and above

