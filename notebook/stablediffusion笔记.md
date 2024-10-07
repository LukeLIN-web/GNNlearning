turbo的model structure啊，至少到每个Layer的级别，所谓layer，不是只有conv层就完了，至少也得知道是3  3的conv？有没有stride?什么的

一文读懂Stable Diffusion 论文原理+代码超详细解读 - 蓝色仙女的文章 - 知乎
https://zhuanlan.zhihu.com/p/640545463

Unet:  https://pic3.zhimg.com/v2-03cf776c6281ff727e157e6088dbb394_r.jpg

深入浅出完整解析Stable Diffusion XL（SDXL）核心基础知识 - Rocky Ding的文章 - 知乎 https://zhuanlan.zhihu.com/p/643420260

目录 https://www.zhihu.com/column/c_1646154470676168704

vae : https://pic3.zhimg.com/v2-a390d53cc59c0e76b0bbc86864f226ac_r.jpg

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





## Unet



跳过路径直接将丰富且相对更多的低级信息从 Di 转发到 Ui 。在 U-Net 架构中的前向传播期间，数据同时通过两条路径遍历：主分支和 skip 分支. 



skip分支代码在哪里?



#### ResnetBlock2D









#### cross attention

Q is projected from noisy data zt, K and V are projected from text condition













## prune



#### 挑战

challenge stems from the step-by-step denoising process required during their reverse phase, limiting parallel decoding capabilities 

方法:  减少采样步骤的数量, 通过模型修剪、蒸馏和量化等方法减少每步的模型推理开销.

需要大规模数据集来重新训练这些轻量级模型.









DeepCache: Accelerating Diffusion Models for Free. CVPR'24

观察到连续步骤之间高级特征的显着时间一致性。我们发现这些高级特征甚至可以缓存，可以计算一次，然后再次检索以进行后续步骤。通过利用 U-Net 的结构特性，可以缓存高级特征，同时保持在每个降噪步骤中更新的低级特征。

是怎么缓存的? 

作者改进出了, https://github.com/VainF/Diff-Pruning



SnapFusion: Text-to-Image Diffusion Model on Mobile Devices within Two Seconds nips 23

fig2 说 UNet 中间部分 参数多, , the slowest parts of UNet are the input and output stages with the largest feature resolution, as spatial cross-attentions have quadratic computation complexity with respect to feature size (tokens).







