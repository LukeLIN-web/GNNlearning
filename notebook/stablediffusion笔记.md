turbo的model structure啊，至少到每个Layer的级别，所谓layer，不是只有conv层就完了，至少也得知道是3  3的conv？有没有stride?什么的

一文读懂Stable Diffusion 论文原理+代码超详细解读 - 蓝色仙女的文章 - 知乎
https://zhuanlan.zhihu.com/p/640545463

Unet:  https://pic3.zhimg.com/v2-03cf776c6281ff727e157e6088dbb394_r.jpg

深入浅出完整解析Stable Diffusion XL（SDXL）核心基础知识 - Rocky Ding的文章 - 知乎 https://zhuanlan.zhihu.com/p/643420260

目录 https://www.zhihu.com/column/c_1646154470676168704

vae : https://pic3.zhimg.com/v2-a390d53cc59c0e76b0bbc86864f226ac_r.jpg



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





turbo就是 https://github.com/Stability-AI/generative-models 

