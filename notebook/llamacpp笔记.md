

## 安装

`make -j` 一分钟。 





llama.cpp 为什么可以下载llama模型？ 这个模型不是写死的吗？ 

models里面没有模型， 怎么拿？ 

就是hugging face 下载， 然后转换模型。 

```
python convert-hf-to-gguf.py Path_To_Qwen
```





#### llama

重点的代码在llama文件夹下面的generation.py和model.py。

ColumnParallelLinear

apply_rotary_emb 是啥 



model.

```
       values = values.transpose(1, 2) # (bs, n_local_heads, cache_len + seqlen, head_dim)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        # values会越来越大？  怎么越来越大呢？ 是拿之前所有的还是新加的？ 
```



kvcache的想法再看看。 

就是把之前的存起来。 

停车200 一个月  在学校， 保险 25岁， 一年4500美元. 麻省查一下沙特驾照能不能直接用。  驾照快的话一个月。 买车。