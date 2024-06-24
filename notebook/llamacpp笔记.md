

## 安装

`make -j` 一分钟。 

models里面没有模型， 怎么拿？  就是hugging face 下载， 然后转换模型。 

```
python convert-hf-to-gguf.py Path_To_Qwen

./llama-cli --hf-repo huggingface的模型 -GGUF --hf-file 对应的.gguf -p "The meaning to life and the universe is"
```





#### llama

重点的代码在llama文件夹下面的generation.py和model.py。

ColumnParallelLinear

apply_rotary_emb 是啥 

```
       values = values.transpose(1, 2) # (bs, n_local_heads, cache_len + seqlen, head_dim)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        # values会越来越大？  怎么越来越大呢？ 是拿之前所有的还是新加的？ 
```



kvcache的想法再看看。 

就是把之前的存起来。 

llama-server和 llama-cli什么区别? 

```
llama_download_file: no previous model file found /Users/linj/Library/Caches/llama.cpp/llama2_7b_chat_uncensored-q4_0.gguf
llama_download_file: curl_easy_perform() failed: Timeout was reached
llama_init_from_gpt_params: error: failed to load model '/Users/linj/Library/Caches/llama.cpp/llama2_7b_chat_uncensored-q4_0.gguf'
main: error: unable to load model
```







