

有了GPT 之后, 想系统学习也许只需要一个目录,  不断问GPT 细节就行. 



## huggingface NLP course

### 第二章

2-2 

https://huggingface.co/learn/nlp-course/chapter2/2?fw=pt

pad是因为不同的batch 不一样长. attention mask也会设为0 表示这些是padding 的.

输出的logits不是概率,  softmax 之后变成概率 

head 是什么?   The model heads take the high-dimensional vector of hidden states as input and project them onto logits.  输出dim是vocab size 就是几个linear.  可以使用相同的架构执行不同的任务，但每个任务都将具有不同的 head 与之关联。  Transformer 模型的输出直接发送到模型头进行处理.  看图  https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter2/transformer_and_head.svg  

所有 🤗 Transformers 模型都输出 logits，因为用于训练的损失函数通常会将最后一个激活函数（如 SoftMax）与实际的损失函数（如交叉熵）融合

quiz: 

- loss是怎么算的?

- logits是什么?


训练自回归模型还是应该用forward而不是generate.

2-3 models

config load的时候可以直接改config. 

2-4 tokenizers 

todo : 用到的时候仔细看看

2-5 Handling multiple sequences

padding必须和 attention mask组合使用不然就出错. 

2-6tokenizers 库

非常强大, 

```
model_inputs = tokenizer(sequences, padding=True, return_tensors="np")# 可以返回torch, TensorFlow和numpy数组. 

model_inputs = tokenizer(sequence) 分词器在开头添加了特殊单词 [CLS]，在结尾添加了特殊单词 [SEP]。这是因为模型是用这些进行预训练的，因此为了获得相同的推理结果需要添加. 
print(model_inputs["input_ids"])

tokens = tokenizer.tokenize(sequence) # 和直接调用不一样!没有CLS了. 
ids = tokenizer.convert_tokens_to_ids(tokens) #就可以看数字了.
print(tokenizer.decode(model_inputs["input_ids"])) #就可以看text了. 用的是同一个tokenizer


特殊token:
llama2等模型的中<s>是 BOS (beginning of a sentence) token
[CLS] 代表分类。之所以在开头添加，是因为这里的训练任务是句子分类。由于他们需要可以表示整个句子含义的输入，因此他们引入了一个新标签。
0 <unk>,  1 <s>, 2 </s>, 
有的会插入 im_start 
tokenizer.special_tokens_map 看
```

3-2 

```
raw_datasets = load_dataset("glue", "mrpc") 
# todo , glue是啥意思? 没找到这个库

raw_train_dataset = raw_datasets["train"]
raw_train_dataset[0] #看长啥样

raw_train_dataset.features  # 这将告诉我们每列的类型：
label 的类型为 ClassLabel，整数到标签 name 的映射存储在 names 文件夹中

tokenizer还可以用逗号传入两个, 返回token type ids来表明是第一个还是第二个.(请注意，如果您选择其他检查点，则标记化输入中不一定包含token_type_ids（例如，如果您使用 DistilBERT 模型，则不会返回它们）。只有当模型知道如何处理它们时，才会返回它们，因为它在预训练期间已经看到了它们。) 还可以传入多个序列.


为了将数据保存为数据集，我们将使用 Dataset.map（） 方法。如果我们需要完成更多的预处理而不仅仅是 tokenization，这也为我们提供了一些额外的灵活性。

def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)
这将允许我们在调用 map（） 时使用选项 batched=True，这将大大加快分词速度。分词器由 Tokenizers 库中用 Rust 编写的🤗分词器提供支持。这个分词器可以非常快，但前提是我们一次给它很多输入。请注意，我们现在在 tokenization 函数中省略了 padding 参数。这是因为将所有样本填充到最大长度效率不高：最好在构建批次时填充样本，因为这样我们只需要填充该批次中的最大长度，而不是整个数据集中的最大长度。当输入的长度非常可变时，这可以节省大量时间和处理能力！

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)


tok("dog walked", add_special_tokens=True)) 的讲解  https://github.com/huggingface/transformers/issues/22935
当设置为 True 时，add_special_tokens用于在输入序列的开头和结尾添加特殊标记。在您的情况下，由于您使用的是单个输入序列，因此分词器将分别在句子的开头和结尾添加特殊标记 [CLS] 和 [SEP]。
请注意，并非所有分词器都支持添加特殊分词。如果分词器不支持添加特殊标记，则将 add_special_tokens 设置为 True 将不起作用。


# todo
继续阅读后面内容, Dynamic padding  
```

3-3 trainer

todo

```
Trainer 里面有一个方法叫 compute_loss：

默认实现是调用 model(**inputs) 得到 outputs，然后从 outputs 里取 loss。
你可以自定义 Trainer，重写 compute_loss，比如直接调用自定义的 loss 计算函数

def compute_loss(self, model, inputs, return_outputs=False):
    outputs = model(**inputs)  # 这里其实就是 forward
    loss = outputs["loss"]     # 或者你可以自定义怎么取 loss
    return (loss, outputs) if return_outputs else loss
```

输出太多了, 但是 `disable_tqdm=True` 会导致 `Trainer` 不触发 `on_log`，从而 wandb 不记录任何日志。  所以还需要手动log

有trainer, wandb 在shell里面初始化最方便, 不用修改任何python代码 .没有trainer就要加很多wandb 的代码.

```
export WANDB_PROJECT=
export WANDB_ENTITY=
 trainer 会自动读取这两个. 
```







 

3-4 A full training 

todo

```

trainer.save_state（） 是一种负责保存训练循环本身的当前状态的方法，而不仅仅是模型权重。这对于能够从上次中断的地方恢复训练至关重要。


优化器状态： 保存优化器的内部状态（例如，Adam/AdamW 的动量缓冲区、运行平均值等）。如果您在没有此功能的情况下重新开始训练，优化器将从头开始，这可能会导致不同的收敛行为。通常保存为 optimizer.pt。
```

### 第七

7-3

您还可以检查 `train` 和 `test` splits 中的标签是否确实是 `0` 或 `1` — 这是一个有用的健全性检查，每个 NLP 从业者都应该在新项目开始时执行

对于自回归和掩码语言建模，一个常见的预处理步骤是连接所有示例，然后将整个语料库拆分为大小相等的块。这与我们通常的方法完全不同，在通常的方法中，我们只是简单地对单个示例进行标记。为什么要把所有东西都连接在一起呢？原因是如果单个示例太长，它们可能会被截断，这会导致丢失可能对语言建模任务有用的信息.  所以不在分词器中设置 `truncation=True`

We’ll also grab the word IDs if they are available ((which they will be if we’re using a fast tokenizer, as described in [Chapter 6](https://huggingface.co/course/chapter6/3)), as we will need them later on to do whole word masking.   啥意思? 

```python
def tokenize_function(examples):
    result = tokenizer(examples["text"])
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))] # word id就是除去特殊字符的每个token的顺序. 
    return result

# Use batched=True to activate fast multithreading!
tokenized_datasets = imdb_dataset.map(
    tokenize_function, batched=True, remove_columns=["text", "label"]
)

为什么要在每句话加上[SEP] 和[CLS]?  因为是多个examples放在一个chunk里了
return_overflowing_tokens是怎么用的? 
```

现在我们已经对电影评论进行了标记化，下一步是将它们全部分组在一起并将结果拆分为块。但是这些块应该有多大呢？这最终将由您可用的 GPU 内存量决定，但一个好的起点是查看模型的最大上下文大小是多少。`tokenizer.model_max_length`  

在实际场景中使用较小的数据块大小可能是有害的，因此您应该使用与您将应用模型的用例相对应的大小。

有的也有点老了, 都还在用p100 和bert.



7-6 Training a causal language model from scratch

mask 是什么? 这里没有讲这些细节.  都用 DataCollatorForLanguageModeling 包装了, `data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)` 就是因果建模

清洗数据: 
对更多使用这些库的训练样本给予更多权重是有意义的。我们可以通过使用 plt、pd、sk、fit 和 predict 等关键字轻松识别这些示例，这些关键字是 matplotlib.pyplot、pandas 和 sklearn 最常见的导入名称，以及后者的 fit/predict 模式。如果它们都表示为单个 token，我们可以轻松检查它们是否出现在 input 序列中。标记可以具有空格前缀，因此我们还将在 tokenizer 词汇表中检查这些版本。

casual llm  forward怎么写?

底层原理

```
    shift_labels = inputs[..., 1:].contiguous() 
我们需要对齐 logits 和 inputs：向右移动 1 的 input 序列形成标签，因为下一个 token 是当前 token 的标签。我们可以通过从输入序列的第二个标记开始标记来实现这一点，因为模型无论如何都不会对第一个标记进行预测。然后我们切断最后一个 logit，因为我们没有遵循完整 input 序列的 token 的标签。这样，我们就可以计算每个样本的损失.
```

input id和labels是对齐的, 在causal LM中会自动偏移算loss.  所以我们不用写上面的操作. 



#### generate

大部分模型 继承了PreTrainedModel , 继承了GenerationMixin, 调用generate, 就可以把self 自己传进去, 这里是调用了 WhisperForConditionalGeneration 中的forward函数。这是因为 PyTorch 的 nn.Module 基类定义了一个 __call__ 方法，当你调用模型实例（即 self）时，它会自动调用这个 __call__ 方法，而这个 __call__ 方法又会调用 forward 方法。

https://techdiylife.github.io/blog/blog.html?category1=c02&blogid=0005

#### 生成的输出

https://huggingface.co/docs/transformers/en/internal/generation_utils

如果模型尚未返回该属性，您将得到 `None`, 不会报错. 

`logits` 是 **模型的原始输出**，未经修改。

`scores` 是 **用于生成 token 的处理后的分数**，可能已经过温度缩放或其他处理。

在标准 `greedy` 或 `beam search` 生成时，`scores` 和 `logits` 可能相同；但在 **Top-k / Top-p / 温度采样** 等情况下，`scores` 可能与 `logits` 有显著区别.

 When you set `output_hidden_states=True` and `return_dict_in_generate=True`, the `language_model_output.hidden_states` will be a tuple of tuples containing the hidden states for each generation step.  就是输出8个token, 会输出一个 ( 8 输出token数,33 层数, hidden size) 这样一个 Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of `torch.FloatTensor` of shape `(batch_size, generated_length, hidden_size)`.  

是不是就算shape不对, 但是

Hidden states怎么append的？

要看看原始代码, 

输入 inputs_embeds 而不是 input_ids,    outputs.sequences 不包含输入的 input_ids.

`tokenizer.add_special_tokens(...)` 会把 `<ACT>` 加入 vocab

**但 model 的 embedding 层大小不会自动更新**

HuggingFace 的 `resize_token_embeddings()` 默认行为只处理 **input embedding**。
 而 **output embedding** 可能是：

1. 被 `tie_weights()` 共享，或者
2. 是一个单独的 `Linear` 层 所以需要手动平均

````
    model.resize_token_embeddings(len(tokenizer))
    output_embeddings = model.language_model.get_output_embeddings().weight.data
    output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
    output_embeddings[-num_new_tokens:] = output_embeddings_avg
````



## attentionmask 问题

可以看看   torchtune的文档. 

https://pytorch.org/torchtune/stable/_modules/torchtune/generation/_generation.html#get_causal_mask_from_padding_mask

推理的时候需要 attention_mask 吗? 

**如果 inputs_embeds 是 padding 后的变长输入**

- LLaMA 2 默认不处理 `padding`，所以 **需要提供 attention_mask，确保模型忽略填充部分**。

**多模态输入场景（如 CLIP + LLaMA）**

- 你可能需要 `attention_mask`，以区分 **文本部分** 和 **多模态部分**（如图像 embeddings

手动控制模型注意力范围,  如果你想让模型关注 **特定 token**（如只对文本部分做注意力计算），可以手动提供 `attention_mask`。 **LLaMA 2 是 decoder-only Transformer**，默认使用 ，即： 因果（causal）masking , 自动屏蔽未来 tokens，确保模型只能看到过去的信息。 这种 mask 在 generate() 过程中是 隐式添加的，不需要额外传 attention_mask。 源码会自动屏蔽 pad id  如果你直接调用 `self.language_model()`（即不使用 `.generate()`，而是直接前向传播 `forward` 方法），**通常需要提供 attention_mask**，具体情况如下.

对于掩码语言模型（masked language model，如BERT），self.language_model确实可以一次预测所有被掩码的token。这是掩码语言模型与自回归语言模型的一个重要区别。

直接使用self.language_model不能原生地一次输出8个token.   需要编写额外的循环和控制逻辑。 .  self.language_model通常是前向传播的核心实现，每次调用只会根据输入上下文预测下一个token的概率分布.

自回归和掩码语言模型区别?

一次预测所有被掩码的token 是做几个prefill 几次decode? 

主要是计算快 双向是真的耗资源. 为什么 自回归模型训练更为高效. masked model不也是一样的token数量吗? 

哦他训练会更慢,因为要算所有token 的loss. ? 

masked model , bert是15%，只有15%的token参与计算loss.

causal model的训练label怎么设置.

# Lora

```
    vla_lora = PeftModel.from_pretrained(base_vla, model_id=cfg.pretrained_checkpoint, subfolder="lora_adapter") 
    vla_lora = vla_lora.merge_and_unload() 
        vla_lora = vla_lora.merge_and_unload() #eval 需要merge. 这样可以读取  config, 不merge不能读取pre ckpt的config. model.config还是 basevla的. 
    
    
# 和下面是不一样的 
    # from peft import get_peft_model, LoraConfig, TaskType
    # lora_rank = 32
    # lora_config = LoraConfig(
    #             r=lora_rank,
    #             # lora_alpha=min(cfg.lora_rank, 16),
    #             lora_alpha=16,  # Xuan: usually, lora_alpha = 2 * lora_rank
    #             lora_dropout=0.0,
    #             target_modules="all-linear",
    #             init_lora_weights="gaussian",
    #         )
    # vla_lora = get_peft_model(base_vla, lora_config)
    # vla_lora.load_adapter(adapter_dir, adapter_name="default") # juyi: 会 assert unnorm_key in norm_stats, 报错.

```

https://huggingface.co/docs/peft/tutorial/peft_model_config

`mlp.up_proj.lora_A.default.weight`   

`merge_and_unload()`之后我们就不是lora模型了, 就是普通模型了. 



