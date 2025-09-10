

有了GPT 之后, 想系统学习也许只需要一个目录,  不断问GPT 细节就行. 



语言模型的数学原理

# physics of LM

https://physics.allen-zhu.com/ 

宣传一下 这本大作, 应该熟练背诵, 汇总一下结论. 

## part1

**SSRN** **paper**: https://ssrn.com/abstract=5250639 

包括说明为什么绝对位置嵌入不如相对嵌入和旋转嵌入

与自回归模型（例如 GPT）相比，仅编码器模型（例如 BERT、DeBERTa）在深度嵌套的 CFG 中遇到困难;将结构或语法噪声注入预训练数据可显着提高对损坏语言提示的鲁棒性。

## part2.1

**SSRN** **paper**: https://ssrn.com/abstract=5250629  (last update: July 2024)
(arxiv link is deprecated) https://arxiv.org/abs/2407.20311

gpt 2 真的会和人一样做数学! 

1. 在我们的例子中，模型**从未见过任何与测试时间长度相同的训练示例。** 这意味着模型可以真正学习一些推理技能，而不是死记硬背解模板。
2. 该模型可以学习生成最短的解决方案，几乎总是避免*不必要的*计算。
3. 我们发现模型在开始任何生成之前预处理了全套必要的参数。同样，人类也会做这个预处理，尽管我们把它写在便签本上。
4. 该模型在预训练后还学习了*不必要但重要的*技能，例如全对依赖性。在提出任何问题之前，它已经（在脑海中）很好地准确地计算出哪些参数依赖于哪些参数，即使有些参数*不是解决数学问题所必需*的。请注意，计算全对依赖关系并不是拟合训练数据中所有解所**必需**的技能。据我们所知，这是语言模型可以*学习有用的技能*的第一个证据，超出了拟合其预训练数据所需的技能。 这可能是 *AGI 中 G 可能来自哪里*的初步信号。
5. 我们解释*为什么会出现错误* 。例如，模型犯了系统错误，可以通过探测其内部状态来解释。有时，这些错误可以在模型生成答案之前进行预测，使其独立于随机生成过程。我们将此与实践联系起来，并指出 GPT-4/4o 也犯了类似的错误.
6. **语言模型的深度对于其推理能力至关重要**。例如， a 16-layer, 576-dim transformer solves harder problems (in reasoning length) than a 4-layer, 1920-dim one。即使使用思维链 （CoT），这也成立。我们主张使用受控的合成数据作为一种更有原则的方法来得出此类声明，这与基于使用互联网预训练数据 的训练损失的“只有大小重要”等预测形成鲜明对比.

## part 2.2 How to Learn From Mistakes 

**SSRN** **paper**: https://ssrn.com/abstract=5250631

https://arxiv.org/abs/2408.16293

- 了解将“纠错”数据直接纳入预训练阶段的有用性.与在相同数量的无错误数据上进行预训练相比，这种类型的**预训练数据**可以帮助语言模型直接实现更高的推理准确性（即通过简单的自回归，无需多轮提示）
- 使用重试数据非常安全：即使在使用高错误率重试数据进行预训练后，模型也**很少出错**，并且无需更改训练过程（简单地自回归，无需标记掩蔽错误）。重试数据教会模型如何在需要时纠正错误，而不是鼓励错误。
- 需要注意的是，这样的*纠错技巧*来之不易。仅使用无错误数据进行预训练的模型不能使用 (1) beam search or (2) retry based on error detection (“retry upon regret”)  来实现可比的性能.  纠错能力**难以在 LoRA** 等参数高效微调 （PEFT） 中学习. 有必要将重试数据添加到语言模型的**预训练**数据中，以真正学习纠正错误的能力。

## Part 3.1, Knowledge Storage and Extraction

**SSRN** **paper**: https://ssrn.com/abstract=5250633 (last updated July 2024)

https://arxiv.org/abs/2309.14316

- **从本质上讲** ，为了可靠地提取知识，必须*在预训练期间*对其进行充分的增强（例如，通过释义、句子打洗、翻译）。如果没有这样的增强，知识可能会被记忆但不可提取，导致准确率为 0%，无论后续指令微调如何。
- **本文为业界的 LLM 预训练提供了几个关键建议：（1） 使用小型辅助模型重写预训练数据以提供知识增强，以及 （2） 在为时已晚之前将更多的指令微调数据纳入预训练阶段。**
- 即使知识增强应用于一部分人，what we call celebrities, 其他人的测试准确性（没有增强）也会显着提高。 (celebrities是需要拥有大量数据还是随机subset 就可以? )
- *encoder-only models akin to BERT* , 无论是混合训练还是预训练然后微调，无论知识增强如何，都无法在微调后提取一个人的知识.

### Part 3.2Knowledge Manipulation

**SSRN** **paper**: https://ssrn.com/abstract=5250621 (last updated July 2024)

https://arxiv.org/abs/2309.14402

- 语言模型在知识检索方面表现出色，但即使在最简单的分类或比较任务中也表现不佳，除非在训练和推理过程中都使用思维链 （CoT）。此外，无论提示如何，它们在逆向知识搜索中的表现几乎为 0%。我们的主要贡献是一个*受控的合成实验* ，证实了这些弱点是语言模型固*有*的. 
- 缓解方法包括, 生成更多的 CoT 数据，并采用检索增强生成（RAG) 和反转训练等方法来帮助反向搜索。我们自己还建议重写训练文档以包含反转数据并引入文档行号以增强逆向搜索功能。

### Part 3.3, Knowledge Capacity Scaling Laws

**SSRN** **paper**: https://ssrn.com/abstract=5250617 (last updated April 2024)

https://arxiv.org/abs/2404.05405

这篇非常重要. 非常有用! 

- 通过多个受控数据集，我们确定语言模型每个参数可以而且只能存储 *2 位知识，即使量化为 int8，并且*可以灵活提取此类知识用于下游应用。因此，一个 7B 模型可以存储 14B 位知识，超过了我们估计的英文维基百科和教科书的总和。研究各种模型大小、深度、宽度、数据大小、类型（合成/半合成）和超参数。所有模型，即使没有 MLP 层，也接近这个比率。
- GPT-2 架构, with rotary embedding,，在*知识存储方面*与 LLaMA/Mistral 架构相当甚至超过，尤其是在较短的训练持续时间内。出现这种情况是因为 LLaMA/Mistral 使用 GatedMLP，它不太稳定且更难训练。
- 在训练数据前面加上域名（例如 wikipedia.org）可以显着提高模型的知识容量。语言模型可以自主识别知识丰富的领域并确定其优先级，从而优化其存储容量。  (然而有些公司出于某些隐私还是合规的原因决定去掉数据集中的域名. )
- 要实现 capacity ratio = 2 bit/param，每个知识片段在训练期间都要暴露/访问 1000 次，称为 *1000-exposure* .在 100 次的情况下， *训练不足*的 GPT2 的capacity ratio降至 1bit/param。换句话说, 在训练期间仅遇到 100 次的*稀有*知识以 1 bit/param存储。 
- 只有100 次时，部分架构出现局限性;值得注意的是，LLaMA/Mistral 的capacity ratio is 1.3x lower than GPT2’s ，即使在最佳调整学习率之后也是如此。(capacity ratio)
- GPTQ 量化到 int8 不会影响模型容量（即使对于 2bit/param 边界上的模型）,  量化到 int4 会将容量减少到 0.7 bit/param。 由于 2bit/param 是在充分训练后获得的，因此训练时间更长*可能不会*进一步提高模型容量， **but quantization can**. 
- 专家混合 （MoE） 模型提供比密集模型更快的推理速度，但通常表现不佳于具有相同总参数数（非有效参数）的密集模型。我们表明，性能下降应该不是缺乏知识存储能力。MoE models, even with 32 experts, only reduce 1.3x in capacity compared to the base scaling laws, despite using just 8.8% of the total parameters during inference.
- 垃圾数据显著  降低了模型容量。例如，“有用与垃圾”训练令牌的比例为 1：7，即使有用知识暴露了 100 次，有用知识的能力*也会损失 20 倍*.
- *有效的缓解措施*是在所有有用知识的前面添加一个特殊标记。这类似于在每个维基百科段落的开头添加一个像 wikipedia.org 这样的域名;该模型无需事先了解有价值的领域即可***自主***识别高质量数据。在上面的示例中，损耗因子从 20 倍提高到 2 倍。

### Part 4.1, Architecture Design and the Magic of Canon Layers

**SSRN** **paper**: https://ssrn.com/abstract=5240330 (**v1.1**, last updated May 19, 2025)

这里开始没有 arxiv 了.

Canon layer, 可以改善模型层内相邻token之间的水平信息流,计算附近token表示的加权总和. 显著提升了NoPE和GLA模型的性能.  有助于模型提高推理能力和可扩展性.









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
model_inputs = tokenizer(sequences, padding=True, return_tensors="np")# 可以返回torch, TensorFlow和numpy数组.model_inputs = tokenizer(sequences, padding=True, return_tensors="np")# 可以返回torch, TensorFlow和numpy数组.q 

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

`self.language_model`通常是前向传播的核心实现，每次调用只会根据输入上下文预测下一个token的概率分布（logits）。

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





# 学习率怎么调

bs大, lr 应该大.

lora , 训练参数少 , LoRA这点小参数，必须**快速学会校正**整个系统,  lr 应该大. 

qlora ,**int4量化**本身就引入了**噪声**，所以训练时系统天然“容忍度变高”，可以允许更大的梯度变化,   lr 应该大.

模型大,梯度求偏导数 平均到每个参数，每个参数变化幅度很小.  lr应该大. 

为什么gradnorm会一直上升呢?
别的模型也是这样, 
. **模型尚未收敛，权重仍在扩大**

在训练过程中，尤其是还没完全收敛时，参数更新的方向会逐渐使梯度范数变大。这可能反映了模型在持续调整其权重范围来更好拟合数据。
- 若 loss 同时在下降，这种缓慢上升是可接受的。

**学习率未衰减 or 衰减过晚**

若你在这段期间还没执行学习率 decay，梯度 norm 也可能呈缓慢上升趋势。尤其你在用较大模型（如 LLaMA2-7B）时，这个现象更明显。

看loss都看不出来效果, 只能eval  .

就是embedding 的lr 可以设大很多 (10x, 20x之类的)，因为他不是个linear，是个look up table. Mamba2就是这么做的。embedding用0.1，然后lm_head用1/sqrt{d}

### muon

https://kexue.fm/archives/10592 

lr 比 adamw 大10倍. 

第一版moonlight，是直接取 0.2 * sqrt(hidden_size) 倍，效果也挺好的，可以参考.

就是有涨点才去搞的，这点显存不是瓶颈.

muon需要加weight decay.

将Muon用于微调（SFT）时，可能会因为预训练没用Muon而不如adam。具体来说，如果预训练和微调都用Muon，那么表现是最好的,



param norm大了除了会overconfident/overfit还有其他坏处吗? 

- overflow, 然后nan了.更可怕的是有outlier , 一个参数吃了99%的norm.
- 学习效率可能会变低, update_rms基本上是固定的,  weight norm 越大, 每一步的更新幅度相对就变小了

olmo经典老坑了。olmo2两个经典结论我记得我看过，一个是说embedding不要wd，一个是rmsnorm gamma不wd，我个人觉得在SP搞法下两个都很坑...其实比l2norm更有信息量的是RMS，帮助你判断每一个数大概多大（不过因为有square，spike会放大整个range）
- without wd的l2norm是2000+，那rms = 2000 / sqrt(4096) ~= 30+ 
- with wd的l2norm是500，那rms = 500 / sqrt(4096) ~= 8+

这样一看显然是with wd的更加健康啊，考虑到bf16的表达精度，0附近显著多很多数。实际上除了embedding RMS，最好是监控lm head logit，那个信息量更大.  可以看llama3的pretrain没用上的special token，embedding rms小的离谱，反推可以得出llama3是在embedding上加了decay的，这些special token由于没有gradients，只有decay，所以被压倒越来越小. 国内有公司在这里吃过亏的，建议多加监控多多观察（尤其是lm head logit和attn logit这些，每一个要走有exp算子前的logit），然后SD下所有参数该decay就decay.

不知道为啥都喜欢记录l2norm，感觉不如abs_mean和rms直观？

问题甚至出在进softmax前，params norm大了之后，你去看进softmax之前的那些logit可能大小爆炸了，比如到了200多之后，可能前后+-20的range，才几百个数，太容易撞了，本来不是一样大小的数也变成一个大小了，那你再进softmax，没区分度了，top1和top2都是一样了. 所以参数不能太大. 





### 减少幻觉

https://openai.com/index/why-language-models-hallucinate/

不要奖励幸运猜测,  所有主要评估指标都需要重新设计，以奖励不确定性的表达。





## 激活函数

silu 在视觉任务中比 ReLU 更不容易梯度截断（ReLU 的 dead neuron 问题） 
Swish 是由 Google 研究人员在 2017 年提出的一种激活函数，结合了 Sigmoid 的平滑性和 ReLU 的稀疏激活特性，实验证明其在多种任务中优于 ReLU，尤其在深层网络中表现优异，也是现在主流LLM中FeedForward层使用的激活函数，只是它们使用的是SiLU.
低精度fp8或更低 swiglu 和gelu 会数值不稳定 , 应该用relu, . relu在 稀疏的时候flops可以减少很多. 
聊一聊Transformer中的FFN - 盘子正的文章 - 知乎 . 改silu不会提高很多. 但是ReLU造成dead neurons，因此在Transformer上逐渐被抛弃。但是cv改激活函数就没啥用. 
https://zhuanlan.zhihu.com/p/685943779
自己也有过一些ViT上的实验 (相信其他人也做过)，两个FC中间会有个hidden dimension的expansion ratio，一般设置为4。把这个地方调小会发现怎么都不如大点好。但是更大memory太费.  
FFN中的activations非低秩 都很难low rank.
应该用transformer.
问题

- 为什么之前vit 不如mlp 16好?  直接用vit, 没有空间结构、也不是 patch，不符合 ViT 的 inductive bias
- 
图像任务上 , ViT-Small参数量少于resnet101，但是微调后性能好于resnet101 

- MLP 是逐维映射，没有 token 间建模能力（无 attention）
- 无局部建模、无时间建模 → MLP 只能学到全局趋势，难建模细节模式

head多,  通常更快（batch 大, GPU 并行好）, 内存占用更大, 更稳定,  **表示力更强，收敛更快**。
但是head太多, `d` 太小时，注意力变“近似点积”，容易梯度爆炸或收敛慢。Jetson 系列在 head 数多时反而吞吐下降。



## KL 散度

【【中字】原来KL散度这么厉害，还有到底怎么落地计算】 https://www.bilibili.com/video/BV1Y57jzfE6N/?share_source=copy_web&vd_source=bb7496f78e4d303270b7c97ae8f69402

可能性更高的, 可以用更少的 bit 来编码.   

用了不同的分布模型, cross entropy 会变大. 

KL 散度就是cross entropy-  entropy. 

KL散度有一个不太好的地方，比方有两个正态分布N(0, 0.001)，N(0.01, 0.001)，虽然这两个分布形状一样，相互之间的偏差也非常小，但由于它们的标准差特别小，这时候算出来的KL散度就会特别大（约摸50），但其实两个分布之间点的距离只有大概0.01而已。这就是为什么有时候会用  wasserstein距离 来衡量分布之间的差距。

forward KL 和 reverse KL 是不同的.

reverse KL 惩罚 Q 在 p 没有质量的地方分配质量, 导致 mode seeking, Q 选择一个峰值,而忽略其他的. 

不可能精确计算, 所以用 monte carlo 估计. 算期望. 也就是取平均, 无偏估计. 

对数平方, 保证 估计量非负, 来减少方差. 但是是有偏估计.

所以提出了一个方式, 换一种方法无偏估计, 





openhands 可以跑 swebench.  terminal bench也挺好用的.

