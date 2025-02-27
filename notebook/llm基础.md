







## 2-2 

https://huggingface.co/learn/nlp-course/chapter2/2?fw=pt

听youtube,

pad是因为不同的batch 不一样长. attention mask也会设为0 表示这些是padding 的.

输出的logits不是概率,  softmax 之后变成概率 

head 是什么?   The model heads take the high-dimensional vector of hidden states as input and project them onto logits. 就是几个linear.  可以使用相同的架构执行不同的任务，但每个任务都将具有不同的 head 与之关联。  Transformer 模型的输出直接发送到模型头进行处理.  看图  https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter2/transformer_and_head.svg  

所有 🤗 Transformers 模型都输出 logits，因为用于训练的损失函数通常会将最后一个激活函数（如 SoftMax）与实际的损失函数（如交叉熵）融合

quiz: 

- loss是怎么算的?

- logits是什么? 

  

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

# todo
继续阅读后面内容, Dynamic padding  
```

3-3 trainer

todo

3-4 A full training 

todo





7-6

```
对更多使用这些库的训练样本给予更多权重是有意义的。我们可以通过使用 plt、pd、sk、fit 和 predict 等关键字轻松识别这些示例，这些关键字是 matplotlib.pyplot、pandas 和 sklearn 最常见的导入名称，以及后者的 fit/predict 模式。如果它们都表示为单个 token，我们可以轻松检查它们是否出现在 input 序列中。标记可以具有空格前缀，因此我们还将在 tokenizer 词汇表中检查这些版本。为了验证它是否有效，我们将添加一个测试令牌，该令牌应拆分为多个令牌：

```





#### generate

大部分模型 继承了PreTrainedModel , 继承了GenerationMixin, 调用generate, 就可以把self 自己传进去, 这里是调用了 WhisperForConditionalGeneration 中的forward函数。这是因为 PyTorch 的 nn.Module 基类定义了一个 __call__ 方法，当你调用模型实例（即 self）时，它会自动调用这个 __call__ 方法，而这个 __call__ 方法又会调用 forward 方法。

https://techdiylife.github.io/blog/blog.html?category1=c02&blogid=0005

