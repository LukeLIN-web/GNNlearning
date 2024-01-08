## Alpa

intra是很难的,通讯成本大

Alpas 是一个编译器,  automates model-parallel training generating execution plans that unify data, operator, and pipeline parallelism.

写了16k python, 6k cpp.

## Unity

FlexFlow属于automatic, intra-op 并行, 但是没考虑inter-op 并行, Alpa说自己考虑了. 

## Orca

Orca: A Distributed Serving System for Transformer-Based Generative Models

首尔国立大学 Gyeong-In Yu and Joo Seong Jeong, *Seoul National University*

不是开源的.We have implemented ORCA with 13K lines of C++, based on the CUDA ecosystem. 这个难点就是cpp 代码才能把算子一个个step分出来. 

最近，为生成任务训练的基于 Transformer 的大规模模型（例如 GPT-3）引起了极大的兴趣，强调了为该系列中的模型提供服务所需的系统支持。由于这些模型以自回归方式生成下一个令牌，因此必须多次运行模型才能处理推理请求，其中模型的每次迭代都会为请求生成一个输出令牌。然而，现有的推理服务系统在这种类型的具有多迭代特性的工作负载上表现不佳，因为它们的调度机制不灵活，无法改变当前正在处理的请求批次;比批处理中的其他请求更早完成的请求无法返回到客户端，而新到达的请求必须等到当前批处理完全完成。

在本文中，我们提出了迭代级调度，这是一种新的调度机制，它以迭代（而不是请求）的粒度调度执行，其中调度器调用执行引擎在批处理上仅运行模型的单次迭代。此外，为了同时将批处理和迭代级调度应用于 Transformer 模型，我们建议选择性批处理，该批处理仅将批处理应用于选定的一组操作。基于这两种技术，我们实现了一个名为ORCA的分布式服务系统，并进行了额外的设计，以扩展到具有数千亿个参数的模型。我们对 GPT-3 175B 模型的评估表明， ORCA 在延迟和吞吐量方面可以显着优于 NVIDIA FasterTransformer：

#### motivation 

您可以加载一次模型参数，然后使用它们来处理多个输入序列，而不是每次都有输入序列时加载新的模型参数。这可以更有效地利用芯片的内存带宽， 因为LLM的参数加载非常费时, 所以效果非常好. 

Orca 不是等到批处理中的每个序列都完成生成，而是实现了迭代级别的调度，其中批处理大小是每次迭代确定的。结果是，一旦批处理中的序列完成生成，就可以在其位置插入一个新序列，从而产生比静态批处理更高的 GPU 利用率。

现实情况比这个简化的模型要复杂一些：由于预填充阶段需要计算，并且具有与生成不同的计算模式，因此它不能轻易地与令牌的生成进行批处理。连续批处理框架目前通过 hyperparameter： waiting_served_ratio 或等待预填充的请求与等待序列结束令牌的请求的比率来管理这一点。

***Note\****: Continuous batching, dynamic batching, and iteration-level scheduling are all close enough in meaning that any one of them can be used to describe the batching algorithm.*

*request-level batching, where an LLM inference server uses a static batch whose size is chosen when the current batch has completely finished generation.* 

input 2D tensor [total_num_tokens, hidden_size]就行，attention 计算那块mask掉不同sequences 之间dependence 就可以了

所以他们是没有依赖? 有依赖的, Q2 计算依赖Q1. 需要concat一下. 

#### selective batching. 

他们解决了不同的sequence长度 batching 的问题.  linear层就flattented 2Dtensor without batch dimension, attention 层就, split 计算attn, 然后merge. 

- 给定形状为 `[(s1, h), (s2, h), ...]` 的输入，我们将它们堆叠成一个形状为 `(sum(si), h)` 的大矩阵。
- 对这个堆叠矩阵应用密集层。
- 将密集层的结果分割回 `[(s1, h), (s2, h), ...]`。
- 对每个序列进行自注意力计算。

把每个layer 切成两半 分给同一个worker的两个GPU.

### 缺点

**ORCA** 使用**first-come-first-served (FCFS)** 处理推理作业, 计划任务持续运行直至完成。 由于 GPU 内存容量有限以及推理对延时敏感，无法通过任意数量的传入函数来增加处理，由此可能会导致队列阻塞。

**FastServe** 使用 **preemptive scheduling**，通过新颖的**0** 程序来最小化延时。 基于 LLM 推理的长度无法确定，调度程序利用输入长度信息来分配适当的初始值每个到达作业要加入的队列。 较高优先级队列跳过加入的队列以减少降级。 设计高效的GPU内存管理机制主动下载和上传 GPU 内存和主机内存之间的中间状态，以进行 LLM 推理。

