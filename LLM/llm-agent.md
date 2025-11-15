

1. 让听众可以参与.
2. 完整讲了论文的细节.
3. QA, 给一些问题. refer一些detail. 回答问题. 

2024年9月9日

reasoning依然非常弱, think step by step work但是没人知道为什么. 

未来可能, 机器人发现明天会下雨, 所以给你伞. 

就是讲自己的论文.

user privacy, 不能让agent说出用户的隐私.

教agent去劝说别人. 

#### 外交agent

22年夏天在FAIR做的, planning + dialogue. planning engine 是symbolic, dialogue agent 转化为verbal. 

难点在于 intents没有标注, 有untruthful messages.

解决:  train an intent model来标注.   用lie score 来判断lie.

在线外交游戏达到rank 2, 每局292条信息,人们没认出是ai. 

#### 劝说人们捐款

 acl 2019 .

可以取代销售员.  bot问, 和人问,  人们认不出来, 捐款量没啥区别.    问私人问题也没事. 不过加州要求必须告诉人们这是bot. 

#### Hugginggpt

浙大Yueting Zhuang的学生在msra的工作, >=24gb的显存, 200多gb的硬盘,  名字很有趣叫jarvis.

#### CompeteAI

提出了一个研究智能体之间竞争的一般框架。用GPT4 实现一个实际的竞争环境, 包括餐厅和客户. 具体来说，餐厅代理相互竞争以吸引更多顾客，竞争鼓励他们转型，例如培养新的运营策略。模拟实验发现一些现象, 与现有的市场和社会学理论非常吻合.  用gpt 研究社会学和市场.

#### Generative Agents: Interactive Simulacra of Human Behavior

斯坦福小镇, 这些代理能够拥有自己的记忆、目标、个性，并能够基于环境和过去的互动调整自己的行为。

**应用场景**：

- **虚拟世界中的角色**：这些生成代理可以用于游戏、虚拟世界中的NPC（Non-Player Characters），提高虚拟角色的智能化和互动感。
- **训练环境**：这些代理也可以用于训练和模拟环境中，例如在医疗、教育或危机管理中的角色扮演。
- **社交模拟**：研究者还探讨了这些代理在社交模拟中的潜力，如何模拟人类互动的复杂性。

reverie.py 526 loc  ReverieServer主要用于流程管理，以及资源管理

maze 327loc 管理地图数据

persona.py  198 loc 寻路, 感知. 

斯坦福小镇核心代码解读 - 小小马的文章 - 知乎
https://zhuanlan.zhihu.com/p/676362983

二创 https://github.com/a16z-infra/ai-town  提供了前后端的JS/TS框架.

#### camel ai

https://github.com/camel-ai/camel 使用初始提示来指导聊天代理完成任务，同时保持与人类意图的一致性。我们展示了如何使用角色扮演来生成对话数据，以研究代理社会的行为和能力.  可以快速产生agent, 让agent对话. 做了infra的工作.  这个或许可以用. 

https://github.com/mistralai/cookbook/blob/main/third_party/CAMEL_AI/camel_roleplaying_scraper.ipynb 这个 加了很多rag, 更强大, 但是也更麻烦, 依赖的东西太多了, 如果需求他没有实现就需要改大量代码. 可以参考. 

####  crab

目前支持安卓和大部分桌面环境. linux windows macos都可用，但效果还要看prompt设计和model性能. 

让agent 控制鼠标键盘, 去触发 mac和windows 的事件. 其实就是网络传输动作和参数，加一个控制mac鼠标键盘的库.

#### ChatDB

ChatDB 探索了使用符号内存增强 LLMs处理任意长度的上下文的方法 . 这样的符号内存框架被实例化为具有一组 SQL 数据库的 LLM。LLM 生成 SQL 指令来自主操作 SQL 数据库（包括插入、选择、更新和删除），旨在完成需要多跳推理和长期符号内存的复杂任务。这与现有的数据库参与形成鲜明对比，在数据库中，数据库被视为整个学习系统之外，被动地存储人类指示的信息。此外，之前的工作主要只关注选择操作，不支持对数据库进行插入、更新和删除操作。

问题是什么? 简单地将所有上下文信息连接起来并将其塞入 LLMs，很容易超出 LLMs并累积错误，导致模型失去对对话的跟踪并产生不太准确的响应。

怎么解决的? using databases as novel symbolic memory for LLMs

#### memory bank

可以用dual-tower retrieval mechanism 来 集成 RAG . 

memory sandbox

users可以控制agent记住什么. 也是可视化的很好, 可以view, edit memory objects.

#### Benchmark Self-Evolving: A Multi-Agent Framework for Dynamic LLM Evaluation

我们利用多代理系统来操纵原始实例的上下文或问题，以高置信度重构新的不断发展的实例，从而动态扩展现有基准。为了实现更具可扩展性、稳健性和细粒度的评估，我们实施了六次重构操作，以构建针对各种查询、数据噪声LLMs，并探测它们解决问题的子能力。

怎么重构的? 

The evaluation of LLMs has emerged as a crucial area. A lot of benchmark datasets have been proposed to evaluate LLMs. However, with the rapid development of LLMs, these static datasets are inadequate. 

There are two problems

1. Dataset inadequate: the previous static datasets used for evaluation are insufficient.
2. Data contamination issues:  In-domain training or even public test data may be unintentionally included during LLM training, resulting in skewed evaluations.  

目标: continual updates of static benchmark datasets, enabling a more dynamic and accurate evaluation of LLMs

过去的方案

1. **Perplexity on re-sampled data?**	-> May not fully capture LLM performance beyond predictive accuracy
2. 基于有向无环图动态合成测试样本，但这种方法难以推广到无法用图表示的任务

怎么解决的? 我们引入了一个基准自我进化框架，它通过修改它们的上下文或问题以及相应的答案，将现有的基准实例重新构建为新的变体以进行动态评估。 

1. 我们通过基于原始上下文创建替代或更复杂的问题来引入**可扩展评估**。 
2. **稳健的评估**。这涉及将各种扰动合并到原始实例的上下文中，包括释义、添加噪声和反转极性    

instance verfier

validate the correctness of both the new generated instance (Ce,Qe,Ae) and its corresponding incorrect alternative (Ce,Qe,Ow).

结果如何? 

这个分数是怎么算的?  他evaluation不是有好几个模块吗? 

这几个agent有训练吗?  oneshot是啥意思? 

#### chateval

