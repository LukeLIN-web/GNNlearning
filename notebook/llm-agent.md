

10% 课堂参与, 发言五次就行.  11 课. 上课了要让TA 知道. 上5个课就行.

25% 展示

1. 让听众可以参与.
2. 完整讲了论文的细节.
3. QA, 给一些问题. refer一些detail. 回答问题. 
4. 

65%  project.



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

提出了一个研究智能体之间竞争的一般框架。然后，我们使用 GPT-4 实现一个实际的竞争环境，以模拟一个具有两种类型的代理的虚拟城镇，包括餐厅和客户 .具体来说，餐厅代理相互竞争以吸引更多顾客，竞争鼓励他们转型，例如培养新的运营策略。模拟实验发现 一些现象, 与现有的市场和社会学理论非常吻合.  用gpt 研究社会学和市场.





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



### camel ai

https://github.com/camel-ai/camel  可以快速产生agent, 让agent对话. 做了infra的工作.  这个或许可以用. 

https://github.com/mistralai/cookbook/blob/main/third_party/CAMEL_AI/camel_roleplaying_scraper.ipynb 这个 加了很多rag, 更强大, 但是也更麻烦, 依赖的东西太多了, 如果需求他没有实现就需要改大量代码. 可以参考. 

####  crab

目前支持安卓和大部分桌面环境. linux windwos macos都可用，但效果还要看prompt设计和model性能. 

让agent 控制鼠标键盘, 去触发 mac和windows 的事件. 其实就是网络传输动作和参数，加一个控制mac鼠标键盘的库.

