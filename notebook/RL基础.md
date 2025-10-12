

https://huggingface.co/learn/deep-rl-course/unit4/hands-on



# policy gradient

| 类别            | 代表算法       | 是否用RL梯度 | 特点              |
| --------------- | -------------- | ------------ | ----------------- |
| Policy Gradient | PPO, REINFORCE | ✅            | 最常见，用于 RLHF |



GRPO 的思想是，在一个批次（group）内，对同一个问题生成的多个候选回答进行比较，计算它们相对于该批次平均奖励的“相对优势”（relative advantage）。性能越好的回答（相对优势越高），其生成概率被提升；性能差的回答则被抑制。





# Preference-based RL



### DPO (Direct Preference Optimization)

- **思路**：不通过 RL，只通过最大化 “preferred response 比 rejected response 概率更高” 来隐式等价优化策略。
- **公式近似等价于**：优化 KL-regularized RL objective，但无需 rollouts。
- **优点**：稳定、无需 RL 框架。
- **代表论文**：
  - *“Direct Preference Optimization: Your Language Model is Secretly a Reward Model” (2023)*
  - 后续：GRPO 



GRPO 

重要性采样的本质是：我们希望在新的分布下计算期望，但数据却来自旧分布。为此，我们使用新旧策略在同一动作上的概率比作为修正权重.

这样就可以利用离线数据（来自旧策略）来评估新策略的期望，避免每次更新都重新采样，从而降低成本。然而，如果新旧策略差异过大，权重的方差会非常高，容易导致训练不稳定。