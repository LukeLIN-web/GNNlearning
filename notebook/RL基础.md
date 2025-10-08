

https://huggingface.co/learn/deep-rl-course/unit4/hands-on



# policy gradient

| 类别            | 代表算法       | 是否用RL梯度 | 特点              |
| --------------- | -------------- | ------------ | ----------------- |
| Policy Gradient | PPO, REINFORCE | ✅            | 最常见，用于 RLHF |









# Preference-based RL



### DPO (Direct Preference Optimization)

- **思路**：不通过 RL，只通过最大化 “preferred response 比 rejected response 概率更高” 来隐式等价优化策略。
- **公式近似等价于**：优化 KL-regularized RL objective，但无需 rollouts。
- **优点**：稳定、无需 RL 框架。
- **代表论文**：
  - *“Direct Preference Optimization: Your Language Model is Secretly a Reward Model” (2023)*
  - 后续：GRPO 