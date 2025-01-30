## Fundamental concepts of statistical learning

Uniform convergence (learning finite, realizable hypothesis spaces)

#### 迁移学习

- How does the difference between β(1) and β(2) affect transfer learning performance?   这个叫model shift.

- How does the difference between the feature vectors of source task and target task affect transfer learning?  这个叫 covariate shift

**Hard Transfer** assumes a single shared parameter vector (β\betaβ) for both tasks.

**Soft Transfer** allows separate parameters for each task but aligns them through regularization

Probably Approximately Correct Learning,  PAC learning

h：是一个理论上的候选假设，可能不一定是最佳的。

h^\hat：是通过训练优化得到的假设，用来对新数据进行预测。

Empirical Risk Minimization, ERM

## HW1

神经切线核

McDiarmid. 有界差分不等式，是概率论中的强大工具。它提供了一种方法来限制独立随机变量函数显著偏离其预期值的概率

Proof of McDiarmid’s inequality

Martingale Definition

A martingale is like a "fair game": no matter what happens up to a given point, your expected future value is equal to your current value.

Di的期望是0 .  因为 Zi会等于 Zi-1.

Rademacher Complexity 是一种用于衡量假设类（hypothesis class）容量的统计学习理论工具。它用于量化模型在随机标签（噪声数据）上的拟合能力，进而用于评估模型的泛化能力。

若 **Rademacher Complexity 低**，说明该假设类对随机噪声拟合能力弱，模型更可能具有较好的泛化能力。

在深度学习中，Rademacher Complexity 也用于分析神经网络的复杂度，并指导如何使用正则化（如 Dropout、权重衰减）来控制过拟合。







## matrix completion.

矩阵的 **核范数**（Nuclear Norm），也叫做 **迹范数**，是矩阵的奇异值的和。给定一个矩阵，其核范数定义为矩阵的所有奇异值的和. 

直观上它可以用来控制矩阵的低秩结构。较小的核范数表示矩阵的秩较低。

我们通常希望找到一个低秩矩阵来近似一个部分已知的矩阵。然而，优化矩阵的秩是一个非凸优化问题，难以直接求解。

为了简化这个问题，我们使用 **核范数松弛**，即将目标函数中的矩阵秩用核范数来代替。通过最小化矩阵的核范数，我们可以得到一个近似的低秩矩阵。

1. **矩阵元素有界**：
   - 这是一个合理的假设，因为在实际应用中（如电影评分），矩阵元素通常是有界的（例如评分在 1 到 5 之间）。

对于极端情况（如矩阵中只有一个非常大的元素），如果没有采样到关键元素，恢复可能会失败。因此，合理的假设（如元素有界性和不相干性）是核范数最小化成功的关键。

## 网络泛化误差

激活函数, 一般都是 1-Lipschitz, Sublinear function, 不能增长太快. 

### **与 VC 维的关系**

- VC 维（Vapnik-Chervonenkis 维度）也是衡量假设类复杂度的工具，但 Rademacher Complexity 具有更强的适用性，尤其适用于神经网络等假设类容量较大的情况。
- 一般来说，较高的 VC 维和较高的 Rademacher Complexity 都意味着较差的泛化能力。



