

https://hanlab.mit.edu/courses/2024-fall-65940

Mit 6.5940







## pruning



80层prune到32layer, 28762 到14336 intermediate dim, 加速2.5x.



prune一点, train 一下, 再prune 一些, 再train, 效果很好. 



2d weight, 随机选一些是很难加速的, irregular, 

2:4 sparsity , 连续4个elements, 2个被prune. 50% sparsity. ampere GPU 可以支持, 两倍加速



哪些weight 应该被prune?

- prune 绝对值最小的weight.