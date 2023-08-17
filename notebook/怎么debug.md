1. 在cpu上 debug.  model不要放在gpu上.  
2. pdb step也挺麻烦, 不如print exit. 
3. vscode的报错可能比终端要少. 终端执行的报错比pytest少.  vscode可以用 debug直接输入某个数值查看. 
4. 出错原因:为什么?  然后往上一层函数排查. 
5. 大图这样太难debug了.  思想就是搞个小图, 不要搞这么大的图.  最短的代码复现. 都先看最小的, 小的对了再看大的. 
6. 做一些检查确定你的假设成立, 比如list确实sorted了.
7. 用assert, 不要肉眼看, 出错了再肉眼看. 让电脑来判断,  gdb的话不用在前面看, 就直接在出错的地方看. 
8.    可以写在  utils/ 和/test   自己写了test ,  可以用pytest 自动化测试.

