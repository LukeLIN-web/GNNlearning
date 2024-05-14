







```

nohup /Applications/WeChat.app/Contents/MacOS/WeChat > /dev/null 2>&1 & # 双开微信
sinfo # 看机器
srun --gres=gpu:a100:4 --pty bash -i # 申请
```

#### ibex

https://docs.google.com/presentation/d/1xWHrX-qy35K5vePpXkromIDNWaBUZ_N5ytcEckab0ts/edit?usp=sharing

Some of you asked for the summary of some commands:

**# To check your priority on pending jobs**
`sprio -u USERNAME`

**# To check currently running jobs**
`squeue --me` OR `squeue -u USERNAME`

To check available GPUs:  `ginfo`

**# Mount Directory**
If you have a Linux system or if you install `sshfs` on your Macbooks, you can run the following command for mounting Ibex files into your machine,
`sshfs ibex:/path/to/mount mountIbex``mountIbex` (or whatever name you decide to call it) is a file you create with `mkdir mountIbex` on your home directly on your machine (not Ibex). Note that using "ibex" was only possible because of the things we run during the tutorial today!

1. 怎么把计算节点中的文件移动到前端节点的空间？可以直接移动吗？
2.  在login节点 load 有什么用？

`srun --pty --time=1:00 --gres=gpu:p100:2 bash -l` 分配gpu，module load cuda就可以了。nvcc就可以找到。 







