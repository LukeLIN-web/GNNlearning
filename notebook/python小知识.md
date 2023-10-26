vscode pdb 的Watch  可以在调试的过程中查看, 不用手动print. 非常方便. 

#### module

https://docs.python.org/3/tutorial/modules.html

Python 有一种方法可以将定义放入文件中，并在脚本或解释器的交互式实例中使用它们。这样的文件称为 *模块*；模块中的定义可以*导入*其他模块或*主*模块（您可以在顶层执行的脚本和计算器模式下访问的变量集合）。

模块是包含 Python 定义和语句的文件。文件名是附加后缀的模块名`.py`。在模块中，模块的名称（作为字符串）可用作全局变量的值 `__name__`。

init的原理

需要这些`__init__.py`文件才能使 Python 将包含该文件的目录视为包。这可以防止具有通用名称的目录，例如`string`无意隐藏模块搜索路径中稍后出现的有效模块。在最简单的情况下，`__init__.py`可以只是一个空文件，但它也可以执行包的初始化代码或设置`__all__`变量

每个模块都有自己的私有命名空间，它被模块中定义的所有函数用作全局命名空间。因此，模块的作者可以在模块中使用全局变量，而不必担心与用户的全局变量发生意外冲突。

需要这些`__init__.py`文件才能使 Python 将包含该文件的目录视为包。  在最简单的情况下，`__init__.py`可以只是一个空文件，但它也可以执行包的初始化代码 ( 比如import)或设置`__all__`变量(暴露些class)。

使用 `from package import item`时，item 可以是包的子模块（或子包），也可以是包中定义的其他名称，如函数、类或变量。import  语句首先测试该项目是否在包中定义；如果不是，它假定它是一个模块并尝试加载它。如果找不到它， 则会引发 ImportError异常   

相反，当使用类似`import item.subitem.subsubitem`的语法时，除了最后一项之外的每一项都必须是一个包；最后一项可以是模块或包，但不能是前一项中定义的类或函数或变量。

if a package’s `__init__.py` code defines a list named `__all__`, it is taken to be the list of module names that should be imported when `from package import *` is encountered. 

You can also write relative imports, with the `from module import name` form of import statement. These imports use leading dots to indicate the current and parent packages involved in the relative import. From the `surround` module for example, you might use:

```
from . import echo
from .. import formats
from ..filters import equalizer
```

Note that relative imports are based on the name of the current module. Since the name of the main module is always `"__main__"`, modules intended for use as the main module of a Python application must always use absolute imports.

怎么引用隔壁文件夹的 python文件?  好神奇, 当我打开目录的时候, 我的库就不是从conda 引入的了, 就是用隔壁文件夹引入了. 不过pytest似乎还是有问题.

#### 计时

https://stackoverflow.com/questions/17579357/time-time-vs-timeit-timeit

#### args

num_workers=args.num_workers,传入参数是中间横线`argparser.add_argument('--num-workers', default=2, type=int)`, 为啥没关系? 就是没关系的. 这个参数ignore中间短横和下划线区别.
