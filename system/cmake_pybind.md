#### cmake

https://zhuanlan.zhihu.com/p/367808125

cpp修改了, make了之后会在build生成.so , 但是不能直接用, 要重新pip install -e .

怎么让他make 更快?

1. https://github.com/pybind/pybind11/discussions/4345 

conda环境
cmake: error while loading shared libraries: librhash.so.0: cannot open shared object file: No such file or directory

conda安装的很老, pip安装比较新. conda基本上只能做环境隔离用, 下载软件都用pip就好.   

#### pybind 入门

```shell
mkdir build #和src,pybind11同级
cd build
cmake ..
make mymath
Scanning dependencies of target mymath
#[ 50%] Building CXX object CMakeFiles/mymath.dir/src/mymath.cpp.o
#[100%] Linking CXX static library libmymath.a
#[100%] Built target mymath
make cmake_example
#[ 50%] Built target mymath
#Scanning dependencies of target cmake_example
#[ 75%] Building CXX object CMakeFiles/cmake_example.dir/src/binder.cpp.o
#[100%] Linking CXX shared module cmake_example.cpython-38-x86_64-linux-gnu.so
#[100%] Built target cmake_example
```

 ```cmake
 set_target_properties(mymath PROPERTIES POSITION_INDEPENDENT_CODE ON)
 add_subdirectory(pybind11)
 pybind11_add_module(cmake_example src/binder.cpp)
 target_link_libraries(cmake_example PRIVATE mymath)
 不能 add_library(mymath SHARED src/mymath.cpp) 因为这样写会找不到依赖的so文件, 
 ```

然后运行 setup.py

setup.py is a file that is executed by pip. run `pip install -e .`  

```py
python setup.py develop
python setup.py bdist_wheel # 这个产生dist文件夹, 文件夹里有.whl文件,  还会产生egg-info然后 pip install .whl文件,就会把.so放在site-packages中.
pip wheel .
```

##### 基础

https://pybind11.readthedocs.io/en/stable/basics.html

```python
example.add(i=1, j=2) # 可以使用关键字参数调用函数，如果带有许多参数的函数可以用
m.def("add", &add, "description",py::arg("i") = 1, py::arg("j") = 2); # 默认参数 
 py::class_<Pet>(m, "Pet") #为自定义类型创建绑定
  #可以绑定lambda函数
  .def_readwrite("name", &Pet::name)# 可以python添加cpp类的属性
```



```cpp
  m.attr("the_answer") = 42;// 让python可以访问cpp里面的值. 
```



##### build

https://pybind11.readthedocs.io/en/stable/compiling.html 仔细读一遍文档可以少问很多问题. 

```python
ext_modules = [
    Pybind11Extension(
        "python_example",
        sorted(glob("src/*.cpp")),  # Sort source files for reproducibility
    ),
]
```

内容

1. 怎么跳过rebuild和怎么强制rebuild.

2. 用`pyproject.toml` 创建虚拟环境
3. cmake怎么找到pybind11 库,找到python

https://pybind11.readthedocs.io/en/stable/advanced/functions.html

CUDAExtension和  Pybind11Extension区别是什么? 

cmake example用的是CMakeExtension, python example用的是Pybind11Extension. gnnlab用的是CUDAExtension

为什么没有`#include <pybind11/pybind11.h>`  也能用`PYBIND11_MODULE `?

内容

1. [Return value policies](https://pybind11.readthedocs.io/en/stable/advanced/functions.html#return-value-policies) 
2. [Keep alive](https://pybind11.readthedocs.io/en/stable/advanced/functions.html#keep-alive) [Call guard](https://pybind11.readthedocs.io/en/stable/advanced/functions.html#call-guard) 
3.  [Python objects as arguments](https://pybind11.readthedocs.io/en/stable/advanced/functions.html#python-objects-as-arguments)
4. [Accepting *args and **kwargs](https://pybind11.readthedocs.io/en/stable/advanced/functions.html#accepting-args-and-kwargs) 

##### 项目学习

https://medium.com/practical-coding/setting-up-a-c-python-project-with-pybind11-and-cmake-8de391494fca

### setuptools

#### develop 模式

1. 更新cpp, 重新make是不是就可以用了? 

https://packaging.python.org/en/latest/guides/distributing-packages-using-setuptools/#working-in-development-mode  还需要重新pip -e . 

setup怎么写:   https://zhuanlan.zhihu.com/p/276461821

```
pip install -e 和 python setup.py develop 有什么区别? 
```

建议用Pip install -e 

因为直接调用`setup.py`会对许多依赖项做错误的事情，例如拉预发布和不兼容的包版本，或者使包难以卸载`pip`.

可编辑模式可以参考:

https://packaging.python.org/en/latest/guides/distributing-packages-using-setuptools/#working-in-development-mode 

`python3 setup.py build  python3 setup.py clean`是速度快的原因吗? 这两个已经被弃用了. 

**build**命令运行后（无论是显式运行，还是 **install**命令为您完成），**install** 命令的工作相对简单：它所要做的就是将 `build/lib`(or ) 下的所有内容复制到您选择的安装目录中。`build/lib.*plat*`
