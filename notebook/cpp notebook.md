





#### make

make install 



extern "C"

使用extern "C"来修饰变量和函数，它是一种连接声明（linkage declaration），被它修饰的变量和函数是按照C语言的方式编译和连接的。





#### CMAKE

find_package 会做什么? 

`find_package` 是去 `CMAKE_MODULE_PATH` 中查找 `Findxxx.cmake` 文件，然后在这个文件提供的路径下去寻找相应的库

set可以设置变量. 

每次改了代码都要重新cmake? 不用, 就重新make build就行. 
