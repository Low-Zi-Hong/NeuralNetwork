# 1. 选底座：拉取一个极其轻量级的 Linux (Ubuntu 基础) 并且带 Python 环境
FROM python:3.10-slim

# 2. 装编译器：在 Linux 容器里安装 GCC/G++ 编译器和 Make 工具
RUN apt-get update && apt-get install -y g++ make

# 3. 建车间：在容器里建个工作目录
WORKDIR /app

# 4. 搬代码：把你电脑上的 C++ 源码、数据集复制进 Linux 容器里
COPY scr/ ./scr/
COPY main.cpp ./main.cpp
COPY train-images.idx3-ubyte ./train-images.idx3-ubyte
COPY train-labels.idx1-ubyte ./train-labels.idx1-ubyte

# 5. 当场编译！让 Linux 的 g++ 编译器把你的 C++ 源码编译成 Linux 原生程序（取名叫 my_neural_net）
# 注意：-O3 是开启 C++ 最高级别的性能优化，跑神经网络必备！
RUN g++ -O3 main.cpp scr/EMath.cpp scr/FileManager.cpp scr/NeuralNetwork.cpp -o my_neural_net

# 6. 点火发射：运行刚刚编译出来的 Linux 原生程序
CMD ["./my_neural_net"]