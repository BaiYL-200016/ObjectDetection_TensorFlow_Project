# ObjectDection 
## 1. 环境搭建
### 1.1 下载并安装Anaconda
<https://www.anaconda.com>

---
### 1.2 使用Anaconda安装Python3
1. 创建虚拟环境

打开Anaconda Prompt，输入：

（1）在指定路径下创建虚拟环境

```bash
$ conda create -p <...\object_detection\env_name> python=3.8 -y
```
（2）在Anaconda安装路径下创建虚拟环境

```bash
$ conda create -n <env_name> python=3.8 -y
````
`<env_name>`为虚拟环境名称。

附：```$ conda env list```查看当前虚拟环境。
2. 激活虚拟环境

```bash
$ conda activate <...\object_detection\env_name>或<env_name>
```
可以看到虚拟环境已经激活（命令行前带有环境名称或路径），可以在虚拟环境下运行Python命令。
3. 安装TensorFlow包

```bash
$ conda install tensorflow=2.6.0/tensorflow-gpu=2.6.0 -y
```
这里根据自己的python版本安装指定版本TensorFlow包，查询网址<https://tensorflow.google.cn/install/source_windows#tested_build_configurations>，若需要选择GPU版本，如果不需要GPU，则使用CPU版本。
4. 安装需求包

```bash 
$ pip install -r <...\object_detection\requirements.txt>
```
5. 安装jupyter notebook内核

```bash
$ python -m ipykernel install --user --name=<kernel_name>
```
`<kernel_name>`为内核名称，可以随意指定，但是不能重复，否则会报错。

附：```$ jupyter kernelspec list```查看当前内核。

### 1.3 运行jupyter notebook

```bash