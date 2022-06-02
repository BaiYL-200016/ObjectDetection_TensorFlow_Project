
# 环境搭建  

## 一. 下载并安装Anaconda  

1. 下载：

- 法一：进入[Anaconda官网](https://www.anaconda.com)进行下载安装(网络问题可能无法访问，建议使用法二)；

<img src="https://github.com/BaiYL-200016/ObjectDetection_TensorFlow_Project/blob/master/snap/download.png" width="60%" align="middle">


- 法二：通过[清华大学镜像源](https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/?C=M&O=D)进行下载安装。

![清华大学镜像源](\snap\Tsinghua.png)  

1. 安装：  

安装过程需要注意两点：

- 注意安装路径：选择合适的安装路径；
 ![Anaconda安装路径](\snap\install_01.png)
- 注意选项：建议只勾选第二个选项。
 ![Anaconda安装选项](\snap\install_02.png)  

其他一直下一步就好。

3. 配置环境变量：

安装完成后，依次点击**此电脑**——**属性**——**高级系统设置**——**高级**——**环境变量**——**path(系统变量)**——**编辑**——**新建**。
分别将以下几个路径添加到环境变量中(这里是我的anaconda路径，请根据自己情况自行修改)
 >       D:\Program Files\Anaconda3
 >       D:\Program Files\Anaconda\Scripts
 >       D:\Program Files\Anaconda\Library\bin
 >       D:\Program Files\Anaconda\Library\usr\bin
 >       D:\Program Files\Anaconda\Library\mingw-w64\bin

4. 检验是否安装配置成功：

`win+R`打开运行，输入`cmd`，打开命令提示符，输入`python -V`，能看到python版本信息则说明配置成功。

5. 更换conda镜像源：

由于国内访问默认镜像源地址较慢，所以需要将默认镜像源替换为国内源，这里以清华镜像源为例。
先介绍一些简单常用的conda命令(使用Anaconda Prompt运行)：

![Anaconda Prompt](\snap/anaconda%20prompt.png)

```bash
$ conda info: 显示有关当前 conda 安装的信息
$ conda -h, --help: 显示帮助信息
```
接下来进行换源，首先执行`$ conda config --set show_channel_urls yes`生成`.condarc`文件，一般此文件在用户目录下，通过`$ conda info`命令可以看到此文件位置。

![conda info](\snap/conda%20info.png)

使用记事本打开`.condarc`文件，将下面的内容添加到文件中并保存：

```text
channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  msys2: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  menpo: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch-lts: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  simpleitk: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
```

最后运行`$ conda clean -i`清除索引缓存，保证用的是镜像站提供的索引。
至此换源完成。

---

## 二. 使用Anaconda创建Python3虚拟环境

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
$ pip install tensorflow==2.8.0 -y
```
这里根据自己的python版本安装指定版本TensorFlow包，查询网址<https://tensorflow.google.cn/install/source_windows?hl=en#tested_build_configurations>，若需要选择GPU版本，如果不需要GPU，则使用CPU版本。

4. 安装需求包

```bash 
$ pip install -r <...\object_detection\requirements.txt>
```
5. 创建jupyter notebook内核

```bash
$ python -m ipykernel install --user --name=<kernel_name>
```
`<kernel_name>`为内核名称，可以随意指定，但是不能重复，否则会报错。

附：```$ jupyter kernelspec list```查看当前内核。

---