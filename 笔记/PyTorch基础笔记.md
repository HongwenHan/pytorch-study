# PyTorch简介

> PyTorch最大的优点是动态的建立神经网络，相对的Tensorflow是静态的。更匹配动态的RNN，底层代码比Tensorflow更清洗一点。

## 安装PyTorch

- 官网：https://pytorch.org/
- 安装教程：https://pytorch.org/get-started/locally/

```vim
// windows pip 安装pytorch 无GPU加速
pip install torch==1.7.0+cpu torchvision==0.8.1+cpu torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html

pip install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
```

```py
// 安装失败版本不一致或者查不到响应版本可以查看当前python版本对应要求的结构：
import setuptools.pep425tags
print(setuptools.pep425tags.get_supported())

[('cp37', 'cp37m', 'win_amd64'), ('cp37', 'none', 'win_amd64'), ('py3', 'none', 'win_amd64'), ('cp37', 'none', 'any'), ('cp3', 'none', 'any'), ('py37', 'none', 'any'), ('py3', 'none', 'any'), ('py36', 'none', 'any'), ('py35', 'none', 'any'), ('py34', 'none', 'any'), ('py33', 'none', 'any'), ('py32', 'none', 'any'), ('py31', 'none', 'any'), ('py30', 'none', 'any')]
```

> 如上可以看到当前python3.7支持('cp37', 'cp37m', 'win_amd64')组合，即可到`https://download.pytorch.org/whl/torch_stable.html`网站去下载cp37和cp37m组合的离线包进行下载，所以下载了如下两个离线包进行安装即可：

- torch-1.7.0+cpu-cp37-cp37m-win_amd64.whl
- torchvision-0.8.0-cp37-cp37m-win_amd64.whl

## PyTorch和Numpy对比

### 一些运算

- numpy转torch：torch.from_numpy(np_data)
- torch转numpy：torch_data.numpy()

### tensor运算

- tensor = torch.FloatTensor(data)
- tensor.abs()
- tensor.sin()
- tensor.mean()
- tensor.dot()
- ...

### 一些文档

- `torch api`开发文档：https://pytorch.org/docs/stable/torch.html

## 变量(Variable)

> 在 Torch 中的 `Variable` 就是一个存放会变化的值的地理位置. 里面的值会不停的变化。
Variable 计算时, 它在背景幕布后面一步步默默地搭建着一个庞大的系统, 叫做计算图, computational graph. 这个图是用来干嘛的? 原来是将所有的计算步骤 (节点) 都连接起来, 最后进行误差反向传递的时候, 一次性将所有 variable 里面的修改幅度 (梯度) 都计算出来。

## 激励函数(Activation)

- 非线性的函数；
- 激励函数是可微分的；
- 梯度爆炸、梯度消失；
- 卷及神经网络：relu，循环神经网络：tanh
- PyTorch的激励函数：relu/sigmoid/tanh/softplus...

> 一句话概括 `Activation`: 就是让神经网络可以描述非线性问题的步骤, 是神经网络变得更强大。
