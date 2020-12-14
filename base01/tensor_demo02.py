import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.utils.data as Data
import torch.nn.functional as F  # nn是神经网络模块
import matplotlib.pyplot as plt
import time


def test01():
    # unsqueeze 以为变二维
    # x data (tensor), shape=(100, 1)
    x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
    # x.pow(2)：x的二次方  0.2*torch.rand(x.size())：添加噪点
    # noisy y data (tensor), shape=(100, 1)
    y = x.pow(2) + 0.2*torch.rand(x.size())
    # y = x.pow(2)  # noisy y data (tensor), shape=(100, 1)
    x, y = Variable(x), Variable(y)

    # 画图
    # plt.scatter(x.data.numpy(), y.data.numpy())
    # plt.show()
    class Net(torch.nn.Module):

        def __init__(self, n_feature, n_hidden, n_output):
            super(Net, self).__init__()
            self.hidden = torch.nn.Linear(n_feature, n_hidden)  # 隐藏层线性输出
            self.predict = torch.nn.Linear(n_hidden, n_output)  # 输出层线性输出

        def forward(self, x):  # 重写Module中forward功能
            # 正想传播输入值，神经网络分析出输出值
            x = F.relu(self.hidden(x))  # 激励函数（隐藏层的线性值）
            return self.predict(x)  # 输出值

    net = Net(n_feature=1, n_hidden=10, n_output=1)
    # print(net)  # net 的结构

    # optimizer 是torch训练的工具
    optimizer = torch.optim.SGD(net.parameters(), lr=0.5)  # 传入net的所有参数，学习率
    loss_func = torch.nn.MSELoss()  # 用在回归问题：预测值和真实值的误差计算公式

    for t in range(100):
        prediction = net(x)  # 喂入训练数据x，输出预测值
        loss = loss_func(prediction, y)  # 计算预测值和实际值的误差
        #  优化步骤
        optimizer.zero_grad()  # 清空上一步的残余更新参数值
        loss.backward()  # 误差反向传播，计算参数更新值，计算梯度
        optimizer.step()  # 将参数值指甲刀net的parameters上，优化梯度

        # 接着上面来
        if t % 5 == 0:
            # plot and show learning process
            plt.cla()
            plt.scatter(x.data.numpy(), y.data.numpy())
            plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
            plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(),
                     fontdict={'size': 20, 'color':  'red'})
            plt.pause(0.1)
            time.sleep(0.5)


def test02():
    # 假数据
    n_data = torch.ones(100, 2)         # 数据的基本形态
    x0 = torch.normal(2*n_data, 1)      # 类型0 x data (tensor), shape=(100, 2)
    y0 = torch.zeros(100)               # 类型0 y data (tensor), shape=(100, )
    x1 = torch.normal(-2*n_data, 1)     # 类型1 x data (tensor), shape=(100, 1)
    y1 = torch.ones(100)                # 类型1 y data (tensor), shape=(100, )

    # 注意 x, y 数据的数据形式是一定要像下面一样 (torch.cat 是在合并数据)
    # FloatTensor = 32-bit floating
    x = torch.cat((x0, x1), 0).type(torch.FloatTensor)
    # LongTensor = 64-bit integer
    y = torch.cat((y0, y1),).type(torch.LongTensor)

    # plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[
    #             :, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
    # plt.show()

    # 画图
    # plt.scatter(x.data.numpy(), y.data.numpy())
    # plt.show()
    class Net(torch.nn.Module):

        def __init__(self, n_feature, n_hidden, n_output):
            super(Net, self).__init__()
            self.hidden = torch.nn.Linear(n_feature, n_hidden)  # 隐藏层线性输出
            self.predict = torch.nn.Linear(n_hidden, n_output)  # 输出层线性输出

        def forward(self, x):  # 重写Module中forward功能
            # 正想传播输入值，神经网络分析出输出值
            x = F.relu(self.hidden(x))  # 激励函数（隐藏层的线性值）
            return self.predict(x)  # 输出值

    net = Net(n_feature=2, n_hidden=10, n_output=2)

    # optimizer 是torch训练的工具
    optimizer = torch.optim.SGD(net.parameters(), lr=0.2)  # 传入net的所有参数，学习率
    loss_func = torch.nn.CrossEntropyLoss()  # 用在分类问题：预测值和真实值的误差计算公式

    for t in range(100):
        out = net(x)  # 喂入训练数据x，输出预测值  F.softmax转换成概率
        loss = loss_func(out, y)  # 计算预测值和实际值的误差
        optimizer.zero_grad()  # 清空上一步的残余更新参数值
        loss.backward()  # 误差反向传播，计算参数更新值
        optimizer.step()  # 将参数值指甲刀net的parameters上

        # 接着上面来
        if t % 2 == 0:
            plt.cla()
            # 过了一道 softmax 的激励函数后的最大概率才是预测值
            prediction = torch.max(F.softmax(out), 1)[1]
            pred_y = prediction.data.numpy().squeeze()
            target_y = y.data.numpy()
            plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[
                        :, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
            accuracy = sum(pred_y == target_y)/200.  # 预测中有多少和真实值一样
            plt.text(1.5, -4, 'Accuracy=%.2f' %
                     accuracy, fontdict={'size': 20, 'color':  'red'})
            plt.pause(0.1)
            time.sleep(0.2)


def test03():
    # unsqueeze 以为变二维
    # x data (tensor), shape=(100, 1)
    x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
    # x.pow(2)：x的二次方  0.2*torch.rand(x.size())：添加噪点
    # noisy y data (tensor), shape=(100, 1)
    y = x.pow(2) + 0.2*torch.rand(x.size())
    # y = x.pow(2)  # noisy y data (tensor), shape=(100, 1)
    x, y = Variable(x), Variable(y)

    # 画图
    # plt.scatter(x.data.numpy(), y.data.numpy())
    # plt.show()
    # 方式一：
    # class Net(torch.nn.Module):

    #     def __init__(self, n_feature, n_hidden, n_output):
    #         super(Net, self).__init__()
    #         self.hidden = torch.nn.Linear(n_feature, n_hidden)  # 隐藏层线性输出
    #         self.predict = torch.nn.Linear(n_hidden, n_output)  # 输出层线性输出

    #     def forward(self, x):  # 重写Module中forward功能
    #         # 正想传播输入值，神经网络分析出输出值
    #         x = F.relu(self.hidden(x))  # 激励函数（隐藏层的线性值）
    #         return self.predict(x)  # 输出值

    # net = Net(n_feature=1, n_hidden=10, n_output=1)
    # print(net)  # net 的结构
    # 方式二：
    net = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )

    # optimizer 是torch训练的工具
    optimizer = torch.optim.SGD(net.parameters(), lr=0.5)  # 传入net的所有参数，学习率
    loss_func = torch.nn.MSELoss()  # 用在回归问题：预测值和真实值的误差计算公式

    for t in range(100):
        prediction = net(x)  # 喂入训练数据x，输出预测值
        loss = loss_func(prediction, y)  # 计算预测值和实际值的误差
        optimizer.zero_grad()  # 清空上一步的残余更新参数值
        loss.backward()  # 误差反向传播，计算参数更新值
        optimizer.step()  # 将参数值指甲刀net的parameters上

        # 接着上面来
        if t % 5 == 0:
            # plot and show learning process
            plt.cla()
            plt.scatter(x.data.numpy(), y.data.numpy())
            plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
            plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(),
                     fontdict={'size': 20, 'color':  'red'})
            plt.pause(0.1)
            time.sleep(0.5)


def test04():
    x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
    y = x.pow(2) + 0.2*torch.rand(x.size())
    x, y = Variable(x), Variable(y)

    def save():
        net1 = torch.nn.Sequential(
            torch.nn.Linear(1, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 1)
        )
        optimizer = torch.optim.SGD(net1.parameters(), lr=0.5)
        loss_func = torch.nn.MSELoss()
        for i in range(100):
            prediction = net1(x)
            loss = loss_func(prediction, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        torch.save(net1, 'net1.pkl')
        torch.save(net1.state_dict(), 'net1_params.pkl')

        plt.figure(1, figsize=(10, 3))
        plt.subplot(131)
        plt.title('Net1')
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)

    def restore_net():
        net2 = torch.load('net1.pkl')
        prediction = net2(x)
        plt.subplot(132)
        plt.title('Net2')
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)

    def restore_params():
        net3 = torch.nn.Sequential(
            torch.nn.Linear(1, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 1)
        )
        net3.load_state_dict(torch.load('net1_params.pkl'))
        prediction = net3(x)
        plt.subplot(133)
        plt.title('Net3')
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)

    save()
    restore_net()
    restore_params()
    plt.show()


def test05():
    x = torch.linspace(1, 10, 10)
    y = torch.linspace(10, 1, 10)
    BATCH_SIZE = 5

    torch_dataset = Data.TensorDataset(x, y)
    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )

    for s in range(2):
        for i, (batch_x, batch_y) in enumerate(loader):
            # 打出来一些数据
            print('Epoch: ', s, '| Step: ', i, '| batch x: ',
                  batch_x.numpy(), '| batch y: ', batch_y.numpy())


def test06():
    # 定义数据集
    x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)
    y = x.pow(2) + 0.1 * torch.normal(torch.zeros(*x.size()))
    # plt.scatter(x.numpy(), y.numpy())
    # plt.show()

    # 超参数：hyper params
    LR = 0.01
    BATCH_SIZE = 32
    EPOCH = 12
    # 定义批训练
    torch_dataset = Data.TensorDataset(x, y)
    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2)

    # 定义神经网络

    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.hidden = torch.nn.Linear(1, 20)   # hidden layer
            self.predict = torch.nn.Linear(20, 1)   # output layer

        def forward(self, x):
            x = F.relu(self.hidden(x))
            return self.predict(x)             # linear output
    net_SGD = Net()
    net_Momentum = Net()
    net_RMSprop = Net()
    net_Adam = Net()
    nets = (net_SGD, net_Momentum, net_RMSprop, net_Adam)

    # 定义优化器:optimizers
    opt_SGD = torch.optim.SGD(net_SGD.parameters(), lr=LR)
    opt_Momentum = torch.optim.SGD(
        net_Momentum.parameters(), lr=LR, momentum=0.8)
    opt_RMSprop = torch.optim.RMSprop(
        net_RMSprop.parameters(), lr=LR, alpha=0.9)
    opt_Adam = torch.optim.Adam(
        net_Adam.parameters(), lr=LR, betas=(0.9, 0.99))
    opts = (opt_SGD, opt_Momentum, opt_RMSprop, opt_Adam)
    # 定义误差计算公式：均方差
    loss_func = torch.nn.MSELoss()
    # 用来存储每一步误差值
    losses_his = ([], [], [], [])

    # 训练神经网络
    for epoch in range(EPOCH):
        for step, (batch_x, batch_y) in enumerate(loader):
            bx, by = Variable(x), Variable(y)
            for net, opt, loss_his in zip(nets, opts, losses_his):
                out = net(bx)
                loss = loss_func(out, by)
                opt.zero_grad()
                loss.backward()
                opt.step()
                loss_his.append(loss.data.numpy())

    lables = ['SGD', 'Momentum', 'RMSprop', 'Adam', ]
    for i, loss_his in enumerate(losses_his):
        plt.plot(loss_his, label=lables[i])
    plt.legend(loc='best')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.ylim((0, 0.2))
    plt.show()


def custom01():
    def custom_save01():
        x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)
        y = x.pow(2) + 0.1 * torch.normal(torch.zeros(*x.size()))

        torch_dataset = Data.TensorDataset(x, y)
        torch_loader = Data.DataLoader(
            dataset=torch_dataset, batch_size=32, shuffle=True, num_workers=2)

        net = nn.Sequential(nn.Linear(1, 10), nn.ReLU(), nn.Linear(10, 1))
        opt = torch.optim.Adam(net.parameters(), lr=0.2, betas=(0.9, 0.99))
        loss_func = nn.MSELoss()

        for epoch in range(12):
            for step, (batch_x, batch_y) in enumerate(torch_loader):
                bx, by = Variable(batch_x), Variable(batch_y)
                out = net(bx)
                loss = loss_func(out, by)
                opt.zero_grad()
                loss.backward()
                opt.step()
                print(epoch, step, loss.data.numpy())
        torch.save(net.state_dict(), 'pox2.pkl')
        
    def custom_restore01():
        x = torch.unsqueeze(torch.linspace(-1, 1, 10), dim=1)
        print(x.numpy())
        net = nn.Sequential(nn.Linear(1, 10), nn.ReLU(), nn.Linear(10, 1))
        net.load_state_dict(torch.load('pox2.pkl'))
        print(net(x))

if __name__ == "__main__":
    # test01()
    # test02()
    # test03()
    # test04()
    # test05()
    # test06()
    # custom01()
    
