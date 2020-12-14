import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torch.nn.functional as F  # nn是神经网络模块
import torchvision  # 包含demo用的训练数据集
import matplotlib.pyplot as plt
import numpy as np
import time


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output


def CNN_demo01():

    # hyper params
    EPOCH = 1
    LR = 0.001
    BATCH_SIZE = 50
    DOWNLOAD_MNIST = False
    # 下载训练数据集
    train_data = torchvision.datasets.MNIST(
        root='./mnist',  # 存放到mnist路径
        transform=torchvision.transforms.ToTensor(),  # 转换成tensor格式
        train=True,  # 获取的是否是训练数据集
        download=DOWNLOAD_MNIST  # 是否需要下载，若果是False则直接从mnist目录获取
    )
    print(train_data.train_data.size(0))
    # 获取第100个值并展示
    # plt.imshow(train_data.train_data[100].numpy(), cmap='gray')
    # plt.title(train_data.train_labels[100])
    # plt.show()
    test_data = torchvision.datasets.MNIST(
        root='./mnist', train=False)

    train_loader = Data.DataLoader(
        dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    test_x = torch.unsqueeze(test_data.test_data, dim=1).type(
        torch.FloatTensor)[:2000] / 255.
    test_y = test_data.test_labels[:2000]

    cnn = CNN()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
    loss_func = nn.CrossEntropyLoss()

    for step, (batch_x, batch_y) in enumerate(train_loader):
        out = cnn(batch_x)
        loss = loss_func(out, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            test_out = cnn(test_x)
            pred_y = torch.max(test_out, 1)[1].data.squeeze()
            accuracy = sum(pred_y == test_y) / test_y.size(0)
            print(step, loss.data.numpy(), '%.2f' % accuracy)

    torch.save(cnn.state_dict(), 'cnn_params_01.pkl')

    test_out = cnn(test_x[:10])
    pred_y = torch.max(test_out, 1)[1].data.squeeze()
    print(pred_y.numpy(), '\n', test_y[:10].numpy())


def custom_restore01():
    test_data = torchvision.datasets.MNIST(
        root='./mnist', train=True)
    test_x = torch.unsqueeze(test_data.test_data, dim=1).type(
        torch.FloatTensor)[:2000] / 255.
    test_y = test_data.test_labels[:2000]
    cnn = CNN()
    cnn.load_state_dict(torch.load('cnn_params_01.pkl'))
    test_out = cnn(test_x[100:110])
    pred_y = torch.max(test_out, 1)[1].data.squeeze()
    print(pred_y.numpy(), '\n', test_y[100:110].numpy())


if __name__ == "__main__":
    # CNN_demo01()
    custom_restore01()
