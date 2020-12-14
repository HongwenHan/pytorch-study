import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torch.nn.functional as F  # nn是神经网络模块
import torchvision  # 包含demo用的训练数据集
import matplotlib.pyplot as plt
import numpy as np
import time


def test01():
    # hyper parameters
    EPOCH = 1
    LR = 0.01
    BATCH_SIZE = 64
    TIME_STEP = 28
    INPUT_SIZE = 28
    DOWNLOAD_MNIST = False

    train_dataset = torchvision.datasets.MNIST(
        root='./mnist/', transform=torchvision.transforms.ToTensor(), train=True, download=DOWNLOAD_MNIST)
    train_loader = Data.DataLoader(
        dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    test_data = torchvision.datasets.MNIST(
        root='./mnist/', train=False)
    test_x = torch.unsqueeze(test_data.test_data, dim=1).type(
        torch.FloatTensor)[:2000] / 255.
    test_y = test_data.test_labels[:2000]

    class RNN(nn.Module):
        def __init__(self):
            super(RNN, self).__init__()
            self.rnn = nn.LSTM(
                input_size=INPUT_SIZE,
                hidden_size=64,
                num_layers=1,
                batch_first=True
            )
            self.out = nn.Linear(64, 10)

        def forward(self, x):
            r_out, (h_n, h_c) = self.rnn(x, None)
            out = self.out(r_out[:, -1, :])
            return out

    rnn = RNN()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
    loss_func = nn.CrossEntropyLoss()

    for epoch in range(EPOCH):
        for step, (batch_x, batch_y) in enumerate(train_loader):
            bx = batch_x.view(-1, TIME_STEP, INPUT_SIZE)
            out = rnn(bx)
            loss = loss_func(out, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 50 == 0:
                test_out = rnn(test_x.view(-1, TIME_STEP, INPUT_SIZE))
                pred_y = torch.max(test_out, 1)[1].data.squeeze()
                accuracy = sum(pred_y == test_y) / test_y.size(0)
                print(step, loss.data.numpy(), '%.2f' % accuracy)

    test_out = rnn(test_x[1000: 1010].view(-1, TIME_STEP, INPUT_SIZE))
    pred_y = torch.max(test_out, 1)[1].data.squeeze()
    print(pred_y, 'prediction number')
    print(test_y[1000:1010], 'real number')


def test02():

    # hyper parameters
    LR = 0.02
    TIME_STEP = 10
    INPUT_SIZE = 1
    DOWNLOAD_MNIST = False

    class RNN(nn.Module):

        def __init__(self):
            super(RNN, self).__init__()
            self.rnn = nn.RNN(
                input_size=INPUT_SIZE,
                hidden_size=32,
                num_layers=1,
                batch_first=True)
            self.out = nn.Linear(32, 1)

        def forward(self, x, h_state):
            r_out, h_state = self.rnn(x, h_state)
            outs = []
            for step in range(r_out.size(1)):
                outs.append(self.out(r_out[:, step, :]))
            return torch.stack(outs, dim=1), h_state

        # def forward(self, x, h_state):
        #     r_out, h_state = self.rnn(x, h_state)
        #     r_out = r_out.view(-1, 32)
        #     outs = self.out(r_out)
        #     return outs.view(-1, 32, TIME_STEP), h_state

    rnn = RNN()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
    loss_func = nn.MSELoss()

    h_state = None

    for step in range(100):
        start, end = step * np.pi, (step + 1) * np.pi  # time steps
        # sin 预测 cos
        steps = np.linspace(start, end, TIME_STEP, dtype=np.float32)
        y_sin = np.sin(steps)    # float32 for converting torch FloatTensor
        y_cos = np.cos(steps)
        # shape (batch, time_step, input_size)
        x = torch.from_numpy(y_sin[np.newaxis, :, np.newaxis])
        y = torch.from_numpy(y_cos[np.newaxis, :, np.newaxis])

        prediction, h_state = rnn(x, h_state)   # rnn output
        # !! next step is important !!
        # repack the hidden state, break the connection from last iteration
        h_state = h_state.data

        loss = loss_func(prediction, y)         # calculate loss
        optimizer.zero_grad()                   # clear gradients for this training step
        loss.backward()                         # backpropagation, compute gradients
        optimizer.step()                        # apply gradients
        time.sleep(0.1)
        # plotting
        plt.plot(steps, y_cos.flatten(), 'r-')
        plt.plot(steps, prediction.data.numpy().flatten(), 'b-')
        plt.draw()
        plt.pause(0.05)

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    # test01()
    test02()
