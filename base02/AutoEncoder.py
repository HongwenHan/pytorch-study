import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torch.nn.functional as F  # nn是神经网络模块
import torchvision  # 包含demo用的训练数据集
import matplotlib.pyplot as plt
import numpy as np
import time


def auto_encoder():

    # hyper params
    EPOCH = 20
    LR = 0.05
    BATCH_SIZE = 64
    DOWNLOAD_MNIST = False
    N_TEST_IMG = 5

    train_dataset = torchvision.datasets.MNIST(
        root='./mnist/', transform=torchvision.transforms.ToTensor())
    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE,
                                   shuffle=True, num_workers=2)

    class AutoEncoder(nn.Module):

        def __init__(self):
            super(AutoEncoder, self).__init__()

            self.encoder = nn.Sequential(
                nn.Linear(28 * 28, 14 * 14),
                nn.Tanh(),
                nn.Linear(14 * 14, 9 * 9),
                nn.Tanh(),
                nn.Linear(9 * 9, 6 * 6),
                nn.Tanh(),
                nn.Linear(6 * 6, 3 * 3),
                nn.Tanh(),
                nn.Linear(3 * 3, 3),   # 压缩成3个特征, 进行 3D 图像可视化
            )
            # 解压
            self.decoder = nn.Sequential(
                nn.Linear(3, 3 * 3),
                nn.Tanh(),
                nn.Linear(3 * 3, 6 * 6),
                nn.Tanh(),
                nn.Linear(6 * 6, 9 * 9),
                nn.Tanh(),
                nn.Linear(9 * 9, 14 * 14),
                nn.Tanh(),
                nn.Linear(14 * 14, 28 * 28),
                nn.Sigmoid()       # 激励函数让输出值在 (0, 1)
            )

        def forward(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return encoded, decoded

    autoencoder = AutoEncoder()
    optimizers = torch.optim.Adam(autoencoder.parameters(), lr=0.01)
    loss_func = nn.MSELoss()

    f, a = plt.subplots(2, N_TEST_IMG, figsize=(5, 2))
    plt.ion()   # continuously plot
    # original data (first row) for viewing
    view_data = train_dataset.train_data[:N_TEST_IMG].view(
        -1, 28*28).type(torch.FloatTensor)/255.
    for i in range(N_TEST_IMG):
        a[0][i].imshow(np.reshape(view_data.data.numpy()
                                  [i], (28, 28)), cmap='gray')
        a[0][i].set_xticks(())
        a[0][i].set_yticks(())
    for epoch in range(EPOCH):
        for step, (x, b_label) in enumerate(train_loader):
            encoded, decoded = autoencoder(x.view(-1, 28 * 28))
            loss = loss_func(decoded, x.view(-1, 28 * 28))
            optimizers.zero_grad()
            loss.backward()
            optimizers.step()
            if step % 100 == 0:
                print('Epoch: ', epoch, '| train loss: %.4f' %
                      loss.data.numpy())

                # plotting decoded image (second row)
                _, decoded_data = autoencoder(view_data)
                for i in range(N_TEST_IMG):
                    a[1][i].clear()
                    a[1][i].imshow(np.reshape(decoded_data.data.numpy()[
                                   i], (28, 28)), cmap='gray')
                    a[1][i].set_xticks(())
                    a[1][i].set_yticks(())
                plt.draw()
                plt.pause(0.05)

    plt.ioff()
    plt.show()

    # visualize in 3D plot
    view_data = train_data.train_data[:200].view(-1, 28*28).type(torch.FloatTensor)/255.
    encoded_data, _ = autoencoder(view_data)
    fig = plt.figure(2); ax = Axes3D(fig)
    X, Y, Z = encoded_data.data[:, 0].numpy(), encoded_data.data[:, 1].numpy(), encoded_data.data[:, 2].numpy()
    values = train_data.train_labels[:200].numpy()
    for x, y, z, s in zip(X, Y, Z, values):
        c = cm.rainbow(int(255*s/9)); ax.text(x, y, z, s, backgroundcolor=c)
    ax.set_xlim(X.min(), X.max()); ax.set_ylim(Y.min(), Y.max()); ax.set_zlim(Z.min(), Z.max())
    plt.show()


if __name__ == "__main__":
    auto_encoder()
