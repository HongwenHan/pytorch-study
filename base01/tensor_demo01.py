import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F  # nn是神经网络模块
import matplotlib.pyplot as plt
import time


def test01():
    np_data = np.arange(6).reshape((2, 3))
    print(np_data)
    print(torch.from_numpy(np_data))
    print(torch.from_numpy(np_data).numpy())


def test02():
    data01 = [-1, 1, 2, 3, 4, 5]
    tensor = torch.FloatTensor(data01)
    print(tensor)
    print(
        '\nabs',
        '\nnumpy: ', np.abs(data01),
        '\ntorch: ', torch.abs(tensor)
    )
    print(
        '\nsin',
        '\nnumpy: ', np.sin(data01),
        '\ntorch: ', torch.sin(tensor)
    )
    print(
        '\nmean',
        '\nnumpy: ', np.mean(data01),
        '\ntorch: ', torch.mean(tensor)
    )


def test03():
    tensor = torch.FloatTensor([[1, 2], [3, 4]])
    variable = Variable(tensor, requires_grad=True)
    print(tensor, '\n', variable)
    t_out = torch.mean(tensor * tensor)
    v_out = torch.mean(variable * variable)
    print(t_out, '\n', v_out)
    v_out.backward()
    print(variable.grad)
    print(variable.data)
    print(variable.data.numpy())


def test04():
    data = torch.linspace(-5, 5, 200)
    x = Variable(data)
    x_np = x.data.numpy()
    y_relu = F.relu(x).data.numpy()
    y_sigmoid = F.sigmoid(x).data.numpy()
    y_tanh = F.tanh(x).data.numpy()
    y_softplus = F.softplus(x).data.numpy()

    plt.figure(1, figsize=(8, 6))
    plt.subplot(221)
    plt.plot(x_np, y_relu, c='red', label='relu')
    plt.ylim((-1, 5))
    plt.legend(loc='best')

    plt.subplot(222)
    plt.plot(x_np, y_sigmoid, c='red', label='sigmoid')
    plt.ylim((-0.2, 1.2))
    plt.legend(loc='best')

    plt.subplot(223)
    plt.plot(x_np, y_tanh, c='red', label='tanh')
    plt.ylim((-1.2, 1.2))
    plt.legend(loc='best')

    plt.subplot(224)
    plt.plot(x_np, y_softplus, c='red', label='softplus')
    plt.ylim((-0.2, 6))
    plt.legend(loc='best')

    plt.show()


if __name__ == "__main__":
    # test01()
    # test02()
    # test03()
    # test04()
