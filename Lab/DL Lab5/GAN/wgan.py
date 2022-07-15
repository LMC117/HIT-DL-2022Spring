import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import draw
import argparse
from scipy.io import loadmat

# 构建输入参数
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=256)
parser.add_argument('--epoch', default=320)
parser.add_argument('--lr', default=0.00005)
parser.add_argument('--train_size', default=7000, help='训练集的大小')
parser.add_argument('--input_size', default=2, help='生成器输入的噪声维度')
parser.add_argument('--clamp', default=0.1, help='WGAN的权值限制')
args = parser.parse_args()

# 超参数
model_name = 'wgan'
batch_size = args.batch_size
epoch = args.epoch
train_size = args.train_size
input_size = args.input_size
lr = args.lr
clamp = args.clamp

# 设定设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载数据集
data = loadmat("./points.mat")['xx']
np.random.shuffle(data)
# 拆分数据集
train_set = data[:train_size]
test_set = data[train_size:]


# 生成网络
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 256),
            nn.BatchNorm1d(256, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        return self.net(x)


# 判别器
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.net(x)


# 定义网络
D = Discriminator().to(device)
G = Generator().to(device)
# 定义优化器
optimizer_D = torch.optim.RMSprop(D.parameters(), lr=lr)
optimizer_G = torch.optim.RMSprop(G.parameters(), lr=lr)


def train():
    for ep in range(epoch):
        loss_D = 0
        loss_G = 0
        for i in range(int(train_size / batch_size)):
            real = torch.from_numpy(train_set[i * batch_size: (i + 1) * batch_size]).float().to(device)
            G_input = torch.randn(batch_size, input_size).to(device)
            fake = G(G_input)
            # 计算discriminator判别fake的概率
            outFake = D(fake)
            # 计算generator的loss（不使用log）
            loss_G = -torch.mean(outFake)
            # 更新生成器
            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()
            # 计算real的概率
            outReal = D(real)
            # 重新计算
            outFake = D(fake.detach())
            # 计算discriminator的loss（不使用log）
            loss_D = torch.mean(outFake - outReal)
            # 更新判别器
            optimizer_D.zero_grad()
            loss_D.backward()
            optimizer_D.step()
            # 限制D中参数数值
            for p in D.parameters():
                p.data.clamp_(-clamp, clamp)
        print("epoch: {:d}     d_loss: {:.3f}     g_loss: {:.3f} ".format(ep + 1, loss_D, loss_G))
        if (ep + 1) % 10 == 0:
            test_generator()
            plt.savefig(os.path.join('./result', model_name, str(ep + 1)))
            plt.cla()


def test_generator():
    """测试generator"""
    G_input = torch.randn(1200, input_size).to(device)
    G_out = G(G_input)
    G_data = np.array(G_out.cpu().data)
    # 画出测试集的点分布和生成器输出的点分布
    draw.draw_scatter(test_set, '#6699A1', 'original')
    draw.draw_scatter(G_data, '#F17C67', 'generated')
    return


if __name__ == '__main__':
    train()
    test_generator()
