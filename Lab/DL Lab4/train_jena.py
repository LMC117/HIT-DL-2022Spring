import argparse
import numpy as np
import torch
import torch.cuda
from OnlineShoppingNet import RNN
from JenaNet import JenaNet
from torch.utils.tensorboard import SummaryWriter
from utils.load_online_shopping import load_online_shopping
from utils.load_jena import load_jena
from sklearn.metrics import precision_recall_fscore_support, classification_report
import matplotlib.pyplot as plt

# 构建输入参数
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', default=100)
parser.add_argument('--batch_size', default=16)
parser.add_argument('--hidden_dim', default=256)
parser.add_argument('--mode', choices=['train', 'test'], default='train')
parser.add_argument('--best_save_path', default='model/Jena_best(GRU).ckpt')

args = parser.parse_args()

# 超参数
batch_size = args.batch_size
epoch = args.epoch

# 设定设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 实例化tensorboard writer
tb_writer = SummaryWriter(log_dir="./runs_jena")
tags = ["train_loss", "val_loss", "learning rate"]

jena_input_dim = 720
jena_embd_dim = 5
jena_out_dim = 288


def train_jena(train_loader, test_loader, val_loader, model, path=args.best_save_path):
    print("Begin training...")
    best_loss = 1000000000
    # 定义损失函数和优化器
    loss_func = torch.nn.MSELoss().cuda()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)
    # 开始训练
    for i in range(epoch):
        model.train()
        train_loss = 0.0
        for data, target in train_loader:
            optimizer.zero_grad()
            # 部署数据至GPU并计算输出
            data, target = data.to(device), target.to(device)
            output = model(data)
            # 计算损失
            loss = loss_func(output, target)
            # 反向传播
            loss.backward()
            # 将参数更新至网络中
            optimizer.step()
            # 计算损失
            train_loss += loss.item() * data.size(0)
        train_loss /= len(train_loader.dataset)
        # 每完成一个epoch的训练，都在验证集上测试准确率
        val_loss, _, __ = test_jena(val_loader, model)
        # 打印输出
        print('Epoch:{}/{}   '.format(i + 1, epoch), end='')
        print('Train Loss: {:.4f}   Val Loss: {:.4f}   '.format(train_loss, val_loss))
        # 将相关参数写入tensorboard
        tb_writer.add_scalar(tags[0], train_loss, i)
        tb_writer.add_scalar(tags[1], val_loss, i)
        tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], i)
        # 保存最佳模型
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), path)
    print("Finished {} epoch".format(epoch))
    # 在测试集上验证结果
    test_loss, y_true, y_pred = test_jena(test_loader, model)
    print("Test Loss:{:.4f}".format(test_loss))
    plot_results(y_true, y_pred)


def test_jena(test_loader, model):
    # 初始化参量，定义损失函数
    correct = 0.0
    test_loss = 0.0
    loss_f = torch.nn.MSELoss().cuda()
    # 模型验证
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            # 部署数据
            data, target = data.to(device), target.to(device)
            # 测试数据
            output = model(data)
            # 计算测试损失
            test_loss += loss_f(output, target).item() * data.size(0)
            # 获取某段时间的准确温度与预测温度
            y_true = target.cpu()[0, :]
            y_pred = output.cpu()[0, :]
        test_loss /= len(test_loader.dataset)
    return test_loss, y_true, y_pred


def load_model_jena(test_loader, model, path=args.best_save_path):
    """载入模型进行测试"""
    model.load_state_dict(torch.load(path))
    model = model.cuda()
    loss, y_true, y_pred = test_jena(test_loader, model)
    print("Load model | Loss:{:.4f}".format(loss))
    plot_results(y_true, y_pred)


def train_model_jena(train_loader, testloader, val_loader, model):
    # 将模型放到GPU上
    model = model.cuda()
    train_jena(train_loader, testloader, val_loader, model)


def plot_results(y_true, y_pred):
    """打印误差，画图"""
    y_true = y_true.tolist()
    y_pred = y_pred.tolist()
    deviation = [y_true[i] - y_pred[i] for i in range(len(y_true))]
    print("Avg error: {:.4f}".format(np.mean(deviation)))
    print("Median error: {:.4f}".format(np.median(deviation)))
    print("Figure shows the prediction on certain period of test set.")
    x = [i for i in range(len(y_true))]
    fig, ax = plt.subplots()  # 创建图实例
    ax.plot(x, y_true, label='true', color='#EB7A77')
    ax.plot(x, y_pred, label='pred', color='#6A8372')
    ax.set_xlabel('time')
    ax.set_ylabel('temperature')
    ax.set_ylim((-3, 3))
    ax.legend()
    plt.show()


if __name__ == '__main__':
    train_loader, test_loader = load_jena(path="data/jena_climate/jena_climate_2009_2016.csv", batch_size=batch_size,
                                          shuffle=True)
    val_loader = test_loader

    # 构建网络
    model = JenaNet(jena_embd_dim, args.hidden_dim, jena_out_dim)
    model.to(device)

    if args.mode == 'train':
        train_model_jena(train_loader, test_loader, val_loader, model)
    elif args.mode == 'test':
        load_model_jena(test_loader, model, path="./model/RNN_best.ckpt")
