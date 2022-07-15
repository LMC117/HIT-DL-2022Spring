import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# 超参数
batch_size = 128
epoch = 40
device = torch.device('cuda')

# 加载MNIST数据集
transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))])
train_data = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()  # 调用父类的构造函数
        layer_1 = 512
        layer_2 = 512
        self.fc1 = nn.Linear(28 * 28, layer_1)
        self.fc2 = nn.Linear(layer_1, layer_2)
        self.fc3 = nn.Linear(layer_2, 10)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.view(-1, 28 * 28)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        x = F.log_softmax(self.fc3(x), dim=1)
        return x


def train(model):
    # 保存最佳acc
    best_acc = 0
    # 保存loss和acc值（画图用）
    loss_list = []
    acc_list = []
    # 定义损失函数和优化器
    loss_f = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    # 开始训练
    for i in range(epoch):
        train_loss = 0.0
        for data, target in train_loader:
            optimizer.zero_grad()
            # 部署数据至GPU并计算输出
            data, target = data.to(device), target.to(device)
            output = model(data)
            # 计算损失
            loss = loss_f(output, target)
            # 反向传播
            loss.backward()
            # 将参数更新至网络中
            optimizer.step()
            # 计算损失
            train_loss += loss.item() * data.size(0)
        train_loss /= len(train_loader.dataset)
        print('Epoch:{}   Training Loss: {:.4f}'.format(i + 1, train_loss), end='')
        loss_list.append(train_loss)
        # 每完成一个epoch的训练，都在测试集上测试准确率
        acc = test(model)
        acc_list.append(acc)
        print("   Acc:{:.4f}".format(acc))
        # 保存最佳模型
        if acc > best_acc:
            best_acc = acc
            path = "./model/MLP_best.ckpt"
            torch.save(model.state_dict(), path)
    print("Finished")
    # 绘制loss和acc曲线
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 7))
    ax1.set_title('Loss')
    ax2.set_title('Accuracy')
    ax1.set_ylim([0, 0.5])
    ax2.set_ylim([0.8, 1])
    ax1.plot(loss_list, color='#DB4D6D')
    ax2.plot(acc_list, color='#E9A368')
    plt.show()


def test(model):
    # 初始化参量，定义损失函数
    correct = 0.0
    test_loss = 0.0
    loss_f = torch.nn.CrossEntropyLoss().cuda()
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
            # 找到概率值最大的下标
            pred = output.argmax(dim=1)
            # 累计正确率
            correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        acc = correct / len(test_loader.dataset)
    return acc


def load_model(path="./model/MLP_best.ckpt"):
    """载入模型进行测试"""
    MLP_model = MLP()
    MLP_model.load_state_dict(torch.load(path))
    MLP_model = MLP_model.cuda()
    acc = test(MLP_model)
    print("Load model | Acc:{:.4f}".format(acc))


def train_model():
    MLP_model = MLP()
    # 将模型放到GPU上
    MLP_model = MLP_model.cuda()
    train(MLP_model)


if __name__ == "__main__":
    # train_model()
    load_model(path="./model/MLP_best.ckpt")
