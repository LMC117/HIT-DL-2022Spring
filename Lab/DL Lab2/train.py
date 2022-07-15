import argparse
import torch
import torch.cuda
from alexnet import AlexNet
from caltech101 import Caltech101
from torchvision import transforms
from utils.load_img_data import load_img_data
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 构建输入参数
parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'test'], default='test')
args = parser.parse_args()

# 超参数
batch_size = 64
epoch = 50

# 设定设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 实例化tensorboard writer
tb_writer = SummaryWriter(log_dir="./runs")
tags = ["train_loss", "accuracy", "learning rate"]


def train(train_loader, test_loader, val_loader, model, path="./model/AlexNet_best.ckpt"):
    # 保存最佳acc
    best_acc = 0
    # 保存loss和acc值（画图用）
    # loss_list = []
    # acc_list = []
    # 定义损失函数和优化器
    loss_func = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0005)
    # 开始训练
    for i in range(epoch):
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
        mean_loss = train_loss / len(train_loader.dataset)
        print('Epoch:{}   Train Loss: {:.4f}'.format(i + 1, mean_loss), end='')
        # 每完成一个epoch的训练，都在测试集上测试准确率
        acc = test(val_loader, model)
        print("   Acc:{:.4f}".format(acc))
        # 保存最佳模型
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), path)
        # 将相关参数写入tensorboard
        tb_writer.add_scalar(tags[0], mean_loss, i)
        tb_writer.add_scalar(tags[1], acc, i)
        tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], i)
    print("Finished")
    # 在测试集上验证结果
    test_acc = test(test_loader, model)
    print("Test Set Acc:{:.4f}".format(test_acc))
    # # 绘制loss和acc曲线
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 7))
    # ax1.set_title('Loss')
    # ax2.set_title('Accuracy')
    # ax1.set_ylim([0, 0.5])
    # ax2.set_ylim([0.8, 1])
    # ax1.plot(loss_list, color='#DB4D6D')
    # ax2.plot(acc_list, color='#E9A368')
    # plt.show()


def test(test_loader, model):
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
        acc = correct / len(test_loader.dataset)
        test_loss /= len(test_loader.dataset)
    return acc


def load_model(test_loader, model, path="./model/AlexNet_best.ckpt"):
    """载入模型进行测试"""
    model.load_state_dict(torch.load(path))
    model = model.cuda()
    acc = test(test_loader, model)
    print("Load model | Acc:{:.4f}".format(acc))


def train_model(train_loader, testloader, val_loader, model):
    # 将模型放到GPU上
    model = model.cuda()
    train(train_loader, testloader, val_loader, model)


if __name__ == '__main__':
    train_path, train_label, test_path, test_label, val_path, val_label = load_img_data()

    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),  # 将给定图像随机裁剪为不同的大小和宽高比，然后缩放所裁剪得到的图像为指定大小
            transforms.RandomHorizontalFlip(),  # 以给定的概率（默认为0.5）水平（随机）翻转图像
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
        "test": transforms.Compose([
            transforms.RandomResizedCrop(224),  # 将给定图像随机裁剪为不同的大小和宽高比，然后缩放所裁剪得到的图像为指定大小
            transforms.RandomHorizontalFlip(),  # 以给定的概率（默认为0.5）水平（随机）翻转图像
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
        "val": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    }

    train_dataset = Caltech101(img_path=train_path, img_class=train_label, transform=data_transform["train"])
    test_dataset = Caltech101(img_path=test_path, img_class=test_label, transform=data_transform["test"])
    val_dataset = Caltech101(img_path=val_path, img_class=val_label, transform=data_transform["val"])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # 构建网络
    model = AlexNet(num_classes=101, dropout=0.5)
    model.to(device)
    # 将模型结构写入tensorboard
    init_img = torch.zeros((1, 3, 224, 224), device=device)
    tb_writer.add_graph(model, init_img)

    if args.mode == 'train':
        train_model(train_loader, test_loader, val_loader, model)
    elif args.mode == 'test':
        load_model(test_loader, model, path="./model/AlexNet_best.ckpt")
