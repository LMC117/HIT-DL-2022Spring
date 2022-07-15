import argparse
import numpy as np
import torch
import torch.cuda
from OnlineShoppingNet import RNN, LSTM, Bi_LSTM, GRU
from JenaNet import JenaNet
from torch.utils.tensorboard import SummaryWriter
from utils.load_online_shopping import load_online_shopping
from utils.load_jena import load_jena
from sklearn.metrics import classification_report

# 构建输入参数
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', default=60)
parser.add_argument('--max_len', default=50)
parser.add_argument('--batch_size', default=64)
parser.add_argument('--hidden_dim', default=256)
parser.add_argument('--mode', choices=['train', 'test'], default='train')
parser.add_argument('--model', choices=['RNN', 'LSTM', 'GRU', 'Bi_LSTM'], default='Bi_LSTM')
parser.add_argument('--npz_path', default='data/online_shopping/w2v.npz')
parser.add_argument('--best_save_path', default='model/os_BiLSTM.ckpt')

args = parser.parse_args()

# 超参数
max_len = args.max_len
batch_size = args.batch_size
epoch = args.epoch

# 设定设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 实例化tensorboard writer
tb_writer = SummaryWriter(log_dir="./runs")
tags = ["train_loss", "val_loss", "accuracy", "learning rate"]

# 数据标签
data_class = {
    "书籍": 0, "平板": 1, "手机": 2, "水果": 3, "洗发水": 4, "热水器": 5, "蒙牛": 6, "衣服": 7, "计算机": 8, "酒店": 9
}


def train(train_loader, test_loader, val_loader, model, path=args.best_save_path):
    print("Begin training...")
    # 保存最佳acc
    best_acc = 0
    # 保存loss和acc值（画图用）
    # loss_list = []
    # acc_list = []
    # 定义损失函数和优化器
    loss_func = torch.nn.CrossEntropyLoss().cuda()
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
        acc, val_loss, _, __ = test(val_loader, model)
        # 打印输出
        print('Epoch:{}/{}   '.format(i + 1, epoch), end='')
        print('Train Loss: {:.4f}   Val Loss: {:.4f}   '.format(train_loss, val_loss), end='')
        print('Acc:{:.4f}'.format(acc))
        # 将相关参数写入tensorboard
        tb_writer.add_scalar(tags[0], train_loss, i)
        tb_writer.add_scalar(tags[1], val_loss, i)
        tb_writer.add_scalar(tags[2], acc, i)
        tb_writer.add_scalar(tags[3], optimizer.param_groups[0]["lr"], i)
        # 保存最佳模型
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), path)
    print("Finished {} epoch".format(epoch))
    # 在测试集上验证结果
    test_acc, test_loss, y_true, y_pred = test(test_loader, model)
    print("Test Acc:{:.4f}   Test Loss:{:.4f}".format(test_acc, test_loss))
    print("Online shopping evaluation results:")
    # print(precision_recall_fscore_support(y_true, y_pred, average=None, labels=[i for i in range(0, 10)]))
    print(classification_report(y_true, y_pred, labels=[i for i in range(0, 10)],
                                target_names=["书籍", "平板", "手机", "水果", "洗发水", "热水器", "蒙牛", "衣服", "计算机", "酒店"]))


def test(test_loader, model):
    # 初始化参量，定义损失函数
    correct = 0.0
    test_loss = 0.0
    y_true = np.zeros((1,), dtype='int')
    y_pred = np.zeros((1,), dtype='int')
    loss_f = torch.nn.CrossEntropyLoss().cuda()
    # 模型验证
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            y_true = np.concatenate((y_true, target.numpy()))
            # 部署数据
            data, target = data.to(device), target.to(device)
            # 测试数据
            output = model(data)
            # 计算测试损失
            test_loss += loss_f(output, target).item() * data.size(0)
            # 找到概率值最大的下标
            pred = output.argmax(dim=1)
            y_pred = np.concatenate((y_pred, pred.cpu().numpy()))
            # 累计正确率
            correct += pred.eq(target.view_as(pred)).sum().item()
        acc = correct / len(test_loader.dataset)
        test_loss /= len(test_loader.dataset)
        y_true = y_true[1:]
        y_pred = y_pred[1:]
    return acc, test_loss, y_true, y_pred


def load_model(test_loader, model, path=args.best_save_path):
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
    loaders = load_online_shopping(path="data/online_shopping", max_len=max_len, batch_size=batch_size,
                                   shuffle=True)
    train_loader = loaders[0]
    val_loader = loaders[1]
    test_loader = loaders[2]

    # 读取词向量矩阵
    print("Load .npz file...")
    loaded = np.load(args.npz_path)
    embeddings = torch.FloatTensor(loaded['embeddings'])
    embedding_dim = embeddings.shape[1]
    a = embeddings.dim()
    print("- finished")

    # 构建网络
    if args.model == 'RNN':
        model = RNN(embeddings, embedding_dim, args.hidden_dim, len(data_class))
    elif args.model == 'LSTM':
        model = LSTM(embeddings, embedding_dim, args.hidden_dim, len(data_class))
    elif args.model == 'GRU':
        model = GRU(embeddings, embedding_dim, args.hidden_dim, len(data_class))
    elif args.model == 'Bi_LSTM':
        model = Bi_LSTM(embeddings, embedding_dim, args.hidden_dim, len(data_class))
    model.to(device)
    # 将模型结构写入tensorboard
    # init_img = torch.zeros((1, 100, 300), device=device)
    # tb_writer.add_graph(model, init_img)

    if args.mode == 'train':
        train_model(train_loader, test_loader, val_loader, model)
    elif args.mode == 'test':
        load_model(test_loader, model, path="./model/os_{}.ckpt".format(args.model))
