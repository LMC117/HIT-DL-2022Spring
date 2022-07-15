import os
import json
import jieba
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from custom_dataset import OnlineShopping

filepath = 'data/caltech101/101_ObjectCategories'


def load_img_data(save=True):
    caltech101_class = [i for i in os.listdir(filepath) if os.path.isdir(os.path.join(filepath, i))]  # 获取类别标签
    caltech101_class.remove("BACKGROUND_Google")
    caltech101_class.sort()  # 定序
    class_index = dict((v, k) for k, v in enumerate(caltech101_class))

    if save:
        json_file = json.dumps(dict((v, k) for k, v in class_index.items()), indent=4)
        with open('class.json', 'w') as f:
            f.write(json_file)

    train_path = []
    train_label = []
    test_path = []
    test_label = []
    val_path = []
    val_label = []
    cls_num = []  # 保存每一类的样本数

    suffix = ['.jpg', '.JPG', '.png', '.PNG']

    # 遍历文件夹
    for cls in caltech101_class:
        cls_path = os.path.join(filepath, cls)
        imgs = [os.path.join(cls_path, i) for i in os.listdir(cls_path) if
                os.path.splitext(i)[-1] in suffix]
        img_class = class_index[cls]
        cls_num.append(len(imgs))

        # 划分训练集，测试集，开发集
        img_train, others = train_test_split(imgs, test_size=0.2, shuffle=True)
        img_test, img_val = train_test_split(others, test_size=0.5, shuffle=True)

        for img in imgs:
            if img in img_train:
                train_path.append(img)
                train_label.append(img_class)
            elif img in img_test:
                test_path.append(img)
                test_label.append(img_class)
            elif img in img_val:
                val_path.append(img)
                val_label.append(img_class)

    return train_path, train_label, test_path, test_label, val_path, val_label


def online_shopping_loader(path="data/online_shopping_10_cats.csv", save=True, batch_size=64, shuffle=True):
    """读取online shopping数据集"""
    # 读取csv
    csv = pd.read_csv(path, low_memory=False)  # 防止弹出警告
    csv_df = pd.DataFrame(csv)
    del csv_df["label"]

    # 处理标签
    data_class = {
        "书籍": 0, "平板": 1, "手机": 2, "水果": 3, "洗发水": 4, "热水器": 5, "蒙牛": 6, "衣服": 7, "计算机": 8, "酒店": 9
    }
    csv_df["cat"] = csv_df["cat"].map(data_class)

    # 将标签保存为json文件
    if save:
        json_file = json.dumps(dict((v, k) for k, v in data_class.items()), indent=4)
        with open('data/online_shopping.json', 'w', encoding="utf-8") as f:
            f.write(json_file)

    # 切分数据集
    train_text = []
    train_label = []
    test_text = []
    test_label = []
    val_text = []
    val_label = []
    count = 0
    # 统计词频（仅训练集）与标签数
    vocab_dict = {}
    label_dict = {}
    for _, row in csv_df.iterrows():
        count += 1
        # 对文本做分词处理
        label = row['cat']
        text = row['review']
        split_row = jieba.lcut(text)
        if count % 5 == 0:  # 测试集
            test_text.append(split_row)
            test_label.append(label)
        elif count % 5 == 4:  # 验证集
            val_text.append(split_row)
            val_label.append(label)
        else:  # 训练集
            train_text.append(split_row)
            train_label.append(label)
            # 加入词频与标签数统计
            for word in split_row:
                if word == '\ufeff' or '':
                    pass
                elif word in vocab_dict:
                    vocab_dict[word] += 1
                else:
                    vocab_dict[word] = 1
            if label in label_dict:
                label_dict[label] += 1
            else:
                label_dict[label] = 1

    # 将数据放入dataloader
    train_dataset = OnlineShopping(train_text, train_label)
    test_dataset = OnlineShopping(test_text, test_label)
    val_dataset = OnlineShopping(val_text, val_label)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)

    return train_loader, test_loader, val_loader


def jena_loader(path="data/online_shopping_10_cats.csv", save=True, batch_size=64, shuffle=True):
    """读取jena_climate数据集"""
    # TODO: 设计如何读取这个数据集(哪个是label？时间是否需要做sin或cos变换？)
