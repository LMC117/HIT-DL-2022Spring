import os
import json
import jieba
import pandas as pd


def build_words(path="../data/online_shopping/online_shopping_10_cats.csv", save_json=True,
                save_path="../data/online_shopping"):
    """读取online shopping数据集"""
    # 读取csv
    csv = pd.read_csv(path, low_memory=False, encoding='utf-8-sig')  # 防止弹出警告与出现\ufeff
    csv_df = pd.DataFrame(csv)
    del csv_df["label"]

    # 处理标签
    data_class = {
        "书籍": 0, "平板": 1, "手机": 2, "水果": 3, "洗发水": 4, "热水器": 5, "蒙牛": 6, "衣服": 7, "计算机": 8, "酒店": 9
    }
    csv_df["cat"] = csv_df["cat"].map(data_class)

    # 将标签保存为json文件
    if save_json:
        json_file = json.dumps(dict((v, k) for k, v in data_class.items()), indent=4)
        with open('../data/online_shopping/labels.json', 'w', encoding="utf-8") as f:
            f.write(json_file)

    # 切分数据集
    train_text = []
    train_label = []
    test_text = []
    test_label = []
    val_text = []
    val_label = []
    count = 0

    # 统计参数
    # csv_df['len'] = csv_df.apply(lambda row: len(str(row['review'])), axis=1)
    # max_len = csv_df.max()[1]  # 2876
    # avg_len = csv_df.mean()[1]  # 58.414
    for _, row in csv_df.iterrows():
        count += 1
        # 对文本做分词处理
        label = row['cat']
        text = str(row['review'])
        if text.strip() != '':
            split_row = [' '.join(jieba.cut(text.strip()))]
        if count % 5 == 0:  # 测试集
            test_text.append(split_row)
            test_label.append(label)
        elif count % 5 == 4:  # 验证集
            val_text.append(split_row)
            val_label.append(label)
        else:  # 训练集
            train_text.append(split_row)
            train_label.append(label)

    # 保存结果
    save_results(train_text, train_label, mode='train')
    save_results(test_text, test_label, mode='test')
    save_results(val_text, val_label, mode='val')

    return


def save_results(text, label, mode, path="../data/online_shopping"):
    with open(os.path.join(path, '{}.words.txt'.format(mode)), 'w', encoding='utf-8') as w:
        for i in range(len(text)):
            w.write('{},{}\n'.format(label[i], ''.join(text[i])))


if __name__ == '__main__':
    build_words(path="../data/online_shopping/online_shopping_10_cats.csv")
