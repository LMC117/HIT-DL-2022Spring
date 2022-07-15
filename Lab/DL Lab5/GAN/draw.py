import matplotlib.pyplot as plt
import numpy as np
import torch
import imageio
import os
import re

# 使用 cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 画散点图
def draw_scatter(data, color, label):
    plt.xlim(-0.25, 1.5)
    plt.ylim(0, 0.8)
    plt.title('Result')
    plt.xlabel("X", fontsize=14)
    plt.ylabel("Y", fontsize=14)
    plt.scatter(data[:, 0], data[:, 1], c=color, s=6, label=label)
    plt.legend(loc="upper left")


def sort_key(s):
    """自定义sort key"""
    c = re.findall('\d+', s)[0]  # 匹配开头数字序号
    return int(c)


# 合成动图
def img_show(model):
    imgs = []
    temp = os.listdir("./result/{}".format(model))
    img_name = [x for x in temp if 'png' in x]
    img_name.sort(key=sort_key)
    for fileName in img_name:
        imgs.append(imageio.imread(os.path.join('./result/gan', fileName)))
    imageio.mimsave("./gif/{}.gif".format(model), imgs, fps=4)
    print("Finished generating gif for {}".format(model))


if __name__ == '__main__':
    img_show("gan")
    img_show("wgan")
    img_show("wgan_gp")
