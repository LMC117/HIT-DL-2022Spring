from utils.load_data import online_shopping_loader
from utils.load_online_shopping import load_online_shopping
from utils.load_jena import load_jena

# 测试online shopping数据的读取
# train_loader, test_loader, val_loader = online_shopping_loader()
# for text, label in train_loader:
#     count = 0

# loaders = load_online_shopping()
# train_loader = loaders[0]
# val_loader = loaders[1]
# test_loader = loaders[2]
# for text, label in train_loader:
#     count = 0

# count = 0
# for i in range(1, 1):
#     count += 1
# print(count)

load_jena()
