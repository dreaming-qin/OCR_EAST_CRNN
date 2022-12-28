import os

# 数据集相关
dataset_path='./data/CRNN'
train_path=os.path.join(dataset_path,'train')
test_path=os.path.join(dataset_path,'test')

# 图片输入格式
r'''是否保证比率。论文中是H为32, W尽量保证同等比率缩放但是不少于100像素, 
这里的设置是不少于120像素'''
keep_ratio=False
imgW=120
imgH=32

# pth相关
pth_path='./pth/CRNN'
# 结果存放路径
result_path='./result/CRNN'

# 训练超参数
epoch=600
batch=50
workers=0
lr=1e-3


# crnn相关超参数
hidden_size=256