import os

# 数据集相关
dataset_path='./data/EAST'
train_path=os.path.join(dataset_path,'train')
test_path=os.path.join(dataset_path,'test')

# 图片输入格式
imgW=120
imgH=32

# 模型相关
save_model_path='./pth/CRNN'

# 训练超参数
epoch=600
batch=50
workers=0
lr=1e-3

result_path='./result/CRNN'