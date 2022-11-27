import os
# 数据集相关
dataset_path='./data/EAST'
train_path=os.path.join(dataset_path,'train')
test_path=os.path.join(dataset_path,'test')

# 图片输入格式
imgW=512
imgH=512

# 模型相关
pth_path='./pth/EAST'

# 训练超参数
epoch=600
batch=2
workers=0
lr=1e-3

result_path='./result/EAST'