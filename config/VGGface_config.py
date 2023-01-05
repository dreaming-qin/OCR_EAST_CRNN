import os
# 数据集相关
dataset_path='./data/VGGface'
train_path=os.path.join(dataset_path,'train')
test_path=os.path.join(dataset_path,'test')

# 模型相关
pth_path='./pth/VGGface'

# 图片输入大小
img_size=(224,224)

# 输出结果存储
result_path='./result/VGGface'

# 置信度阈值
# 欧式距离置信度
euc=0.62
# 余弦相似度置信度s
cos=0.82