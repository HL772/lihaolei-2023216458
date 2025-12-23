# 用于显示保存的纹理特征文件

import numpy as np

# 1. 加载 .npy 文件
data = np.load('texture_features.npy')

# 2. 查看数据信息
print("数据类型:", data.dtype)  # 例如: int32
print("数据形状:", data.shape)  # 例如: (256,) 表示有256个数字的一维数组
print("数据内容:", data)        # 打印具体的数值