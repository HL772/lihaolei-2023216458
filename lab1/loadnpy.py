# 用于显示保存的纹理特征文件

import numpy as np
import os

# 当前运行目录下 result 文件夹里的 texture_features.npy
file_path = os.path.join('result', 'texture_features.npy')

# 2. 加载 .npy 文件
data = np.load(file_path)

# 3. 查看数据信息
print(f"成功加载文件: {file_path}")
print("数据类型:", data.dtype)
print("数据形状:", data.shape)
print("数据内容:", data)
