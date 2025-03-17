import h5py

# 打开HDF5文件
file_path = r'C:\Users\Elysia\Desktop\typhoonpredict\MINI-TCSA.h5'
file = h5py.File(file_path, 'r')    # 'r'表示为只读模式

# 列出文件中所有的数据集
print("Datasets in the file:")
print(list(file.keys()))    # 显示文件中的顶层对象，例如info matrix

# 访问特定的数据集
dataset = file['matrix']  # 将 'matrix' 替换为实际存在的数据集名称
data = dataset[:]

# 关闭文件
file.close()

print("Data in the dataset:")
print(data)

import h5py
import pandas as pd
import matplotlib.pyplot as plt

# 打开HDF5文件
file_path = r'C:\Users\Elysia\Desktop\typhoonpredict\MINI-TCSA.h5'
file = h5py.File(file_path, 'r')

# 访问特定的数据集
dataset = file['matrix']  # 使用实际数据集名称
data = dataset[:]

# 关闭文件
file.close()

# 将三维数组转换为二维数组
flattened_data = data.reshape(-1, data.shape[-1])

# 转换为DataFrame
# DataFrame:二维表格型数据结构
df = pd.DataFrame(flattened_data, columns=['Temperature', 'Humidity', 'Pressure', 'Wind Speed'])

# 打印DataFrame
print(df)

# 可视化示例
df.plot(subplots=True, figsize=(10, 8))
plt.show()
