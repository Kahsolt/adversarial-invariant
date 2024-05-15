import numpy as np

# 初始化为空的NumPy数组
all_arrays = None

# 假设我们要添加5个大小为4的数组
for _ in range(5):
    new_array = np.random.rand(4)  # 创建一个大小为4的随机数组
    if all_arrays is None:
        all_arrays = new_array
    else:
        all_arrays = np.vstack((all_arrays, new_array))  # 堆叠新数组
    #breakpoint()

# 计算平均值，axis=0 表示沿列计算平均值（即每个位置的平均）
breakpoint()
mean_values = np.mean(all_arrays, axis=0)

print("平均值:", mean_values)
