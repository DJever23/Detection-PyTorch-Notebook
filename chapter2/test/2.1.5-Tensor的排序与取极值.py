import torch

'''
比较重要的是排序函数sort()，选择沿着指定维度进行排序，返
回排序后的Tensor及对应的索引位置。max()与min()函数则是沿着指
定维度选择最大与最小元素，返回该元素及对应的索引位置。
'''
a = torch.randn(3, 3)
print('a', a)
# 按照第0维即按行排序，每一列进行比较，True代表降序，False代表升序
print('a.sort(0, True)[0]', a.sort(0, True)[0])  # 返回排序后的Tensor
print('a.sort(0, True)[1]', a.sort(0, True)[1])  # 返回对应的索引位置
# 按照第0维即按行选取最大值，即将每一列的最大值选取出来
print('a.max(0)', a.max(0))
print('a.max(1)', a.max(1))