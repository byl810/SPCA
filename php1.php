https://blog.csdn.net/weixin_40903057/article/details/95314169
https://blog.csdn.net/seanzhen52/article/details/73740615
稀疏主成分分析案例
https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.SparsePCA.html#sklearn.decomposition.SparsePCA
官网
https://blog.csdn.net/Forever_pupils/article/details/88572281
给你推荐一个讲稀疏原理的

from sklearn.decomposition import PCA, SparsePCA
# 主成分分析算法包和稀疏主成分分析
from sklearn.decomposition import sparse_pca
# 稀疏主成分分析算法包
import numpy as np
import xlrd

# 读取xlsx文件
# data_spca = np.load('F:\PythonCodes/apple.project\Apple.xlsx')
data_spca = np.loadtxt('F:\PythonCodes/apple.project\Apple1.txt')
data_spca1 = data_spca.T#对数据集转置
print(data_spca1)# 输出转置后的数据矩阵11*29
print(np.shape(data_spca1))#输出数据集的类型（11*29）
print(111111111111111111111)

# 加载PCA算法，降维后为主成分数为11
pca_op = PCA(n_components=11)
pca_op.fit(data_spca)#对数据进行降维，
XX = pca_op.fit_transform(data_spca)#对数据进行降维，得到参数
print(XX)
print(np.shape(XX))#(29*11)
# a为3x3的数据
# X: array-like, shape (n_samples, n_features)
#print(pca_op.fit_transform(data_spca))
# print(2222222222222222222222222)
print(pca_op.explained_variance_) # 在PCA中有特征值的输出啊
print(sum(pca_op.explained_variance_ratio_)) #累计贡献率
print(            33333333333333333333333333333)

# 加载SPCA算法，降维后为主成分数为4
spca_op = SparsePCA(n_components=5)
spca_op.fit(data_spca)#得到参数
X_trans = spca_op.transform(data_spca)# transform（X自己）数据的最小二乘投影到稀疏分量上。
print(X_trans)
print(X_trans.shape)
print(np.mean(spca_op.components_ == 0))#稀疏性

# spca_value = spca_op.fit_transform(data_spca1)#对原始数据进行spca 降维分析，保存
# print(spca_value)
# print(np.shape(spca_value))
print(44444444444444444444)
spca_co = spca_op.components_# components_ 从数组中提取的稀疏成分
print(spca_co)
print(np.shape(spca_co))
print(555555555555555555555555555)
print(np.mean(spca_op.components_))
np.savetxt('a.txt',X_trans)
np.savetxt('b.txt',spca_co)#存为文本文件

""""""""""""""""
from sklearn.decomposition import PCA
from sklearn.decomposition import sparse_pca
import numpy as np
data_spca = np.loadtxt('F:/作业/主成分分析/vxq0qvyy/TRD.txt')
data_spca1 = data_spca.T
print(data_spca1)
print(np.shape(data_spca1))
pca_op = PCA(n_components=4)
pca_op.fit(data_spca)
# a为3x3的数据
# X: array-like, shape (n_samples, n_features)
#print(pca_op.fit_transform(data_spca))
print(pca_op.explained_variance_) #在PCA中有特征值的输出啊
print(sum(pca_op.explained_variance_ratio_)) #累计贡献率
spca_op = SparsePCA(n_components=4)
spca


import numpy as np
from sklearn.datasets import make_friedman1
from sklearn.decomposition import SparsePCA
data_spca = np.loadtxt('C:\\Users\\ASUS\\shuju.txt')
print(np.shape(data_spca))

data_spca, _ = make_friedman1(n_samples=45, n_features=12,random_state = 0)
transformer = SparsePCA(n_components=12,alpha=3.12,random_state = 0)
transformer.fit(data_spca)
X_transformed = transformer.transform(data_spca)
print(X_transformed.shape)
print (transformer)
a = (transformer.components_)
print(a)
print(np.shape(a))
# most values in the components_ are zero (sparsity)
print(np.mean(transformer.components_ == 0))
np.savetxt('b.txt',a)


注释：
alpha：惩罚系数、
transformer = SparsePCA(n_components=12,alpha=3.12,random_state = 0)这个是提取的稀疏主元数，惩罚系数
data_spca, _ = make_friedman1(n_samples=45, n_features=12,random_state = 0)  这个是导入数据集的维度，组数
结果：
Value of k Corresponding Variable 
1 Storage temperature 
2 Storage temperature, Total soluble solids 
3 Storage temperature,Total soluble solids,Starch 
4 Storage temperature,Total soluble solids,Starch, Reducing ascorbic acid 
5 Storage temperature,Total soluble solids,Starch, Reducing ascorbic acid, Titratable acidity 
6 Storage temperature,Total soluble solids,Starch, Reducing ascorbic acid, Titratable acidity, Color change △ E 
7 Storage temperature,Total soluble solids,Starch, Reducing ascorbic acid, Titratable acidity, Color change △ E, Loss of weight 
8 Storage temperature,Total soluble solids,Starch, Reducing ascorbic acid, Titratable acidity, Color change △ E, Loss of weight,Color change b *, 
9 Storage temperature,Total soluble solids,Starch, Reducing ascorbic acid, Titratable acidity, Color change △ E, Loss of weight,Color change b *,Firmness 
10 Storage temperature,Total soluble solids,Starch, Reducing ascorbic acid, Titratable acidity, Color change △ E, Loss of weight,Color change b *,Firmness, Color change a * 
11 Storage temperature,Total soluble solids,Starch, Reducing ascorbic acid, Titratable acidity, Color change △ E, Loss of weight,Color change b *,Firmness, Color change a *, Color change L * 
12 Storage temperature,Total soluble solids,Starch, Reducing ascorbic acid, Titratable acidity, Color change △ E, Loss of weight,Color change b *,Firmness, Color change a *, Color change L *, Color change c 
这是稀疏主成分得到的结果，一共12组变量，所以就用这这些变量作为神经网络的输入变量，看哪些变量得到的结果比较好
