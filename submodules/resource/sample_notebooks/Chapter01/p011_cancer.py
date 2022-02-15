from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = load_breast_cancer()
cancers = load_breast_cancer()
X = cancers.data       #获取特征值
Y = cancers.target     #获取标签
print(X.shape)         #查看特征形状
print(Y.shape)         #查看标签形状

print(X)
print(Y)

#dict_keys(['data', 'target', 'target_names', 'DESCR', 'feature_names', 'filename'])
print(cancers.keys())


print(data.DESCR)  #查看数据集描述

print('特征名称')
print(data.feature_names)  # 特征名
print('分类名称')
print(data.target_names)  # 标签类别名

# 注意返回值： 训练集train，x_train，y_train，测试集test，x_test，y_test
# x_train为训练集的特征值，y_train为训练集的目标值，x_test为测试集的特征值，y_test为测试集的目标值
# 注意，接收参数的顺序固定
# 训练集占80%，测试集占20%
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
print('训练集的特征值和目标值：', x_train, y_train)
print('测试集的特征值和目标值：', x_test, y_test)



X = data.data
y = data.target

print(X.shape)
X = X[:, :10]
print(X.shape)
