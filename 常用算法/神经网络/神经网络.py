#MLPClassifier 只支持交叉熵损失函数，通过运行 predict_proba 方法进行概率估计
import numpy as np
import matplotlib.pyplot as plt
#多层感知器
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
data = [
[-0.017612, 14.053064, 0],[-1.395634, 4.662541, 1],[-0.752157, 6.53862, 0],[-1.322371, 7.152853, 0],[0.423363, 11.054677, 0],
    [0.406704, 7.067335, 1],[0.667394, 12.741452, 0],[-2.46015, 6.866805, 1],[0.569411, 9.548755, 0],[-0.026632, 10.427743, 0],
    [0.850433, 6.920334, 1],[1.347183, 13.1755, 0],[1.176813, 3.16702, 1],[-1.781871, 9.097953, 0],[-0.566606, 5.749003, 1],
    [0.931635, 1.589505, 1],[-0.024205, 6.151823, 1],[-0.036453, 2.690988, 1],[-0.196949, 0.444165, 1],[1.014459, 5.754399, 1],
    [1.985298, 3.230619, 1],[-1.693453, -0.55754, 1],[-0.576525, 11.778922, 0],[-0.346811, -1.67873, 1],[-2.124484, 2.672471, 1],
    [1.217916, 9.597015, 0],[-0.733928, 9.098687, 0],[1.416614, 9.619232, 0],[1.38861, 9.341997, 0],[0.317029, 14.739025, 0]
]

dataMat=np.array(data)
X=dataMat[:,0:2]
y=dataMat[:,2]
#数据标准化
scaler=StandardScaler()
scaler.fit(X)
X=scaler.transform(X)
#神经网络分类
# solver='lbfgs',  MLP的求解方法：L-BFGS 在小数据上表现较好，Adam 较为鲁棒，SGD在参数调整较优时会有最佳表现（分类效果与迭代次数）；SGD标识随机梯度下降。
# alpha:L2的参数：MLP是可以支持正则化的，默认为L2，具体参数需要调整
# hidden_layer_sizes=(5, 2) hidden层2层,第一层5个神经元，第二层2个神经元)，2层隐藏层，也就有3层神经网络
clf=MLPClassifier(solver='lbfgs',alpha=1e-5,hidden_layer_sizes=(5,2),random_state=1)
clf.fit(X,y)
print('每层网络层系数矩阵维度：\n',[coef.shape for coef in clf.coefs_])
y_pred=clf.predict([[0.317029, 14.739025]])
print('预测结果:',y_pred)
y_pred_pro=clf.predict_proba([[0.317029, 14.739025]])
print('预测结果概率:',y_pred_pro)

cengindex=0
for wi in clf.coefs_:
    cengindex+=1
    print('第%d层网络层:' % cengindex)
    print('权重矩阵维度:',wi.shape)
    print('系数矩阵:\n',wi)

#绘制分割区域
x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
xx1,xx2=np.meshgrid(np.arange(x_min,x_max,0.01),np.arange(y_min,y_max,0.01))
Z=clf.predict(np.c_[xx1.ravel(),xx2.ravel()])
Z=Z.reshape(xx1.shape)
#绘制区域网格图
plt.pcolormesh(xx1,xx2,Z,cmap=plt.cm.Paired)
plt.scatter(X[:,0],X[:,1],c=y)
plt.show()
