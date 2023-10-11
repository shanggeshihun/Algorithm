# -*- coding: utf-8 -*-
"""
Created on 20190521

@author: dell
"""

#coding:utf-8
#导入鸢尾花数据集
from sklearn import datasets    #导入内置数据集模块
iris=datasets.load_iris()   #导入鸢尾花的数据集
x=iris.data[:,0:2]    #样本数据共150个，取前两个特征，即花瓣长、宽

#可视化数据集
import matplotlib.pyplot as plt
plt.scatter(x[:,0],x[:,1]) #x的第0列绘制在横轴，x的第1列绘制在纵轴
plt.show()

#训练模型
from sklearn.neighbors import LocalOutlierFactor
model = LocalOutlierFactor(n_neighbors=4, contamination=0.1)    #定义一个LOF模型，异常比例是10%
model.fit(x)

#预测模型
y = model._predict(x) #若样本点正常，返回1，不正常，返回-1

#可视化预测结果
plt.scatter(x[:,0],x[:,1],c=y)  #样本点的颜色由y值决定
plt.show()


#模型其他指标
temp1 = model.kneighbors(x) #找出每个样本点的最近k个邻居，返回两个数组，一个是邻居的距离，一个是邻居的索引
temp2 = temp1[0]    #k个最近邻居的距离
temp3 = temp1[1]    #k个最近邻居在样本中的序号
temp3 = temp2.max(axis=1)   #取最大值，即第k个邻居的距离
temp4 = -model._decision_function(x)   #得到每个样本点的异常分数，注意这里要取负
