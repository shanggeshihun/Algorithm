# -*- coding: utf-8 -*-
"""
随机森林主要是应用于回归和分类这两种场景，又侧重于分类。研究表明，组合分类器比单一分类器的分类效果好，在上述中我们知道，随机森林是指利用多棵决策树对样本数据进行训练、分类并预测的一种方法，它在对数据进行分类的同时，还可以给出各个变量（基因）的重要性评分，评估各个变量在分类中所起的作用。随机森林的构建大致如下：首先利用bootstrap方法又放回的从原始训练集中随机抽取n个样本，并构建n个决策树；然后假设在训练样本数据中有m个特征，那么每次分裂时选择最好的特征进行分裂 每棵树都一直这样分裂下去，直到该节点的所有训练样例都属于同一类；接着让每颗决策树在不做任何修剪的前提下最大限度的生长；最后将生成的多棵分类树组成随机森林，用随机森林分类器对新的数据进行分类与回归。对于分类问题，按多棵树分类器投票决定最终分类结果；而对于回归问题，则由多棵树预测值的均值决定最终预测结果。

在正式应用随机森林之前，要了解一下随机森林有几个超参数，这几个参数有的是增强模型的预测能力，有的是提高模型计算能力。

1、n_estimators：它表示建立的树的数量。 一般来说，树的数量越多，性能越好，预测也越稳定，但这也会减慢计算速度。一般来说在实践中选择数百棵树是比较好的选择，因此，一般默认是100。

2、n_jobs：超参数表示引擎允许使用处理器的数量。 若值为1，则只能使用一个处理器。 值为-1则表示没有限制。设置n_jobs可以加快模型计算速度。

3、oob_score :它是一种随机森林交叉验证方法，即是否采用袋外样本来评估模型的好坏。默认是False。推荐设置为True，因为袋外分数反应了一个模型拟合后的泛化能力。
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
RF=RandomForestClassifier(n_estimators=100,n_jobs=-1,oob_score=True)
iris=load_iris()
x=iris.data[:,:2]
y=iris.target
RF.fit(x,y)
#step size in the mesh
h=0.02
#Create color maps
cmap_light=ListedColormap(['#FFAAAA','#AAFFAA','#AAAAFF'])
cmap_bold=ListedColormap(['#FF0000','#00FF00','#0000FF'])
for weight in ['uniform','distance']:
    x_min,x_max=x[:,0].min()-1,x[:,0].max()+1
    y_min,y_max=x[:1].min()-1,x[:,1].max()+1
    xx,yy=np.meshgrid(np.arange(x_min,x_max,h),
                      np.arange(y_min,y_max,h))
    Z=RF.predict(np.c_[xx.ravel(),yy.ravel()])
    Z=Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx,yy,Z,cmap=cmap_light)
    plt.scatter(x[:,0],x[:,1],c=y,cmap=cmap_bold,edgecolor='k',s=20)
    plt.xlim(xx.min(),xx.max())
    plt.title('RandomForestClassifier')
plt.show()
