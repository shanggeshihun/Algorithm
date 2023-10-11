# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 14:50:38 2019

@author: dell
"""

#2.1随机森林回归器的使用Demo1
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import numpy as np

from sklearn.datasets import load_iris
    #４个属性是：萼片宽度　萼片长度　花瓣宽度　花瓣长度　标签是花的种类：setosa versicolour virginica
iris=load_iris()
print(iris['target'].shape)
    #使用默认设置
rf=RandomForestRegressor()
rf.fit(iris.data[:150],iris.target[:150])

    #随机挑选两个预测不相同的样本
instance=iris.data[[100,109]]
print(instance)
rf.predict(instance[[0]])
print('instance 0 predict:',rf.predict(instance[[0]]))
print('instance 1 predict:',rf.predict(instance[[1]]))


#2.2 随机森林分类器、决策树、extra树分类器的比较Demo2
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
    #生成的样本数据集,样本数据集的标签
X,y=make_blobs(n_samples=10000,n_features=10,centers=100,random_state=0)

clf=DecisionTreeClassifier(max_depth=None,min_samples_split=2,random_state=0)
scores=cross_val_score(clf,X,y)
print(scores.mean())

clf=RandomForestClassifier(n_estimators=10,max_depth=None,min_samples_split=2,random_state=0)
scores=cross_val_score(clf,X,y)
print(scores.mean())

clf=ExtraTreesClassifier(n_estimators=10,max_depth=None,min_samples_split=2,random_state=0)
scores=cross_val_score(clf,X,y)
print(scores.mean())

#2.3 随机森林回归器regressor-实现特征选择
from sklearn.tree import DecisionTreeRegressor
import numpy as np

from sklearn.datasets import load_iris
iris=load_iris()

from sklearn.model_selection import cross_val_score,ShuffleSplit
X=iris['data']
Y=iris['target']
names=iris['feature_names']

rf=RandomForestRegressor()
scores=[]
#    单根据每个特征进行分类
for i in range(X.shape[1]):
    score=cross_val_score(rf,X[:,i:i+1],Y,scoring='r2',cv=ShuffleSplit(len(X),3,0.3))
    scores.append((round(np.mean(score),3),names[i]))
print(sorted(scores,reverse=True))





