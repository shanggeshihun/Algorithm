# -*- coding: utf-8 -*-
a=[1,2,3]
b=[4,5,6]
c=[4,5,6,7,8]
zipped=zip(a,b)# 打包为元组的列表
#[(1, 4), (2, 5), (3, 6)]


seq=['one','two','three']
for i,element in enumerate(seq):#同时列出数据和数据下标
    print(i,element)


import numpy as np
import matplotlib.pyplot as plt
cluster1=np.random.uniform(0.5,1.5,(2,10))
cluster2=np.random.uniform(3.5,4.5,(2,10))
#cluster1 2行10列，行数不变
X=np.hstack((cluster1,cluster2)).T
plt.figure()
plt.axis([0,5,0,5])
plt.grid(True)
plt.plot(X[:,0],X[:,1],'k.')
#计算K值从1到10对应的平均畸变程度
from sklearn.cluster import KMeans
#用scipy求解距离
from scipy.spatial.distance import cdist
K=range(1,10)
meandistortions=[]
for k in K:
    kmeans=KMeans(n_clusters=k)
    kmeans.fit(X)
    meandistortions.append(sum(np.min(cdist(X,kmeans.cluster_centers_,'euclidean'),axis=1))/X.shape[0])
plt.plot(K,meandistortions,'bx-')
plt.xlabel('k')
plt.ylabel('平均畸变程度',fontproperties=font)
plt.title('用肘部法则来确定最佳的K值',fontproperties=font)

import numpy as np
x1=np.array([1, 2, 3, 1, 5, 6, 5, 5, 6, 7, 8, 9, 7, 9])
x2=np.array([1, 3, 2, 2, 8, 6, 7, 6, 7, 1, 2, 1, 1, 3])
X=np.array(list(zip(x1,x2))).reshape(len(x1),2)
plt.figure()
plt.axis([0,10,0,10])
plt.grid(True)
plt.plot(X[:,0],X[:,1],'k.')


#聚类想过评价
#轮廓系数(Silhouette Coefficient):s=ba/max(a,b)
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
plt.figure(figsize=(8,10))
plt.subplot(3,2,1)
x1=np.array([1, 2, 3, 1, 5, 6, 5, 5, 6, 7, 8, 9, 7, 9])
x2=np.array([1, 3, 2, 2, 8, 6, 7, 6, 7, 1, 2, 1, 1, 3])
X=np.array(list(zip(x1,x2))).reshape(len(x1),2)
plt.xlim([0,10])
plt.ylim([0,10])
plt.title('样本',fontproperties='font')
plt.scatter(x1,x2)
colors=['b','g','r','c','m','y','k','b']
markers=['o','s','D','v','^','p','*','+']
tests=[2,3,4,5,6]
subplot_counter=1
for t in tests:
    subplot_counter+=1
    plt.subplot(3,2,subplot_counter)
    kmeans_model=KMeans(n_clusters=t).fit(X)
    for i,l in enumerate(kmeans_model.labels_):
        plt.plot(x1[i],x2[i],color=colors[l],
                marker=markers[l],
                ls='None')
        plt.xlim([0,10])
        plt.ylim([0,10])
        plt.title('K=%s,轮廓系数=%.03f' %(t,metrics.silhouette_score(X,kmeans_model.labels_,metric='euclidean')),fontproperties='font')



import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
def kmeans_building(x1,x2,type_num,types,colors,shapes):
    X=np.array(list(zip(x1,x2))).reshape(len(x1),2)
    kmeans_model=KMeans(n_clusters=types_num).fit(X)
    x1_result=[]
    x2_result=[]
    for i in range(types_num):
        temp=[]
        temp1=[]
        x1_result.append(temp)
        x2_result.append(temp1)
    for i,l in enumerate(kmeans_model.labels_):
        x1_result[1].append(x1[i])
        x2_result[1].append(x2[i])
        plt.scatter(x1[i],x2[i],c=color[l],marker=shapes[l])
    for i in range(len(list(kmeans_model.cluster_centers_))):
        plt.scatter(list(list(kmeans_model.cluster_centers_)[i])[0],list(list(kmeans_model.cluster_centers_)[i])[1],c=colors[i],marker=shapes[i],label=types[i])
    plt.legend()
    return kmeans_model,x1_result,x2_result


import matplotlib.pyplot as plt
import kmeans
plt.figure(figsize=(8, 6))
x1 = [1, 2, 3, 1, 5, 6, 5, 5, 6, 7, 8, 9, 7, 9] # x坐标列表
x2 = [1, 3, 2, 2, 8, 6, 7, 6, 7, 1, 2, 1, 1, 3] # y坐标列表
colors = ['b', 'g', 'r'] # 颜色列表，因为要分3类，所以该列表有3个元素
shapes = ['o', 's', 'D'] # 点的形状列表，因为要分3类，所以该列表有3个元素
labels=['A','B','C'] # 画图的标签内容，A, B, C分别表示三个类的名称
kmeans_model,x1_result,x2_result=kmeans.kmeans_building(x1, x2, 3, labels, colors, shapes) # 本例要分3类，所以传入一个3
print(kmeans_model) 
print(x1_result) 
print(x2_result)

