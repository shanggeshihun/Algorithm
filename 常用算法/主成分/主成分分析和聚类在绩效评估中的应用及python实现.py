# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 00:05:48 2019

@author: Administrator
"""

#主成分分析和聚类在绩效评估中的应用及python实现
"""
绩效管理是企业进行人力资源管理的重要内容之一，通过绩效考核发现员工在工作中存在的问题，并为员工解决这些问题，实现员工在工作能力上的提高，现实中企业一般对员工的考核指标较多，且把绩效结果反馈给员工时员工并不详细的知道自己和别人存在的差异，所以我们用聚类分析把员工进行分类，所谓的物以类聚，对不同群组的员工分类培训，不但降低了人事部门的工作量，也让员工可以看到自己和别人之间的差距，在聚类之前使用主成分分析是因为在指标复杂性和多样性存在的情况下，人为评价绩效时很难设定权重，主成分分析根据其算法特性，能克服一些主观设置造成的偏差，根据载荷矩阵系数大小判断不同主成分反应的主要问题，简而言之主成分就相当于对指标的聚类，同时也有降维作用，对主成分得分进行聚类更容易在大量复杂考核指标中发现员工在哪方面存在的不足。
"""
#读取数据
import pandas as pd
file='c:/users/administrator/desktop/02.xls'
data=pd.read_excel(file)
#缺失值处理
data.head()
explore=data.describe(percentile=[],include='all').T
explore['null']=len(data)-explore['count']
#纵向缺失值占比
colrate=explore['null']/len(data)
#对缺失值不做处理，仅删除占比超过80%的变量

#数据标准化
import numpy as np 
b=list(np.std(data,ddof=1))
mu=list(data.mean())
for i in range(1,13):
    for j in range(0,len(data.iloc[:,i])):
        data.iloc[j,i]=(data.iloc[j,i]-mu[i-1])/b[i-1]
outputfile='c:/users/gh/desktop/021.xls'
data.to_excel(outputfile,index=False)

#主成分
import pandas as pd
inputfile='c:/users/gh/desktop/021.xls'
outputfile='c:/users/gh/desktop/022.xls'
data46=pd.read_excel(inputfile)
from sklearn.decomposition import PCA
pca=PCA()
data=data46.iloc[:,1:13]
pca.fit(data)
#返回模型的特征向量
pca.components_
#返回各个成分的方差百分比
ratio=pca.explained_variance_ratio_
ratio*100
#保留前5个主成分
pca=PCA(5)
pca.fit(data)
#用data来训练PCA模型，同时返回降维后的数据
low_d=pca.fit_transform(data)
pd.DataFrame(low_d).to_excel(outputfile)

#聚类分析
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn import metrics
import matplotlib.pyplot as plt
x=pd.DataFrame({'a':[1,2,3,4,5,4,5,6],'b':[6,7,8,9,10,7,5,7],'c':[6,7,8,9,10,8,7,8]},index=pd.date_range('2019-01-01','2019-01-08'))
#判断聚类个数
K=range(2,3)
meandistortion=[]
sc_score=[]
for k in K:
    kmeans=KMeans(n_clusters=k)
    kmeans.fit(x)
    meandistortion.append(sum(np.min(cdist(x,kmeans.cluster_centers_,'euclidean'),axis=1))/x.shape[0])
    sc_score_k=metrics.silhouette_score(x, kmeans.labels_, metric='euclidean')
    sc_score.append(sc_score_k)
print(sc_score)
plt.subplot(2,1,2)
plt.plot(K,meandistortion,'bx-')
plt.xlabel('k')
plt.ylabel('平均畸变程度')
plt.title(u'用肘部法则确定的最佳k值')
#聚类效果评估
from sklearn.cluster import KMeans
from sklearn import metrics
for i in test:
    kmeans_model=KMeans(n_cluster=t).fit(x)
print(u'K=%s,轮廓系数 = %.03f' % (t, metrics.silhouette_score(x, kmeans_model.labels_, metric='euclidean')))
#由上图可知：分为3类
kmeans_model = KMeans(n_clusters=3).fit(x)
kmeans_model.cluster_centers_#查看聚类中心
kmeans_model.labels_#类别标签



#df数据标准化
import pandas as pd
import numpy as np
def df_scale(dataframe):
#    df=pd.DataFrame({'a':[1,2,3,4,5],'b':[6,7,8,9,10],'c':[6,7,8,9,10]},index=pd.date_range('2019-01-01','2019-01-05'))
    col_num=len(dataframe.columns)
    b=list(np.std(dataframe,ddof=1))
    mu=list(dataframe.mean())
#    指标从第2列开始
    for i in range(col_num):
        for j in range(len(dataframe.iloc[:,i])):
            dataframe.iloc[j,i]=(dataframe.iloc[j,i]-mu[i])/b[i]
    return dataframe
if __name__=='__main__':
    df=pd.DataFrame({'a':[1,2,3,4,5],'b':[6,7,8,9,10],'c':[6,7,8,9,10]},index=pd.date_range('2019-01-01','2019-01-05'))
    print(df_scale(df))


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn import metrics
#best_k返回最佳聚类方案
def best_k(scale_df):
    #用scipy求解距离
    K=range(2,8)
    meandistortions=[]
    sc_score=[]
    for k in K:
        kmeans=KMeans(n_clusters=k)
        kmeans.fit(scale_df)
        meandistortions.append(sum(np.min(cdist(scale_df,kmeans.cluster_centers_,'euclidean'),axis=1))/scale_df.shape[0])
        sc_score_k=metrics.silhouette_score(scale_df,kmeans.labels_,metric='euclidean')
        sc_score.append(sc_score_k)
    plt.figure(1,figsize=(10,10))
    plt.subplot(2,1,1)
    plt.plot(K,meandistortions,'bx-')
    plt.xlabel('k')
    plt.ylabel('平均畸变程度',fontproperties='SimHei',fontsize=15)
    plt.title('用肘部法则来确定最佳的K值',fontproperties='SimHei',fontsize=15)

    plt.subplot(2,1,2)
    plt.plot(K,sc_score,'gx-')
    plt.xlabel('k')
    plt.ylabel('轮廓系数',fontproperties='SimHei',fontsize=15)
    plt.title('用轮廓系数来确定最佳的K值',fontproperties='SimHei',fontsize=15)
    plt.show()

if __name__=='__main__':
    df=pd.read_excel(r"C:\Users\dell\Desktop\rfm_login_2.xlsx")
    best_k(df.loc[:,['order_cnt','order_pay']])





import numpy as np
from sklearn.cluster import KMeans #导入Kmeans算法包
from sklearn.metrics import silhouette_score #计算轮廓系数
import matplotlib.pyplot as plt #画图工具
plt.subplot(3,2,1)
x=o_pca_factor
#x1=np.array([1,2,3,1,5,6,5,5,6,7,8,9,7,9])
#x2=np.array([1,3,2,2,8,6,7,6,7,1,2,1,1,3])
#x=np.array((x1,x2)).T
x1=x[:,0]
x2=x[:,1]

plt.xlim([0,10])
plt.ylim([0,10])
plt.title('Instances')
plt.scatter(x1,x2)
#在1号子图做出原始数据点阵的分布
colors=['b','g','r','c','m','y','k','b']
markers=['o','s','D','v','^','p','*','+']
clusters=[2,3,4,5,8]
subplot_counter=1
sc_scores=[]
for t in clusters:
    subplot_counter+=1
    plt.subplot(3,2,subplot_counter)
    kmeans_model=KMeans(n_clusters=t).fit(x)
    sc_score=silhouette_score(x,kmeans_model.labels_,metric='euclidean')
    sc_scores.append(sc_score)

    plt.title('K=%s,silhouette coefficient=%0.03f'%(t,sc_score))
    print(sc_score)
    #绘制轮廓系数与不同类簇数量的直观显示图
plt.figure()
#绘制轮廓系数与不同类簇数量的直观显示图
plt.plot(clusters,sc_scores,'*-')
plt.xlabel('Numbers of clusters')
plt.ylabel('Silhouette Coefficient score')
plt.show()