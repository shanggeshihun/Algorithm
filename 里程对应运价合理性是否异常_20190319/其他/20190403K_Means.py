# -*- coding: utf-8 -*-
"""
Created on 2019/04/03 13:41:37
"""

#coding:utf-8
"""
20190306
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def exe_lower_upper(df):
    Ql=np.percentile(df,25) # p25
    Qu=np.percentile(df,75) # p75
    IQR=Qu-Ql
    lower=Ql-1.5*IQR
    upper=Qu+1.5*IQR
    return lower,upper

import pymysql
h='192.168.0.204'
p=6612
u='bi'
pw='79f6ba05a7e0bbb7dbcc4cc2fbdb15c9'
d='repm'

#从数据库读取表
conn=pymysql.Connect(host=h,port=p,user=u,passwd=pw,db=d,charset='utf8')
cursor=conn.cursor()
#    查询返回结果是tuple
read_tb="select * from rep_op_ord_price_for_warning"
cursor.execute(read_tb)
result=cursor.fetchall()
cursor.close()
conn.close()

col_desc=cursor.description
colname=[]
[colname.append(col_desc[i][0]) for i in range(len(col_desc))]
df_all=pd.DataFrame(list(result))
df_all.columns=colname
df_all['actual_unit_price']=df_all['actual_unit_price'].astype('float')
df_all.info()
df_all.describe()
df_all.dtypes

"""
1 利用分位点寸照异常运价
"""
#    唯一线路、里程、运价
df_unique=df_all.loc[:,['load_address','upload_address','lc','actual_unit_price']].drop_duplicates(keep='first')
df_unique.drop(df_unique[df_unique.lc.isnull()].index,axis=0,inplace=True)
rows=len(df_unique)  # 16426 不同线路不同里程不同运价
df_unique['lc'].max() # 2810907
df_unique.dtypes

#   唯一里程、运价
lc_price_df=df_unique.loc[:,['lc','actual_unit_price']].drop_duplicates(keep='first')
#   pearsonr相关关系(里程与运价高度相关)
lc_price_pearsonr=lc_price_df.corr(method='pearson')
lc_price_spearman=lc_price_df.corr(method='spearman')
plt.figure(figsize=(10,10))
plt.scatter(lc_price_df.lc,lc_price_df.actual_unit_price)
plt.show()

#   唯一线路、里程（5764条）
line_lc_df=df_unique.loc[:,['load_address','upload_address','lc']].drop_duplicates(keep='first')
#   等区间区分
bins=list(range(0,2810907,50000))
line_lc_lc_cut=pd.cut(line_lc_df['lc'],bins=bins)
line_lc_df['lc_cut']=pd.DataFrame(line_lc_lc_cut)
line_lc_df['lc_cut']=line_lc_df['lc_cut'].astype(str)
line_lc_df.groupby('lc_cut').count()
#   分位数区分（保证线路多样性）
line_lc_df_qcut=pd.qcut(line_lc_df['lc'],q=15)
line_lc_df['lc_qcut']=pd.DataFrame(line_lc_df_qcut)
line_lc_df['lc_qcut']=line_lc_df['lc_qcut'].astype(str)
line_lc_df.groupby('lc_qcut').count()


import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
sns.boxplot(df_unique.actual_unit_price,df_unique.lc_cut,orient='h',fliersize=5)
plt.show()

lc_cut_label=list(set(df_unique['lc_cut']))
lc_cut_label_mapping={}
index_lower=[]
index_upper=[]
upper=[]
#    唯一线路、里程、运价
for interval in lc_cut_label:
    df_unique_sub=df_unique[df_unique['lc_cut']==interval]
#    print(df_sub.head(3))
    lower,upper=exe_lower_upper(df_unique_sub['actual_unit_price'])
#    print(lower,upper)
    cut_index_lower=df_unique_sub[df_unique_sub['actual_unit_price']<lower].index
    cut_index_upper=df_unique_sub[df_unique_sub['actual_unit_price']>upper].index
    lc_cut_label_mapping[interval]=(lower,upper)

    index_lower.extend(cut_index_lower)
    index_upper.extend(cut_index_upper)

lower_cnt=len(index_lower)
upper_cnt=len(index_upper)
df_unique.loc[index_lower,'exception']='lower'
df_unique.loc[index_upper,'exception']='upper'

#    array([nan, 'upper', 'lower'], dtype=object)
df_unique['exception'].unique()
#    非nan值
df_unique['exception'].isnull().sum()
df_unique['exception'].notnull().sum()
df_unique['exception'][df_unique['exception'].isnull()]='normal'
df_unique['exception'].unique()

df_unique.groupby('exception').count()
df_unique['exception'].value_counts()

df_unique['lower']=df_unique['lc_cut'].map(lambda k:lc_cut_label_mapping[k][0])
df_unique['upper']=df_unique['lc_cut'].map(lambda k:lc_cut_label_mapping[k][1])
df_unique.head(3)

df_all.info()
df_unique.info()

#KMeans聚类
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
def best_k(scale_df):
    """
    sacle_df:只含有标准化指标数据的DataFrame类型数据
    """
    #用scipy求解距离
    K=range(1,20)
    meandistortions=[]
#    sc_score=[]
#    sc_k=[]
    for k in K:
        kmeans=KMeans(n_clusters=k,random_state=9)
        kmeans.fit(scale_df)
        meandistortions.append(sum(np.min(cdist(scale_df,kmeans.cluster_centers_,'euclidean'),axis=1))/scale_df.shape[0])
#        if k>1:
#            sc_k.append(k)
#            sc_score_k=metrics.silhouette_score(scale_df,kmeans.labels_,metric='euclidean')
#            sc_score.append(sc_score_k)
    plt.figure(1,figsize=(10,10))
    plt.plot(K,meandistortions,'bx-')
    plt.xlabel('k')
    plt.ylabel('平均畸变程度',fontproperties='SimHei',fontsize=15)
    plt.title('用肘部法则来确定最佳的K值',fontproperties='SimHei',fontsize=15)
    plt.show()

X_cols = ["lc", "actual_unit_price"]
data=df_unique.loc[:,X_cols]
best_k(data)

kmeans=KMeans(n_clusters=10,random_state=9)
kmeans.fit(data)
data['label']=kmeans.labels_
cnames =[
'aliceblue',
'antiquewhite',
'aqua',
'aquamarine',
'azure',
'beige',
'bisque',
'black',
'blanchedalmond',
'blue']

for i in set(data['label']):
    X=data.lc[data.label==i].reshape(-1,1)
    y=data.actual_unit_price[data.label==i].reshape(-1,1)
    plt.scatter(X,y,c=cnames[i])
    plt.legend()
plt.show()


