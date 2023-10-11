#coding:utf-8
"""
20190306
"""
# -*- coding: utf-8 -*-
"""
Created on 2019/04/03 13:42:38
1 q_cut分位数划分区间(保证线路多样性)
2 子区间 lc_price相关性
    2.1 相关性高  回归  实际运价高于预测30%则预警 返回预测运价
    2.2 相关性低  分位点  高于1.5*IQR则预警 返回里程lc区间
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
def exe_lower_upper(df):
    Ql=np.percentile(df,15) # p25
    Qu=np.percentile(df,85) # p75
    IQR=Qu-Ql
    lower=Ql-1.5*IQR
    upper=Qu+1.5*IQR
    lower=Ql
    upper=Qu
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

#唯一线路、里程、运价（16426条记录，最高里程2810907）
df_unique=df_all.loc[:,['load_address','upload_address','lc','actual_unit_price']].drop_duplicates(keep='first')
df_unique.drop(df_unique[df_unique.lc.isnull()].index,axis=0,inplace=True)
rows=len(df_unique)
df_unique['lc'].max() # 2810907
df_unique.dtypes

#唯一里程、运价（16134条记录，里程、运价散 高度正相关）
lc_price_df=df_unique.loc[:,['lc','actual_unit_price']].drop_duplicates(keep='first')
#   pearsonr相关关系
lc_price_pearsonr=lc_price_df.corr(method='pearson')
lc_price_spearman=lc_price_df.corr(method='spearman')
plt.figure(figsize=(9,9))
plt.scatter(lc_price_df.lc,lc_price_df.actual_unit_price)
plt.show()


#孤立森林（效果不好）
from sklearn.ensemble import IsolationForest
ilf = IsolationForest(n_estimators=100,
                      n_jobs=-1,          # 使用全部cpu
                      verbose=2,
                      max_samples=2000,
                      contamination=0.05
                      )
IsolationForest()
X_cols = ["lc", "actual_unit_price"]
# 训练
ilf.fit(lc_price_df[X_cols])
shape = lc_price_df.shape[0]
test = lc_price_df[X_cols]
    # 预测
pred = ilf.predict(test)
from collections import Counter
Counter(pred)
print(pred)

lc_price_df['pred'] =pred
plt.figure(figsize=(10,10))
lc_price_df.dtypes
j=0
clst=['r','b']
for i in set(pred):
    X=lc_price_df.lc[lc_price_df.pred==i]
    y=lc_price_df.actual_unit_price[lc_price_df.pred==i]
    plt.scatter(X,y,c=clst[j],label=i)
    j=j+1
    plt.legend()
plt.show()