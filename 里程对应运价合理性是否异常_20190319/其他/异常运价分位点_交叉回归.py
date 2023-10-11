#coding:utf-8
"""
分位点&回归
"""
import numpy as np
import pandas as pd

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

import matplotlib.pyplot as plt

from sklearn import linear_model
def regression(X,y):
    rg=linear_model.LinearRegression()
    rg.fit(X,y)
    y_pred=rg.predict(X)
    plt.figure(figsize=(10,10))
    plt.scatter(X,y,label='实际运价',c='r')
    plt.plot(X,y_pred,label='回归运价',c='g')
    plt.title('拟合程度：'+str(rg.score(X,y)))
    plt.legend()
    plt.show()
    return rg.intercept_,rg.coef_

#从数据库读取表
conn=pymysql.Connect(host=h,port=p,user=u,passwd=pw,db=d,charset='utf8')
cursor=conn.cursor()
#    查询返回结果是tuple
read_tb="select * from rep_op_ord_price_warning"
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
1 不同的里程分组取预测最大值
"""
#    剔除重复值（日期、运单号）
df_unique=df_all.loc[:,['load_address','upload_address','lc','actual_unit_price']].drop_duplicates(keep='first')

rows=len(df_unique)  # 1244 不同线路不同里程不同运价
max_lc=df_unique['lc'].max() #1392784
gap=50000
bins=list(range(0,max_lc,gap))
df_unique_lc_cut=pd.cut(df_unique['lc'],bins=bins)

df_unique['lc_cut_1']=pd.DataFrame(df_unique_lc_cut)
df_unique['lc_cut_1']=df_unique['lc_cut_1'].astype(str)
lc_cut_label=set(df_unique['lc_cut_1'])
df_unique['price_pred_1']=0
df_unique['intercept_1']=0
df_unique['coef_1']=0
for lc_cut_i in lc_cut_label:
    X=df_unique.lc[df_unique['lc_cut_1']==lc_cut_i].reshape(-1,1)
    y=df_unique.actual_unit_price[df_unique['lc_cut_1']==lc_cut_i].reshape(-1,1)
    intercept,coef=regression(X,y)[0],regression(X,y)[1][0]
    df_unique['price_pred_1'][df_unique['lc_cut_1']==lc_cut_i]=intercept+coef*df_unique['lc'][df_unique['lc_cut_1']==lc_cut_i]
    df_unique['intercept_1'][df_unique['lc_cut_1']==lc_cut_i]=intercept
    df_unique['coef_1'][df_unique['lc_cut_1']==lc_cut_i]=coef



max_lc=df_unique['lc'].max() #1392784
gap=80000
bins=list(range(0,max_lc,gap))
df_unique_lc_cut=pd.cut(df_unique['lc'],bins=bins)

df_unique['lc_cut_2']=pd.DataFrame(df_unique_lc_cut)
df_unique['lc_cut_2']=df_unique['lc_cut_2'].astype(str)
lc_cut_label=set(df_unique['lc_cut_2'])
df_unique['price_pred_2']=0
df_unique['intercept_2']=0
df_unique['coef_2']=0
for lc_cut_i in lc_cut_label:
    X=df_unique.lc[df_unique['lc_cut_2']==lc_cut_i].reshape(-1,1)
    y=df_unique.actual_unit_price[df_unique['lc_cut_2']==lc_cut_i].reshape(-1,1)
    intercept,coef=regression(X,y)[0],regression(X,y)[1][0]
    df_unique['price_pred_2'][df_unique['lc_cut_2']==lc_cut_i]=intercept+coef*df_unique['lc'][df_unique['lc_cut_2']==lc_cut_i]
    df_unique['intercept_2'][df_unique['lc_cut_2']==lc_cut_i]=intercept
    df_unique['coef_2'][df_unique['lc_cut_2']==lc_cut_i]=coef



temp_df=df_unique.loc[:,['price_pred_1','price_pred_2']]

df_unique['price_pred']=temp_df.apply(max,axis=1)

df_unique.head(3)

df_pred=pd.merge(df_all,df_unique,on=['load_address','upload_address','lc','actual_unit_price'],how='left')
df_pred.to_excel(r"C:\Users\dell\Desktop\exception_unit_price\df_pred.xlsx")

