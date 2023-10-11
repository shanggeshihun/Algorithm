# -*- coding: utf-8 -*-
"""
Created on 20190416
1 q_cut分位数划分区间(保证线路多样性)
2 子区间 lc_price相关性
    2.1 相关性高  回归  实际运价高于预测30%则预警 返回预测运价
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

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
read_tb="select * from rep_op_ord_price_for_warning_samples"
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

#唯一线路、里程（5764条）
line_lc_df=df_unique.loc[:,['load_address','upload_address','lc']].drop_duplicates(keep='first')
#   分位数区分（保证线路多样性）
line_lc_df_qcut=pd.qcut(line_lc_df['lc'],q=150)
len(line_lc_df_qcut)
line_lc_df['lc_qcut']=pd.DataFrame(line_lc_df_qcut)
line_lc_df['lc_qcut']=line_lc_df['lc_qcut'].astype(str)
#   定义子区间极差
def get_describe(group):
    return {'range':group.max()-group.min(),
            'count':group.count(),
            'mean':group.mean()
            }
#   子区间 区间大小、线路条数、里程均值、
line_lc_df_interval=line_lc_df.lc.groupby(line_lc_df['lc_qcut'])
line_lc_df_interval_desc=line_lc_df_interval.apply(get_describe).unstack()


#线路、里程、运价df_unique 添加里程区间（16426条记录，最高里程2810907）
df_unique_qcut=pd.merge(df_unique,line_lc_df,on=['load_address','upload_address','lc'],how='inner')
#   每个里程区间的运价箱型图
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
plt.figure(figsize=(9,9))
sns.boxplot(df_unique_qcut.actual_unit_price,df_unique_qcut.lc_qcut,orient='h',fliersize=5)
plt.show()

lc_qcut_label=list(set(line_lc_df['lc_qcut']))


lc_qcut_label_mapping={}
index_lower=[]
index_upper=[]
#    唯一线路、里程、运价
lc_price_corr_lst=[]
df_unique_qcut_pred=pd.DataFrame()
for interval in lc_qcut_label:
    print(interval)
    df_unique_qcut_sub=df_unique_qcut[df_unique_qcut['lc_qcut']==interval]
#    子区间拟合预测运价
    reg=linear_model.LinearRegression()
    lc=np.array(df_unique_qcut_sub.lc).reshape(-1,1)
    price=np.array(df_unique_qcut_sub.actual_unit_price).reshape(-1,1)
    reg.fit(lc,price)
    part_price_pred=reg.predict(lc)
    df_unique_qcut_sub['part_price_pred']=part_price_pred
    
    df_unique_qcut_sub['part_warning_label']=0
    df_unique_qcut_sub['part_warning_label'][(df_unique_qcut_sub.actual_unit_price/df_unique_qcut_sub.part_price_pred>1.3)]=1
    
    df_unique_qcut_pred=df_unique_qcut_pred.append(df_unique_qcut_sub,ignore_index=True)
    
#    相关性
    lc_price_corr=df_unique_qcut_sub.loc[:,['lc','actual_unit_price']].corr('pearson').iloc[0,1]
    lc_price_corr_lst.append(lc_price_corr)
#    整体拟合预测运价
    total_price_pred=19.368612570820474 +0.26126693*lc/1000
    df_unique_qcut_sub['total_price_pred']=total_price_pred
#    比较子区间预测与整体预测区别（红点标记运价异常偏高）
#    预警标记warning_label
    df_unique_qcut_sub['total_warning_label']=0
    df_unique_qcut_sub['total_warning_label'][(df_unique_qcut_sub.actual_unit_price/  df_unique_qcut_sub.total_price_pred>1.3)]=1

    plt.figure(figsize=(9,9))
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus']=False
    colorLst=['green','red']
    plt.subplot(2,1,1)
    plt.scatter(df_unique_qcut_sub['lc'][df_unique_qcut_sub['part_warning_label']==0],df_unique_qcut_sub['actual_unit_price'][df_unique_qcut_sub['part_warning_label']==0],c=colorLst[0])
    plt.scatter(df_unique_qcut_sub['lc'][df_unique_qcut_sub['part_warning_label']==1],df_unique_qcut_sub['actual_unit_price'][df_unique_qcut_sub['part_warning_label']==1],c=colorLst[1])
    plt.plot(lc,part_price_pred)
    plt.xlabel('里程(M)')
    plt.ylabel('运价(元/吨)')
    plt.title('子区间拟合',loc='left')
    plt.subplot(2,1,2)
    plt.scatter(df_unique_qcut_sub['lc'][df_unique_qcut_sub['total_warning_label']==0],df_unique_qcut_sub['actual_unit_price'][df_unique_qcut_sub['total_warning_label']==0],c=colorLst[0])
    plt.scatter(df_unique_qcut_sub['lc'][df_unique_qcut_sub['total_warning_label']==1],df_unique_qcut_sub['actual_unit_price'][df_unique_qcut_sub['total_warning_label']==1],c=colorLst[1])
    plt.plot(lc,total_price_pred)
    plt.xlabel('里程(M)')
    plt.ylabel('运价(元/吨)')
    plt.title('整体拟合',loc='left')
    plt.show()
    print('\n\n\n')

#合并子区间运价预测与子区间运价异常
df_unique_qcut=pd.merge(df_unique_qcut,df_unique_qcut_pred,on=['load_address','upload_address','lc','actual_unit_price'],how='inner')
df_unique_qcut.drop(['lc_qcut_y'],axis=1,inplace=True)
df_unique_qcut.rename(columns={'lc_qcut_x':'lc_qcut'},inplace=True)
df_unique_qcut.info()
df_unique_qcut.to_excel(r"C:\Users\dell\Desktop\df_unique_qcut_100.xlsx")





import datetime
the_date=datetime.datetime.strptime('2019-01-01','%Y-%m-%d').date()
type(the_date)
type(df_all['data_dt'].values[1])

df_all_2018=df_all[df_all['company_check_receipt_date']>=the_date]
df_all_merge_exception=pd.merge(df_all_2018,df_unique_qcut,on=['load_address','upload_address','lc','actual_unit_price'],how='inner')
df_all_merge_exception.to_excel(r"C:\Users\dell\Desktop\df_all_merge_exception.xlsx")
