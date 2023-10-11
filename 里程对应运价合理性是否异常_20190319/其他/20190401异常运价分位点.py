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
#获取异常运价标记表与原始表合并
df=pd.merge(df_all,df_unique,on=['load_address','upload_address','lc','actual_unit_price'],how='inner')
df.columns

#最近一天
yesterday=df['company_check_receipt_date'].max()
#历史异常记录
df_exception=df[(df['exception']!='normal')]
#历史异常记录输出到表格
#df_exception.to_excel(r"C:\Users\dell\Desktop\exception_unit_price\exception.xls")


#   异历史常记录的interval
exe_interval=df_exception.loc[:,['lc_cut']].drop_duplicates(keep='first')
#   异常记录的interval历史里程及运价
import matplotlib.pyplot as plt
#   遍历每个异常的interval,输出原始回归预测结果与分位点异常散点图比较
type(exe_interval)
for i in range(len(exe_interval)):
#   该interval的历史记录
    interval_df=df[(df['lc_cut']==exe_interval.iloc[i,0])]
    c=['red','blue','green']
    plt.figure(figsize=(10,10))
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus']=False
    for ci,cate in enumerate(set(interval_df['exception'])):
        x=interval_df['lc'][interval_df['exception']==cate]
        y=interval_df['actual_unit_price'][interval_df['exception']==cate]
        colori=c[ci]
        plt.scatter(x=x,y=y,c=colori,label=cate)
        plt.xlabel('里程(m)')
        plt.ylabel('运价(元/吨)')
    plt.plot(interval_df['lc'],19.368612570820474 +0.26126693*interval_df['lc']/1000,label='Linear_predict')
    plt.legend()
    plt.title('该里程区间%s 里程~运价分布' %(exe_interval.iloc[i,0]),fontsize=15)
plt.show()


#异常记录的load_upload
exe_load_upload=df_exception.loc[:,['load_address','upload_address']].drop_duplicates(keep='first')
#   异常记录的load_upload历史里程及运价
import matplotlib.pyplot as plt
#   遍历每个异常的load_upload
for i in range(len(exe_load_upload)):
#   该interval的历史记录
    load_upload_df=df[(df['load_address']==exe_load_upload.iloc[i,0]) & (df['upload_address']==exe_load_upload.iloc[i,1])]

    path=r"C:\Users\dell\Desktop\exception_unit_price\\"+ exe_load_upload.iloc[i,0] +".xls"
#    load_upload_df.to_excel(path)
#   print(set(load_upload_df['exception']))
    c=['red','blue','green']
    plt.figure(figsize=(10,10))
#   load_upload_df_uni=load_upload_df['exception'].drop_duplicates(keep='first')
    for ci,cate in enumerate(set(load_upload_df['exception'])):
        x=load_upload_df['lc'][load_upload_df['exception']==cate]
        y=load_upload_df['actual_unit_price'][load_upload_df['exception']==cate]
        colori=c[ci]
#   print(colori)
        plt.scatter(x=x,y=y,c=colori,label=cate)
        plt.legend()
plt.show()


df.info()

"""
lst=['lc_cut','load_address','upload_address','lc','actual_unit_price']
#按 lc_cut统计 装货地、卸货地 条数
d1=df.loc[:,lst[:3]].drop_duplicates(keep='first')
d1.groupby('lc_cut').count()
#按lc_cut统计 装货地、卸货地、lc 条数
d1=df.loc[:,lst[:4]].drop_duplicates(keep='first')
d1.groupby('lc_cut').count()
#按lc_cut统计 装货地、卸货地、lc、actual_unit_price 条数1244
d1=df.loc[:,lst[:]].drop_duplicates(keep='first')
d1.groupby('lc_cut').count()
"""

"""
2 结合回归模型寻找异常运价
"""
#按 lc_cut统计 装货地、卸货地 条数 小于10条则用回归
lst=['lc_cut','load_address','upload_address','lc','actual_unit_price']
d1=df.loc[:,lst[:3]].drop_duplicates(keep='first')
load_upload_cnt=d1.groupby('lc_cut').count()
#   小于10条线路的里程区间
lc_cut_below_10=load_upload_cnt[load_upload_cnt['load_address']<10].index
lc_cut_below_10=list(lc_cut_below_10)
#   df_unique小于10条线路的索引
df_unique_line_below10_index=[]
df_unique['line_below10']=''
for i in lc_cut_below_10:
    index_sub=df_unique[df_unique.lc_cut==i].index
    df_unique_line_below10_index.extend(index_sub)
    df_unique['line_below10'][df_unique.lc_cut==i]='Y'
    

#小于10条线路的里程区间  线路 里程 价格
#df_unique_for_linear=df_unique.loc[df_unique_line_below10_index,:]
df_unique_for_linear=df_unique[df_unique['line_below10']=='Y']

from sklearn import linear_model
rg=linear_model.LinearRegression()
X=np.array(df_unique_for_linear.lc).reshape(-1,1)
y=np.array(df_unique_for_linear.actual_unit_price)
import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plt.plot(X,y)
plt.show()
rg.fit(X,y)
intercept,coef=rg.intercept_,rg.coef_
#   help('sklearn.linear_model.LinearRegression')
df_unique_for_linear['line_below10_pred_price']=''
df_unique_for_linear['line_below10_pred_price']=intercept+coef*df_unique_for_linear['lc']
#   实际运价超过预测运价30%则upper，低于30%则lower
df_unique_for_linear.dtypes
df_unique_for_linear['line_below10_pred_price']=df_unique_for_linear['line_below10_pred_price'].astype(float) 

df_unique_for_linear['pred_exception']=''
df_unique_for_linear['pred_exception'][ (df_unique['actual_unit_price']/df_unique_for_linear['line_below10_pred_price']>1.3)]='upper'
df_unique_for_linear['pred_exception'][ (df_unique['actual_unit_price']/df_unique_for_linear['line_below10_pred_price']<0.7)]='lower'
df_unique_for_linear['pred_exception'][ (df_unique['actual_unit_price']/df_unique_for_linear['line_below10_pred_price']>0.7) &  (df_unique['actual_unit_price']/df_unique_for_linear['line_below10_pred_price']<1.3)]='normal'

df_unique_for_linear.loc[:,['actual_unit_price','line_below10_pred_price','pred_exception']]

plt.figure(figsize=(10,10))
X=df_unique_for_linear['lc']
y=df_unique_for_linear['actual_unit_price']
y_pred=df_unique_for_linear['line_below10_pred_price']
plt.plot(X,y,label='实际运价')
plt.plot(X,y_pred,label='预测运价')
plt.title('低于10条线路的里程区间运价预测')
plt.legend()
plt.show()

"""
3 (1 2 结果合并)
"""
df_11=pd.merge(df_all,df_unique,on=['load_address','upload_address','lc','actual_unit_price'],how='left')
df_unique.columns
df_11.columns
df_22=pd.merge(df_11,df_unique_for_linear,on=['load_address','upload_address','lc','actual_unit_price'],how='left')
df_22.columns
df_33=df_22.drop(['lc_cut_y', 'exception_y', 'lower_y', 'upper_y','line_below10_y'],axis=1,inplace=False)
df_33.columns
#df_33.to_excel(r"C:\Users\dell\Desktop\exception_unit_price\df_33.xlsx")
df_33.columns

df_33['final_exception']=np.nan
df_33['final_exception'][~(df_33['lc_cut_x'].isin(lc_cut_below_10))]=df_33['exception_x'][~(df_33['lc_cut_x'].isin(lc_cut_below_10))]
df_33['final_exception'][df_33['lc_cut_x'].isin(lc_cut_below_10)]=df_33['pred_exception'][df_33['lc_cut_x'].isin(lc_cut_below_10)]
#df_33.to_excel(r"C:\Users\dell\Desktop\exception_unit_price\df_33_final.xlsx")

