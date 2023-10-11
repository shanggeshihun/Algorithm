#coding:utf-8
"""
20190306
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
#    剔除重复值（日期、运单号）
df_unique=df_all.loc[:,['load_address','upload_address','lc','actual_unit_price']].drop_duplicates(keep='first')
df_unique.drop(df_unique[df_unique.lc.isnull()].index,axis=0,inplace=True)
rows=len(df_unique)  # 1244 不同线路不同里程不同运价
df_unique['lc'].max() # 2810907
bins=list(range(0,2810907,50000))
df_unique_lc_cut=pd.cut(df_unique['lc'],bins=bins)

df_unique['lc_cut']=pd.DataFrame(df_unique_lc_cut)
df_unique['lc_cut']=df_unique['lc_cut'].astype(str)

df_unique.groupby('lc_cut').count()
df_unique[df_unique.lc_cut=='nan']
df_unique.drop(df_unique[df_unique.lc_cut=='nan'].index,axis=0,inplace=True)

# 箱型图显示里程运价分布
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
# df_exception.to_excel(r"C:\Users\dell\Desktop\exception_unit_price\exception.xls")


#   异历史常记录的interval
exe_interval=df_exception.loc[:,['lc_cut']].drop_duplicates(keep='first')
#   异常记录的interval历史里程及运价
import matplotlib.pyplot as plt
#   遍历每个异常的interval
type(exe_interval)
for i in range(len(exe_interval)):
    i+=1
    if i>2:
        break
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
        # plt.scatter(x=x,y=19.368612570820474 +0.26126693*x,label='预测',c='y')
    # print(interval_df['lc']*0.26126693)
    plt.plot(x=interval_df['lc'],y=19.368612570820474 +0.26126693*interval_df['lc'],linewidth=2.0)
    plt.legend()
    plt.title('该里程区间%s 里程~运价分布' %(exe_interval.iloc[i,0]),fontsize=15)
    plt.show()
    
    plt.figure(figsize=(10,10))
    plt.hist(
plt.show()


