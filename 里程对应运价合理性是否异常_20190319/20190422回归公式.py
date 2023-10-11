# -*- coding: utf-8 -*-
"""
Created on 20190422

@author: dell
"""

# -*- coding: utf-8 -*-
"""
Created on 20190417
训练样本：
1 相同里程取50%分位点运价
2 合并1KM内的运价，里程及运价分别取50%分位点数值，获取按里程划分区间(0.27元/1KM)，

测试样本
1 q_cut分位数划分区间(保证线路多样性)
2 子区间 lc_price相关性
3 精简画图直接输出到MySQL
4 0422 发 肖诗昌

"""

import numpy as np
import pandas as pd
from sklearn import linear_model
help("linear_model.fit")
import pymysql
h='192.168.0.204'
p=6612
u='bi'
pw='79f6ba05a7e0bbb7dbcc4cc2fbdb15c9'
d='repm'

#从数据库读取表
conn=pymysql.Connect(host=h,port=p,user=u,passwd=pw,db=d,charset='utf8')
cursor=conn.cursor()
"""
#    重跑
del_sql="delete from rep_op_ord_price_for_warning_samples"
cursor.execute(del_sql)
conn.commit()
"""
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


#唯一线路、里程、运价（16426条记录，最高里程2810907）
df_unique=df_all.loc[:,['load_address','upload_address','lc','actual_unit_price']].drop_duplicates(keep='first')
df_unique.drop(df_unique[df_unique.lc.isnull()].index,axis=0,inplace=True)
rows=len(df_unique)

#唯一里程、运价（16134条记录，里程、运价散 高度正相关）
lc_price_df=df_unique.loc[:,['lc','actual_unit_price']].drop_duplicates(keep='first')

max_lc=lc_price_df.lc.max()
lc_price_df_cut=pd.cut(lc_price_df.lc,bins=range(0,int(max_lc+1),1000))
lc_price_df['lc_cut']=pd.DataFrame(lc_price_df_cut)
lc_price_df['lc_cut']=lc_price_df['lc_cut'].astype(str)

lc_cut_label=list(set(lc_price_df['lc_cut']))

#生成50%分位点里程及运价(作为样本)
lc_price_new=pd.DataFrame()
for cut_label in lc_cut_label:
    lc_price_sub=lc_price_df[lc_price_df.lc_cut==cut_label]
    sub=lc_price_sub.quantile(0.5)
    lc_price_new=lc_price_new.append(sub,ignore_index=True)

#唯一线路、里程（5764条）
line_lc_df=df_unique.loc[:,['load_address','upload_address','lc']].drop_duplicates(keep='first')
#   分位数区分（保证线路多样性）
line_lc_df_qcut=pd.qcut(line_lc_df['lc'],q=80)
##等距区间
#max_lc_1=line_lc_df.lc.max()
#line_lc_df_qcut=pd.cut(line_lc_df['lc'],bins=range(0,int(max_lc_1+1),20000))
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

lc_qcut_label=list(set(line_lc_df['lc_qcut']))

lc_qcut_label_mapping={}
index_lower=[]
index_upper=[]
#    唯一线路、里程、运价
lc_price_corr_lst=[]
df_unique_qcut_pred=pd.DataFrame()
#    存储回归系数
reg_qcut_parameters_lst=[]
for interval in lc_qcut_label:
    df_unique_qcut_sub=df_unique_qcut[df_unique_qcut['lc_qcut']==interval]
    lc_min=df_unique_qcut_sub.lc.min()
    lc_max=df_unique_qcut_sub.lc.max()

#    子区间拟合预测运价
#    训练
    lc_price_new_sample=lc_price_new[(lc_price_new.lc>=lc_min) & (lc_price_new.lc<=lc_max)]
    reg=linear_model.LinearRegression()
    train_lc=np.array(lc_price_new_sample.lc).reshape(-1,1)
    train_price=np.array(lc_price_new_sample.actual_unit_price).reshape(-1,1)
    reg.fit(train_lc,train_price)
#    存储回归系数
    reg_para=[interval,reg.coef_[0][0],reg.intercept_[0],len(train_lc)]
    reg_qcut_parameters_lst.append(reg_para)



#    预测
    lc=df_unique_qcut_sub.lc.reshape(-1,1)
    part_price_pred=reg.predict(lc)
    df_unique_qcut_sub['part_price_pred']=part_price_pred
    
    df_unique_qcut_sub['part_warning_label']=0
    df_unique_qcut_sub['part_warning_label'][(df_unique_qcut_sub.actual_unit_price/df_unique_qcut_sub.part_price_pred>1.3)]=1
    
    df_unique_qcut_pred=df_unique_qcut_pred.append(df_unique_qcut_sub,ignore_index=True)

#    整体拟合预测运价
    total_price_pred=19.368612570820474 +0.26126693*lc/1000
    df_unique_qcut_sub['total_price_pred']=total_price_pred
#    比较子区间预测与整体预测区别（红点标记运价异常偏高）
#    预警标记warning_label
    df_unique_qcut_sub['total_warning_label']=0
    df_unique_qcut_sub['total_warning_label'][(df_unique_qcut_sub.actual_unit_price/  df_unique_qcut_sub.total_price_pred>1.3)]=1


#合并子区间运价预测与子区间运价异常
df_unique_qcut=pd.merge(df_unique_qcut,df_unique_qcut_pred,on=['load_address','upload_address','lc','actual_unit_price'],how='inner')
df_unique_qcut.drop(['lc_qcut_y'],axis=1,inplace=True)
df_unique_qcut.rename(columns={'lc_qcut_x':'lc_qcut'},inplace=True)
df_unique_qcut.to_excel(r"C:\Users\dell\Desktop\df_unique_qcut_new.xlsx")
#回归参数输出EXCEL
df_qcut_parameters_lst=pd.DataFrame(reg_qcut_parameters_lst,columns=['lc_interval','coef_','intercept_','samples'])
df_qcut_parameters_lst.to_excel(r"C:\Users\dell\Desktop\reg_qcut_parameters_lst.xlsx")




"""
#添加data_dt字段
import datetime
today_date=datetime.date.today()
yesterday_date=today_date-datetime.timedelta(days=1)
yesterday_str=datetime.date.strftime(yesterday_date, '%Y-%m-%d')
df_unique_qcut.insert(0,'data_dt',yesterday_str)


from sqlalchemy import create_engine
yconnect= create_engine('mysql+pymysql://{0}:{1}@{2}:{3}/{4}?charset=utf8'.format(u,pw,h,p,d))

pd.io.sql.to_sql(df_unique_qcut,'rep_op_ord_p_line_price_for_warning', yconnect, schema='repm',index=False, if_exists='append')
"""


import numpy as np
import pandas as pd
from sklearn import linear_model
a=np.array([2,3,4,5,6])
b=np.array([3,6,9,14,19])
a=a.reshape(-1,1)
b=b.reshape(-1,1)
reg.fit(a,b)
reg.coef_[0][0]
reg.intercept_[0]
reg.coef_.values
reg.predict(a)