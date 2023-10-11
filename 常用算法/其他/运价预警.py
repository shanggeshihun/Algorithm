# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 09:18:16 2019

@author: dell
"""

import pandas as pd
#from sklearn import metrics
import pymysql
import datetime

start_time=datetime.datetime.now()
#数据库连接参数
h='192.168.0.204'
p=6612
u='bi'
pw='79f6ba05a7e0bbb7dbcc4cc2fbdb15c9'
d='repm'

#返回查询结果
conn=pymysql.Connect(host=h,port=p,user=u,passwd=pw,db=d,charset='utf8')
cursor=conn.cursor()
sql="select distinct lc,actual_unit_price,case when lc>0 and lc<=50000 then '0_50KM' when lc>50000 and lc<=200000 then '50_200KM' when lc>200000 then '200KM以上' end as lc_interval from rep_op_ord_price_warning where data_dt<'2019-03-05'"
cursor.execute(sql)
result=cursor.fetchall()
col_desc=cursor.description
conn.commit()
col_name=[]
[col_name.append(col_desc[i][0]) for i in range(len(col_desc))]
result=pd.DataFrame(list(result))
conn.close()
result.columns=col_name

from sklearn import linear_model
import numpy as np
df_1=result[result['lc_interval']=='0_50KM']
df_2=result[result['lc_interval']=='50_200KM']
df_3=result[result['lc_interval']=='200KM以上']

clf_1=linear_model.LinearRegression()
clf_1.fit(df_1['lc'].reshape(-1, 1),np.array(df_1['actual_unit_price']))
coef_1,int_1= clf_1.coef_, clf_1.intercept_
#(array([-0.00023352]), 58.99596036142401)

clf_2=linear_model.LinearRegression()
clf_2.fit(df_2['lc'].reshape(-1, 1),np.array(df_2['actual_unit_price']))
coef_2,int_2= clf_2.coef_, clf_2.intercept_
#(array([0.0004037]), 30.73161912853722)

clf_3=linear_model.LinearRegression()
clf_3.fit(df_3['lc'].reshape(-1, 1),np.array(df_3['actual_unit_price']))
coef_3,int_3= clf_3.coef_, clf_3.intercept_
#(array([0.00033278]), 52.63126109636707)

