# -*- coding: utf-8 -*-
"""
备注：
1 检查proc_rep_op_dri_cluster_samples昨日是否正常完成

更改日期：
20190304
"""
import pymysql
import datetime
import time

today=datetime.date.today()
oneday=datetime.timedelta(days=3)
lastday=today-oneday
lastday=lastday.strftime('%Y-%m-%d')

#数据库连接参数
h='192.168.0.204'
p=6612
u='bi'
pw='79f6ba05a7e0bbb7dbcc4cc2fbdb15c9'
d='repm'

#proc_rep_op_dri_cluster_samples昨日是否正常完成
flag=True
while flag:
    conn=pymysql.Connect(host=h,port=p,user=u,passwd=pw,db=d,charset='utf8')
    cursor=conn.cursor()
    sql="select sqlcode from odm.ifs_joblog where proname='repm.proc_rep_op_dri_cluster_samples' and accdate='{0}'".format(lastday)
    cursor.execute(sql)
    result=cursor.fetchall()
    conn.commit()
    conn.close()

    if result==0:
        
        with open(r'E:\SJB\NOTE\Python\algorithm\司机宝司机聚类\聚类_函数封装20190301_所有数据.py') as f:
            exec(f.read())
        break
    else:
        time.sleep(1800)