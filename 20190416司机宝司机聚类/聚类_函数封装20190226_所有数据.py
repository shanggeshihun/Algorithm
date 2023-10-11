# -*- coding: utf-8 -*-
"""
取 煤炭 所有的货源 运单数据
司机服务多样性：使用 县-县 指标
高比例类：再做细分，该类内部再做聚类

聚类_函数封装20190221
+ 现将跑数结果存入数据库，再从数据库取数据
+ 自动判断聚类结果存入数据表

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
#from sklearn import metrics
from sklearn.decomposition import PCA
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
def query_mysql(h,p,u,pw,d,sql):
    conn=pymysql.Connect(host=h,port=p,user=u,passwd=pw,db=d,charset='utf8')
    cursor=conn.cursor()
    cursor.execute(sql)
    result=cursor.fetchall()
    col_desc=cursor.description
    conn.commit()
    col_name=[]
    [col_name.append(col_desc[i][0]) for i in range(len(col_desc))]
    result=pd.DataFrame(list(result))
    conn.close()
    result.columns=col_name
    return result


#best_k返回最佳聚类方案
def best_k(theme,scale_df):
    """
    sacle_df:只含有标准化指标数据的DataFrame类型数据
    """
    #用scipy求解距离
    K=range(1,8)
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
    plt.title(theme + '用肘部法则来确定最佳的K值',fontproperties='SimHei',fontsize=15)
#    plt.subplot(2,1,2)
#    plt.plot(sc_k,sc_score,'gx-')
#    plt.xlabel('sc_k')
#    plt.ylabel('轮廓系数',fontproperties='SimHei',fontsize=15)
#    plt.title('用轮廓系数来确定最佳的K值',fontproperties='SimHei',fontsize=15)
    plt.show()


#df数据标准化
def df_scale(dataframe):
    """
    dataframe:只含有指标数据的DataFrame数据
    """
    col_num=len(dataframe.columns)
    b=list(np.std(dataframe,ddof=1))
    mu=list(dataframe.mean())
#    指标从第2列开始
    for i in range(col_num):
        for j in range(len(dataframe.iloc[:,i])):
            dataframe.iloc[j,i]=(dataframe.iloc[j,i]-mu[i])/b[i]
    return dataframe


#利用pca主成分得分，绘制最佳聚类图，返回pca得分
def pca_best_k_score(theme,data_scale,pca_num):
    """
    data_scale:标准化只含有指标的DataFrame类型数据
    pca_num:提取主成分的个数，默认2
    """
    pca=PCA()
    pca.fit(data_scale)
#   返回模型的特征向量
    pca.components_
#   返回各个成分的方差百分比
    ratio=pca.explained_variance_ratio_*100
    print('PCA贡献率','\n',ratio)
#   保留前2个主成分 
    pca=PCA(pca_num)
    pca.fit(data_scale)
    pca.components_
#   用data_scale来训练PCA模型，同时返回降维后的数据
    pca_factor=pca.fit_transform(data_scale)
#   绘制得分聚类分布图
    best_k(theme,pca_factor)
#   肘部法则判断最佳聚类k值
    return pca_factor



#绘制pca得分分类图，返回类别标签
def pca_classify_plot(theme,pca_num,pca_factor,pca_best_k):
    """
    pca_num：提取的主成分个数
    pca_factor:提取的主成分得分
    pca_best_k:通过主成分得分最佳聚类k
    """
    best_k=pca_best_k
    estimator=KMeans(n_clusters=best_k,random_state=9)
    estimator.fit(pca_factor)
    #通过pca返回聚类标签
    label_pred=estimator.labels_
    pca_factor_labels=np.insert(pca_factor,pca_num,label_pred,axis=1)
    #pca降维聚类分布图
    colors=['royalblue','salmon','lightgreen','c','m','y','k','b']
    markers=['o','s','D','v','^','p','*','+']
    plt.figure(figsize=(10,10))
    try:
        for i in range(pca_num):
    #        第一个pca得分
            temp_x=pca_factor[pca_factor_labels[:,pca_num]==i][:,0]
    #        第二个pca得分
            temp_y=pca_factor[pca_factor_labels[:,pca_num]==i][:,1]
            plt.plot(temp_x,temp_y,color=colors[i],marker=markers[i],ls='None',alpha=0.05)
        plt.title(theme+ '\n pca_cluster_distribution',fontsize=15)
        plt.savefig(r"C:\\Users\\dell\\Desktop\\" + theme +" &pca_cluster_distribution.png")
        plt.show()
    except:
        print('提取主成分个数:',pca_num)
    return label_pred


#绘制每个类样本数量条形图，返回聚类字典
def class_count_plot(theme,label_pred):
    """
    label_pred:聚类标签0,1,2,3……
    """
    labels_dic={}
    for each in set(label_pred):
        labels_dic[each]=list(label_pred).count(each)
    plt.figure(figsize=(10,10))
    plt.bar(list(labels_dic.keys()),list(labels_dic.values()))
    for rect in labels_dic.keys():
        plt.text(rect,labels_dic[rect]+10,labels_dic[rect],ha='center', va= 'bottom')
    plt.xticks(range(len(set(label_pred))))
    plt.title(theme+'\n each label’s samples')
    plt.show()
    return labels_dic


#绘制样本聚类散点图，返回 原始清洗后的添加聚类标签的数据
def class_samples_scatter(theme,dfDropexe,X,label_pred,best_k):
    """
    dfDropexe:原始经过清洗后的数据
    label_pred:聚类标签array
    best_k:最佳聚类类别个数
    """
    dfDropexe['label_pred']=label_pred
    dfDropexegp=dfDropexe.groupby(label_pred).mean()
    #每个类的samples散点图
    plt.figure(1,figsize=(10,10))
    plt.figure(1).suptitle(theme + '\n +the differences between labels',fontsize=15)
    clus_indicator_dic={}
    #i:第i个指标
    for i in range(1,len(X)):
        plt.subplot(len(X)-1,1,i)
        print(dfDropexegp.loc[:,X[i]])
        plt.plot(range(best_k),dfDropexegp.loc[:,X[i]])
        plt.ylabel(X[i])
        plt.xticks(range(best_k))
        clus_indicator_dic_sub={}
        #xj：第xj类
        for xj in range(best_k):
        #第j类dataframe
            df_temp=dfDropexe[dfDropexe.loc[:,'label_pred']==xj]
            xij_max=np.round(np.max(df_temp.loc[:,X[i]]),2)
            xij_min=np.round(np.min(df_temp.loc[:,X[i]]),2)
            xij_center=np.round(dfDropexegp.loc[xj,X[i]],2)
            #主要聚类特征字典
            clus_indicator_dic_sub[xj]=[xij_min,xij_max]
            #添加聚类中心点数据标签
            plt.text(xj+0.1,dfDropexegp.loc[xj,X[i]],xij_center,ha='center', va= 'bottom')
            #添加最值数据标签
            plt.text(xj+0.08,xij_max,xij_max,ha='center', va= 'bottom')
            plt.text(xj+0.08,xij_min,xij_min,ha='center', va= 'bottom')

            plt.scatter(df_temp.loc[:,'label_pred'],df_temp.loc[:,X[i]])
        clus_indicator_dic[X[i]]=clus_indicator_dic_sub
    plt.savefig(r"C:\\Users\\dell\\Desktop\\" +theme + "&label_scatter.png")
    plt.show()
    return dfDropexe,clus_indicator_dic



#原始运单相关数据源
rfm_login="select * from rep_op_dri_rfm_login"
df=query_mysql(h,p,u,pw,d,rfm_login)
#分析异常记录
# 1 剔除 编码null的记录
df1= df.drop(df[pd.isnull(df['driver_code'])].index)

# 2 剔除 离群值 为单个公司超大跑单量司机
plt.figure(figsize=(10,10))
sns.boxplot()
sns.boxplot(x='com_cnt',y='order_cnt',data=df1)
plt.show()

df2=df1.drop(df1[(df1.loc[:,'order_cnt']>=800) & (df1.loc[:,'com_cnt']==1)].index,axis=0)

plt.figure(figsize=(10,10))
sns.boxplot()
sns.boxplot(x='com_cnt',y='order_cnt',data=df2)
plt.show()
df_copy=df2.copy()


#流失
loss_df=df_copy.copy()
#分析异常记录
plt.boxplot(
            loss_df.loc[:,'recent_order_days'],
            notch=False, # box instead of notch shape
            sym='rs',    # red squares for outliers
            showmeans=True,
            vert=True)
plt.xlabel('recent_order_days')
plt.ylabel('days')
plt.show()

loss_X=['driver_code','recent_order_days']
#清洗后的原始数据'
loss_dfDropexe=loss_df.loc[:,loss_X]
loss_theme='loss'
from sklearn import preprocessing
loss_data_scale=preprocessing.scale(loss_dfDropexe.loc[:,loss_X[1:]])
#一般pca提取两个主成分
loss_pca_num=1
#绘制最佳聚类图，返回pca得分
loss_pca_factor=pca_best_k_score(loss_theme,loss_data_scale,loss_pca_num)
#根据最佳最聚类图定义聚类类别数，暂且定义为3
loss_pca_best_k=3
#绘制pca得分分类图，返回类别标签
loss_label_pred=pca_classify_plot(loss_theme,loss_pca_num,loss_pca_factor,loss_pca_best_k)
#绘制每个类样本数量条形图
loss_labels_dic=class_count_plot(loss_theme,loss_label_pred)
#绘制样本聚类散点图，返回 原始清洗后的添加聚类标签的数据、及类别类别中心
loss_dfDropexe=class_samples_scatter(loss_theme,loss_dfDropexe,loss_X,loss_label_pred,loss_pca_best_k)
loss_dfDropexe,loss_clus_indicator_dic=list(loss_dfDropexe)[0],list(loss_dfDropexe)[1]
#修改制定列名
loss_dfDropexe.rename(columns={'label_pred':'loss_label_pred'}, inplace = True)
loss_labels_dic_per={}
for key in loss_labels_dic.keys():
    loss_labels_dic_per[key]=(loss_labels_dic[key],loss_labels_dic[key]/sum(loss_labels_dic.values()))
loss_labels_dic_per

loss_labels={}
for key in loss_clus_indicator_dic.keys():
    for key_sub in loss_clus_indicator_dic[key].keys():
        loss_temp=loss_clus_indicator_dic[key][key_sub]
        loss_labels[key_sub]=str(loss_temp[0])+'-'+str(loss_temp[1])
loss_labels

loss_dfDropexe['loss_labels_define']=''
for loss_key in loss_labels.keys():
    loss_dfDropexe['loss_labels_define'][loss_dfDropexe.loc[:,'loss_label_pred']==loss_key]=loss_labels[loss_key]
    
"""
0 148-406 未跑单 26.7%  》 轻度流失客户
1 407-781 未跑单 4.7%  》 重度流失客户
2 1-147 未跑单 68.5%  》 正常客户

type：煤炭
"""
loss_merge_final=loss_dfDropexe.loc[:,['driver_code','recent_order_days','loss_labels_define']]



#司机年龄
age_df=df_copy.copy()

    #分析异常记录 dri_age异常值
plt.boxplot(age_df.loc[:,'dri_age'])
plt.xlabel('dri_age')
plt.ylabel('age')
plt.show()

    #剔除dri_age异常
age_dfDropexe=age_df[(age_df.loc[:,'dri_age']<100) & (age_df.loc[:,'dri_age']>0)]

plt.boxplot(age_dfDropexe.loc[:,'dri_age'])
plt.xlabel('dri_age')
plt.ylabel('age')
plt.show()

age_X=['driver_code','dri_age']
    #清洗后的原始数据'
age_dfDropexe=age_dfDropexe.loc[:,age_X]
age_theme='age'
from sklearn import preprocessing
age_data_scale=preprocessing.scale(age_dfDropexe.loc[:,age_X[1:]])
    #一般pca提取两个主成分
age_pca_num=1
    #绘制最佳聚类图，返回pca得分
age_pca_factor=pca_best_k_score(age_theme,age_data_scale,age_pca_num)
    #根据最佳最聚类图定义聚类类别数，暂且定义为3
age_pca_best_k=3
    #绘制pca得分分类图，返回类别标签
age_label_pred=pca_classify_plot(age_theme,age_pca_num,age_pca_factor,age_pca_best_k)
    #绘制每个类样本数量条形图
age_labels_dic=class_count_plot(age_theme,age_label_pred)
    #绘制样本聚类散点图，返回 原始清洗后的添加聚类标签的数据
    #绘制样本聚类散点图，返回 原始清洗后的添加聚类标签的数据、及类别类别中心
age_dfDropexe=class_samples_scatter(age_theme,age_dfDropexe,age_X,age_label_pred,age_pca_best_k)
age_dfDropexe,age_clus_indicator_dic=list(age_dfDropexe)[0],list(age_dfDropexe)[1]
    #修改制定列名
age_dfDropexe.rename(columns={'label_pred':'age_label_pred'}, inplace = True)

age_labels_dic_per={}
for key in age_labels_dic.keys():
    age_labels_dic_per[key]=(age_labels_dic[key],age_labels_dic[key]/sum(age_labels_dic.values()))

age_labels={0:'38-45',1:'46-69',2:'21-37'}
age_dfDropexe['age_labels_define']=''
for age_key in age_labels.keys():
    age_dfDropexe['age_labels_define'][age_dfDropexe.loc[:,'age_label_pred']==age_key]=age_labels[age_key]
"""
0 38-45 39.5%  》
1 46-69 26.1%  》 
2 21-37 34.3%  》 
type：煤炭
"""
age_merge_final=age_dfDropexe.loc[:,['driver_code','dri_age','age_labels_define']]
#age_merge_final['age_labels_define'].value_counts()


#活跃频次
active_df=df_copy.copy()

    #分析异常
    #最近跑单与跑单频率 散点图检测异常值
plt.figure(figsize=(10,10))
plt.plot(active_df.loc[:,'order_freq'],active_df.loc[:,'order_cnt'],'o')
plt.xlabel('order_freq')
plt.ylabel('order_cnt')
plt.show()

active_df_50=active_df[active_df.loc[:'order_freq']<=20]
plt.plot(active_df_50.loc[:,'order_freq'],active_df_50.loc[:,'order_cnt'],'o')
plt.xlabel('order_freq_50')
plt.ylabel('order_cnt')
plt.show()

    #至今只跑1单且未签收，order_freq为空
active_dfDropexe=active_df[active_df.loc[:,'order_freq']>=0]

active_X=['driver_code','order_freq']
    #清洗后的原始数据'
active_dfDropexe=active_dfDropexe.loc[:,active_X]
active_theme='active'

from sklearn import preprocessing
active_data_scale=preprocessing.scale(active_dfDropexe.loc[:,active_X[1:]])
    #一般pca提取两个主成分
active_pca_num=1
    #绘制最佳聚类图，返回pca得分
active_pca_factor=pca_best_k_score(active_theme,active_data_scale,active_pca_num)
    #根据最佳最聚类图定义聚类类别数，暂且定义为3
active_pca_best_k=3
    #绘制pca得分分类图，返回类别标签
active_label_pred=pca_classify_plot(active_theme,active_pca_num,active_pca_factor,active_pca_best_k)
    #绘制每个类样本数量条形图
active_labels_dic=class_count_plot(active_theme,active_label_pred)
    #绘制样本聚类散点图，返回 原始清洗后的添加聚类标签的数据、及类别类别中心
active_dfDropexe=class_samples_scatter(active_theme,active_dfDropexe,active_X,active_label_pred,active_pca_best_k)
active_dfDropexe,active_clus_indicator_dic=list(active_dfDropexe)[0],list(active_dfDropexe)[1]
    #修改制定列名
active_dfDropexe.rename(columns={'label_pred':'active_label_pred'}, inplace = True)
active_labels_dic_per={}
for key in active_labels_dic.keys():
    active_labels_dic_per[key]=(active_labels_dic[key],active_labels_dic[key]/sum(active_labels_dic.values()))

active_labels={}
for key in active_clus_indicator_dic.keys():
    for key_sub in active_clus_indicator_dic[key].keys():
        active_temp=active_clus_indicator_dic[key][key_sub]
        active_labels[key_sub]=str(active_temp[0])+'-'+str(active_temp[1])
active_labels

active_dfDropexe['active_labels_define']=''
for active_key in active_labels.keys():
    active_dfDropexe['active_labels_define'][active_dfDropexe.loc[:,'active_label_pred']==active_key]=active_labels[active_key]

#active子聚类
active_max_=0
for key in active_labels_dic_per:
    active_max_=max(active_labels_dic_per[key][1],active_max_)
    if active_labels_dic_per[key][1]==active_max_:
        active_most_label=key
active_most_label

active_dfDropexe_sub=active_dfDropexe[active_dfDropexe.loc[:,'active_label_pred']==active_most_label]
active_dfDropexe_sub=active_dfDropexe_sub.loc[:,active_X]
active_theme_sub='active'
from sklearn import preprocessing
active_data_scale_sub=preprocessing.scale(active_dfDropexe_sub.loc[:,active_X[1:]])
    #一般pca提取两个主成分
active_pca_num_sub=1
    #绘制最佳聚类图，返回pca得分
active_pca_factor_sub=pca_best_k_score(active_theme_sub,active_data_scale_sub,active_pca_num_sub)
    #根据最佳最聚类图定义聚类类别数，暂且定义为3
active_pca_best_k_sub=2
    #绘制pca得分分类图，返回类别标签
active_label_pred_sub=pca_classify_plot(active_theme_sub,active_pca_num_sub,active_pca_factor_sub,active_pca_best_k_sub)
    #绘制每个类样本数量条形图
active_labels_dic_sub=class_count_plot(active_theme_sub,active_label_pred_sub)
    #绘制样本聚类散点图，返回 原始清洗后的添加聚类标签的数据、及类别类别中心
active_dfDropexe_sub=class_samples_scatter(active_theme_sub,active_dfDropexe_sub,active_X,active_label_pred_sub,active_pca_best_k_sub)
active_dfDropexe_sub,active_clus_indicator_sub_dic=list(active_dfDropexe_sub)[0],list(active_dfDropexe_sub)[1]
    #修改制定列名
active_dfDropexe_sub.rename(columns={'label_pred':'active_label_pred_sub'}, inplace = True)
active_labels_dic_per_sub={}
for key in active_labels_dic_sub.keys():
    active_labels_dic_per_sub[key]=(active_labels_dic_sub[key],active_labels_dic_sub[key]/sum(active_labels_dic_sub.values()))

active_dfDropexe_sub['active_labels_define_sub']=''
active_labels_sub={}
for key in active_clus_indicator_sub_dic.keys():
    for key_sub in active_clus_indicator_sub_dic[key].keys():
        active_temp=active_clus_indicator_sub_dic[key][key_sub]
        active_labels_sub[key_sub]=str(active_temp[0])+'-'+str(active_temp[1])
active_labels_sub

for active_key in active_labels_sub.keys():
    active_dfDropexe_sub['active_labels_define_sub'][active_dfDropexe_sub.loc[:,'active_label_pred_sub']==active_key]=active_labels_sub[active_key]

active_merge=pd.merge(active_dfDropexe,active_dfDropexe_sub,left_on='driver_code',right_on='driver_code',how='left')

    #子类定义结果赋值给新类
active_merge['active_define_final']=active_merge['active_labels_define_sub']
active_merge['active_define_final'][active_merge['active_define_final'].isnull()]=active_merge['active_labels_define']

active_merge['active_define_final'].value_counts()

active_merge_final=active_merge.loc[:,['driver_code','order_freq_x','active_define_final']]

active_merge_final.rename(columns={'order_freq_x':'order_freq'},inplace=True)



#司机体量指标volumn
volumn_df=df_copy.copy()

    #单均与下单总额散点图检测异常值
plt.figure(1,figsize=(10,10))
plt.plot(volumn_df.loc[:,'per_order_pay'],volumn_df.loc[:,'order_cnt'],'o')
plt.xlabel('per_order_pay')
plt.ylabel('order_cnt')
plt.show()

plt.figure(1,figsize=(10,10))
volumn_df_temp=volumn_df[volumn_df.loc[:,'per_order_pay']<=250]
plt.plot(volumn_df_temp.loc[:,'per_order_pay'],volumn_df_temp.loc[:,'order_cnt'],'o')
plt.xlabel('per_order_pay')
plt.ylabel('order_cnt')
plt.show()

plt.plot(volumn_df.loc[:,'order_cnt'],volumn_df.loc[:,'order_pay'],'o')
plt.xlabel('order_cnt')
plt.ylabel('order_pay')
plt.show()

    #异常处理
volumn_dfDropexe=volumn_df.copy()

volumn_X=['driver_code','order_pay']
volumn_dfDropexe=volumn_dfDropexe.loc[:,volumn_X]

volumn_theme='driver_volumn'
from sklearn import preprocessing
volumn_data_scale=preprocessing.scale(volumn_dfDropexe.loc[:,volumn_X[1:]])

    #一般pca提取两个主成分
volumn_pca_num=1
    #绘制最佳聚类图，返回pca得分
volumn_pca_factor=pca_best_k_score(volumn_theme,volumn_data_scale,volumn_pca_num)
    #根据最佳最聚类图定义聚类类别数，暂且定义为3
volumn_pca_best_k=3
    #绘制pca得分分类图，返回类别标签
volumn_label_pred=pca_classify_plot(volumn_theme,volumn_pca_num,volumn_pca_factor,volumn_pca_best_k)
    #绘制每个类样本数量条形图
volumn_labels_dic=class_count_plot(volumn_theme,volumn_label_pred)
    #绘制样本聚类散点图，返回 原始清洗后的添加聚类标签的数据、及类别类别中心
volumn_dfDropexe=class_samples_scatter(volumn_theme,volumn_dfDropexe,volumn_X,volumn_label_pred,volumn_pca_best_k)
volumn_dfDropexe,volumn_clus_indicator_dic=list(volumn_dfDropexe)[0],list(volumn_dfDropexe)[1]
    #修改制定列名
volumn_dfDropexe.rename(columns={'label_pred':'volumn_label_pred'}, inplace = True)
volumn_labels_dic_per={}
for key in volumn_labels_dic.keys():
    volumn_labels_dic_per[key]=(volumn_labels_dic[key],volumn_labels_dic[key]/sum(volumn_labels_dic.values()))

volumn_labels={}
for key in volumn_clus_indicator_dic.keys():
    for key_sub in volumn_clus_indicator_dic[key].keys():
        volumn_temp=volumn_clus_indicator_dic[key][key_sub]
        volumn_labels[key_sub]=str(volumn_temp[0])+'-'+str(volumn_temp[1])
volumn_labels

volumn_dfDropexe['volumn_labels_define']=''
for volumn_key in volumn_labels.keys():
    volumn_dfDropexe['volumn_labels_define'][volumn_dfDropexe.loc[:,'volumn_label_pred']==volumn_key]=volumn_labels[volumn_key]

"""
0 下单金额9.55万内--91.4%  》  小体量
1 下单金额40万以上--0.78%  》 大体量
2 下单金额9.55-40万 7.8%  》 中体量
"""

#volumn子聚类
volumn_max_=0
for key in volumn_labels_dic_per:
    volumn_max_=max(volumn_labels_dic_per[key][1],volumn_max_)
    if volumn_labels_dic_per[key][1]==volumn_max_:
        volumn_most_label=key
volumn_most_label

volumn_dfDropexe_sub=volumn_dfDropexe[volumn_dfDropexe.loc[:,'volumn_label_pred']==volumn_most_label]
volumn_dfDropexe_sub=volumn_dfDropexe_sub.loc[:,volumn_X]
volumn_theme_sub='volumn'
from sklearn import preprocessing
volumn_data_scale_sub=preprocessing.scale(volumn_dfDropexe_sub.loc[:,volumn_X[1:]])
    #一般pca提取两个主成分
volumn_pca_num_sub=1
    #绘制最佳聚类图，返回pca得分
volumn_pca_factor_sub=pca_best_k_score(volumn_theme_sub,volumn_data_scale_sub,volumn_pca_num_sub)
    #根据最佳最聚类图定义聚类类别数，暂且定义为3
volumn_pca_best_k_sub=2
    #绘制pca得分分类图，返回类别标签
volumn_label_pred_sub=pca_classify_plot(volumn_theme_sub,volumn_pca_num_sub,volumn_pca_factor_sub,volumn_pca_best_k_sub)
    #绘制每个类样本数量条形图
volumn_labels_dic_sub=class_count_plot(volumn_theme_sub,volumn_label_pred_sub)
    #绘制样本聚类散点图，返回 原始清洗后的添加聚类标签的数据、及类别类别中心
volumn_dfDropexe_sub=class_samples_scatter(volumn_theme_sub,volumn_dfDropexe_sub,volumn_X,volumn_label_pred_sub,volumn_pca_best_k_sub)
volumn_dfDropexe_sub,volumn_clus_indicator_sub_dic=list(volumn_dfDropexe_sub)[0],list(volumn_dfDropexe_sub)[1]
    #修改制定列名
volumn_dfDropexe_sub.rename(columns={'label_pred':'volumn_label_pred_sub'}, inplace = True)

volumn_labels_dic_per_sub={}
for key in volumn_labels_dic_sub.keys():
    volumn_labels_dic_per_sub[key]=(volumn_labels_dic_sub[key],volumn_labels_dic_sub[key]/sum(volumn_labels_dic_sub.values()))

volumn_dfDropexe_sub['volumn_labels_define_sub']=''
volumn_labels_sub={}
for key in volumn_clus_indicator_sub_dic.keys():
    for key_sub in volumn_clus_indicator_sub_dic[key].keys():
        volumn_temp=volumn_clus_indicator_sub_dic[key][key_sub]
        volumn_labels_sub[key_sub]=str(volumn_temp[0])+'-'+str(volumn_temp[1])

for volumn_key in volumn_labels_sub.keys():
    volumn_dfDropexe_sub['volumn_labels_define_sub'][volumn_dfDropexe_sub.loc[:,'volumn_label_pred_sub']==volumn_key]=volumn_labels_sub[volumn_key]

volumn_merge=pd.merge(volumn_dfDropexe,volumn_dfDropexe_sub,left_on='driver_code',right_on='driver_code',how='left')


    #子类定义结果赋值给新列
volumn_merge['volumn_define_final']=volumn_merge['volumn_labels_define_sub']
    #left join 子类定义结果为空的index
volumn_merge['volumn_define_final']=volumn_merge['volumn_labels_define_sub']
volumn_merge['volumn_define_final'][volumn_merge['volumn_define_final'].isnull()]=volumn_merge['volumn_labels_define']

#volumn_merge['volumn_define_final'].value_counts()
volumn_merge_final=volumn_merge.loc[:,['driver_code','order_pay_x','volumn_define_final']]
volumn_merge_final['volumn_define_final'].value_counts()

volumn_merge_final.rename(columns={'order_pay_x':'order_pay'}, inplace = True)



#司机跑单习惯（跑单县-县 ）
sc_df=df_copy.copy()

    #异常处理
sc_dfDropexe=sc_df.copy()


sc_X=['driver_code','loadupload_region']
sc_dfDropexe=sc_dfDropexe.loc[:,sc_X]

sc_theme='serve_county'

from sklearn import preprocessing
sc_data_scale=preprocessing.scale(sc_dfDropexe.loc[:,sc_X[1:]])
    #一般pca提取两个主成分
sc_pca_num=1
    #绘制最佳聚类图，返回pca得分
sc_pca_factor=pca_best_k_score(sc_theme,sc_data_scale,sc_pca_num)
    #根据最佳最聚类图定义聚类类别数，暂且定义为3
sc_pca_best_k=3
    #绘制pca得分分类图，返回类别标签
sc_label_pred=pca_classify_plot(sc_theme,sc_pca_num,sc_pca_factor,sc_pca_best_k)
    #绘制每个类样本数量条形图
sc_labels_dic=class_count_plot(sc_theme,sc_label_pred)
    #绘制样本聚类散点图，返回 原始清洗后的添加聚类标签的数据、及类别类别中心
sc_dfDropexe=class_samples_scatter(sc_theme,sc_dfDropexe,sc_X,sc_label_pred,sc_pca_best_k)
sc_dfDropexe,sc_clus_indicator_dic=list(sc_dfDropexe)[0],list(sc_dfDropexe)[1]
    #修改制定列名
sc_dfDropexe.rename(columns={'label_pred':'sc_label_pred'}, inplace = True)
sc_labels_dic_per={}
for key in sc_labels_dic.keys():
    sc_labels_dic_per[key]=(sc_labels_dic[key],sc_labels_dic[key]/sum(sc_labels_dic.values()))
sc_labels_dic_per

sc_labels={}
for key in sc_clus_indicator_dic.keys():
    for key_sub in sc_clus_indicator_dic[key].keys():
        sc_temp=sc_clus_indicator_dic[key][key_sub]
        sc_labels[key_sub]=str(sc_temp[0])+'-'+str(sc_temp[1])
sc_labels

sc_dfDropexe['sc_labels_define']=''
for sc_key in sc_labels.keys():
    sc_dfDropexe['sc_labels_define'][sc_dfDropexe.loc[:,'sc_label_pred']==sc_key]=sc_labels[sc_key]

sc_dfDropexe.columns
sc_merge_final=sc_dfDropexe.loc[:,['driver_code','loadupload_region','sc_labels_define']]
"""
0 1条县路  -- 63.5%  》 公司及卸货省单一
1 2-3条县路  -- 30.55%  》 公司及卸货省多样
2 4条以上县路  -- 5.9%  》 公司多样卸货省单一

"""


#登陆次数与首次登录时长、最近登录时长散点图检测异常值
lg_df=df_copy.copy()

    #剔除登录指标中含有的nan值得行
lg_df_temp=lg_df.loc[:,['login_times','first_login_days','recent_login_days','login_freq']]

lg_null_index=lg_df_temp.index[np.where(pd.isnull(lg_df_temp))[0]].drop_duplicates()

lg_dfDropexe=lg_df.drop(lg_null_index)

plt.figure(1,figsize=(8,16))
plt.figure(1).suptitle('the login exception dot',fontsize=15)
plt.subplot(411)
plt.plot(lg_dfDropexe.loc[:,'first_login_days'],lg_dfDropexe.loc[:,'login_times'],'o')
plt.xlabel('first_login_days')
plt.ylabel('login_times')
plt.subplot(412)
plt.plot(lg_dfDropexe.loc[:,'recent_login_days'],lg_dfDropexe.loc[:,'login_times'],'o')
plt.xlabel('recent_login_days')
plt.ylabel('login_times')
plt.subplot(413)
plt.plot(lg_dfDropexe.loc[:,'recent_login_days'],lg_dfDropexe.loc[:,'login_freq'],'o')
plt.xlabel('recent_login_days')
plt.ylabel('login_freq')

plt.subplot(414)
plt.boxplot(lg_dfDropexe.loc[:,'login_freq'])
plt.xlabel('login_freq')
plt.ylabel('freq')
plt.show()


lg_X=['driver_code','recent_login_days','login_times','login_freq']
lg_dfDropexe=lg_dfDropexe.loc[:,lg_X]

lg_theme='login'
from sklearn import preprocessing
lg_data_scale=preprocessing.scale(lg_dfDropexe.loc[:,lg_X[1:]])
    #一般pca提取两个主成分
lg_pca_num=3
    #绘制最佳聚类图，返回pca得分
lg_pca_factor=pca_best_k_score(lg_theme,lg_data_scale,lg_pca_num)
    #根据最佳最聚类图定义聚类类别数，暂且定义为3
lg_pca_best_k=3
    #绘制pca得分分类图，返回类别标签
lg_label_pred=pca_classify_plot(lg_theme,lg_pca_num,lg_pca_factor,lg_pca_best_k)
    #绘制每个类样本数量条形图
lg_labels_dic=class_count_plot(lg_theme,lg_label_pred)
    #绘制样本聚类散点图，返回 原始清洗后的添加聚类标签的数据、及类别类别中心
lg_dfDropexe=class_samples_scatter(lg_theme,lg_dfDropexe,lg_X,lg_label_pred,lg_pca_best_k)
lg_dfDropexe,lg_clus_indicator_dic=list(lg_dfDropexe)[0],list(lg_dfDropexe)[1]
    #修改制定列名
lg_dfDropexe.rename(columns={'label_pred':'lg_label_pred'}, inplace = True)

lg_labels_dic_per={}
for key in lg_labels_dic.keys():
    lg_labels_dic_per[key]=(lg_labels_dic[key],lg_labels_dic[key]/sum(lg_labels_dic.values()))

lg_labels={0:'APP粘性低(最低)',1:'APP粘性较低',2:'APP粘性有潜力提升'}

lg_dfDropexe['lg_labels_define']=''
for lg_key in lg_labels.keys():
    lg_dfDropexe['lg_labels_define'][lg_dfDropexe.loc[:,'lg_label_pred']==lg_key]=lg_labels[lg_key]
lg_merge_final=lg_dfDropexe.loc[:,['driver_code','recent_login_days','login_times','login_freq','lg_labels_define']]
"""
0 最近登录距今8月左右 -- 29.6% 》 APP粘性低(最低)
1 最近登录距今2.7月左右 间隔2个月 -- 9.2% 》 APP粘性较低
2 最近登录距今2.3月左右 间隔6.55天--60.0% 》 APP粘性有潜力提升 
"""

#司机搜索查看货源
search_watch="select * from rep_op_dri_search_watch"
sw_df=query_mysql(h,p,u,pw,d,search_watch)

colList=['user_code','search_days','first_search_days','recent_search_days','watch_days','first_watch_days','recent_watch_days']
    #司机编码或者首次搜索距今null的索引
sw_null_index=sw_df.loc[:,['user_code','first_search_days']].index[np.where(pd.isnull(sw_df.loc[:,['user_code','first_search_days']]))[0]].drop_duplicates()
sw_dfDropexe=sw_df.drop(sw_null_index)

    #NA值替换成0
sw_dfDropexe=sw_dfDropexe.fillna(value=0)
data=sw_dfDropexe.loc[:,colList[1:]]
    #corr不适用于ndarray
correlations=data.corr()
    #相关系数矩阵图 first_watch_days与recent_watch_days具有强烈相关关系
    #基于大量的样本数据 只搜索没有查看 即watch=0  暂不作处理
fig=plt.figure(figsize=(10,10))
ax=fig.add_subplot(111)
cax=ax.matshow(correlations,vmin=-1,vmax=1)
fig.colorbar(cax)
ticks=np.arange(0,6,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(colList[1:])
ax.set_yticklabels(colList[1:])
plt.show()

    #首次搜索、最近搜索散点图,首次查看、最近查看散点图 检测异常值  (基本无异常)
plt.figure(1,figsize=(10,10))
plt.subplot(2,1,1)
plt.plot(sw_dfDropexe.loc[:,'first_search_days'],sw_dfDropexe.loc[:,'recent_search_days'],'o')
plt.xlabel('first_search_days')
plt.ylabel('recent_search_days')
plt.subplot(2,1,2)
plt.plot(sw_dfDropexe.loc[:,'first_watch_days'],sw_dfDropexe.loc[:,'recent_watch_days'],'o')
plt.xlabel('first_watch_days')
plt.ylabel('recent_watch_days')
plt.show()

    #剔除first_search_days first_watch_days
    #sw_X=['user_code','search_days','recent_search_days','watch_days','recent_watch_days']
sw_X=['user_code','search_days','first_search_days','recent_search_days','watch_days','recent_watch_days']
sw_theme='search_watch'
from sklearn import preprocessing
sw_data_scale=preprocessing.scale(sw_dfDropexe.loc[:,sw_X[1:]])
    #一般pca提取两个主成分
sw_pca_num=3
    #绘制最佳聚类图，返回pca得分
sw_pca_factor=pca_best_k_score(sw_theme,sw_data_scale,sw_pca_num)
    #根据最佳最聚类图定义聚类类别数，暂且定义为3
sw_pca_best_k=4
    #绘制pca得分分类图，返回类别标签
sw_label_pred=pca_classify_plot(sw_theme,sw_pca_num,sw_pca_factor,sw_pca_best_k)
    #绘制每个类样本数量条形图
sw_labels_dic=class_count_plot(sw_theme,sw_label_pred)
    #绘制样本聚类散点图，返回 原始清洗后的添加聚类标签的数据、及类别类别中心
sw_dfDropexe=class_samples_scatter(sw_theme,sw_dfDropexe,sw_X,sw_label_pred,sw_pca_best_k)
sw_dfDropexe,sw_clus_indicator_dic=list(sw_dfDropexe)[0],list(sw_dfDropexe)[1]
    #修改制定列名
sw_dfDropexe.rename(columns={'label_pred':'sw_label_pred'}, inplace = True)

sw_labels_dic_per={}
for key in sw_labels_dic.keys():
    sw_labels_dic_per[key]=(sw_labels_dic[key],sw_labels_dic[key]/sum(sw_labels_dic.values()))

sw_labels={0:'有意向获取货源，潜在引导型',1:'低意向获取货源，可能流失型',2:'高意向获取货源',3:'低意向获取货源，可能流失型'}
sw_dfDropexe['sw_labels_define']=''
for sw_key in sw_labels.keys():
    sw_dfDropexe['sw_labels_define'][sw_dfDropexe.loc[:,'sw_label_pred']==sw_key]=sw_labels[sw_key]

"""
0 最近搜索货源2.3个月 最近查看货源  --56.7%  查看行为比较新、查看次数低》 有意向获取货源，潜在引导型
1 最近搜索货11个月前 最近查看货源9个月 平均搜索5天次   --15.8%  长时间无搜索行为》 低意向获取货源，可能流失型
2 最近搜索1.5个月 最近查看货源2.5个月 查看搜索次数高  -  2.0%   最近查看比较近》 高意向获取货源
3 最近搜索9个月 最近查看货源0个月                    -- 25.4%  很早有搜索行为 近来有查看》 低意向获取货源，可能流失型
"""
sw_merge_final=sw_dfDropexe.loc[:,['user_code','search_days','first_search_days','recent_search_days','watch_days','recent_watch_days','sw_labels_define']]


#经纪人依赖reply on agent
reply_ag_df=df_copy.copy()

reply_ag_X=['driver_code','agent_order_rate']
reply_ag_dfDropexe=reply_ag_df.loc[:,reply_ag_X]

reply_ag_theme='reply_agent_degree'
from sklearn import preprocessing
reply_ag_data_scale=preprocessing.scale(reply_ag_dfDropexe.loc[:,reply_ag_X[1:]])
    #一般pca提取两个主成分
reply_ag_pca_num=1
    #绘制最佳聚类图，返回pca得分
reply_ag_pca_factor=pca_best_k_score(reply_ag_theme,reply_ag_data_scale,reply_ag_pca_num)
    #根据最佳最聚类图定义聚类类别数，暂且定义为3
reply_ag_pca_best_k=3
    #绘制pca得分分类图，返回类别标签
reply_ag_label_pred=pca_classify_plot(reply_ag_theme,reply_ag_pca_num,reply_ag_pca_factor,reply_ag_pca_best_k)
    #绘制每个类样本数量条形图
reply_ag_labels_dic=class_count_plot(reply_ag_theme,reply_ag_label_pred)
    #绘制样本聚类散点图，返回 原始清洗后的添加聚类标签的数据、及类别类别中心
reply_ag_dfDropexe=class_samples_scatter(reply_ag_theme,reply_ag_dfDropexe,reply_ag_X,reply_ag_label_pred,reply_ag_pca_best_k)
reply_ag_dfDropexe,reply_ag_clus_indicator_dic=list(reply_ag_dfDropexe)[0],list(reply_ag_dfDropexe)[1]
    #修改制定列名
reply_ag_dfDropexe.rename(columns={'label_pred':'reply_ag_label_pred'}, inplace = True)
reply_ag_labels_dic_per={}
for key in reply_ag_labels_dic.keys():
    reply_ag_labels_dic_per[key]=(reply_ag_labels_dic[key],reply_ag_labels_dic[key]/sum(reply_ag_labels_dic.values()))
reply_ag_labels_dic_per

reply_ag_labels={}
for key in reply_ag_clus_indicator_dic.keys():
    for key_sub in reply_ag_clus_indicator_dic[key].keys():
        reply_ag_temp=reply_ag_clus_indicator_dic[key][key_sub]
        reply_ag_labels[key_sub]=str(reply_ag_temp[0])+'-'+str(reply_ag_temp[1])
reply_ag_labels

reply_ag_dfDropexe['reply_ag_labels_define']=''
for reply_ag_key in reply_ag_labels.keys():
    reply_ag_dfDropexe['reply_ag_labels_define'][reply_ag_dfDropexe.loc[:,'reply_ag_label_pred']==reply_ag_key]=reply_ag_labels[reply_ag_key]

"""
0 经纪人运单占比 -- 0.23以内  84.1%》 经纪人依赖程度低
1 经纪人运单占比 --0.72以上  10.5%》 经纪人依赖程度高 
2 经纪人运单占比 -- 0.23-0.71  5.3%》 经纪人依赖程度中等
"""
reply_ag_merge_final=reply_ag_dfDropexe.loc[:,['driver_code','agent_order_rate','reply_ag_labels_define']]

df_copy_1=df_copy.loc[:,['driver_code','com_cnt','upload_province','order_cnt','per_order_pay','per_order_lc']]

df_merge_1=pd.merge(df_copy_1,loss_merge_final,left_on ='driver_code',right_on = 'driver_code',how ='left')
df_merge_1=pd.merge(df_merge_1,age_merge_final,left_on ='driver_code',right_on = 'driver_code',how ='left')
df_merge_1=pd.merge(df_merge_1,active_merge_final,left_on ='driver_code',right_on = 'driver_code',how ='left')
df_merge_1=pd.merge(df_merge_1,volumn_merge_final,left_on ='driver_code',right_on = 'driver_code',how ='left')
df_merge_1=pd.merge(df_merge_1,sc_merge_final,left_on ='driver_code',right_on = 'driver_code',how ='left')
df_merge_1=pd.merge(df_merge_1,lg_merge_final,left_on ='driver_code',right_on = 'driver_code',how ='left')
df_merge_1=pd.merge(df_merge_1,sw_merge_final,left_on ='driver_code',right_on = 'user_code',how ='left')
df_merge_1=pd.merge(df_merge_1,reply_ag_merge_final,left_on ='driver_code',right_on = 'driver_code',how ='left')


import pandas as pd
from sqlalchemy import create_engine
##将数据写入mysql的数据库，但需要先通过sqlalchemy.create_engine建立连接,且字符编码设置为utf8，否则有些latin字符不能处理
yconnect= create_engine('mysql+pymysql://{0}:{1}@{2}:{3}/{4}?charset=utf8'.format(u,pw,h,p,d))

pd.io.sql.to_sql(df_merge_1,'rep_op_driver_clusters_result', yconnect, schema='repm',index=False, if_exists='append')

end_time=datetime.datetime.now()

waste_seconds=(end_time-start_time).seconds
print(waste_seconds)

