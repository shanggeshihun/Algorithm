# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
#from sklearn import metrics
from sklearn.decomposition import PCA
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
#    df=pd.DataFrame({'a':[1,2,3,4,5],'b':[6,7,8,9,10],'c':[6,7,8,9,10]},index=pd.date_range('2019-01-01','2019-01-05'))
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
    #返回模型的特征向量
    pca.components_
    #返回各个成分的方差百分比
    ratio=pca.explained_variance_ratio_*100
    print('PCA贡献率','\n',ratio)
    #保留前2个主成分 
    pca=PCA(pca_num)
    pca.fit(data_scale)
    pca.components_
    #用data_scale来训练PCA模型，同时返回降维后的数据
    pca_factor=pca.fit_transform(data_scale)
#    绘制得分聚类分布图
    best_k(theme,pca_factor)
    #肘部法则判断最佳聚类k值
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
    plt.figure(figsize=(8,4))
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
    plt.figure(1,figsize=(15,15))
    plt.figure(1).suptitle(theme + '\n +the differences between labels',fontsize=15)
    #i:第i个指标
    for i in range(1,len(X)):
        plt.subplot(len(X)-1,1,i)
        plt.plot(range(best_k),dfDropexegp.loc[:,X[i]])
        plt.ylabel(X[i])
        plt.xticks(range(best_k))
    #    xj：第xj类
        for xj in range(best_k):
#            第j类dataframe
            df_temp=dfDropexe[dfDropexe.loc[:,'label_pred']==xj]
            xij_max=np.round(np.max(df_temp.loc[:,X[i]]),2)
            xij_min=np.round(np.min(df_temp.loc[:,X[i]]),2)
            xij_center=np.round(dfDropexegp.loc[xj,X[i]],2)
#            添加聚类中心点数据标签
            plt.text(xj+0.1,dfDropexegp.loc[xj,X[i]],xij_center,ha='center', va= 'bottom')
#            添加最值数据标签
            plt.text(xj+0.08,xij_max,xij_max,ha='center', va= 'bottom')
            plt.text(xj+0.08,xij_min,xij_min,ha='center', va= 'bottom')

            plt.scatter(df_temp.loc[:,'label_pred'],df_temp.loc[:,X[i]])
    plt.savefig(r"C:\\Users\\dell\\Desktop\\" +theme + "&label_scatter.png")
    plt.show()
    return dfDropexe

#原始数据源
df=pd.read_excel(r"C:\Users\dell\Desktop\rfm_login_2.xlsx")
df = df.drop(df[pd.isnull(df['driver_code'])].index)
df_copy=df.copy()


#流失客户
loss_df=df_copy.copy()

#服务公司数量与运单量散点图检测异常值
plt.plot(loss_df.loc[:,'com_cnt'],loss_df.loc[:,'order_cnt'],'o')
plt.xlabel('com_cnt')
plt.ylabel('order_cnt')
plt.show()

#服务公司1家，攻击跑单超过2000单很可能是异常值
dfDropexe=loss_df[(loss_df.loc[:,'order_cnt']<2000) & (loss_df.loc[:,'dri_age']>0)]
#司机年龄区间0-100
dfDropexe=dfDropexe[(dfDropexe.loc[:,'dri_age']<100) & (dfDropexe.loc[:,'dri_age']>0)]


######################流失
#loss_X=['driver_code','first_order_days','recent_order_days']
loss_X=['driver_code','recent_order_days']
#清洗后的原始数据'
loss_dfDropexe=dfDropexe.loc[:,loss_X]
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
#绘制样本聚类散点图，返回 原始清洗后的添加聚类标签的数据
loss_dfDropexe=class_samples_scatter(loss_theme,loss_dfDropexe,loss_X,loss_label_pred,loss_pca_best_k)
#修改制定列名
loss_dfDropexe.rename(columns={'label_pred':'loss_label_pred'}, inplace = True)
loss_labels_dic_per={}
for key in loss_labels_dic.keys():
    loss_labels_dic_per[key]=(loss_labels_dic[key],loss_labels_dic[key]/sum(loss_labels_dic.values()))
loss_labels_dic_per
loss_labels={0:'145-402|轻度流失客户',1:'403-778|重度流失客户',2:'1-144|正常客户'}
loss_dfDropexe['loss_labels_define']=''
for loss_key in loss_labels.keys():
    loss_dfDropexe['loss_labels_define'][loss_dfDropexe.loc[:,'loss_label_pred']==loss_key]=loss_labels[loss_key]
"""
0 145-402未跑单 25.37%  》 轻度流失客户
1 403-778未跑单 4.49%  》 重度流失客户
2 1-144未跑单 70.1%  》 正常客户

跑单时间：17后
type：煤炭
"""




######################司机年龄
age_df=df_copy.copy()
#ri_age散点图检测异常值
plt.boxplot(age_df.loc[:,'dri_age'])
plt.show()

#剔除dri_age异常
age_dfDropexe=age_df[(age_df.loc[:,'dri_age']<100) & (age_df.loc[:,'dri_age']>0)]
plt.boxplot(age_dfDropexe.loc[:,'dri_age'])
plt.show()

age_X=['driver_code','dri_age']
#清洗后的原始数据'
age_dfDropexe=dfDropexe.loc[:,age_X]
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
age_dfDropexe=class_samples_scatter(age_theme,age_dfDropexe,age_X,age_label_pred,age_pca_best_k)
#修改制定列名
age_dfDropexe.rename(columns={'label_pred':'age_label_pred'}, inplace = True)
age_labels_dic_per={}
for key in age_labels_dic.keys():
    age_labels_dic_per[key]=(age_labels_dic[key],age_labels_dic[key]/sum(age_labels_dic.values()))
age_labels_dic_per
age_labels={0:'38-45',1:'46-69',2:'21-37'}
age_dfDropexe['age_labels_define']=''
for age_key in age_labels.keys():
    age_dfDropexe['age_labels_define'][age_dfDropexe.loc[:,'age_label_pred']==age_key]=age_labels[age_key]
"""
0 38-45 39.5%  》 
1 46-69 26.1%  》 
2 21-37 34.3%  》 

跑单时间：17后
type：煤炭
"""



########################活跃频次
active_df=df_copy.copy()

#服务公司数量与运单量散点图检测异常值
plt.plot(active_df.loc[:,'com_cnt'],active_df.loc[:,'order_cnt'],'o')
plt.xlabel('com_cnt')
plt.ylabel('order_cnt')
plt.show()

#服务公司1家，攻击跑单超过2000单很可能是异常值
dfDropexe=active_df[(active_df.loc[:,'order_cnt']<2000) & (active_df.loc[:,'dri_age']>0)]
#司机年龄区间0-100
dfDropexe=dfDropexe[(dfDropexe.loc[:,'dri_age']<100) & (dfDropexe.loc[:,'dri_age']>0)]

#最近跑单与跑单频率 散点图检测异常值
plt.plot(dfDropexe.loc[:,'recent_order_days'],dfDropexe.loc[:,'order_freq'],'o')
plt.xlabel('recent_order_days')
plt.ylabel('order_freq')
plt.show()
#异常处理
active_dfDropexe=dfDropexe.copy()

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
#绘制样本聚类散点图，返回 原始清洗后的添加聚类标签的数据
active_dfDropexe=class_samples_scatter(active_theme,active_dfDropexe,active_X,active_label_pred,active_pca_best_k)
#修改制定列名
active_dfDropexe.rename(columns={'label_pred':'active_label_pred'}, inplace = True)
active_labels_dic_per={}
for key in active_labels_dic.keys():
    active_labels_dic_per[key]=(active_labels_dic[key],active_labels_dic[key]/sum(active_labels_dic.values()))
active_labels_dic_per
active_labels={0:'0-28|高频',1:'28-107|中频',2:'107-|低频'}
active_dfDropexe['active_labels_define']=''
for active_key in active_labels.keys():
    active_dfDropexe['active_labels_define'][active_dfDropexe.loc[:,'active_label_pred']==active_key]=active_labels[active_key]
"""
0 频率0-28天 -- 86.2%   》 高频
1 频率28-107天 -- 12.3%  》 中频
2 频率107-天 -- 1.4%  》 低频
跑单时间：17后
type：煤炭
"""


#################司机体量指标volumn
volumn_df=df_copy.copy()
#跑单数量与下单总额散点图检测异常值
plt.figure(1,figsize=(10,10))
plt.plot(volumn_df.loc[:,'per_order_pay'],volumn_df.loc[:,'order_cnt'],'o')
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
volumn_dfDropexe=dfDropexe.loc[:,volumn_X]
volumn_dfDropexe.head(3)

volumn_theme='driver_volumn'
from sklearn import preprocessing
volumn_data_scale=preprocessing.scale(volumn_dfDropexe.loc[:,volumn_X[1:]])
#一般pca提取两个主成分
volumn_pca_num=1
#绘制最佳聚类图，返回pca得分
volumn_pca_factor=pca_best_k_score(volumn_theme,volumn_data_scale,volumn_pca_num)
#根据最佳最聚类图定义聚类类别数，暂且定义为3
volumn_pca_best_k=4
#绘制pca得分分类图，返回类别标签
volumn_label_pred=pca_classify_plot(volumn_theme,volumn_pca_num,volumn_pca_factor,volumn_pca_best_k)
#绘制每个类样本数量条形图
volumn_labels_dic=class_count_plot(volumn_theme,volumn_label_pred)
#绘制样本聚类散点图，返回 原始清洗后的添加聚类标签的数据
volumn_dfDropexe=class_samples_scatter(volumn_theme,volumn_dfDropexe,volumn_X,volumn_label_pred,volumn_pca_best_k)
#修改制定列名
volumn_dfDropexe.rename(columns={'label_pred':'volumn_label_pred'}, inplace = True)
volumn_labels_dic_per={}
for key in volumn_labels_dic.keys():
    volumn_labels_dic_per[key]=(volumn_labels_dic[key],volumn_labels_dic[key]/sum(volumn_labels_dic.values()))
volumn_labels_dic_per
volumn_labels={0:'9.6-39万|中体量',1:'9.6万内|小体量',2:'39万以上|大体量'}
volumn_dfDropexe['volumn_labels_define']=''
for volumn_key in volumn_labels.keys():
    volumn_dfDropexe['volumn_labels_define'][volumn_dfDropexe.loc[:,'volumn_label_pred']==volumn_key]=volumn_labels[volumn_key]
"""
0 下单金额9.6-39万  8.3%  》 中体量
1 下单金额9.6万内--90.9%  》  小体量
2 下单金额39万以上--0.78%  》 大体量
"""


##############司机跑单习惯（为多少公司跑单serve_company，目的地一般有多少个省份）
sc_df=df_copy.copy()
#服务公司数量与卸货地省份数量 散点图检测异常值
plt.figure(1,figsize=(10,10))
plt.plot(sc_df.loc[:,'com_cnt'],sc_df.loc[:,'upload_province'],'o')
plt.xlabel('com_cnt')
plt.ylabel('upload_province')
plt.show()

#异常处理
sc_dfDropexe=sc_df.copy()

sc_X=['driver_code','com_cnt','upload_province']
sc_dfDropexe=dfDropexe.loc[:,sc_X]
#跑单数量与下单总额散点图检测异常值
plt.figure(1,figsize=(10,10))
plt.plot(sc_dfDropexe.loc[:,'com_cnt'],sc_dfDropexe.loc[:,'upload_province'],'o')
plt.xlabel('com_cnt')
plt.ylabel('upload_province')
plt.show()

sc_theme='serve_company'
from sklearn import preprocessing
sc_data_scale=preprocessing.scale(sc_dfDropexe.loc[:,sc_X[1:]])
#一般pca提取两个主成分
sc_pca_num=2
#绘制最佳聚类图，返回pca得分
sc_pca_factor=pca_best_k_score(sc_theme,sc_data_scale,sc_pca_num)
#根据最佳最聚类图定义聚类类别数，暂且定义为3
sc_pca_best_k=1
#绘制pca得分分类图，返回类别标签
sc_label_pred=pca_classify_plot(sc_theme,sc_pca_num,sc_pca_factor,sc_pca_best_k)
#绘制每个类样本数量条形图
sc_labels_dic=class_count_plot(sc_theme,sc_label_pred)
#绘制样本聚类散点图，返回 原始清洗后的添加聚类标签的数据
sc_dfDropexe=class_samples_scatter(sc_theme,sc_dfDropexe,sc_X,sc_label_pred,sc_pca_best_k)
#修改制定列名
sc_dfDropexe.rename(columns={'label_pred':'sc_label_pred'}, inplace = True)
sc_labels_dic_per={}
for key in sc_labels_dic.keys():
    sc_labels_dic_per[key]=(sc_labels_dic[key],sc_labels_dic[key]/sum(sc_labels_dic.values()))
sc_labels_dic_per
sc_labels={0:'公司及卸货省单一',2:'公司多样卸货省单一',1:'公司及卸货省多样'}
sc_dfDropexe['sc_labels_define']=''
for sc_key in sc_labels.keys():
    sc_dfDropexe['sc_labels_define'][sc_dfDropexe.loc[:,'sc_label_pred']==sc_key]=sc_labels[sc_key]

"""
0 单个公司单个省份  -- 80.0%  》 公司及卸货省单一
1 多个公司单个省份  -- 10.9%  》 公司多样卸货省单一
2 多个公司多个省份  -- 9.0%  》 公司及卸货省多样
"""



################登陆次数与首次登录时长、最近登录时长散点图检测异常值
lg_df=df_copy.copy()

plt.figure(1,figsize=(8,8))
plt.figure(1).suptitle('the login exception dot',fontsize=15)
plt.subplot(211)
plt.plot(lg_df.loc[:,'first_login_days'],lg_df.loc[:,'login_times'],'o')
plt.xlabel('first_login_days')
plt.ylabel('login_times')
plt.subplot(212)
plt.plot(lg_df.loc[:,'recent_login_days'],lg_df.loc[:,'login_times'],'o')
plt.xlabel('recent_login_days')
plt.ylabel('login_times')
plt.show()

#异常处理
#剔除最近登录时间异常值(>2500)  剔除登录次数异常值(超过4000)
lg_dfDropexe=lg_df[(lg_df.loc[:,'login_times']<4000) & (lg_df.loc[:,'recent_login_days']<2500) &  (lg_df.loc[:,'recent_login_days']>0)]
#暂时直接删除缺失值
lg_dfDropexe=lg_dfDropexe.dropna(axis=0)
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
#绘制样本聚类散点图，返回 原始清洗后的添加聚类标签的数据
lg_dfDropexe=class_samples_scatter(lg_theme,lg_dfDropexe,lg_X,lg_label_pred,lg_pca_best_k)
#修改制定列名
lg_dfDropexe.rename(columns={'label_pred':'lg_label_pred'}, inplace = True)
lg_labels_dic_per={}
for key in lg_labels_dic.keys():
    lg_labels_dic_per[key]=(lg_labels_dic[key],lg_labels_dic[key]/sum(lg_labels_dic.values()))
lg_labels_dic_per
lg_labels={0:'APP粘性有潜力提升',1:'APP粘性低(最低)',2:'APP粘性较低'}
lg_dfDropexe['lg_labels_define']=''
for lg_key in lg_labels.keys():
    lg_dfDropexe['lg_labels_define'][lg_dfDropexe.loc[:,'lg_label_pred']==lg_key]=lg_labels[lg_key]

"""
0 最近登录距今2.3月左右 间隔6天天--61.0% 》 APP粘性有潜力提升 
1 最近登录距今8月左右 -- 29.6% 》 APP粘性低(最低)
2 最近登录距今2.7月左右 间隔2个月 -- 9.2% 》 APP粘性较低
"""

lg_dfDropexe.to_excel(r"C:\Users\dell\Desktop\temp.xlsx")


##################司机搜索查看货源
df=pd.read_excel(r"C:\Users\dell\Desktop\search_watch_2.xlsx")
colList=['user_code','search_days','first_search_days','recent_search_days','watch_days','first_watch_days','recent_watch_days']
#NA值替换成0
dfFillna=df.fillna(value=0)
data=dfFillna.loc[:,colList[1:]]
#corr不适用于ndarray
correlations=data.corr()
#相关系数矩阵图 first_watch_days与recent_watch_days具有强烈相关关系
#基于大量的样本数据 只搜索没有查看 即watch=0  暂不作处理
fig=plt.figure(figsize=(15,15))
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
plt.figure(1)
plt.subplot(2,1,1)
plt.plot(dfFillna.loc[:,'first_search_days'],dfFillna.loc[:,'recent_search_days'],'o')
plt.xlabel('first_search_days')
plt.ylabel('recent_search_days')
plt.subplot(2,1,2)
plt.plot(dfFillna.loc[:,'first_watch_days'],dfFillna.loc[:,'recent_watch_days'],'o')
plt.xlabel('first_watch_days')
plt.ylabel('recent_watch_days')
plt.show()
sw_dfDropexe=dfFillna.copy()
#剔除first_search_days first_watch_days
sw_X=['user_code','search_days','recent_search_days','watch_days','recent_watch_days']
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
#绘制样本聚类散点图，返回 原始清洗后的添加聚类标签的数据
sw_dfDropexe=class_samples_scatter(sw_theme,sw_dfDropexe,sw_X,sw_label_pred,sw_pca_best_k)
#修改制定列名
sw_dfDropexe.rename(columns={'label_pred':'sw_label_pred'}, inplace = True)
sw_labels_dic_per={}
for key in sw_labels_dic.keys():
    sw_labels_dic_per[key]=(sw_labels_dic[key],sw_labels_dic[key]/sum(sw_labels_dic.values()))
sw_labels_dic_per
sw_labels={0:'有意向获取货源，潜在引导型',1:'高意向获取货源',2:'低意向获取货源，可能流失型',3:'低意向获取货源，可能流失型'}
sw_dfDropexe['sw_labels_define']=''
for sw_key in sw_labels.keys():
    sw_dfDropexe['sw_labels_define'][sw_dfDropexe.loc[:,'sw_label_pred']==sw_key]=sw_labels[sw_key]

"""
0 最近搜索货源2.3个月 最近查看货源1个月  --56.7%  查看行为比较新、查看次数低》 有意向获取货源，潜在引导型
1 最近搜索1.5个月 最近查看货源2.5个月 查看搜索次数高  -  2.0%   最近查看比较近》 高意向获取货源
2 最近搜索货源8个月前 最近查看货源9个月 平均搜索5天次   --15.8%  长时间无搜索行为》 低意向获取货源，可能流失型
3 最近搜索9个月 最近查看货源0个月                    -- 25.4%  很早有搜索行为 近来有查看》 低意向获取货源，可能流失型
"""


###########################经纪人依赖reply on agent
reply_ag_df=df_copy.copy()

#服务公司数量与运单量散点图检测异常值
plt.plot(reply_ag_df.loc[:,'com_cnt'],reply_ag_df.loc[:,'order_cnt'],'o')
plt.xlabel('com_cnt')
plt.ylabel('order_cnt')
plt.show()

#服务公司1家，攻击跑单超过2000单很可能是异常值
dfDropexe=reply_ag_df[(reply_ag_df.loc[:,'order_cnt']<2000) & (reply_ag_df.loc[:,'dri_age']>0)]
#司机年龄区间0-100
dfDropexe=dfDropexe[(dfDropexe.loc[:,'dri_age']<100) & (dfDropexe.loc[:,'dri_age']>0)]


reply_ag_X=['driver_code','agent_order_rate']
reply_ag_dfDropexe=dfDropexe.loc[:,reply_ag_X]

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
#绘制样本聚类散点图，返回 原始清洗后的添加聚类标签的数据
reply_ag_dfDropexe=class_samples_scatter(reply_ag_theme,reply_ag_dfDropexe,reply_ag_X,reply_ag_label_pred,reply_ag_pca_best_k)
#修改制定列名
reply_ag_dfDropexe.rename(columns={'label_pred':'reply_ag_label_pred'}, inplace = True)
reply_ag_labels_dic_per={}
for key in reply_ag_labels_dic.keys():
    reply_ag_labels_dic_per[key]=(reply_ag_labels_dic[key],reply_ag_labels_dic[key]/sum(reply_ag_labels_dic.values()))
reply_ag_labels_dic_per
reply_ag_labels={0:'占比23%以内|经纪人依赖程度低',1:'占比73%以上|经纪人依赖程度高',2:'占比23-72%|经纪人依赖程度中等'}
reply_ag_dfDropexe['reply_ag_labels_define']=''
for reply_ag_key in reply_ag_labels.keys():
    reply_ag_dfDropexe['reply_ag_labels_define'][reply_ag_dfDropexe.loc[:,'reply_ag_label_pred']==reply_ag_key]=reply_ag_labels[reply_ag_key]

reply_ag_dfDropexe.to_excel(r"C:\Users\dell\Desktop\search_watch.xlsx")
"""
0 经纪人运单占比 -- 0.23以内  84.1%》 经纪人依赖程度低
1 经纪人运单占比 --0.73以上  10.5%》 经纪人依赖程度高 
2 经纪人运单占比 -- 0.23-0.72 5.3%》 经纪人依赖程度中等

"""



df_copy.head(2)
loss_dfDropexe.head(3)
age_dfDropexe.head(3)
active_dfDropexe.head(3)
volumn_dfDropexe.head(3)
sc_dfDropexe.head(3)
lg_dfDropexe.head(3)
sw_dfDropexe.head(3)
reply_ag_dfDropexe.head(3)

loss_labels_dic_per
active_labels_dic_per
volumn_labels_dic_per
sc_labels_dic_per
lg_labels_dic_per
sw_labels_dic_per
reply_ag_labels_dic_per



df_merge=pd.merge(df_copy,loss_dfDropexe,left_on ='driver_code',right_on = 'driver_code',how ='left')
df_merge=pd.merge(df_merge,reply_ag_dfDropexe,left_on ='driver_code',right_on = 'driver_code',how ='left')
df_merge=pd.merge(df_merge,age_dfDropexe,left_on ='driver_code',right_on = 'driver_code',how ='left')
df_merge=pd.merge(df_merge,active_dfDropexe,left_on ='driver_code',right_on = 'driver_code',how ='left')
df_merge=pd.merge(df_merge,volumn_dfDropexe,left_on ='driver_code',right_on = 'driver_code',how ='left')
df_merge=pd.merge(df_merge,sc_dfDropexe,left_on ='driver_code',right_on = 'driver_code',how ='left')
df_merge=pd.merge(df_merge,lg_dfDropexe,left_on ='driver_code',right_on = 'driver_code',how ='left')
df_merge=pd.merge(df_merge,sw_dfDropexe,left_on ='driver_code',right_on = 'user_code',how ='left')

df_merge.head(3).columns
df_merge.to_excel(r"C:\Users\dell\Desktop\df_merge_5.xlsx")