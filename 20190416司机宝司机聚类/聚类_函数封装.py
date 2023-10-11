# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
        kmeans=KMeans(n_clusters=k)
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
#    k_delta=[]
#    for i in range(len(meandistortions)-1):
#        k_delta.append(meandistortions[i]-meandistortions[i+1])
#
#    kk_delta=[]
#    for i in range(len(k_delta)-1):
#        kk_delta.append(k_delta[i]-k_delta[i+1])
#    k_2=kk_delta.index(max(kk_delta))+2
#    return k_2


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


#绘制前2个pca得分分类图，返回类别标签
def pca_classify_plot(theme,pca_num,pca_factor,pca_best_k):
    """
    pca_num：提取的主成分个数
    pca_factor:提取的主成分得分
    pca_best_k:通过主成分得分最佳聚类k
    """
    best_k=pca_best_k
    estimator=KMeans(n_clusters=best_k)
    estimator.fit(pca_factor)
    #通过pca返回聚类标签
    label_pred=estimator.labels_
    pca_factor_labels=np.insert(pca_factor,pca_num,label_pred,axis=1)
    #pca降维聚类分布图
    colors=['royalblue','salmon','lightgreen','c','m','y','k','b']
    markers=['o','s','D','v','^','p','*','+']
    
    for i in range(2):
        temp_x=pca_factor[pca_factor_labels[:,pca_num]==i][:,0]
        temp_y=pca_factor[pca_factor_labels[:,pca_num]==i][:,1]
        plt.plot(temp_x,temp_y,color=colors[i],marker=markers[i],ls='None',alpha=0.05)
    plt.title(theme+ '\n pca_cluster_distribution',fontsize=15)
    plt.savefig(r"C:\\Users\\dell\\Desktop\\" + theme +" &pca_cluster_distribution.png")
    plt.show()
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
            plt.text(xj+0.1,dfDropexegp.loc[xj,X[i]],np.int(dfDropexegp.loc[xj,X[i]]),ha='center', va= 'bottom')
            df_temp=dfDropexe[dfDropexe.loc[:,'label_pred']==xj]
            plt.scatter(df_temp.loc[:,'label_pred'],df_temp.loc[:,X[i]])
    plt.savefig(r"C:\\Users\\dell\\Desktop\\" +theme + "&label_scatter.png")
    plt.show()
    return dfDropexe



#司机跑单数量、最近跑单时间、最初跑单时间、登录时间
df=pd.read_excel(r"C:\Users\dell\Desktop\rfm_login.xlsx")
df_copy=df.copy()
dfDropna=df.dropna(axis=0)
#服务公司数量与运单量散点图检测异常值
plt.plot(dfDropna.loc[:,'com_cnt'],dfDropna.loc[:,'order_cnt'],'o')
plt.xlabel('com_cnt')
plt.ylabel('order_cnt')
plt.show()

#服务公司1家，攻击跑单超过2000单很可能是异常值
dfDropexe=dfDropna[(dfDropna.loc[:,'order_cnt']<2000) & (dfDropna.loc[:,'dri_age']>0)]
#剔除dri_age  在不同类之间表现出来的差异性不大
#剔除dri_age  在不同类之间表现出来的差异性不大
#X=['driver_code','order_cnt','first_order_days','recent_order_days']
X=['driver_code','first_order_days','recent_order_days']
#清洗后的原始数据'
o_dfDropexe=dfDropexe.loc[:,X]
o_theme='create_order'
from sklearn import preprocessing
o_data_scale=preprocessing.scale(o_dfDropexe.loc[:,X[1:]])
#一般pca提取两个主成分
o_pca_num=2
#绘制最佳聚类图，返回pca得分
o_pca_factor=pca_best_k_score(o_theme,o_data_scale,o_pca_num)
#根据最佳最聚类图定义聚类类别数，暂且定义为3
o_pca_best_k=3
#绘制pca得分分类图，返回类别标签
o_label_pred=pca_classify_plot(o_theme,o_pca_num,o_pca_factor,o_pca_best_k)
#绘制每个类样本数量条形图
o_labels_dic=class_count_plot(o_theme,o_label_pred)
#绘制样本聚类散点图，返回 原始清洗后的添加聚类标签的数据
o_dfDropexe=class_samples_scatter(o_theme,o_dfDropexe,X,o_label_pred,o_pca_best_k)
#修改制定列名
o_dfDropexe.rename(columns={'label_pred':'o_label_pred'}, inplace = True)
o_labels_dic_per={}
for key in o_labels_dic.keys():
    o_labels_dic_per[key]=(o_labels_dic[key],o_labels_dic[key]/sum(o_labels_dic.values()))
o_labels_dic_per
o_labels={1:'以新客为主的跑单潜力客户',0:'以老客为主的跑单流失或可能流失客户',2:'以老客为主的跑单频繁客户'}
o_dfDropexe['o_labels_define']=''
for o_key in o_labels.keys():
    o_dfDropexe['o_labels_define'][o_dfDropexe.loc[:,'o_label_pred']==o_key]=o_labels[o_key]
"""
1 首次跑单5个月 平均跑单5单 最近跑单距今3月左右 --86.7%  首跑时间短、跑单少》以新客为主的跑单潜力客户
0 首次跑单19个月 平均跑单5单 最近跑单距今14月左右 -- 11.8%  首跑时间长、跑单少、最近跑单久远》以老客为主的跑单流失或可能流失客户
2 首次跑单10个月 平均跑单115单 最近跑单距今2月左右 -- 1.47%  首跑时间中长、跑单多、最近跑单近》以老客为主的跑单频繁客户

跑单时间：17后
type：煤炭
"""


#司机体量指标volumn
v_X=['driver_code','order_cnt','per_order_pay','order_pay']
v_dfDropexe=dfDropexe.loc[:,v_X]
v_dfDropexe.head(3)
#跑单数量与下单总额散点图检测异常值
plt.figure(1,figsize=(10,10))
plt.subplot(2,1,1)
plt.plot(v_dfDropexe.loc[:,'order_cnt'],v_dfDropexe.loc[:,'order_pay'],'o')
plt.xlabel('order_cnt')
plt.ylabel('order_pay')
plt.subplot(2,1,2)
plt.plot(v_dfDropexe.loc[:,'order_cnt'],v_dfDropexe.loc[:,'per_order_pay'],'o')
plt.xlabel('order_cnt')
plt.ylabel('per_order_pay')
plt.show()

v_theme='driver_volumn'
from sklearn import preprocessing
v_data_scale=preprocessing.scale(v_dfDropexe.loc[:,v_X[1:]])
#一般pca提取两个主成分
v_pca_num=2
#绘制最佳聚类图，返回pca得分
v_pca_factor=pca_best_k_score(v_theme,v_data_scale,v_pca_num)
#根据最佳最聚类图定义聚类类别数，暂且定义为3
v_pca_best_k=3
#绘制pca得分分类图，返回类别标签
v_label_pred=pca_classify_plot(v_theme,v_pca_num,v_pca_factor,v_pca_best_k)
#绘制每个类样本数量条形图
v_labels_dic=class_count_plot(v_theme,v_label_pred)
#绘制样本聚类散点图，返回 原始清洗后的添加聚类标签的数据
v_dfDropexe=class_samples_scatter(v_theme,v_dfDropexe,v_X,v_label_pred,v_pca_best_k)
#修改制定列名
v_dfDropexe.rename(columns={'label_pred':'v_label_pred'}, inplace = True)
v_labels_dic_per={}
for key in v_labels_dic.keys():
    v_labels_dic_per[key]=(v_labels_dic[key],v_labels_dic[key]/sum(v_labels_dic.values()))
v_labels_dic_per
v_labels={2:'以单均万级为主，小体量',1:'单均6K为主，大体量',0:'单均5K为主，小体量'}
v_dfDropexe['v_labels_define']=''
for v_key in v_labels.keys():
    v_dfDropexe['v_labels_define'][v_dfDropexe.loc[:,'v_label_pred']==v_key]=v_labels[v_key]
"""
2 跑单3单 下单金额3.9万 单均1.1万--32.2%  》 以单均万级为主，小体量
1 跑单77单 下单金额34.7万 单均0.6万--2.9%  》 单均6K为主，大体量
0 跑单5单 下单金额2.5万 单均0.5万--64.8%  》 单均5K为主，小体量
"""



#司机跑单习惯（为多少公司跑单serve_company，目的地一般有多少个省份）
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
sc_pca_best_k=3
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
sc_labels={0:'公司目的省单一化',1:'公司目的省多样化',2:'公司多样化目的省份单一化'}
sc_dfDropexe['sc_labels_define']=''
for sc_key in sc_labels.keys():
    sc_dfDropexe['sc_labels_define'][sc_dfDropexe.loc[:,'sc_label_pred']==sc_key]=sc_labels[sc_key]

"""
0 单个公司单个省份  -- 80.5%  》 公司目的省单一化
1 多个公司多个省份  -- 10.6%  》 公司目的省多样化
2 多个公司单个省份  -- 8.8%  》 公司多样化目的省份单一化
"""




#登陆次数与首次登录时长、最近登录时长散点图检测异常值
plt.figure(1,figsize=(8,8))
plt.figure(1).suptitle('the login exception dot',fontsize=15)
plt.subplot(211)
plt.plot(df.loc[:,'first_login_days'],df.loc[:,'login_times'],'o')
plt.xlabel('first_login_days')
plt.ylabel('login_times')
plt.subplot(212)
plt.plot(df.loc[:,'recent_login_days'],df.loc[:,'login_times'],'o')
plt.xlabel('recent_login_days')
plt.ylabel('login_times')
plt.show()
#剔除最近登录时间异常值(>2500)  剔除登录次数异常值(超过4000)
dfDropexe=df[(df.loc[:,'login_times']<4000) & (df.loc[:,'recent_login_days']<2500)]
#暂时直接删除缺失值
dfDropexe=dfDropexe.dropna(axis=0)
lg_X=['driver_code','first_login_days','recent_login_days','login_times']
lg_dfDropexe=dfDropexe.loc[:,lg_X]

lg_theme='login'
from sklearn import preprocessing
lg_data_scale=preprocessing.scale(lg_dfDropexe.loc[:,lg_X[1:]])
#一般pca提取两个主成分
lg_pca_num=2
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
lg_labels={0:'APP粘性有潜力再提升',2:'APP粘性低',1:'APP粘性高'}
lg_dfDropexe['lg_labels_define']=''
for lg_key in lg_labels.keys():
    lg_dfDropexe['lg_labels_define'][lg_dfDropexe.loc[:,'lg_label_pred']==lg_key]=lg_labels[lg_key]

"""
0 首次登录5个月 平均登录9天次 最近登录距今3月左右 --82.3%  使用APP年限短、月均登录1次》 APP粘性有潜力再提升 
2 首次登录17个月 平均登录21天次 最近登录距今12.6月左右 -- 16.3% 使用APP年限长、最近登录久远》 APP粘性低
1 首次登录15个月 平均登录332天次 最近登录距今2.5月左右 -- 1.3%  使用APP年限长、最近登录久远》 APP粘性高
"""



#司机搜索查看货源
df=pd.read_excel(r"C:\Users\dell\Desktop\search_watch_1.xlsx")
colList=['user_code','search_days','first_search_days','recent_search_days','watch_days','first_watch_days','recent_watch_days','search_freq','watch_freq']
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
sw_X=['user_code','search_days','first_search_days','recent_search_days','search_freq']
sw_theme='search_watch'
from sklearn import preprocessing
sw_data_scale=preprocessing.scale(sw_dfDropexe.loc[:,sw_X[1:]])
#一般pca提取两个主成分
sw_pca_num=3
#绘制最佳聚类图，返回pca得分
sw_pca_factor=pca_best_k_score(sw_theme,sw_data_scale,sw_pca_num)
#根据最佳最聚类图定义聚类类别数，暂且定义为3
sw_pca_best_k=3
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
sw_labels={0:'获取货源 意向流失型',1:'有意向获取货源，潜在引导型',2:'高意向获取货源，忠诚型',3:'不稳定意向获取货源'}
sw_dfDropexe['sw_labels_define']=''
for sw_key in sw_labels.keys():
    sw_dfDropexe['sw_labels_define'][sw_dfDropexe.loc[:,'sw_label_pred']==sw_key]=sw_labels[sw_key]

"""
0 首次搜索货源10个月 最近搜索货源9.5个月前   --15.7%  长时间无搜索行为》 获取货源 意向流失型
1 首次搜索3.5个月 最近搜索货源2.5个月   --52.9%  搜索行为比较新、查看次数低》 获取货源 意向引导型
2 首次搜索10个月 最近搜索货源2.5个月  -- 31.3%  很早有搜索行为 查看次数高》 获取货源，意向忠诚型
"""
df_copy.head(2)
o_dfDropexe.head(3)
sc_dfDropexe.head(3)
lg_dfDropexe.head(3)
sw_dfDropexe.head(3)
v_dfDropexe.head(3)

df_merge=pd.merge(df_copy,o_dfDropexe,left_on ='driver_code',right_on = 'driver_code',how ='left')
df_merge=pd.merge(df_merge,sc_dfDropexe,left_on ='driver_code',right_on = 'driver_code',how ='left')
df_merge=pd.merge(df_merge,lg_dfDropexe,left_on ='driver_code',right_on = 'driver_code',how ='left')
df_merge=pd.merge(df_merge,sw_dfDropexe,left_on ='driver_code',right_on = 'user_code',how ='left')
df_merge=pd.merge(df_merge,v_dfDropexe,left_on ='driver_code',right_on = 'driver_code',how ='left')
df_merge.head(3)
df_merge.to_excel(r"C:\Users\dell\Desktop\df_merge_3.xlsx")


1+2