# coding:utf-8
# _*_coding:utf-8 _*_
# @Time　　 : 2020/03/08   0:00
# @Author　 : zimo
# @File　   :
# @Software :PyCharm

# 逻辑回归回归分类器
from sklearn.linear_model import LogisticRegression
# 带交叉验证的逻辑回归分类器
from sklearn.linear_model import LogisticRegressionCV
# 计算Logistic回归模型以获得正则化参数的列表
from sklearn.linear_model import logistic_regression_path
# 利用梯度下降求解的线性分类器
from sklearn.linear_model import SGDClassifier
# 利用梯度下降最小化正则化后的损失函数的线性回归模型
from sklearn.linear_model import SGDRegressor

# metrics.log_loss对数损失，又称逻辑损失或交叉熵损失


# metrics.confusion_matrix 混淆矩阵，模型评估指标之一
# metrics.roc_auc_score ROC曲线，模型评估指标之一
# metrics.accuracy_score 精确性，模型评估指标之一


# 二元逻辑回归的损失函数
# 损失函数是衡量参数theta的优劣的评估指标，用来求解最优参数的工具
# 追求能让损失函数最小化的参数组合

# 不求解参数的算法，无损失函数

# 损失函数由极大似然法估计推导

# 由于追求损失函数的最小值，让模型在训练集上表现最优，可能引发，在训练集上优秀却在测试集上表现糟糕，模型就会出现过拟合。

# 对逻辑回归中拟合的控制，通过正则化实现。L1正则化和L2正则化。

"""
J(theta)L1=C*J(theta)+所有theta绝对值之和
J(theta)L2=C*J(theta)+所有theta平方的和再开方
"""

# 截距不参与正则化

# penalty:默认是L2.若选择L1，则参数solver仅能够使用求解方式liblinear和saga；若选择L2，则参数solver中所有的求解方式都可以使用

# C正则化强度的倒数，必须是一个大于0的浮点数，不填默认是1。即正则化与随时函数的比值是1:1。C越小，损失函数会越小，模型对损失函数的惩罚越重，正则化的效力越强

# L1正则化会将参数压缩为0，L2正则化只会让参数尽量小，不会取到0
# L1掌握参数的稀疏性，特征选择的过程

from sklearn.linear_model import LogisticRegression as LR
from sklearn.datasets import load_breast_cancer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data=load_breast_cancer()
X=data.data
y=data.target
X.shape

lrl1=LR(penalty='l1',solver='liblinear',C=0.5,max_iter=1000)
lrl2=LR(penalty='l2',solver='liblinear',C=0.5,max_iter=1000)

# 逻辑回归的重要属性coef_，查看每个特征对应参数
lrl1=lrl1.fit(X,y)
lrl1.coef_
(lrl1.coef_!=0).sum(axis=1)

lrl2=lrl2.fit(X,y)
lrl2.coef_
(lrl2.coef_!=0).sum(axis=1)

# 研究哪个C参数最好？
l1=[]
l2=[]
l1test=[]
l2test=[]

x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=22)

for i in np.linspace(0.05,1,19):
    lrl1=LR(penalty='l1',solver='liblinear',C=i,max_iter=1000)
    lrl2=LR(penalty='l2',solver='liblinear',C=i,max_iter=1000)

    lrl1=lrl1.fit(x_train,y_train)
    l1.append(accuracy_score(lrl1.predict(x_train),y_train))
    l1test.append(accuracy_score(lrl1.predict(x_test),y_test))

    lrl2=lrl2.fit(x_train,y_train)
    l2.append(accuracy_score(lrl2.predict(x_train),y_train))
    l2test.append(accuracy_score(lrl2.predict(x_test),y_test))

graph=[l1,l2,l1test,l2test]
color=['green','black','lightgreen','gray']
label=['l1','l2','l1test','l2test']

plt.figure(figsize=(6,6))
for i in range(len(graph)):
    plt.plot(np.linspace(0.05,1,19),graph[i],color[i],label=label[i])
plt.legend(loc=4)
plt.show()
# C取0.9比较好





#========= 逻辑回归中的特征工程=========
# 1 业务选择
# 2 PCA和SVD一般不用，降维结果不可解释，因此一旦将降维后，我们无法保持特征原貌
# 3 统计方法可以使用，但是不是非常必要。既然降维算法不能使用，我们要用的就是特征选择方法。逻辑回归对数据的要求低于线性回归，逻辑回归不使用最小二乘法求解，所以逻辑回归对数据的总体分布和方差没有要求，也不要排除特征之间的共线性。
# 4 高效的嵌入法embedded.

from sklearn.linear_model import LogisticRegression as LR
from sklearn.datasets import load_breast_cancer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel

data=load_breast_cancer()
X=data.data
y=data.target
X.shape

LR_=LR(penalty='l1',solver='liblinear',C=0.8,max_iter=1000)
# 交叉验证
cross_val_score(LR_,data.data,data.target,cv=10).mean()
#========高效的嵌入法embedded========
# 运用l1范式删选特征矩阵，模型会删掉在L1范式下判断为无效的特征
X_embedded=SelectFromModel(LR_,norm_order=1).fit_transform(data.data,data.target)
# 特征数量减少到了10个
X_embedded.shape
# X_embedded代到交叉验证中
cross_val_score(LR_,X_embedded,data.target,cv=10).mean()


#======== 模型比较优化 调整=========

# 通过调整特征选择的threshold画学习曲线
fullx=[]
fsx=[]
# 此时，我们使用的判断指标，就不是L1范式，而是逻辑回归中的系数了
threshold=np.linspace(0,abs(LR_.fit(data.data,data.target).coef_).max(),20)

k=0
for i in threshold:
    X_embedded=SelectFromModel(LR_,threshold=i).fit_transform(data.data,data.target)
    fullx.append(cross_val_score(LR_,data.data,data.target,cv=5).mean())
    fsx.append(cross_val_score(LR_,X_embedded,data.target,cv=5).mean())
    print(threshold[k],X_embedded.shape[1])
    k+=1

plt.figure(figsize=(20,5))
plt.plot(threshold,fullx,label='full')
plt.plot(threshold,fsx,label='feature selection')
plt.xticks(threshold)
plt.legend()
plt.show()


# 第二种调整方法，是逻辑回归的类LR_，通过画C的学习曲线实现
fullx=[]
fsx=[]
C=np.arange(0.01,10.01,0.5)
for i in C:
    LR_=LR(solver='liblinear',C=i,random_state=420)
    fullx.append(cross_val_score(LR_,data.data,data.target,cv=10).mean())
    X_embedded=SelectFromModel(LR_,norm_order=1).fit_transform(data.data,data.target)
    fsx.append(cross_val_score(LR_,X_embedded,data.target,cv=10).mean())
print(max(fullx),C[fsx.index(max(fsx))])

plt.figure(figsize=(20,5))
plt.plot(C,fullx,label='full')
plt.plot(C,fsx,label='feature selection')
plt.xticks()
plt.legend()
plt.show()


#========将C的范围精细化调整========
# 第二种调整方法，是逻辑回归的类LR_，通过画C的学习曲线实现 更精细的C
fullx=[]
fsx=[]
C=np.arange(5.55,6.06,0.03)
for i in C:
    LR_=LR(solver='liblinear',C=i,random_state=420)
    fullx.append(cross_val_score(LR_,data.data,data.target,cv=10).mean())
    X_embedded=SelectFromModel(LR_,norm_order=1).fit_transform(data.data,data.target)
    fsx.append(cross_val_score(LR_,X_embedded,data.target,cv=10).mean())
print(max(fullx),C[fsx.index(max(fsx))])

plt.figure(figsize=(20,5))
plt.plot(C,fullx,label='full')
plt.plot(C,fsx,label='feature selection')



#=========梯度下降：重要参数max_iter========
# Gradient decsent
# 求解梯度，是在损失函数J(theta1,theta2)上对损失函数自身的自变量theta1,theta2求偏导，而这两个自变量，刚好是逻辑回归的预测函数的参数
# theta(j+1)=thetaj-alpha*d(j)

#========步长的概念和解惑========
# 步长不是任何物理距离，它甚至不是梯度下降过程中任何距离的而直接变化，他是梯度向量的大小d上的一个比例，影响着参数向量theta每次迭代后改变的部分

# 在损失函数降低的方向上，步长越长，theta（横轴）的变动就越大，下降过程可能跳过损失函数的最低点，无法获得最优值。步长太小，迭代速度缓慢，迭代次数需要很多

# max_iter的学习曲线
l2=[]
l2test=[]
x_train,x_test,y_train,y_test=train_test_split(data.data,data.target,random_state=420)
for i in np.arange(1,201,10):
    lrl2=LR(penalty='l2',solver='liblinear',C=0.9,max_iter=i)
    lrl2=lrl2.fit(x_train,y_train)
    l2.append(accuracy_score(lrl2.predict(x_train),y_train))
    l2test.append(accuracy_score(lrl2.predict(x_test),y_test))
graph=[l2,l2test]
color=['black','gray']
label=['l2','l2test']
plt.figure(figsize=(20,5))
for i in range(len(graph)):
    plt.plot(np.arange(1,201,10),graph[i],color[i],label=label[i])
plt.legend()
plt.xticks(np.arange(1,201,10))
plt.show()

# 我们还可以使用属性n_iter来调用本次求解中真正实现的迭代次数
lr=LR(penalty='l2',solver='liblinear',C=0.9,max_iter=300).fit(x_train,y_train)
lr.iter_




#======== 案例：用逻辑回归制作评分卡========
# 对于个人而言，有4张卡判断个人信用程度：A卡，B卡，C卡，和F卡。而众人常说评分卡指的是A卡，又称为 申请者评级模型，主要评估新用户的主体评级

# 完整地模型开发：
# 1 获取数据
# 2 数据清洗特征工程
# 3 模型开发
# 4 模型检验和评估
# 5 模型上线
# 6 检测与报告


# 导库，获取数据
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression as LR
data=pd.read_csv(r"D:\learn\software_learn\NOTE\Python\dataset\cs-training.csv",index_col=0)
data.head()

# 数据结构
data.shape

data.info()

# 去重重复值
data.drop_duplicates(inplace=True)# 替换原数据
data.info
# 删除之后千万不索引要忘记恢复
data.shape
data.index.max()
# 恢复索引
data.index=range(data.shape[0])
# 缺失值
# 每个字段的缺失值
data.isnull().sum()
# 每个字段的确实比例
data.isnull().sum()/data.shape[0]
data.isnull().mean()

# 填补缺失值
data['NumberOfDependents'].fillna(data['NumberOfDependents'].mean(),inplace=True)
data.isnull().sum()

# 收入缺失值 随机森林填补
# 特征T不缺失的值对应的其他n-1个特征+本来的标签 x_train
# 特征T不缺失的值：y_train
# 特征T缺失的值对应的其他的n-1个特征+本来的标签x_test
# 特征T缺失的值：未知，我们需要预测的y_test

# 这种做法，对于一个特征大量缺失，其他特征却很完整的，非常实用。

def fill_missing_rf(X,y,to_fill):
    """
    使用随机森林填补一个特征缺失值的函数
    :param x:要填补的特征矩阵
    :param y:完整的，没有缺失值的标签
    :param to_fill:字符串，要填补的那一列的名称
    :return:
    """
    # 构建我们的新特征矩阵和新标签
    df=X.copy()
    fill=df.loc[:,to_fill]
    df=pd.concat([df.loc[:,df.columns != to_fill],pd.DataFrame(y)],axis=1)
    # 找出我们的训练集合测试集
    y_train=fill[fill.notnull()]
    y_test=fill[fill.isnull()]
    x_train=df.iloc[y_train.index,:]
    x_test=df.iloc[y_test.index,:]

    # 用随机森林回归来填补缺失值
    from sklearn.ensemble import RandomForestRegressor as rfr
    rfr=rfr(n_estimators=100)
    rfr=rfr.fit(x_train,y_train)
    y_predict=rfr.predict(x_test)
    return y_predict

X=data.iloc[:,1:]
y=data['SeriousDlqin2yrs']

x_pred=fill_missing_rf(X,y,'MonthlyIncome')

data.loc[data.loc[:,'MonthlyIncome'].isnull(),'MonthlyIncome']=x_pred


#========描述性统计处理异常值==========
# 处理异常值：箱型图和3thegma
# 描述性统计
data.describe()
data.describe([0,0.1,0.25,0.5,0.75,1]).T
# 查看偏态，以及其他异常值
(data['age']==0).sum()

data=data[data['age']!=0]

data[data.loc[:,'NumberOfTimes90DaysLate']>90].count()
data.loc[:,'NumberOfTimes90DaysLate'].value_counts()
data=data[data.loc[:,'NumberOfTimes90DaysLate']<90]

# 恢复索引
data.index=range(data.shape[0])


#========处理偏态======
# 评分卡需要给数据分档，不需要量纲化



#========样本不均衡========
X=data.iloc[:,1:]
y=data.iloc[:0]
y.value_counts()
n_1_sample=y.value_counts()[1]
n_0_sample=y.value_counts()[0]
# 样本不均衡（违约的人少数）

# 逻辑回归使用最多的是采样方法来平衡样本
import imblearn
# imblearn是专门来处理不平衡数据集的库，在处理样本不均衡问题中性能高过sklearn
# imblearn里面也是一个个的类，也需要进行实例化，fit拟合


import imblearn
from imblearn.over_sampling import SMOTE
# 实例化
sm=SMOTE(random_state=42)
# 返回已上采样完毕后的特征矩阵和标签
X,y=sm.fit_resample(X,y)
n_sample=X.shape[0]
n_1_sample=y.value_counts()[1]
n_0_sample=y.value_counts()[0]
X.shape
y.value_counts()

#========分训练街和测试集========
from  sklearn.model_selection import train_test_split
X=pd.DataFrame(X)
y=pd.DataFrame(y)
x_train,x_vali,y_train,y_vali=train_test_split(X,y,test_size=0.3,random_state=42)
model_data=pd.concat([y_train,x_train],axis=1)
model_data.index=range(model_data.shape[0])
model_data.columns=data.columns
model_data.shape
vali_data=pd.concat([y_vali,x_vali],axis=1)
vali_data.index=range(vali_data.shape[0])
vali_data.columns=data.columns
vali_data.shape

# 保存训练街和测试集
model_data.to_csv(r"D:\learn\software_learn\NOTE\Python\dataset\model_data.csv")
vali_data.to_csv(r"D:\learn\software_learn\NOTE\Python\dataset\vali_data.csv")




#========分箱========
# IV衡量特征上的信息量以及特征对预测函数的贡献
# WOE这个 证据权重，是对于一个箱子来说的，WOE越大，代表了这个箱子的优质客户越多，而IV是对于整个特征来说的。

# IV
# <0.03 特征几乎不带有效信息，对模型没有贡献，这种特征可以删除
# 0.03-0.09 有效信息很少，对模型的贡献度低
# 0.1-0.29 有效信息一般，对模型的贡献度中等
# >=0.5 有效信息非常多，对模型的贡献度超高并且可疑

# IV并非越大越好，我们想要找到IV的大小和箱子个数的平衡点。所以对特征分箱，然后计算每个特征在每个箱子数目下的WOE值，利用IV值得曲线，找到合适的分箱个数


# 分箱要达成什么样的效果
# 希望同一个箱子内的人的属性是尽量相同，不同箱子的人的属性尽可能不同，即 组间差异大，组内差异小。对于评分卡来说，希望每个箱子内的人违约的概率是类似的，而不同箱子的人违约概率差距很大，即WOE差距要大，并且每个箱子中坏客户所占的比重也要不同。那我们可以使用卡方检验对比两个箱子之间的相似性。如果两个箱子之间卡方检验的P值很大，则说明相似，即可合并成一个箱子



#=========等频分箱=========
# age为例
model_data['qcut'],updown=pd.qcut(model_data['age'],retbins=True,q=20)

#========每个箱子的0,1=========
count_y0=model_data[model_data['SeriousDlqin2yrs']==0].groupby(by='qcut').count()['SeriousDlqin2yrs']
count_y1=model_data[model_data['SeriousDlqin2yrs']==1].groupby(by='qcut').count()['SeriousDlqin2yrs']

# num_bins值分别为每个区间的上界，下界，0出现的次数，1出现的次数
num_bins=[*zip(updown,updown[1:],count_y0,count_y1)]
# 确保每个箱子中都有0和1


# ========定义ＷＯＥ和ＩＶ函数=========
# 计算woe和bad rate
# bad rate是一个箱中，坏的样本所占比例

def get_woe(num_bins):
    # 通过num_bins数据计算woe
    columns=['min','max','count_0','count_1']
    df=pd.DataFrame(num_bins,columns=columns)
    # 一个箱子中所有的样本数
    df['total']=df.count_0+df.count_1
    df['percentage']=df.total/df.total.sum()
    df['bad_rate']=df.count_1/df.total
    # 该箱子中的好样本数量/所有箱子中的好样本的数量
    df['good%']=df.count_1/df.count_1.sum()
    df['bad%']=df.count_0/df.count_0.sum()
    df['woe']=np.log(df['good%']/df['bad%'])
    return df


def get_iv(bins_df):
    rate=bins_df['good%']-bins_df['bad%']
    iv=np.sum(rate*bins_df.woe)
    return iv


#========卡方检验，合并箱体，画出IV曲线=========
num_bins_=num_bins.copy()
import matplotlib.pyplot as plt
import scipy
IV=[]
axisx=[]
while len(num_bins_)>2:
    pvs=[]
    # 获取num_bins_两两之间的卡方检验的置信度
    for i in  range(len(num_bins_)-1):
        x1=num_bins_[i][2:]
        x2=num_bins_[i+1][2:]
        # 0返回chi2值，1返回p值
        pv=scipy.stats.chi2_contingency([x1,x2])[1]
        # chi2=scipy.stats.chi2_contingency([x1,x2])[0]
        pvs.append(pv)
    # 通过p值进行处理，合并p值最大的两组
    i=pvs.index(max(pvs))
    num_bins_[i:i+2]=[(
        num_bins_[i][0],
        num_bins_[i+1][1],
        num_bins_[i][2]+num_bins_[i+1][2],
        num_bins_[i][3]+num_bins_[i+1][3]
    )]
    bins_df=get_woe(num_bins_)
    axisx.append(len(num_bins_))
    IV.append(get_iv(bins_df))

plt.figure(figsize=(10,10))
plt.plot(axisx,IV)
plt.xticks(axisx)
plt.xlabel('boxes')
plt.ylabel('the iv of the boxes')
plt.show()


#========用最佳分箱个数分箱，并验证分箱结果========
# 将合并箱体的部分定义为函数，并实现分箱
def get_bin(num_bins_,n):
    while len(num_bins_)>n:
        pvs=[]
        for i in range(len(num_bins_)-1):
            x1=num_bins_[i][2:]
            x2=num_bins_[i+1][2:]
            pv=scipy.stats.chi2_contingency([x1,x2])[1]
            pvs.append(pv)
        i=pvs.index(max(pvs))
        num_bins_[i:i + 2] = [(
            num_bins_[i][0],
            num_bins_[i + 1][1],
            num_bins_[i][2] + num_bins_[i + 1][2],
            num_bins_[i][3] + num_bins_[i + 1][3]
        )]
    return num_bins_

afterbins=get_bin(num_bins,6)


#========将选取最佳分箱个数的过程包装成函数=========
def graphforbestbin(df,X,y,n=5,q=20,graph=True):
    """
    自动最优分箱函数，基于卡方检验的分箱
    :param df:需要分箱的数据
    :param X:需要分箱的列名
    :param y:分箱数据对应的标签 列名
    :param n:保留分箱个数
    :param q:初始分箱的个数
    :param graph:是否要输出IV图像
    :return:
    """
    df=df[[X,y]].copy()
    df['qcut'],bins=pd.qcut(df[X],retbins=True,q=q,duplicates='drop')
    count_y0=df.loc[df[y]==0].groupby(by='qcut').count()[y]
    count_y1=df.loc[df[y]==1].groupby(by='qcut').count()[y]
    num_bins=[*zip(bins,bins[1:],count_y0,count_y1)]

    for i in range(q):
        if 0 in num_bins_[0][2:]:
            num_bins[0:2]=[(
                num_bins[0][0],
                num_bins[1][1],
                num_bins[0][2]+num_bins[1][2],
                num_bins[0][3]+num_bins[1][3]
            )]
            break
        for i in range(len(num_bins)):
            if 0 in num_bins[i][2:]:
                num_bins_[i:i + 2] = [(
                    num_bins[i][0],
                    num_bins[i + 1][1],
                    num_bins[i][2] + num_bins[i + 1][2],
                    num_bins[i][3] + num_bins[i + 1][3]
                )]
                break
        else:
            break
    def get_woe(num_bins):
        columns=['min','max','count_0','count_1']
        df=pd.DataFrame(num_bins,columns=columns)
        df['total'] = df.count_0 + df.count_1
        df['percentage'] = df.total / df.total.sum()
        df['bad_rate'] = df.count_1 / df.total
        # 该箱子中的好样本数量/所有箱子中的好样本的数量
        df['good%'] = df.count_1 / df.count_1.sum()
        df['bad%'] = df.count_0 / df.count_0.sum()
        df['woe'] = np.log(df['good%'] / df['bad%'])
        return df

    def get_iv(df):
        rate = df['good%'] - df['bad%']
        iv = np.sum(rate * df.woe)
        return iv

    IV = []
    axisx = []
    while len(num_bins) > n:
        pvs = []
        # 获取num_bins_两两之间的卡方检验的置信度
        for i in range(len(num_bins) - 1):
            x1 = num_bins[i][2:]
            x2 = num_bins[i + 1][2:]
            # 0返回chi2值，1返回p值
            pv = scipy.stats.chi2_contingency([x1, x2])[1]
            # chi2=scipy.stats.chi2_contingency([x1,x2])[0]
            pvs.append(pv)
        # 通过p值进行处理，合并p值最大的两组
        i = pvs.index(max(pvs))
        num_bins[i:i + 2] = [(
            num_bins[i][0],
            num_bins[i + 1][1],
            num_bins[i][2] + num_bins[i + 1][2],
            num_bins[i][3] + num_bins[i + 1][3]
        )]
        bins_df = get_woe(num_bins)
        axisx.append(len(num_bins))
        IV.append(get_iv(bins_df))
    if graph:
        plt.figure(figsize=(10, 10))
        plt.plot(axisx, IV)
        plt.xticks(axisx)
        plt.xlabel('boxes')
        plt.ylabel('the iv of the boxes')
        plt.show()
    return bins_df


model_data.columns

for i in model_data.columns[1:-1]:
    print(i)
    b=graphforbestbin(model_data,i,'SeriousDlqin2yrs',n=2,q=20,graph=True)
    print(b)
# 部分特征的分箱曲线是直线：有些特征是分类特征、需要手动分箱

# 可自动分箱的特征
auto_col_bins={
    'RevolvingUtilizationOfUnsecuredLines':6,
    'age':5,
    'DebtRatio':4,
    'MonthlyIncome':3,
    'NumberOfOpenCreditLinesAndLoans':5
}

# 不能使用自动分箱的变量
hand_bins={
    'NumberOfTime30-59DaysPastDueNotWorse':[0,1,2,13],
    'NumberOfTimes90DaysLate':[0,1,2,17],
    'NumberRealEstateLoansOrLines':[0,1,2,4,54],
    'NumberOfTime60-89DaysPastDueNotWorse':[0,12,8],
    'NumberOfDependents':[0,1,2,3]
}
# 保证区间覆盖使用np.inf替换最大值，用-np.inf替换最小值
hand_bins={k:[-np.inf,*v[:-1],np.inf] for k,v in hand_bins.items()}

bins_of_col={}
# 生成自动分箱的分箱区间和分箱后的IV值
for col in auto_col_bins:
    bins_df=graphforbestbin(model_data,col,'SeriousDlqin2yrs',n=auto_col_bins[col],q=20,graph=True)
    bins_list=sorted(set(bins_df['min']).union(bins_df['max']))
    # 保证区间覆盖所有np.inf替换最大值， -np.inf替换最小值
    bins_list[0],bins_list[-1]=-np.inf,np.inf
    bins_of_col[col]=bins_list




