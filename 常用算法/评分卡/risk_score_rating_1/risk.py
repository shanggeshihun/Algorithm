
# _*_coding:utf-8 _*_
# @Time　　 :2019/7/23   14:55
# @Author　 : Antipa
#@ File　　 :123.py
#@Software  :PyCharm

# 一、项目流程
# 信用风险评级模型的主要开发流程如下：
# （1） 数据获取，包括获取存量客户及潜在客户的数据。存量客户是指已经在证券公司开展相关融资类业务的客户，包括个人客户和机构客户；潜在客户是指未来拟在证券公司开展相关融资类业务的客户，主要包括机构客户，这也是解决证券业样本较少的常用方法，这些潜在机构客户包括上市公司、公开发行债券的发债主体、新三板上市公司、区域股权交易中心挂牌公司、非标融资机构等。
# （2） 数据预处理，主要工作包括数据清洗、缺失值处理、异常值处理，主要是为了将获取的原始数据转化为可用作模型开发的格式化数据。
# （3） 探索性数据分析，该步骤主要是获取样本总体的大概情况，描述样本总体情况的指标主要有直方图、箱形图等。
# （4） 变量选择，该步骤主要是通过统计学的方法，筛选出对违约状态影响最显著的指标。主要有单变量特征选择方法和基于机器学习模型的方法 。
# （5） 模型开发，该步骤主要包括变量分段、变量的WOE（证据权重）变换和逻辑回归估算三部分。
# （6） 模型评估，该步骤主要是评估模型的区分能力、预测能力、稳定性，并形成模型评估报告，得出模型是否可以使用的结论。
# （7） 信用评分，根据逻辑回归的系数和WOE等确定信用评分的方法。将Logistic模型转换为标准评分的形式。
# （8） 建立评分系统，根据信用评分方法，建立自动信用评分系统。

# 二、数据获取
# 数据来自Kaggle的Give Me Some Credit，有15万条的样本数据，大致情况如下：
# 数据属于个人消费类贷款，只考虑信用评分最终实施时能够使用到的数据应从如下一些方面获取数据：
# –            基本属性：包括了借款人当时的年龄。
# –            偿债能力：包括了借款人的月收入、负债比率。
# –            信用往来：两年内35-59天逾期次数、两年内60-89天逾期次数、两年内90
# 天或高于90天逾期的次数。
# –            财产状况：包括了开放式信贷和贷款数量、不动产贷款或额度数量。
# –            贷款属性：暂无。
# –            其他因素：包括了借款人的家属数量（不包括本人在内）。
# –            时间窗口：自变量的观察窗口为过去两年，因变量表现窗口为未来两年。

# 附：数据下载网址 https://www.kaggle.com/c/GiveMeSomeCredit/data

# 三、数据预处理

# 在对数据处理之前，需要对数据的缺失值和异常值情况进行了解。Python内有describe()函数，可以了解数据集的缺失值、均值和中位数等。

import numpy as np
import pandas as pd

# Viz
import matplotlib.pyplot as plt
import seaborn as sns

# settings
import warnings
warnings.filterwarnings("ignore")

from  sklearn.ensemble import RandomForestClassifier
from  sklearn.ensemble import RandomForestRegressor

from scipy.stats import stats
#载入数据
data = pd.read_csv(r'D:\learn\software_learn\NOTE\Python\algorithm\risk_score_rating\GiveMeSomeCredit\cs-training.csv')
data.describe()
data.dtypes
#数据集缺失和分布情况
data.describe().to_csv(r'D:\learn\software_learn\NOTE\Python\algorithm\risk_score_rating\GiveMeSomeCredit\DataDescribe.csv')
data.columns

# 将ID列设置为索引列
data=data.set_index('Unnamed: 0',drop=True)   #设置 Unnamed: 0 列为索引列

# np.set_printoptions(suppress=True) # 搜索可用该语句取消MonthlyIncome的科学计数法显示，但是此处没有生效。

# 3.1 缺失值处理
# 这种情况在现实问题中非常普遍，这会导致一些不能处理缺失值的分析方法无法应用，因此，在信用风险评级模型开发的第一步我们就要进行缺失值处理。
# 缺失值处理的方法，包括如下几种。
# （1） 直接删除含有缺失值的样本。
# （2） 根据样本之间的相似性填补缺失值。
# （3） 根据变量之间的相关关系填补缺失值。

# 变量MonthlyIncome缺失率比较大，所以我们根据变量之间的相关关系填补缺失值data.isnull.any()，我们采用随机森林法：
data_1=data.copy()
# 查看存在null的列（MonthlyIncome，NumberOfDependents）
data_1.isnull().any()
data_1[data.MonthlyIncome.isnull()]
# 重排列名
data_1.drop(['NumberOfDependents'],axis=1,inplace=True)
order=['SeriousDlqin2yrs', 'RevolvingUtilizationOfUnsecuredLines', 'age',
       'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio',
       'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate',
       'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse', 'MonthlyIncome']
data_1=data_1[order]

data_1_not_null=data_1[data_1.MonthlyIncome.isnull()==False]
data_m_not_null=data_1_not_null.as_matrix()

data_1_is_null=data_1[data_1.MonthlyIncome.isnull()==True]
data_m_is_null=data_1_is_null.as_matrix()

X_train=data_m_not_null[:,:-1]
y_train=data_m_not_null[:,-1]

from sklearn.ensemble import RandomForestRegressor
rfr=RandomForestRegressor(random_state=32)
rfr.fit(X_train,y_train)
y_train_pred=rfr.predict(X_train)


X_test=data_m_is_null[:,:-1]
y_test_pred=rfr.predict(X_test)

plt.figure(figsize=(10,10))
plt.subplot(2,1,1)
plt.boxplot(y_train_pred)
plt.subplot(2,1,2)
plt.boxplot(y_test_pred)
plt.show()

data_1.MonthlyIncome[data_1.MonthlyIncome.isnull()==True]=y_test_pred

data_1_unique=data_1.drop_duplicates()
len(data_1_unique)

data.MonthlyIncome.min()
# 检查经过处理后的数据是否存在缺失值
data_1_unique.isnull().any()

data_1_unique.shape

# 3.2 异常值处理
# 缺失值处理完毕后，我们还需要进行异常值处理。
# 异常值是指明显偏离大多数抽样数据的数值，比如个人客户的年龄为0时，通常认为该值为异常值。找出样本总体中的异常值，通常采用离群值检测的方法。
# 首先，我们发现变量age中存在0，显然是异常值，直接剔除：

# 年龄等于0的异常值进行剔除
plt.figure()
data_1_unique['age'].plot(kind='box',title='age')
plt.show()

data_1_unique = data_1_unique[data_1_unique['age'] > 0]


# 对于变量NumberOfTime30-59DaysPastDueNotWorse、NumberOfTimes90DaysLate、NumberOfTime60-89DaysPastDueNotWorse这三个变量，
# 由下面的箱线图图3-2可以看出，均存在异常值，且由unique函数可以得知均存在96、98两个异常值，因此予以剔除。
# 同时会发现剔除其中一个变量的96、98值，其他变量的96、98两个值也会相应被剔除。
plt.figure(figsize=(16,12))
fig,axes = plt.subplots(1,3)
# boxes表示箱体，whisker表示触须线
# medians表示中位数，caps表示最大与最小值界限
color = dict(boxes='DarkGreen', whiskers='DarkOrange',
              medians='DarkBlue', caps='Red')
datatemp1=data_1_unique[["NumberOfTime30-59DaysPastDueNotWorse","NumberOfTimes90DaysLate","NumberOfTime60-89DaysPastDueNotWorse"]]
datatemp1.plot(kind='box',ax=axes,subplots=True,title='3 Different boxplots',color=color,sym='r+')
# sym参数表示异常值标记的方式
axes[0].set_ylabel('NumberOfTime30-59DaysPastDueNotWorse')
axes[1].set_ylabel('NumberOfTimes90DaysLate')
axes[2].set_ylabel('NumberOfTime60-89DaysPastDueNotWorse')
fig.subplots_adjust(wspace=3,hspace=1)  # 调整子图之间的间距
plt.show()
# 查看上述三个变量的不重复值。
print(np.unique(datatemp1["NumberOfTime30-59DaysPastDueNotWorse"]))
print(np.unique(datatemp1["NumberOfTimes90DaysLate"]))
print(np.unique(datatemp1["NumberOfTime60-89DaysPastDueNotWorse"]))

data_1_unique.head(2)
# 剔除变量NumberOfTime30-59DaysPastDueNotWorse、NumberOfTimes90DaysLate、NumberOfTime60-89DaysPastDueNotWorse的异常值。
# 另外，数据集中好客户为0，违约客户为1，考虑到正常的理解，能正常履约并支付利息的客户为1，所以我们将其取反。

#剔除异常值
data_1_unique = data_1_unique[data_1_unique['NumberOfTime30-59DaysPastDueNotWorse'] < 90]

#变量SeriousDlqin2yrs取反
data_1_unique['SeriousDlqin2yrs']=1-data_1_unique['SeriousDlqin2yrs']
data_1_unique.head(2)

# 3.3 数据切分

# 为了验证模型的拟合效果，我们需要对数据集进行切分，分成训练集和测试集。
from sklearn.cross_validation import train_test_split
data_1_unique.columns
Y=data_1_unique['SeriousDlqin2yrs']
X=data_1_unique.iloc[:,1:]
# X_train,Y_train dataframe类型
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=0)

train = pd.concat([Y_train, X_train], axis=1)
test = pd.concat([Y_test, X_test], axis=1)
test.head()

test['SeriousDlqin2yrs'].head()
test['SeriousDlqin2yrs'].value_counts()
clasTest = test.groupby(['SeriousDlqin2yrs'])['SeriousDlqin2yrs'].count()

train.to_csv(r'D:\learn\software_learn\NOTE\Python\risk_score_rating\TrainData.csv',index=False)
test.to_csv(r'D:\learn\software_learn\NOTE\Python\risk_score_rating\TestData.csv',index=False)

plt.figure(figsize=(8,8))
bins=30
plt.subplot(211)
ax1= plt.hist(data_1_unique.age, bins, color="green", alpha=0.8,rwidth=0.9)
plt.title("Age distribution")
plt.ylabel('# of age', fontsize=12)
plt.xlabel('age', fontsize=12)

plt.subplot(212)
ax2= plt.hist(data_1_unique.MonthlyIncome,bins,color="green",alpha=0.8,rwidth=0.9)
plt.title("MonthlyIncome distribution")
plt.ylabel('# of MonthlyIncome', fontsize=12)
plt.xlabel('MonthlyIncome', fontsize=12)
plt.show()

# 而我们这里呈现的收入分布集中在一条柱子上，看不清分布。我们猜测是异常值影响所致。接下来用箱线图找找异常值。
datatemp2=data_1_unique["MonthlyIncome"]
datatemp2.plot(kind='box',title='MonthlyIncome Distribution',sym='r+');

print(data_1_unique[data_1_unique['MonthlyIncome'] > 50000].count()) # 回头查看下面被剔除了多少异常值

# 如前文处理'NumberOfTime30-59DaysPastDueNotWorse'异常值的方式，直接剔除异常值
data_1_unique = data_1_unique[data_1_unique['MonthlyIncome'] < 50000]
# 上述语句的阈值，从100万尝试到50万尝试到10万，再到6万，5万，最终确定5万。

# 重新查看收入直方图分布
plt.figure(figsize=(15,5))
plt.hist(data_1_unique.MonthlyIncome,bins,color="green",alpha=0.8,rwidth=0.9)
plt.title("MonthlyIncome distribution")
plt.ylabel('# of MonthlyIncome', fontsize=12)
plt.xlabel('MonthlyIncome', fontsize=12)
plt.show()

# 所以，从客户收入分布图看出，月收入也大致呈正态分布，符合统计分析的需要。
# 另外，剔除掉的月收入异常值，有301个，在12万的数据量中可忽略不计。
# 来个快速版的直方图
# 发现挺多变量含有异常值影响了直方图分布。
data_1_unique.hist(bins=50, figsize=(20,15))
plt.show()


# 用箱线图看看异常值
plt.figure(figsize=(16,8))
data_1_unique.plot(kind='box',title='Various Var Distribution',sym='r+')
plt.show()
# 因为各个变量的数量级相差较大，直接放一起，无法观察。后面如有需要，再分开观察。此处不过多赘述。

# 五、变量选择
# 特征变量选择(排序)对于数据分析、机器学习从业者来说非常重要。
# 好的特征选择能够提升模型的性能，更能帮助我们理解数据的特点、底层结构，这对进一步改善模型、算法都有着重要作用。
# 至于Python的变量选择代码实现可以参考结合Scikit-learn介绍几种常用的特征选择方法。

# 在本文中，我们采用信用评分模型的变量选择方法，通过WOE分析方法，即是通过比较指标分箱和对应分箱的违约概率来确定指标是否符合经济意义。
# 首先我们对变量进行离散化（分箱）处理。

# 5.1 分箱处理
# 变量分箱（binning）是对连续变量离散化（discretization）的一种称呼。
# 信用评分卡开发中一般有常用的等距分段、等深分段、最优分段。
# 其中等距分段（Equval length intervals）是指分段的区间是一致的，比如年龄以十年作为一个分段；
# 等深分段（Equal frequency intervals）是先确定分段数量，然后令每个分段中数据数量大致相等；
# 最优分段（Optimal Binning）又叫监督离散化（supervised discretizaion），使用递归划分（Recursive Partitioning）将连续变量分为分段，背后是一种基于条件推断查找较佳分组的算法。

data_1_unique=data_1_unique.drop_duplicates(subset=None,keep='first',inplace=False)
data_1_unique.shape

# 5.2 WOE

# WoE分析， 是对指标分箱、计算各个档位的WoE值并观察WoE值随指标变化的趋势。其中WoE的数学定义是:
# woe=ln(goodattribute/badattribute)
# 在进行分析时，我们需要对各指标从小到大排列，并计算出相应分档的WoE值。其中正向指标越大，WoE值越小；反向指标越大，WoE值越大。正向指标的WoE值负斜率越大，反响指标的正斜率越大，则说明指标区分能力好。WoE值趋近于直线，则意味指标判断能力较弱。若正向指标和WoE正相关趋势、反向指标同WoE出现负相关趋势，则说明此指标不符合经济意义，则应当予以去除。
# woe函数实现在上一节的mono_bin()函数里面已经包含，这里不再重复。

# 5.3 相关性分析和IV筛选
# 接下来，我们会用经过清洗后的数据看一下变量间的相关性。注意，这里的相关性分析只是初步的检查，进一步检查模型的VI（证据权重）作为变量筛选的依据。

# 数据集各变量的相关性。
# 相关性图我们通过Python里面的seaborn包，调用heatmap()绘图函数进行绘制，实现代码如下：

corr = data_1_unique.corr()#计算各变量的相关性系数
xticks = ['x0','x1','x2','x3','x4','x5','x6','x7','x8','x9']#x轴标签
yticks = list(corr.index)#y轴标签
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
sns.heatmap(corr, annot=True, cmap='rainbow', ax=ax1, annot_kws={'size': 9, 'weight': 'bold', 'color': 'blue'})#绘制相关性系数热力图
ax1.set_xticklabels(xticks, rotation=0, fontsize=10)
ax1.set_yticklabels(yticks, rotation=0, fontsize=10)
plt.show()

# 由下图可以看出，各变量之间的相关性是非常小的。NumberOfOpenCreditLinesAndLoans和NumberRealEstateLoansOrLines的相关性系数为0.43。

# 接下来，我进一步计算每个变量的Infomation Value（IV）。IV指标是一般用来确定自变量的预测能力。 其公式为：
# IV=sum((goodattribute-badattribute)*ln(goodattribute/badattribute))
# 通过IV值判断变量预测能力的标准是：
# < 0.02: unpredictive
# 0.02 to 0.1: weak
# 0.1 to 0.3: medium
# 0.3 to 0.5: strong
# > 0.5: suspicious

# IV的实现放在mono_bin()函数里面，代码实现如下：

# 定义自动分箱函数
def mono_bin(Y, X, n = 20):
    """
    :param Y:
    :param X:
    :param n:
    :return: d1[X，Y，Bucket],d4[Bucket,woe],iv
    """
    r = 0
    good=Y.sum()
    bad=Y.count()-good
    while np.abs(r) < 1:
        d1 = pd.DataFrame({"X": X, "Y": Y, "Bucket": pd.qcut(X, n)})
        d2 = d1.groupby('Bucket', as_index = True)
        r, p = stats.spearmanr(d2.mean().X, d2.mean().Y)
        n = n - 1
    d3 = pd.DataFrame(d2.X.min(), columns = ['min'])
    d3['min']=d2.min().X
    d3['max'] = d2.max().X
    d3['sum'] = d2.sum().Y
    d3['total'] = d2.count().Y
    d3['rate'] = d2.mean().Y
    d3['woe']=np.log((d3['rate']/(1-d3['rate']))/(good/bad))
    d3['goodattribute']=d3['sum']/good
    d3['badattribute']=(d3['total']-d3['sum'])/bad
    iv=((d3['goodattribute']-d3['badattribute'])*d3['woe']).sum()
    d3['Bucket']=d3.index
    return d1,d3,iv

#自定义分箱函数
# ——该定义函数参考另一篇帖子：https://blog.csdn.net/sunyaowu315/article/details/82981216
def self_bin(Y,X,cat):
    """
    :param Y:
    :param X:
    :param cat:
    :return: d1[X，Y，Bucket],d4[Bucket,woe],iv
    """
    good=Y.sum()
    bad=Y.count()-good
    d1=pd.DataFrame({'X':X,'Y':Y,'Bucket':pd.cut(X,cat)})
    d2=d1.groupby('Bucket', as_index = True)
    d3 = pd.DataFrame(d2.X.min(), columns=['min'])
    d3['min'] = d2.min().X
    d3['max'] = d2.max().X
    d3['sum'] = d2.sum().Y
    d3['total'] = d2.count().Y
    d3['rate'] = d2.mean().Y
    d3['woe'] = np.log((d3['rate'] / (1 - d3['rate'])) / (good / bad))
    d3['goodattribute'] = d3['sum'] / good
    d3['badattribute'] = (d3['total'] - d3['sum']) / bad
    iv = ((d3['goodattribute'] - d3['badattribute']) * d3['woe']).sum()
    d3['Bucket']=d3.index
    return d1,d3, iv

d1,dfx1, ivx1= mono_bin(data_1_unique.SeriousDlqin2yrs, data_1_unique.RevolvingUtilizationOfUnsecuredLines,n=10)
d2,dfx2, ivx2= mono_bin(data_1_unique.SeriousDlqin2yrs, data_1_unique.age, n=10)
d4,dfx4, ivx4=mono_bin(data_1_unique.SeriousDlqin2yrs, data_1_unique.DebtRatio, n=20)
d5,dfx5, ivx5=mono_bin(data_1_unique.SeriousDlqin2yrs, data_1_unique.MonthlyIncome, n=10)

# 连续变量离散化
pinf = float('inf') #正无穷大
ninf = float('-inf') #负无穷大

cutx3 = [ninf, 0, 1, 3, 5, pinf]
cutx6 = [ninf, 1, 2, 3, 5, pinf]
cutx7 = [ninf, 0, 1, 3, 5, pinf]
cutx8 = [ninf, 0,1,2, 3, pinf]
cutx9 = [ninf, 0, 1, 3, pinf]
d3,dfx3, ivx3 = self_bin(data_1_unique.SeriousDlqin2yrs, data_1_unique['NumberOfTime30-59DaysPastDueNotWorse'], cutx3)
d6,dfx6, ivx6= self_bin(data_1_unique.SeriousDlqin2yrs, data_1_unique['NumberOfOpenCreditLinesAndLoans'], cutx6)
d7,dfx7, ivx7 = self_bin(data_1_unique.SeriousDlqin2yrs, data_1_unique['NumberOfTimes90DaysLate'], cutx7)
d8,dfx8, ivx8 = self_bin(data_1_unique.SeriousDlqin2yrs, data_1_unique['NumberRealEstateLoansOrLines'], cutx8)
d9,dfx9, ivx9 = self_bin(data_1_unique.SeriousDlqin2yrs, data_1_unique['NumberOfTime60-89DaysPastDueNotWorse'], cutx9)

# 生成的IV图代码：
ivlist=[ivx1,ivx2,ivx3,ivx4,ivx5,ivx6,ivx7,ivx8,ivx9]#各变量IV
index=['x1','x2','x3','x4','x5','x6','x7','x8','x9']#x轴的标签
fig1 = plt.figure(1)
ax1 = fig1.add_subplot(1, 1, 1)
x = np.arange(len(index))+1
ax1.bar(x, ivlist, width=0.4)#生成柱状图
ax1.set_xticks(x)
ax1.set_xticklabels(index, rotation=0, fontsize=12)
ax1.set_ylabel('IV(Information Value)', fontsize=14)
#在柱状图上添加数字标签
for a, b in zip(x, ivlist):
    plt.text(a, b + 0.01, '%.4f' % b, ha='center', va='bottom', fontsize=10)
plt.show()

# 输出的各变量IV图，如下。
# 可以看出，DebtRatio(x4)、MonthlyIncome(x5)、NumberOfOpenCreditLinesAndLoans(x6)、NumberRealEstateLoansOrLines(x8)和NumberOfDependents(x10)变量的IV值明显较低，所以予以删除。
data_1_unique.columns  # 用于对比上条语句结果，查看x1、x2、x3、x6、x8、x9具体对应哪些字段

# 一般选取IV大于0.02的特征变量进行后续训练，从以上可以看出x1,x2,x3,x6,x7,x8,x9满足


df_new=pd.DataFrame()
df_new["好坏客户"]=pd.merge(dfx1,d1,how='inner')["Y"]
df_new["可用额度比值"]=pd.merge(dfx1,d1,how='inner')["woe"]
df_new["年龄"]=pd.merge(dfx2,d2,how='inner')["woe"]
df_new["逾期30-59天笔数"]=pd.merge(dfx3,d3,how='inner')["woe"]
df_new["逾期90天笔数"]=pd.merge(dfx6,d6,how='inner')["woe"]
df_new["固定资产贷款量"]=pd.merge(dfx7,d7,how='inner')["woe"]
df_new["逾期60-89天笔数"]=pd.merge(dfx8,d8,how='inner')["woe"]
df_new["月收入"]=pd.merge(dfx9,d9,how='inner')["woe"]


# 模型训练
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
x=df_new.iloc[:,1:]
y=df_new.iloc[:,:1]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=2)
model=LogisticRegression()
clf=model.fit(x_train,y_train)
print(clf.score(x_test,y_test))

# 特征权值系数coef_
coe=clf.coef_

# 模型评估（AUC和K-S值）
from sklearn.metrics import roc_curve,auc
