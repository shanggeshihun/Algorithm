
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
def datasets_demo():
    iris=load_iris()
    print('鸢尾花数据集:',iris)
    print(iris['DESCR'])
    print(iris.feature_names)
    print(iris.data)
    print(iris.target)
    print(iris.data.shape)

    # 数据集划分
    x_train,x_test,y_train,y_test=train_test_split(iris.data,iris.target,test_size=0.2,random_state=22)
    print('训练集的特征值:\n',x_train,x_train.shape)

if __name__ == '__main__':
    datasets_demo()
	
# 数据和特征决定了机器学习的上限，而模型和算法只是逼近这个上限而已


# 特征工程
sklearn 专门做特征工程
pandas数据清洗
机器学习算法=统计方法

将任意数据转换成可用于机器学习的数字特征
1 字典特征提取
2 文本特征提取
3 图像特征提取


sklearn.feature_extraction

#=======字典特征抽取=======
字典特征提取》类别》One-hot编码
import sklearn
sklearn.feature_extraction.DictVectorizer(sparse=True)
vector 数学：向量 物理：矢量
矩阵 matrix 二维数据
向量 vector 一维数据

DictVectorizer.fit_transform(X) X 字典或者包含字典的迭代器返回值：返回sparse矩阵
	sparse稀疏，将非零值按照位置表示出来，可以节省内存，提高效率
DictVectorizer.inverse_transform(X) X array数组或者sparse矩阵 返回值：转换之前数据格式

DictVectorizer.get_feature_names() 返回类别名称

	

from sklearn.feature_extraction import DictVectorizer
def dict_demo():
    """
    字典特征抽取
    :return:
    """
    data=[
        {'city':'北京','temperature':100},
        {'city':'伤害','temperature':60},
        {'city':'深圳','temperature':77}
    ]
    # 实例化一个转换器类
    transfer=DictVectorizer(sparse=False)
    # 调用fit_transform()
    data_new=transfer.fit_transform(data)
    print('data_new:\n',data_new)
    print('特征名称:\n',transfer.get_feature_names())
    return None
if __name__ == '__main__':
    dict_demo()# 稀疏矩阵

	
	
#=======文本特征抽取CountVectorize=======
# 文本特征提取：对文本数据特征化
单词 作为特征
句子、短语、单词、字母
特征：特征码

sklearn.feature_extraction.text.CountVectorize(stop_word=[])
返回词频矩阵

CountVectorizer.fit_transform(X) X文本或者包含文本字符串的可迭代对象，返回值：返回sparse矩阵

CountVectorizer.get_feature_names() 返回值：单词列表

sklearn.feature_extraction.text.TfidfVectorizer



# CountVectorizer统计每个样本特征词出现的个数
from sklearn.feature_extraction.text import CountVectorizer
def count_demo():
    """
    文本特征提取 Count
    :return:
    """
    data=[
        'life is short ,i like python',
        'life is too long,i dislike python'
    ]
    # 实例化一个转换类
    transfer=CountVectorizer()
    # 调用fit_transform
    data_new=transfer.fit_transform(data)
    print('data_new \n',data_new.toarray())
    print('特征名称:\n',transfer.get_feature_names())

if __name__ == '__main__':
    count_demo()
	


#========中文文本特征抽取=======
# CountVectorizer统计每个样本特征词出现的个数
from sklearn.feature_extraction.text import CountVectorizer
def count_chinese_demo():
    """
    文本特征提取 Count
    :return:
    """
    data=[
        '我爱天安门',
        '天安门太阳'
    ]
    # 实例化一个转换类
    transfer=CountVectorizer()
    # 调用fit_transform
    data_new=transfer.fit_transform(data)
    print('data_new \n',data_new.toarray())
    print('特征名称:\n',transfer.get_feature_names())

if __name__ == '__main__':
    count_chinese_demo()
    
    
stop_word停用词
停用词表

from sklearn.feature_extraction.text import CountVectorizer
def count_demo():
    data=[
        'i love you,',
        'i like you'
    ]
    transfer=CountVectorizer(stop_words=['i'])
    data_new=transfer.fit_transform(data)
    print('data_new:\n',data_new.toarray())
    print('特征名称:\n',transfer.get_feature_names())
if __name__ == '__main__':
    count_demo()
    
    
from sklearn.feature_extraction.text import  CountVectorizer
import jieba
def count_chinese_demo():
    """
    中文文本特征，自动分词（分词库）
    :return:
    """
    data=[
        '我爱天安门','我喜欢去北京天安门'
    ]
    data_new=[]
    for sent in data:
        data_new.append(cut_word(sent))
    transfer=CountVectorizer()
    print(data_new)
    data_new=transfer.fit_transform(data_new)
    print(data_new.toarray())
    print(transfer.get_feature_names())

def cut_word(text):
    """
    进行中文分词
    :param text:
    :return:
    """
    a=jieba.cut(text) # 生成器
    a=' '.join(list(a))# 空格连接
    return a
if __name__ == '__main__':
    count_chinese_demo()
    
    

#=======文本特征抽取TfidfVectorizer=======

# 关键词：在某一类的文章中，出现的次数很多，但是在其他类别的文章中出现较少

# 文本特征抽取第2种方法：Tf-idf,衡量一个词的重要程度
# tf term-freqency：某一个给定的词语在该文件中出现的频率
# idf inverse doucument frequency：是一个词语普遍重要性的度量。某一个稳定词语的idf，可以由总文件数目除以包含该词语之文件的数目。再将得到的商取以10为底的对数得到
两个词 经济 非常
1000篇文章-语料库
100篇文章-非常
10篇文章-经济

两个文章
文章A（100个词）：10次经济 Tf-idf:
    Tf  10/100=0.1
    idf log(1000/10)=2
    tf-idf=0.1*2=0.2
文章B（100个词）：10次非常 Tf-idf
    Tf 10/100=0.1
    idf  log(1000/100)=1
    df-idf=0.1*1=0.1
说明 经济更重要



from sklearn.feature_extraction.text import TfidfVectorizer
import  jieba
def tfidf_demo():
    """
    使用TF-IDF进行文本特征抽取
    :return:
    """
    data = [
        '我爱天安门', '我喜欢去北京天安门'
    ]
    data_new=[]
    for sent in data:
        data_new.append(cut_word(sent))
    transfer=TfidfVectorizer()
    data_final=transfer.fit_transform(data_new)
    print(data_new)
    print('data_final:\n',data_final.toarray())
    print('特征名字:\n',transfer.get_feature_names())

def cut_word(text):
    return ' '.join(list(jieba.cut(text)))

if __name__ == '__main__':
    tfidf_demo()
    
    
    
#========数据预处理-归一化=======

# 通过一些转换函数将数据转换成更加适合算法模型的特征数据过程
归一化
标准化

无量纲化将不同规格的数据转换成统一规格

特征的单位或者大小相差较大，后者某特征的方差相比其他的特征要大出几个数量级，容易影响（支配）目标结果，似的一些算法无法学习到其他的特征

归一化：
x'=(x-min)/(max-min)
x''=x'*(mx-mi)+mi
作用于每一列，max为一列的最大值，min为一列的最小值;
mx,mi分别是指定区间值默认mx是1，mi是0

from sklearn.preprocessing import MinMaxScaler
import pandas as pd
def minmax_demo():
    """
    归一化
    :return:
    1 获取数据
    2 实例化一个转化器类
    3 调用fit_transfrom
    """
    data=pd.read_csv(r'.txt')
    data=data.iloc[:,:3]
    transfer=MinMaxScaler(feature_range=[0,1])
    data_new=transfer.fit_transform(data)
    print(data_new)
    
    

#=======数据预处理-标准化========
异常值一般是最大值、最小值
归一化容易受到异常值影响，归一化鲁棒性不够强

标准化：通过对原始数据进行变换把数据变换成均值为0标准差为1范围内

x'=(x-mena)/st

from sklearn.preprocessing import StandardScaler
import pandas as pd

def stand_demo():
    """
        标准化
        :return:
        1 获取数据
        2 实例化一个转化器类
        3 调用fit_transfrom
        """
    data = pd.read_csv(r'.txt')
    data = data.iloc[:, :3]
    transfer = StandardScaler()
    data_new = transfer.fit_transform(data)
    print(data_new)
if __name__ == '__main__':
    stand_demo()
    
    
    
#=======什么是降维 特征降维=======
ndarray
维数：嵌套的层数
0 维 标量
1 维 向量
2 维矩阵
3 维
n维
二维数组
此处的降维即降低特征或者说是变量的个数

降维是指在某些限定条件下，降低随机变量（特征）个数，得到一组不相关主变量的过程

效果：要求特征与特征之间不相关

降维方法：特征选择，主成分分析


#========删除低方差特征和相关系数=========
filter过滤式：
方差选择法：低方差特征过滤

相关系数：特征之间的相关程度

嵌入式：
决策时
正则化
深度学习


from sklearn.feature_selection import VarianceThreshold

import pandas as pd
def variance_demo():
    """
    低方差特征过滤
    :return:
    1 获取数据
    2 实例化一个转换器类
    3 调用fit_transfrom
    """
    data=pd.read_csv()
    data=data.iloc[:,1:-2]
    transfer=VarianceThreshold(threshold=5)
    data_new=transfer.fit_transform(data)
    print(data_new,data_new.shape)

if __name__ == '__main__':
    variance_demo()
    
    
from scipy.stats import pearsonr
import pandas as pd

def pearson_demo():
    data=pd.read_csv()
    data=data.iloc[:,1:-2]
    r=pearsonr(data['per_ratio'],data['pb_ratio'])
if __name__ == '__main__':
    pearson_demo()
    
    
    

#========主成分分析=======
sklearn.decomposition.PCA(n_compoents=None)
将数据分解为较低维数空间
n_components:
小数 表示保留百分之几的信息
整数 减少到多少特征


from sklearn.decomposition import PCA
def pca_demo():
    """
    PCA降维
    :return:
    1 实例化一个转换器类
    2 调用fit_transform
    """
    data=[
        [2,3,4,5],
        [6,3,7,4],
        [7,2,9,3]
    ]
    transfer=PCA(n_components=2)
    data_new=transfer.fit_transform(data)
    print('data_new:',data_new)

if __name__ == '__main__':
    pca_demo()
    
    
#========instacart降维案例========
order_product_prior.csv 订单与商品信息

product.csv 商品信息

orders.csv 用户的订单信息

aisles.csv 商品所属具体物品类型

1 需要将user_id和aisles放在一张表
2 找到user_id和aisles交叉表和透视表
3 特征冗余过多 pca降维


# 1 获取数据
# 2 合并表
# 3 找到user_id和aisles的关系
# 4 pca降维

import pandas as pd
order_products=pd.read_csv()
products=pd.rread_csv()
orders=pd.read_csv()
aisles=pd.read_csv()

# 合并aisles和product
tab1=pd.merge(aisles,products,on=['aisles_id','aisles_id'])
tab2=pd.merge(tab1,order_products,on=['product_id','product_id'])
tab3=pd.merge(tab2,orders,on=['order_id','order_id'])

# 找到user_id和aisles的关系
table=pd.crosstab(tab3['user_id'],tab3['aisles'])

# pca降维
from sklearn.decomposition import PCA
transfer=PCA(n_components=0.95)
data_new=transfer.fit_transform(table)
data_new.shape


#========总结=========
什么是机器学习：
数据》模型》预测

机器学习开发流程：
获取数据》数据处理》特征工程》机器学习算法训练+模型》模型评估

机器学习算法分类：
监督学习：分类+回归
无监督学习：聚类

特征工程：
数据集》特征抽取》特征预处理》特征降维

数据集：load_*小规模数据集，fetch_*大规模的数据集，bunch类型，数据集划分 sklearn.model_selection.train_test_split

特征抽取feature_extraction:字典特征抽取，DictVectorizer,sparse矩阵节省空间；文本特征抽取CountVectorizer,Tf-idf

特征预处理：无量纲化。归一化MinMaxScaler(),标准化StandardScaler()

特征选择：过滤式：删除低方差特征VarianceThreshold(),相关系数

特征降维：主成分分析PCA降维




分类算法 目标值：类别
1 sklearn转换器与预估器
2 KNN算法
3 模型选择与调优
4 朴素贝叶斯算法
5 随机森林

#========转换器与预估器=========
转换器：特征工程的父类
1 实例化（实例化一个转换器类Transformer）
2 调用fit_transform(对于建立文档分类词频矩阵，不能同时调用)
fit()+transform()

预估器：estimator
1 实例化一个estimator
2 estimator.fit(x_train,y_train)计算
--调用完毕，模型生成
3 模型评估
  1）直接对比真实值和预测值
    y_predict=estimator.predict(x_test)
  2）计算准确率
    estimator.score(x_test,y_text)
    
#========K-紧邻算法 KNN
核心思想：根据邻居推算你的类型
如果一个样本在特征空间中的k个最相似的样本中的大多数属于某一个类别，则该样本属于该找个类型

k=1 易受到异常值影响

如何确定谁是邻居，计算距离：
欧氏距离，马哈顿距离（绝对值距离），闵可夫斯基距离

电影类型分析(6个样本)：
电影名称，打斗镜头数，接吻镜头数，电影类型
推算电影类型

电影的分类：
k=1 易受到异常值影响
k=2
k=6 k取过大容易分错，当样本不均衡的时候。

无量纲化处理：标准化

from sklearn.neighbors import KNeighborsClassifier
n_neighbors=5 K值
algorithm-auto

案例：鸢尾花数据集

1 获取数据集
2 数据预处理（此处可不做）
3 特征工程（标准化）
4 KNN 预估器流程
5 模型评估

from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
def knn_iris():
    """
    用knn对鸢尾花进行分类
    :return:
    """
    # 获取数据
    iris=load_iris()
    # 划分数据集
    x_train,x_test,y_train,y_test=train_test_split(iris.data,iris.target,random_state=6)
    # 特征工程，标准化
    transfer=StandardScaler()
    x_train=transfer.fit_transform(x_train)
    x_test=transfer.transform(x_test)
    # knn算法预估器
    estimator=KNeighborsClassifier(n_neighbors=3)
    estimator.fit(x_train,y_train)
    # 模型评估
    y_predict=estimator.predict(x_test)
    print(y_predict)
    print('直接比对真实值和预测值：',y_test==y_predict)

    # 计算准确率
    score=estimator.score(x_test,y_test)
    print(score)
if __name__ == '__main__':
    knn_iris()
    
    
    
    
#========模型选择与调优========
cross validation
交叉验证：将拿到的训练数据，分为训练和验证集。以下图为例：将数据分成4份，其中一份作为验证集，然后经过4次的测试，每次都更换不同的验证集，即得到4组模型的结果，取平均值作为最终二级果，又称4折交叉验证。

数据分为训练集和测试集，但为了训练得到模型结果更加准确，做以下处理：
训练街：训练集+验证集
测试集：测试集


超参数搜索-网络搜索（grid search)
有很多参数是需要手动指定的（如k-近邻算法中的k值），这种叫超参数。手动过程复杂，所以需对模型预设集中超参数集合。每组超参数都采用交叉验证进行评估。最后选出最优参数组合建立模型。

K值 k=3 k=5 k=7
模型 模型1 模型2 模型3

模型选择与调优API

sklearn.model_selection.GridSearchCV(estimator,param_grid=None,cv=None)
对估计器的指定参数值进行详尽搜索
estimator 估计器对象
param_grid 估计器参数（dict） 'n_neighbors':{1,3,5}
cv 指定几折交叉验证
fit() 输入训练数据
score 准确率

结果分析
最佳参数 best_params_
最佳结果 best_score_
最佳估计器 best_estimator_
交叉验证结果 cv_results_



from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
def knn_iris_gscv():
    """
    用knn算法对鸢尾花进行分类、添加网络搜索和交叉验证
    :return:
    """
    # 1 获取数据
    iris=load_iris()
    # 2 划分数据集
    x_train,x_test,y_train,y_test=train_test_split(iris.data,iris.target,random_state=22)
    # 3 特征工程：标准化
    transfer=StandardScaler()
    x_train=transfer.fit_transform(x_train)
    x_test=transfer.transform(x_test)
    # 4 knn算法预估器
    estimator=KNeighborsClassifier()
    # 加入网络搜索和交叉验证
    # 参数准备
    param_dict={'n_neighbors':[1,3,5,7,9,11]}
    estimator=GridSearchCV(estimator,param_grid=param_dict,cv=10)
    estimator.fit(x_train,y_train)
    # 5 模型评估
    # 方法1：直接对比真实值和预测值
    y_predict=estimator.predict(x_test)
    print('y_predict:\n',y_predict)
    print('直接比对真实值和预测值:',y_test==y_predict)
    # 方法2：计算准确率
    score=estimator.score(x_test,y_test)
    print('准确率为:',score)

    print('最佳参数：',estimator.best_params_)
    print('最佳结果：',estimator.best_score_)
    print('最佳估计器:',estimator.best_estimator_)
    print('交叉验证结果：',estimator.cv_results_)


if __name__ == '__main__':
    knn_iris_gscv()
    
#========facebook案例流程分析========
row_id:id of the check in event登记时间的id
x,y: coordinates 坐标
accuracy:location accuracy 定位准确率
time:timestamp 时间戳
place_id:id of the business,this is the target you are predicting(目标值)

数据介绍：将根据用户的位置，准确性和时间戳预测用户正在查看的业务


流程分析：
1 获取数据
2 数据处理
目的：
特征值 x
目标值 y
缩小数据范围：2<x<2.5,1.0<y<1.5

time 年月日时分秒
过滤签到次数少的地

数据集划分
3 特征工程：标准化
4 knn算法预估流程
5 模型选择与调优
6 模型评估


#========facebook代码实现========
import pandas as pd

# 1 获取数据
data=pd.read_csv()
# 2 基本的数据处理
# 缩小数据范围
data=data.query('x<2.5 & x>2 & y<1.5 & y>1')
time_value=pd.to_datetime(data['time'],unit='s')
date=pd.DatetimeIndex(time_value)
# 星期几
date.week
date.day
date.month

data['day']=date.day
date['weekday']=date.weekday
date['hour']=date.hour

# 过滤签到次数少的place_id
place_count=data.groupby('place_id').count()['row_id']
data_final=data[data['place_id'].isin(place_count[place_count>3].index.values)]

# 筛选特征值和目标值
x=data_final[['x','y','accuracy','day','weekday','hour']]
y=data_final['place_id']

# 数据集划分
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y)

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

# 3 特征工程 标准化
transfer=StandardScaler()
x_train=transfer.fit_transform(x_train)
x_test=transfer.transform(x_test)

# 4 knn算法预估器
estimator=KNeighborsClassifier()

# 加入网格搜索与交叉验证
# 参数准备
param_dict={'n_neighbors':{1,3,4,7}}
estimator=GridSearchCV(estimator,param_grid=param_dict,cv=3)
estimator.fit(x_train,y_train)

# 5 模型评估
# 方法1：直接比对
y_predict=estimator.predict(x_test)
print('y_predict:\n',y_predict)
print('直接比对真实值和预测值:\n',y_test==y_predict)

# 方法2：计算准去率
score=estimator.score(x_test,y_test)
print('准确率为:\n',score)

# 最佳参数：
prints('最佳参数:\n',estimator.best_params_)

# 最佳预估器
print('最佳预估器:\n',estimator.best_estimator_)

# 最佳结果
print('最佳结果:\n',estimator.best_result_)





#========朴素贝叶斯算法原理=========

联合概率：包含多个条件，且所有条件同事成立的概率

条件概率：事件A在另一个事件B已经发生条件下的发生概率

相互对立：P(AB)=P(A)P(B)

条件概率：已知小明是产品经理，体重超重，是否会被女生喜欢

朴素：假设特征与特征之间是相互独立的

朴素贝叶斯：文本分类



#=========朴素贝叶斯对文本分类==========

预测样本出现新的特征值，导致概率为0。
拉普拉斯平滑系数：（Ni+a)/(N+am)
m为训练文档中统计出特征词个数


from sklearn.naive_bayes import MultinomialNB(alpha=1.0)

朴素贝叶斯分类
alpha：拉普拉斯平滑系数，默认1

案例：20类新闻分类

分析流程：sklearn
划分数据集
特征工程：文本特征抽取
朴素贝叶斯预估器流程
模型评估

from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
def nb_news():
    """
    用朴素贝叶斯对数据进行分类
    :return:
    """
    # 1 获取数据
    news=fetch_20newsgroups(subset='all')
    # 2 划分数据集
    x_train, x_test, y_train, y_test = train_test_split(news.data, news.target)
    # 3 特征工程 文本特征抽取 tf-idf
    transfer=TfidfVectorizer()
    x_train=transfer.fit_transform(x_train)
    x_test=transfer.transform(x_test)
    # 4 朴素贝叶斯预估器
    estimator=MultinomialNB()
    estimator.fit(x_train,y_train)
    # 5 模型评估
    # 方法一：
    y_predict=estimator.predict(x_test)
    print('y_predict:\n',y_predict)
    print('直接对比:\n',y_test==y_predict)

    # 方法二
    score=estimator.score(x_test,y_test)
    print('准确率:\n',score)
    
if __name__ == '__main__':
    nb_news()