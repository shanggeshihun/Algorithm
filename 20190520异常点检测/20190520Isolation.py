#coding:utf-8
import pandas as pd
import matplotlib.pyplot as plt
path=r"C:\Users\dell\Desktop\hh.xlsx"
df=pd.read_excel(path) #DataFrame
#剔除distance为0记录
df.drop(df[df['distance']==0].index,inplace=True)
#衍生特征
df['speed']=df['distance']/df['gap_hours']
df['unit_distance_price']=df['actual_unit_price']/df['distance']
df.reset_index(drop=True,inplace=True)
samples=len(df)
df.describe()
df.dtypes
plt.figure()
plt.scatter(df.distance,df.actual_unit_price)
plt.show()




#IsolationForest
    #变量权重
cols=['order_number','unit_err_rate','unit_distance_price','speed']
dfw=df[cols]
dfw.dtypes
from EmtropyForWeight import EmtropyMethod
index_name=dfw['order_number']

dataframe=dfw.iloc[:,1:]
dataframe.dtypes

positive=list(dataframe.columns)
negative=[]

em=EmtropyMethod(dataframe,positive,negative,index_name)

uniform=em.uniform()
weight=em.calc_weight()
print(uniform)
print(weight)

    #异常点输出
df0=df[cols]
samples=len(df0)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
clf=IsolationForest(n_estimators=100,
                    max_samples=3000,
                    contamination=0.05)
X=df0.iloc[:,1:]
clf.fit(X)
X_pred=clf.predict(X) #numpy.ndarray
X_score=clf.decision_function(X)

from collections import Counter
Counter(X_pred)

df['X_pred']=X_pred
df['X_score']=X_score


#降维
from sklearn.decomposition import PCA
pca=PCA()
pca.fit(uniform)
#返回各个成分各自的方差百分比
ratio=pca.explained_variance_ratio_*100

pca=PCA(3)
pca.fit(uniform)
#返回模型的各个特征向量
pca.components_

cp1=pca.components_[0]
cp1_max_idx=np.argwhere(abs(cp1)==max(abs(cp1)))[0][0]+1
cp1_max_idx_name=cols[cp1_max_idx]

cp2=pca.components_[1]
cp2_max_idx=np.argwhere(abs(cp2)==max(abs(cp2)))[0][0]+1
cp2_max_idx_name=cols[cp2_max_idx]

cp3=pca.components_[2]
cp3_max_idx=np.argwhere(abs(cp3)==max(abs(cp3)))[0][0]+1
cp3_max_idx_name=cols[cp3_max_idx]

components_theme={0:cp1_max_idx_name,1:cp2_max_idx_name,2:cp3_max_idx_name}

#降维后的数据
cLst=list(map(lambda x:'r' if x==-1 else 'c',X_pred))
new_uniform=pca.fit_transform(uniform)
factor1=new_uniform[:,0]
factor2=new_uniform[:,1]
factor3=new_uniform[:,2]

#每个案例的代表成分
sample_theme=[]
sample_theme_factor=[]
for i in range(samples):
    tmp_lst=[new_uniform[i,0],new_uniform[i,1],new_uniform[i,2]]
    max_idx=list(map(abs,tmp_lst)).index(max(map(abs,tmp_lst)))
    max_factor=new_uniform[i,max_idx]
    sample_theme.append(max_idx)
    sample_theme_factor.append(max_factor)
df['sample_theme']=[components_theme[i] for i in sample_theme]
df['sample_theme_factor']=sample_theme_factor
df.to_csv(r"C:\Users\dell\Desktop\hh_pred_isolation.csv")

from mpl_toolkits.mplot3d.axes3d import Axes3D
fig = plt.figure()#此处fig是二维
axes3d = Axes3D(fig)#将二维转化为三维
axes3d.scatter(factor1,factor2,factor3,c=cLst)
plt.show()


