import pandas as pd
path=r"C:\Users\dell\Desktop\hh.xlsx"
df=pd.read_excel(path) #DataFrame
df0=df[['order_number','unit_err_rate','original_unit_price','actual_unit_price','unit_price_err_rate','speed']]
samples=len(df)#3045
df.describe()
df.dtypes

#变量权重
from EmtropyForWeight import EmtropyMethod
index=df0.iloc[:,1:]
print(len(index))
positive=['unit_err_rate','original_unit_price','actual_unit_price','unit_price_err_rate','speed']
negative=[]
row_name=df['order_number']
print(len(row_name))
em=EmtropyMethod(index,positive,negative,row_name)
weight=em.calc_weight()
print(weight)


X=df0.loc[:,['unit_err_rate','original_unit_price','actual_unit_price','speed']]
from sklearn.neighbors import LocalOutlierFactor
model=LocalOutlierFactor(n_neighbors=30,contamination=0.05)
model.fit(X)
dis,sample_idx=model.kneighbors(X)
dis[0].max(axis=0)
len(dis[0]) #30
len(sample_idx[0]) #30

X_pred=model._predict(X)
X_score=-model._decision_function(X)

from collections import Counter
Counter(X_pred)

df['X_pred']=X_pred
df.to_excel(r"C:\Users\dell\Desktop\hh_pred_lof.xlsx")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
fig=plt.figure()
x1=X['unit_err_rate']
x2=X['original_unit_price']
x3=X['speed']
axes3d=Axes3D(fig)
axes3d.scatter(x1,x2,x3,c=X_pred)
plt.show()
