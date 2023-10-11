# encoding:utf-8
import sklearn.datasets as skdt
bos=skdt.load_boston()
data=bos['data']
target=bos['target']

import sklearn.model_selection as skms
X_train,X_test,y_train,y_test=skms.train_test_split(data,target,test_size=0.3,random_state=42)
import sklearn.linear_model as sklm
lr=sklm.LinearRegression()
lr.fit(X_train,y_train)
lr.predict(X_train)
lr.score(X_train,y_train)
lr_score=lr.score(X_test,y_test)

sgdr=sklm.SGDRegressor()
sgdr.fit(X_train,y_train)
sgdr.predict(X_train)
sgdr.score(X_train,y_train)
sgdr_score=sgdr.score(X_test,y_test)

print(lr_score,sgdr_score)

# sklearn官网建议，训练数据规模超过10万，推荐使用随机梯度法估计参数模型。