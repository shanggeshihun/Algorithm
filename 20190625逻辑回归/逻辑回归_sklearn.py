# -*- coding: utf-8 -*-
"""
Created on 20190625

@author: dell
"""

import sklearn.datasets as skdt
iris=skdt.load_iris()
iris_data=iris['data']
iris_target=iris['target']
import sklearn.model_selection as skms
X_train,X_test,y_train,y_test=skms.train_test_split(iris_data,iris_target,test_size=0.3,random_state=42)
import sklearn.linear_model as sklm
lr_model=sklm.LogisticRegression()
lr_model.fit(X_train,y_train)
print(lr_model.predict_log_proba(X_train)[0])
print(lr_model.predict(X_train)[0])
print(lr_model.predict_proba(X_train)[0])
print(lr_model.score(X_train,y_train))
print(lr_model.score(X_test,y_test))
