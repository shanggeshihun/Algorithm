# _*_coding:utf-8 _*_
# @Time　　 :2020/12/7/007   16:20
# @Author　 : Antipa
# @File　　 :forest_decision_tree_classifer.py
# @Theme    :PyCharm
# DataConversionWarning: A column-vector y was passed when a 1d array was expected

import numpy as np
import pandas as pd
# import pandas_profiling as pdf
import os
import matplotlib.pyplot as plt
from_file_path=os.path.join(os.getcwd(),'file','result.xlsx')
df=pd.read_excel(from_file_path)
df.fillna(value=0,inplace=True)


feature_columns=[ 'package_cnt', 'kh_sample_cnt', 'kh_black_sample_cnt','kh_black_sample_cnt_per','sample_cnt', 'black_sample_cnt','black_sample_cnt_per', 'tuia_sdk_samples','tuia_sdk_samples_flag', 'meishu_sdk_samples','meishu_sdk_samples_flag','tuia_meishu_sdk_flag', 'ad_value_cnt', 'risk_value_cnt','risk_value_cnt_per']
label_columns=['type']
df_feature=df[feature_columns]
df_label=df[label_columns]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(df_feature, df_label,test_size=0.1,random_state=1)
print('训练样本',len(X_train),';测试样本',len(X_test))


from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE # 导入SMOTE算法模块

forest_clf=RandomForestClassifier(n_estimators=1000, random_state=42, n_jobs=-1) # 随机森林

# 处理不平衡数据
sm = SMOTE(random_state=42)    # 处理过采样的方法
X, y = sm.fit_resample(X_train, y_train.values.ravel())

forest_clf.fit(X,y)

# 特征重要性
forest_importances = pd.DataFrame({'feature':X_train.columns,'forest_importances':np.round(forest_clf.feature_importances_,4)})
forest_importances = forest_importances.sort_values('forest_importances',ascending=False)
print('随机森林_特征重要性:\n',forest_importances)

score = forest_clf.score(X_test, y_test)
print('随机森林_准确率',score)

