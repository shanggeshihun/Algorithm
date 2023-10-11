# _*_coding:utf-8 _*_
# @Time　　 :2020/12/8/008   10:18
# @Author　 : Antipa
# @File　　 :ad_decision_tree_classifer_v_1.py
# @Theme    :不平衡样本处理


import numpy as np
import pandas as pd
# import pandas_profiling as pdf
import os
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)

from_file_path=os.path.join(os.getcwd(),'file','result.xlsx')
df=pd.read_excel(from_file_path)
df.fillna(value=0,inplace=True)

feature_columns=[ 'package_cnt', 'kh_sample_cnt', 'kh_black_sample_cnt','kh_black_sample_cnt_per','sample_cnt', 'black_sample_cnt','black_sample_cnt_per', 'tuia_sdk_samples','tuia_sdk_samples_flag', 'meishu_sdk_samples','meishu_sdk_samples_flag','tuia_meishu_sdk_flag', 'ad_value_cnt', 'risk_value_cnt','risk_value_cnt_per']
label_columns=['type']
df_feature=df[feature_columns]
df_label=df[label_columns]

# 不平衡样本处理
from imblearn.combine import SMOTETomek

cs = SMOTETomek(random_state=0)  # 综合采样
X_feature, y_label = cs.fit_resample(df_feature, df_label)
print('平衡样本处理后','黑样本',len(df_label==0),'；白样本',len(df_label==1))

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X_feature, y_label,test_size=0.3,random_state=1)
print('训练样本',len(X_train),';测试样本',len(X_test))

from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor

clf = DecisionTreeClassifier(max_depth=4) # 决策树
clf.fit(X_train, y_train)

# 特征重要性
importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(clf.feature_importances_,4)})
importances = importances.sort_values('importance',ascending=False)
print('特征重要性:\n',importances)

score = clf.score(X_test, y_test)
print('准确率',score)

from six import StringIO
import pydotplus
from sklearn import tree

dot_data = StringIO()
# 单独安装graphviz.msi 软件
tree.export_graphviz(clf,
                     out_file=dot_data,
                     max_depth=4,
                     feature_names=feature_columns,
                     class_names=['1','0'],
                     filled=True,
                     rounded=True,
                     special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

tree_out_file=os.path.join(os.getcwd(),'file','tree.png')
tree_out_file_pdf=os.path.join(os.getcwd(),'file','tree.pdf')

graph.write_png(tree_out_file)  #当前文件夹生成out.png
graph.write_pdf(tree_out_file_pdf)  #当前文件夹生成out.png
