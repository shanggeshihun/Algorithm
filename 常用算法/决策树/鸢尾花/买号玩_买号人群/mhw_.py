# _*_coding:utf-8 _*_
# @Time　　 :2020/12/7/007   12:38
# @Author　 : Antipa
# @File　　 :iris_decision_tree_classifier.py
# @Theme    :PyCharm

import sys
from sklearn.datasets import load_iris
from sklearn import tree
from six import StringIO
import pydotplus
import os
import pandas as pd
import numpy as np


data = load_iris()  # 载入数据集
# df = pd.DataFrame(data.data, columns=data.feature_names)
# df['target'] = data.target
df = pd.read_excel("./mhw_user.xlsx")

df.fillna(value = 0 ,inplace=True)
df['is_paid'] =df.is_paid.astype(str)
df['is_ordered'] =df.is_ordered.astype(str)
print(df.info())
# 将数据拆分为训练和测试集


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(df.iloc[:,3:10], df['is_ordered'],test_size = 0.3, random_state=0)

feature_names,target_names = df.columns[3:10], ['0','1']

X_train,Y_train =df.iloc[:,3:10],df['is_ordered']

clf = tree.DecisionTreeClassifier(max_depth=4)  # 算法模型
clf = clf.fit(X_train, Y_train,)  # 模型训练

# 输出决策树
dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data,
                         feature_names=feature_names,
                         class_names=target_names,
                         filled=True, rounded=True,
                         special_characters=True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
save_path=os.path.join(os.getcwd(),'mhw_user_tree_result.pdf')
graph.write_pdf(save_path)  #

#
#
# # Predict for 1 observation
# clf.predict(X_test.iloc[0].values.reshape(1, -1))
# # Predict for multiple observations
# clf.predict(X_test[0:10])
#
# # （正确预测的分数）：正确预测 / 数据点总数
# # The score method returns the accuracy of the model
# score = clf.score(X_test, Y_test)
# print(score)
#
# # 调整树的深度
# # List of values to try for max_depth:
# max_depth_range = list(range(1, 6))
# # List to store the average RMSE for each value of max_depth:
# accuracy = []
# for depth in max_depth_range:
#     clf = tree.DecisionTreeClassifier(max_depth=depth,
#                                  random_state=0)
#     clf.fit(X_train, Y_train)
#     score = clf.score(X_test, Y_test)
#     accuracy.append(score)
#
# # 特征重要性
# importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(clf.feature_importances_,3)})
# importances = importances.sort_values('importance',ascending=False)
#
