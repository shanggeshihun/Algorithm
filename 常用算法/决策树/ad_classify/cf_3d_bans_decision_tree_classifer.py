# _*_coding:utf-8 _*_
# @Time　　 :2020/12/8/008   10:18
# @Author　 : Antipa
# @File　　 :cf_3d_bans_decision_tree_classifer.py
# @Theme    :不平衡样本处理-SMOTETomek（综合采样）

import numpy as np
import pandas as pd
# import pandas_profiling as pdf
import os
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)

from_file_path = os.path.join(os.getcwd(), 'file', 'lock_zh_statistic.xlsx')
df = pd.read_excel(from_file_path, sheet_name='Sheet2')
df.fillna(value=0, inplace=True)

feature_columns = ['order_days', 'orders', 'citys', 'ips', 'hards', 'macs']
label_columns = ['lock_3d']
df_feature = df[feature_columns]
df_label = df[label_columns]


# 一、不平衡样本处理-综合采样 SMOTETomek
print('------------ 一、不平衡样本处理-综合采样 SMOTETomek ------------')
from imblearn.combine import SMOTETomek
cs = SMOTETomek(random_state=0)  # 综合采样
X_feature, y_label = cs.fit_resample(df_feature, df_label)
print('平衡样本处理后', '黑样本', len(df_label == 1), '；白样本', len(df_label == 0))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X_feature, y_label,test_size=0.3, random_state=1)
print('训练样本', len(X_train), ';测试样本', len(X_test))

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
clf = DecisionTreeClassifier(max_depth=3) # 决策树
clf.fit(X_train, y_train)

# 特征重要性
importances = pd.DataFrame({'feature': X_train.columns, 'importance': np.round(clf.feature_importances_, 4)})
importances = importances.sort_values('importance', ascending=False)
print('训练集-特征重要性:\n', importances)

score = clf.score(X_test, y_test)
print('测试集-准确率', score)

from six import StringIO
import pydotplus
from sklearn import tree

dot_data = StringIO()
# 单独安装graphviz.msi 软件
tree.export_graphviz(
    clf,
    out_file=dot_data,
    max_depth=3,
    feature_names=feature_columns,
    class_names=['1','0'],
    filled=True,
    rounded=True,
    special_characters=True
)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

tree_out_file = os.path.join(os.getcwd(), 'file', 'lock_zh_statistic_v1.png')
tree_out_file_pdf = os.path.join(os.getcwd(), 'file', 'lock_zh_statistic_v2.pdf')

graph.write_png(tree_out_file)  #当前文件夹生成out.png
graph.write_pdf(tree_out_file_pdf)  #当前文件夹生成out.png




# 二、不平衡样本处理-过采样 Borderline SMOTE
"""
Borderline SMOTE是在SMOTE基础上改进的过采样算法，该算法仅使用边界上的少数类样本来合成新样本，从而改善样本的类别分布
"""
print('------------ 二、不平衡样本处理-过采样 Borderline SMOTE ------------')
from imblearn.over_sampling import BorderlineSMOTE
cs = BorderlineSMOTE(random_state=0)
X_feature, y_label = cs.fit_resample(df_feature, df_label)
print('平衡样本处理后', '黑样本', len(df_label == 1), '；白样本', len(df_label == 0))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X_feature, y_label,test_size=0.3, random_state=1)
print('训练样本', len(X_train), ';测试样本', len(X_test))

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
clf = DecisionTreeClassifier(max_depth=3) # 决策树
clf.fit(X_train, y_train)

# 特征重要性
importances = pd.DataFrame({'feature': X_train.columns, 'importance': np.round(clf.feature_importances_, 4)})
importances = importances.sort_values('importance', ascending=False)
print('训练集-特征重要性:\n', importances)

score = clf.score(X_test, y_test)
print('测试集-准确率', score)

from six import StringIO
import pydotplus
from sklearn import tree

dot_data = StringIO()
# 单独安装graphviz.msi 软件
tree.export_graphviz(
    clf,
    out_file=dot_data,
    max_depth=3,
    feature_names=feature_columns,
    class_names=['1','0'],
    filled=True,
    rounded=True,
    special_characters=True
)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

tree_out_file = os.path.join(os.getcwd(), 'file', 'lock_zh_statistic_v2.png')
tree_out_file_pdf = os.path.join(os.getcwd(), 'file', 'lock_zh_statistic_v2.pdf')

graph.write_png(tree_out_file)  #当前文件夹生成out.png
graph.write_pdf(tree_out_file_pdf)  #当前文件夹生成out.png



# 三、不平衡样本处理-过采样 ADASYN
"""
ADASYN自适应合成抽样，与Borderline SMOTE相似，对不同的少数类样本赋予不同的权重，从而生成不同数量的样本
"""
print('------------ 三、不平衡样本处理-过采样 ADASYN ------------')
from imblearn.over_sampling import ADASYN
cs = BorderlineSMOTE(random_state=0)
X_feature, y_label = cs.fit_resample(df_feature, df_label)
print('平衡样本处理后', '黑样本', len(df_label == 1), '；白样本', len(df_label == 0))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X_feature, y_label,test_size=0.3, random_state=1)
print('训练样本', len(X_train), ';测试样本', len(X_test))

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
clf = DecisionTreeClassifier(max_depth=3) # 决策树
clf.fit(X_train, y_train)

# 特征重要性
importances = pd.DataFrame({'feature': X_train.columns, 'importance': np.round(clf.feature_importances_, 4)})
importances = importances.sort_values('importance', ascending=False)
print('训练集-特征重要性:\n', importances)

score = clf.score(X_test, y_test)
print('测试集-准确率', score)

from six import StringIO
import pydotplus
from sklearn import tree

dot_data = StringIO()
# 单独安装graphviz.msi 软件
tree.export_graphviz(
    clf,
    out_file=dot_data,
    max_depth=3,
    feature_names=feature_columns,
    class_names=['1','0'],
    filled=True,
    rounded=True,
    special_characters=True
)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

tree_out_file = os.path.join(os.getcwd(), 'file', 'lock_zh_statistic_v3.png')
tree_out_file_pdf = os.path.join(os.getcwd(), 'file', 'lock_zh_statistic_v3.pdf')

graph.write_png(tree_out_file)  #当前文件夹生成out.png
graph.write_pdf(tree_out_file_pdf)  #当前文件夹生成out.png





# 四、不平衡样本处理-过采样 SMOTE
"""
SMOTE 合成少数类过采样技术，是在随机采样的基础上改进的一种过采样算法。SMOTE实现简单，但其弊端也很明显，由于SMOTE对所有少数类样本一视同仁，并未考虑近邻样本的类别信息，往往出现样本混叠现象，导致分类效果不佳
"""
print('------------ 四、不平衡样本处理-过采样 SMOTE ------------')
from imblearn.over_sampling import ADASYN
cs = BorderlineSMOTE(random_state=0)
X_feature, y_label = cs.fit_resample(df_feature, df_label)
print('平衡样本处理后', '黑样本', len(df_label == 1), '；白样本', len(df_label == 0))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X_feature, y_label,test_size=0.3, random_state=1)
print('训练样本', len(X_train), ';测试样本', len(X_test))

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
clf = DecisionTreeClassifier(max_depth=3) # 决策树
clf.fit(X_train, y_train)

# 特征重要性
importances = pd.DataFrame({'feature': X_train.columns, 'importance': np.round(clf.feature_importances_, 4)})
importances = importances.sort_values('importance', ascending=False)
print('训练集-特征重要性:\n', importances)

score = clf.score(X_test, y_test)
print('测试集-准确率', score)

from six import StringIO
import pydotplus
from sklearn import tree

dot_data = StringIO()
# 单独安装graphviz.msi 软件
tree.export_graphviz(
    clf,
    out_file=dot_data,
    max_depth=3,
    feature_names=feature_columns,
    class_names=['1','0'],
    filled=True,
    rounded=True,
    special_characters=True
)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

tree_out_file = os.path.join(os.getcwd(), 'file', 'lock_zh_statistic_v4.png')
tree_out_file_pdf = os.path.join(os.getcwd(), 'file', 'lock_zh_statistic_v4.pdf')

graph.write_png(tree_out_file)  #当前文件夹生成out.png
graph.write_pdf(tree_out_file_pdf)  #当前文件夹生成out.png



# 五、不平衡样本处理-不处理
print('------------ 五、不平衡样本处理-不处理 ------------')

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(df_feature, df_label,test_size=0.3, random_state=1)
print('训练样本', len(X_train), ';测试样本', len(X_test))

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
clf = DecisionTreeClassifier(max_depth=3) # 决策树
clf.fit(X_train, y_train)

# 特征重要性
importances = pd.DataFrame({'feature': X_train.columns, 'importance': np.round(clf.feature_importances_, 4)})
importances = importances.sort_values('importance', ascending=False)
print('训练集-特征重要性:\n', importances)

score = clf.score(X_test, y_test)
print('测试集-准确率', score)

from six import StringIO
import pydotplus
from sklearn import tree

dot_data = StringIO()
# 单独安装graphviz.msi 软件
tree.export_graphviz(
    clf,
    out_file=dot_data,
    max_depth=3,
    feature_names=feature_columns,
    class_names=['1','0'],
    filled=True,
    rounded=True,
    special_characters=True
)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

tree_out_file = os.path.join(os.getcwd(), 'file', 'lock_zh_statistic_v5.png')
tree_out_file_pdf = os.path.join(os.getcwd(), 'file', 'lock_zh_statistic_v5.pdf')

graph.write_png(tree_out_file)  #当前文件夹生成out.png
graph.write_pdf(tree_out_file_pdf)  #当前文件夹生成out.png