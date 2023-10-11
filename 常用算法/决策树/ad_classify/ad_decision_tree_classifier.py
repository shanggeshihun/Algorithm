# _*_coding:utf-8 _*_
# @Time　　 :2020/12/4/004   12:46
# @Author　 : Antipa
# @File　　 :ad_decision_tree_classifier.py
# @Theme    :PyCharm

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from_file_path = os.path.join(os.getcwd(), 'file', 'result.xlsx')
df = pd.read_excel(from_file_path)
df.fillna(value=0, inplace=True)

feature_columns = ['package_cnt', 'kh_sample_cnt', 'kh_black_sample_cnt', 'kh_black_sample_cnt_per', 'sample_cnt',
                   'black_sample_cnt', 'black_sample_cnt_per', 'tuia_sdk_samples', 'tuia_sdk_samples_flag',
                   'meishu_sdk_samples', 'meishu_sdk_samples_flag', 'tuia_meishu_sdk_flag', 'ad_value_cnt',
                   'risk_value_cnt', 'risk_value_cnt_per']
label_columns = ['type']
df_feature = df[feature_columns]
df_label = df[label_columns]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df_feature, df_label, test_size=0.3, random_state=1)
print('训练样本', len(X_train), ';测试样本', len(X_test))

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

clf = DecisionTreeClassifier(max_depth=4)  # 决策树
clf.fit(X_train, y_train)

# 特征重要性
importances = pd.DataFrame({'feature': X_train.columns, 'importance': np.round(clf.feature_importances_, 4)})
importances = importances.sort_values('importance', ascending=False)
print('特征重要性:\n', importances)

score = clf.score(X_test, y_test)
print('准确率', score)

from six import StringIO
import pydotplus
from sklearn import tree

dot_data = StringIO()
# 单独安装graphviz.msi 软件
tree.export_graphviz(clf,
                     out_file=dot_data,
                     max_depth=4,
                     feature_names=feature_columns,
                     class_names=['1', '0'],
                     filled=True,
                     rounded=True,
                     special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

tree_out_file = os.path.join(os.getcwd(), 'file', 'tree.png')
tree_out_file_pdf = os.path.join(os.getcwd(), 'file', 'tree.pdf')

graph.write_png(tree_out_file)  # 当前文件夹生成out.png
graph.write_pdf(tree_out_file_pdf)  # 当前文件夹生成out.png

# 自动化输出规则到Excel
from sklearn.tree import _tree
def Tree_Rules(clf, X):
    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold
    value = clf.tree_.value

    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, 0)]

    while len(stack) > 0:

        node_id, depth = stack.pop()
        node_depth[node_id] = depth

        is_split_node = children_left[node_id] != children_right[node_id]

        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            is_leaves[node_id] = True
    feature_name = [
        X.columns[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in clf.tree_.feature]

    ways = []
    depth = []
    feat = []
    nodes = []
    rules = []
    for i in range(n_nodes):
        if is_leaves[i]:
            while depth[-1] >= node_depth[i]:
                depth.pop()
                ways.pop()
                feat.pop()
                nodes.pop()
            if children_left[i - 1] == i:
                a = '{f}<={th}'.format(f=feat[-1], th=round(threshold[nodes[-1]], 4))
                ways[-1] = a
                last = ' & '.join(ways) + ':' + str(value[i][0][0]) + ':' + str(value[i][0][1])
                rules.append(last)
            else:
                a = '{f}>{th}'.format(f=feat[-1], th=round(threshold[nodes[-1]], 4))
                ways[-1] = a
                last = ' & '.join(ways) + ':' + str(value[i][0][0]) + ':' + str(value[i][0][1])
                rules.append(last)

        else:
            if i == 0:
                ways.append(round(threshold[i], 4))
                depth.append(node_depth[i])
                feat.append(feature_name[i])
                nodes.append(i)
            else:
                while depth[-1] >= node_depth[i]:
                    depth.pop()
                    ways.pop()
                    feat.pop()
                    nodes.pop()
                if i == children_left[nodes[-1]]:
                    w = '{f}<={th}'.format(f=feat[-1], th=round(threshold[nodes[-1]], 4))
                else:
                    w = '{f}>{th}'.format(f=feat[-1], th=round(threshold[nodes[-1]], 4))
                ways[-1] = w
                ways.append(round(threshold[i], 4))
                depth.append(node_depth[i])
                feat.append(feature_name[i])
                nodes.append(i)
    return rules


# 对决策树规则进行提取
Rules = Tree_Rules(clf, X_train)
# list转为表结构
df = pd.DataFrame(Rules)
# 导出为excel
df.to_excel('output.xlsx', index=False)
