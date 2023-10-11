# _*_coding:utf-8 _*_
# @Time　　 :2020/12/8/008   10:18
# @Author　 : Antipa
# @File　　 :cf_3d_bans_decision_tree_classifer.py
# @Theme    :不平衡样本处理-不处理，遍历天数查看决策树差异

import numpy as np
import pandas as pd
# import pandas_profiling as pdf
import os
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)

from_file_path = os.path.join(os.getcwd(), 'file', 'lock_zh_statistic.xlsx')
df_0 = pd.read_excel(from_file_path, sheet_name='Sheet3')
df_0.fillna(value=0, inplace=True)

for i in range(0,32):
    df = df_0[df_0.accm_days == i]
    df.reset_index(drop=True)
    feature_columns = ['order_days', 'orders', 'citys', 'ips', 'hards', 'macs']
    label_columns = ['lock_3d']
    df_feature = df[feature_columns]
    df_label = df[label_columns]


    # 五、不平衡样本处理-不处理
    print(i, '------------ 五、不平衡样本处理-不处理 ------------')

    from sklearn.model_selection import train_test_split
    from sklearn import metrics
    X_train, X_test, y_train, y_test=train_test_split(df_feature, df_label, test_size=0.3, random_state=1)
    print('训练样本', len(X_train), ';测试样本', len(X_test))

    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    clf = DecisionTreeClassifier(max_depth=3) # 决策树
    clf.fit(X_train, y_train)

    # 特征重要性
    importances = pd.DataFrame({'feature': X_train.columns, 'importance': np.round(clf.feature_importances_, 4)})
    importances = importances.sort_values('importance', ascending=False)
    print('训练集-特征重要性:\n', importances)


    score = clf.score(X_test, y_test)
    # print('测试集-准确率', score)

    y_test_pred = clf.predict(X_test)
    acc = metrics.accuracy_score(y_test, y_test_pred)
    recall = metrics.recall_score(y_test, y_test_pred)
    precision = metrics.precision_score(y_test, y_test_pred)
    print('准确率', acc, '\t召回率', recall, '\t精确度', precision)

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
        # class_names=['1', '0'],
        filled=True,
        rounded=True,
        special_characters=True
    )
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

    # tree_out_file = os.path.join(os.getcwd(), 'file', 'lock_zh_statistic_v5_{}.png'.format(i))
    tree_out_file_pdf = os.path.join(os.getcwd(), 'file', 'lock_zh_statistic_v5_{}.pdf'.format(i))

    # graph.write_png(tree_out_file)  #当前文件夹生成out.png
    graph.write_pdf(tree_out_file_pdf)  #当前文件夹生成out.png
