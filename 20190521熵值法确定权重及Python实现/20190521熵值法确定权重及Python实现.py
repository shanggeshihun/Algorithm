# -*- coding: utf-8 -*-
"""
Created on 20190521

"""

import numpy as np
import pandas as pd
#熵值法确定权重及Python实现
class EmtropyMethod:
    def __init__(self,dataframe,positive,negative,index_name):
        """
        dataframe:DataFrame
        positive:正向指标(越高越好)
        negative:负向指标(越低越好)
        """
        if len(dataframe)!=len(index_name):
            raise Exception('数据指标行数与行名称数不符')
        if sorted(dataframe.columns)!=sorted(positive+negative):
            raise Exception('正项指标加负向指标不等于数据指标的条目数')
        self.dataframe = dataframe.copy().astype('float64')
        self.positive = positive
        self.negative = negative
        self.index_name = index_name.copy()

    #数据归一化
    def uniform(self):
        uniform_mat = self.dataframe.copy()
        min_dataframe = {column:min(uniform_mat[column]) for column in uniform_mat.columns}
        max_dataframe = {column:max(uniform_mat[column]) for column in uniform_mat.columns}
        for i in range(len(uniform_mat)):
            for column in uniform_mat.columns:
                if column in self.negative:
                    if max_dataframe[column]==min_dataframe[column]:
                        uniform_mat[column][i]=0
                    else:
                        uniform_mat[column][i] = (uniform_mat[column][i] - min_dataframe[column]) / (max_dataframe[column] - min_dataframe[column])
                else:
                    if max_dataframe[column]==min_dataframe[column]:
                        uniform_mat[column][i]=0
                    else:
                        print('\n',max_dataframe[column],uniform_mat[column][i],min_dataframe[column])
                        uniform_mat[column][i] = (max_dataframe[column] - uniform_mat[column][i]) / (max_dataframe[column] - min_dataframe[column])
                        print(uniform_mat[column][i])
                        print(i)
        self.uniform_mat = uniform_mat
        return uniform_mat

    #指标比重
    def calc_probability(self):
        try:
            p_mat = self.uniform_mat.copy()
        except AttributeError:
            raise Exception('你还没进行归一化处理，请先调用uniform方法')
        for column in p_mat.columns:
            sigma_x_1_n_j = sum(p_mat[column])
            p_mat[column] = p_mat[column].apply(lambda x_i_j: x_i_j / sigma_x_1_n_j if x_i_j / sigma_x_1_n_j != 0 else 1e-6)

        self.p_mat = p_mat
        return p_mat
    
        pass

    #计算熵值
    def calc_emtropy(self):
        try:
            self.p_mat.head(0)
        except AttributeError:
            raise Exception('你还没计算比重，请先调用calc_probability方法')
        import numpy as np
        e_j  = -(1 / np.log(len(self.p_mat)+1)) * np.array([sum([pij*np.log(pij) for pij in self.p_mat[column]]) for column in self.p_mat.columns])
        ejs = pd.Series(e_j, index=self.p_mat.columns, name='指标的熵值')

        self.emtropy_series = ejs
        return self.emtropy_series

    #计算信息熵冗余度
    def calc_emtropy_redundancy(self):
        try:
            self.d_series = 1 - self.emtropy_series
            self.d_series.name = '信息熵冗余度'
        except AttributeError:
            raise Exception('你还没计算信息熵，请先调用calc_emtropy方法')
        return self.d_series

    #计算权值
    def calc_weight(self):
        self.uniform()
        self.calc_probability()
        self.calc_emtropy()
        self.calc_emtropy_redundancy()
        self.weight = self.d_series / sum(self.d_series)
        self.weight.name = '权值'
        return self.weight

    def calc_score(self):
        self.calc_weight()
        self.score = pd.Series([np.dot(np.array(self.dataframe[row:row+1])[0], np.array(self.weight)) for row in range(len(self.dataframe))],index=self.index_name, name='得分').sort_values(ascending=False)
        return self.score


if __name__=='__main__':
    #读入数据
    import pandas as pd
    path=r"E:\SJB\NOTE\Python\algorithm\20190521熵值法确定权重及Python实现\province_eco.txt"
    lst=[]
    with open(path,'r',encoding='utf-8') as f:
        for line in f.readlines():
            the_lst=line.strip().split('\t')
            lst.append(the_lst)
    cols=lst[0]
    df=pd.DataFrame(lst[1:],columns=cols)
    df=df.dropna().reset_index(drop=True)
    df[cols[1:]]=df[cols[1:]].astype(float)
    
    indexs=["GDP总量增速", "人口总量", "人均GDP增速", "地方财政收入总额", "固定资产投资", "社会消费品零售总额增速", "进出口总额","城镇居民人均可支配收入", "农村居民人均可支配收入"]
    
    positive=indexs
    negative=[]
    
    index_name=df['地区']
    dataframe=df[indexs]
    df.dtypes
    em=EmtropyMethod(dataframe, negative, positive, index_name)
    em.uniform()
    em.calc_probability()
    em.calc_emtropy()
    em.calc_emtropy_redundancy()
    em.calc_weight()
    em.calc_score()
    print(em.calc_score())