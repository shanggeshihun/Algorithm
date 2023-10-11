#coding:utf-8
import pandas as pd
import matplotlib.pyplot as plt

def deal_df():
    """
    return：读取并清洗后的数据
    """
    path=r"C:\Users\dell\Desktop\hh.xlsx"
    df=pd.read_excel(path) #DataFrame
    #剔除distance为0记录
    df.drop(df[df['distance']==0].index,inplace=True)
    #衍生特征
    df['speed']=df['distance']/df['gap_hours']
    df['unit_distance_price']=df['actual_unit_price']/df['distance']
    original_df=df.reset_index(drop=True)
    return original_df

class AbnormalOrderPred():
    """
    cols:案例及指标形成列表
    dataframe:数据集
    """
    def __init__(self,original_df,target_cols,isolation_percent):
        self.target_cols=target_cols
        self.original_df=original_df
        self.isolation_percent=isolation_percent

    def uniform_and_weight(self):
        original_df=self.original_df
        from EmtropyForWeight import EmtropyMethod
        index_name=original_df[target_cols[0]]
        dataframe=original_df[target_cols[1:]]
        positive_column=target_cols[1:].copy()
        negative_column=[]
        em=EmtropyMethod(dataframe,positive_column,negative_column,index_name)
        self.uniform=em.uniform()
        self.weight=em.calc_weight()
        return self.uniform,self.weight

    def pred(self):
        from sklearn.ensemble import IsolationForest
        clf=IsolationForest(n_estimators=100,
                            max_samples=3000,
                            contamination=0.05)
        dataframe=original_df[target_cols[1:]]
        clf.fit(dataframe)
        self.pred=clf.predict(dataframe) #numpy.ndarray
        self.pred_of_score=clf.decision_function(dataframe)
        return self.pred,self.pred_of_score

    def pca(self):
        uniform=self.uniform
        from sklearn.decomposition import PCA
        pca=PCA(3)
        pca.fit(uniform)
        self.new_uniform=pca.fit_transform(uniform)
        self.pca_components=pca.components_
        return self.new_uniform,self.pca_components
    
    def pca_plot(self):
        pred=self.pred
        new_uniform=self.new_uniform
        cLst=list(map(lambda x:'r' if x==-1 else 'c',pred))
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.axes3d import Axes3D
        cLst=list(map(lambda x:'r' if x==-1 else 'c',pred))
        factor1=new_uniform[:,0]
        factor2=new_uniform[:,1]
        factor3=new_uniform[:,2]
        fig = plt.figure()#此处fig是二维
        axes3d = Axes3D(fig)#将二维转化为三维
        axes3d.scatter(factor1,factor2,factor3,c=cLst)
        plt.show()

import numpy as np
if __name__=='__main__':
    original_df=deal_df()
    samples=len(original_df)
    isolation_percent=0.1
    target_cols=['order_number','unit_err_rate','unit_distance_price','speed']
    aop=AbnormalOrderPred(original_df,target_cols,isolation_percent)
    uniform,weight=aop.uniform_and_weight()
    pred,pred_of_score=aop.pred()
    aop.pca()
    aop.pca_plot()

#降维
from sklearn.decomposition import PCA
pca=PCA()
pca.fit(uniform)
#返回各个成分各自的方差百分比
ratio=pca.explained_variance_ratio_*100

pca=PCA(3)
pca.fit(uniform)
#返回模型的各个特征向量
pca.components_

cp1=pca.components_[0]
cp1_max_idx=np.argwhere(abs(cp1)==max(abs(cp1)))[0][0]+1
cp1_max_idx_name=target_cols[cp1_max_idx]

cp2=pca.components_[1]
cp2_max_idx=np.argwhere(abs(cp2)==max(abs(cp2)))[0][0]+1
cp2_max_idx_name=target_cols[cp2_max_idx]

cp3=pca.components_[2]
cp3_max_idx=np.argwhere(abs(cp3)==max(abs(cp3)))[0][0]+1
cp3_max_idx_name=target_cols[cp3_max_idx]

components_theme={0:cp1_max_idx_name,1:cp2_max_idx_name,2:cp3_max_idx_name}

#降维后的数据
cLst=list(map(lambda x:'r' if x==-1 else 'c',pred))
new_uniform=pca.fit_transform(uniform)


#每个案例的代表成分
sample_theme=[]
sample_theme_factor=[]
for i in range(samples):
    tmp_lst=[new_uniform[i,0],new_uniform[i,1],new_uniform[i,2]]
    max_idx=list(map(abs,tmp_lst)).index(max(map(abs,tmp_lst)))
    max_factor=new_uniform[i,max_idx]
    sample_theme.append(max_idx)
    sample_theme_factor.append(max_factor)
original_df['sample_theme']=[components_theme[i] for i in sample_theme]
original_df['sample_theme_factor']=sample_theme_factor
original_df['pred']=pred
original_df.to_csv(r"C:\Users\dell\Desktop\hh_pred_isolation.csv")

original_df.head(3)

from mpl_toolkits.mplot3d.axes3d import Axes3D
factor1=new_uniform[:,0]
factor2=new_uniform[:,1]
factor3=new_uniform[:,2]
fig = plt.figure()#此处fig是二维
axes3d = Axes3D(fig)#将二维转化为三维
axes3d.scatter(factor1,factor2,factor3,c=cLst)
plt.show()


