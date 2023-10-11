# 关键字：内部节点（属性），叶节点（类）

# 三个过程：特征选择，决策树生成，决策树剪枝（分类效果好/泛化能力差/易于过拟合）

# 信息论中熵（entropy）是表示随机变量不确定性的度量，不确定性越大，不纯度越高，熵越大

# ID3：信息增益 log(2) 自然对数

# coding:utf-8
def createDataSet():
    dataSet = [[0, 0, 0, 0, 'no'],                        #数据集
            [0, 0, 0, 1, 'no'],
            [0, 1, 0, 1, 'yes'],
            [0, 1, 1, 0, 'yes'],
            [0, 0, 0, 0, 'no'],
            [1, 0, 0, 0, 'no'],
            [1, 0, 0, 1, 'no'],
            [1, 1, 1, 1, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [2, 0, 1, 2, 'yes'],
            [2, 0, 1, 1, 'yes'],
            [2, 1, 0, 1, 'yes'],
            [2, 1, 0, 2, 'yes'],
            [2, 0, 0, 0, 'no']]
    labels = ['年龄', '有工作', '有自己的房子', '信贷情况']        #特征标签
    return dataSet, labels                             #返回数据集和分类属性

# 计算数据集的信息熵
def calcshan(dataSet):
    lenDataSet=len(dataSet)
    p={}
    H=0.0
    for data in dataSet:
        currentLabel=data[-1]
        if currentLabel not in p.keys():
            p[currentLabel]=0
        p[currentLabel]+=1
    for key in p:
        px=float(p[key])/float(lenDataSet)
        H-=px*log(px,2)
    return H
    
# 根据某一个特征特征值分类数据集
def splitDataSet(dataSet,axis,value):
    subDataSet=[]
    for data in dataSet:
        subData=[]
        if data[axis]==value:
            subData=data[:axis]
            subData.extend(data[axis:])
            subDataSet.append(subData)
    return subDataSet
    

#ID3：信息增益 选择最优的特征
def chooseBestFeature(dataSet):
    lenFeature=len(dataSet[0])-1
    # 母树的信息熵
    shanInit=calcshan(dataSet)
    feature=[]
    inValue=0.0
    bestFeature=0
    for i in range(lenFeature):
        shanCarry=0.0
        # 第i个特征的特征值(价格i：高，中，低)
        feature=[example[i] for example in dataSet]
        feature=set(feature)
        # 第i个特征的信息增益  信息熵
        for feat in feature:
            subData=splitDataSet(dataSet,i,feat)
            prob=float(len(subData))/float(len(dataSet))
            shanCarry+=prob*calcshan(subData)
        outValue=shanInit-shanCarry
        if outValue>inValue:
            inValue=outValue
            bestFeature=i
    return bestFeature

# 统计classList中出现此处最多的元素（类标签）
def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classList.keys():
            classCount[vote]=0
        else:
            classCount[vote]+=1
    # 根据字典的值降序排序        
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]
    
    
# 创建决策树
def createTree(dataSet,label,featLabels):
    classList=[example[-1] for example in dataSet]
    # 如果类别完全相同则停止划分
    if classList.count(classList[0])==len(classList):
        return classList[0]
    # 如果遍历完了所有的特征值，类别可以不一样  
    if len(dataSet[0])==1:
        return majorityCnt(classList)
    # 选择最优特征
    bestFeat=chooseBestFeature(dataSet)
    bestFeatLabel=labels[bestFeat]
    featLabels.append(bestFeatLabel)
    #根据最优特征的标签生成树
    myTree={bestFeatLabel:{}}
     #删除已经使用特征标签
    del(labels[bestFeat])
    #得到训练集中所有最优特征的特征值
    featValues=[example[bestFeat] for example in dataSet]
    #去掉重复的特征值
    uniqueVals=set(featValues)
    for value in uniqueVals:
        myTree[bestFeatLabel][value]=createTree(splitDataSet(dataSet,bestFeat,value),label,featLabels)
    return myTree
    
if __name__=='__main__':
    dataSet,labels=createDataSet()
    featLabel=[]
    myTree=createTree(dataSet,labels,featLabels)
    print(myTree)
    
    