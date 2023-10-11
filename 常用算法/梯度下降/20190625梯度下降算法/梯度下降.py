import numpy as np
dataPath=r"E:\SJB\NOTE\Python\algorithm\20190625梯度下降算法\Linear_Regression.csv"
dataSet=np.genfromtxt(dataPath,delimiter=',',skip_header=1)

def getData(dataSet):
    m,n=np.shape(dataSet)
    # 在原有自变量的基础上，需要主观添加一个均为1的偏移量，即公式中的x0
    trainData=np.ones((m,n))
    trainData[:,:-1]=dataSet[:,:-1]
    trainLabel=dataSet[:,-1]
    return trainData,trainLabel

def batchGradientDescent(x,y,theta,alpha,m,maxIterations):
    xTrain=x.transpose()
    for i in range(0,maxIteraions):
        hypothesis=np.dot(x,theta)
        loss=hypothesis-y
        gradient=np.dot(xTrain,loss)/m
        theta=theta-alpha*gradient
    return theta

trainData,trainLabel=getData(dataSet)
m,n=np.shape(trainData)
theta=np.ones(n)
alpha=0.05
maxIteraions=1000

theta=batchGradientDescent(trainData,trainLabel,theta,alpha,m,maxIteraions)
print(theta)

