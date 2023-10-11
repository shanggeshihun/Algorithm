# -*- coding: utf-8 -*-
"""
Created on 20190506

输出：用户u对 与其最近邻的K个用户的物品的评分
"""

#https://blog.csdn.net/anryyang/article/details/23563237
from math import sqrt
 
 
def loadData():
    trainSet = {}
    testSet = {}
    movieUser = {}
    u2u = {}

 
    TrainFile =r"E:\SJB\NOTE\Python\algorithm\recommend\ml-100k\u1.base"   #指定训练集 
    TestFile =r"E:\SJB\NOTE\Python\algorithm\recommend\ml-100k\u1.test"    #指定测试集
    #加载训练集
    for line in open(TrainFile):
        (userId, itemId, rating, timestamp) = line.strip().split('\t')   
        trainSet.setdefault(userId,{})
        trainSet[userId].setdefault(itemId,float(rating))
 
        movieUser.setdefault(itemId,[])
        movieUser[itemId].append(userId.strip())
    #加载测试集
    for line in open(TestFile): 
        (userId, itemId, rating, timestamp) = line.strip().split('\t')   
        testSet.setdefault(userId,{})
        testSet[userId].setdefault(itemId,float(rating))
 
    #生成用户用户共有电影矩阵
    for m in movieUser.keys():
        for u in movieUser[m]:
            u2u.setdefault(u,{})
            for n in movieUser[m]:
                if u!=n:
                    u2u[u].setdefault(n,[])
                    u2u[u][n].append(m)
    return trainSet,testSet,u2u
      
  
 
#计算一个用户的平均评分  
def getAverageRating(user):
    trainSet,testSet,u2u = loadData()
    average = (sum(trainSet[user].values())*1.0) / len(trainSet[user].keys())  
    return average
 
#计算用户相似度  
def getUserSim(u2u,trainSet):
    userSim = {}
    # 计算用户的用户相似度  
    for u in u2u.keys(): #对每个用户u
        userSim.setdefault(u,{})  #将用户u加入userSim中设为key，该用户对应一个字典
        average_u_rate = getAverageRating(u)  #获取用户u对电影的平均评分
        for n in u2u[u].keys():  #对与用户u相关的每个用户n             
            userSim[u].setdefault(n,0)  #将用户n加入用户u的字典中
 
            average_n_rate = getAverageRating(n)  #获取用户n对电影的平均评分
              
            part1 = 0  #皮尔逊相关系数的分子部分
            part2 = 0  #皮尔逊相关系数的分母的一部分
            part3 = 0  #皮尔逊相关系数的分母的一部分
            for m in u2u[u][n]:  #对用户u和用户n的共有的每个电影  
                part1 += (trainSet[u][m]-average_u_rate)*(trainSet[n][m]-average_n_rate)*1.0  
                part2 += pow(trainSet[u][m]-average_u_rate, 2)*1.0  
                part3 += pow(trainSet[n][m]-average_n_rate, 2)*1.0  
                  
            part2 = sqrt(part2)  
            part3 = sqrt(part3)  
            if part2 == 0 or part3 == 0:  #若分母为0，相似度为0
                userSim[u][n] = 0
            else:
                userSim[u][n] = part1 / (part2 * part3)
    return userSim
  
 
#寻找用户最近邻并生成推荐结果
def getRecommendations(N,trainSet,userSim):
    pred = {}
    for user in trainSet.keys():    #对每个用户
        pred.setdefault(user,{})    #生成预测空列表
        interacted_items = trainSet[user].keys() #获取该用户评过分的电影  
        average_u_rate = getAverageRating(user)  #获取该用户的评分平均分
        userSimSum = 0
        simUser = sorted(userSim[user].items(),key = lambda x : x[1],reverse = True)[0:N]
        for n, sim in simUser:  
            average_n_rate = getAverageRating(n)
            userSimSum += sim   #对该用户近邻用户相似度求和
            for m, nrating in trainSet[n].items():  
                if m in interacted_items:  
                    continue  
                else:
                    pred[user].setdefault(m,0)
                    pred[user][m] += (sim * (nrating - average_n_rate))
        for m in pred[user].keys():  
                pred[user][m] = average_u_rate + (pred[user][m]*1.0) / userSimSum
    return pred

#计算预测分析准确度
def getMAE(testSet,pred):
    MAE = 0
    rSum = 0
    setSum = 0
 
    for user in pred.keys():    #对每一个用户
        for movie, rating in pred[user].items():    #对该用户预测的每一个电影    
            if user in testSet.keys() and movie in testSet[user].keys() :   #如果用户为该电影评过分
                setSum = setSum + 1     #预测准确数量+1
                rSum = rSum + abs(testSet[user][movie]-rating)      #累计预测评分误差
    MAE = rSum / setSum
    return MAE
 
 
if __name__ == '__main__':
    print('正在加载数据...')
    trainSet,testSet,u2u = loadData()
    print('正在计算用户间相似度...')
    userSim = getUserSim(u2u,trainSet)
    print('正在寻找最近邻...')
    for N in (5,10,20,30,40,50,60,70,80,90,100):            #对不同的近邻数
        pred = getRecommendations(N,trainSet,userSim)   #获得推荐
        mae = getMAE(testSet,pred)  #计算MAE
        print('邻居数为：N= %d 时 预测评分准确度为：MAE=%f'%(N,mae))
