# -*- coding: utf-8 -*-
"""
Created on 20190507

输出：用户u对 与其最近邻的K个用户的物品的兴趣度
https://pan.baidu.com/s/1uE6-r2aV1ABrkSbJmoQLOQ 84c8


ItemCF算法并不利用物品的内容属性计算物品之间的相似度，它主要通过分析用户的行为记录计算物品之间的相似度；
亚马逊在iPhone商品界面上提供的与iPhone相关的商品，而相关商品都是购买iPhone的用户也经常购买的其他商品；
Hulu在个性化视频推荐利用ItemCF给每个推荐结果提供了一个推荐解释，而用于解释的视频都是用户之前观看或者收藏过的视频；

基于物品的协同过滤算法主要分为两步：
(1) 计算物品之间的相似度。
(2) 根据物品的相似度和用户的历史行为给用户生成推荐列表。
"""

"""

"""


#=================基于物品的协同过滤算法======================
def ItemSimilarity(train):
    #calculate co-rated users between items
    C = dict()#物品-物品的共现矩阵 
    N = dict()#物品被多少个不同用户购买
    for u, items in train.items():
        for i in items:
            N[i] += 1
            for j in items:
                if i == j:
                    continue
                C[i][j] += 1
                #calculate finial similarity matrix W
                W = dict()
                for i,related_items in C.items():
                    for j, cij in related_items.items():
                        W[i][j] = cij / math.sqrt(N[i] * N[j])
    return W 

import operator
def Recommendation(train, user_id, W, K):
    rank = dict()
    ru = train[user_id]
    for i,pi in ru.items():
        for j, wj in sorted(W[i].items(),key=operator.itemgetter(1),reverse=True)[0:K]:
            if j in ru:
                continue
            rank[j] += pi * wj
    return rank 

#假设有这么一个用户，他是开书店的，并且买了当当网上80%的书准备用来自己卖。那么，他的购物车里包含当当网80%的书。假设当当网有100万本书，也就是说他买了80万本。从前面对ItemCF的讨论可以看到，这意味着因为存在这么一个用户，有80万本书两两之间就产生了相似度，也就是说，内存里即将诞生一个80万乘80万的稠密矩阵。另外可以看到，这个用户虽然活跃，但是买这些书并非都是出于自身的兴趣，而且这些书覆盖了当当网图书的很多领域，所以这个用户对于他所购买书的两两相似度的贡献应该远远小于一个只买了十几本自己喜欢的书的文学青年。John S. Breese在论文①中提出了一个称为IUF（Inverse User Frequence），即用户活跃度对数的倒数的参数，他也认为活跃用户对物品相似度的贡献应该小于不活跃的用户，他提出应该增加IUF参数来修正物品相似度
import math
def ItemSimilarity(train):
    #calculate co-rated users between items
    C = dict()
    N = dict()
    for u, items in train.items():
        for i in items:
            N[i] += 1
            for j in items:
                if i == j:
                    continue
                C[i][j] += 1 / math.log(1 + len(items) * 1.0)
                #calculate finial similarity matrix W
                W = dict()
                for i,related_items in C.items():
                    for j, cij in related_items.items():
                        W[i][j] = cij / math.sqrt(N[i] * N[j])
    return W 