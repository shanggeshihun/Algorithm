# -*- coding: utf-8 -*-
"""
Created on 20190507

输出：用户u对 与其最近邻的K个用户的物品的兴趣度
https://pan.baidu.com/s/1uE6-r2aV1ABrkSbJmoQLOQ 84c8

算法缺点：的用户数目越来越大，计算用户兴趣相似度矩阵将越来越困难，其运算时间复杂度和空间复杂度的增长和用户数的增长近似于平方关系。其次，基于用户的协同过滤很难对推荐结果作出解释

"""


"""
覆盖率：
热门排行榜的推荐覆盖率是很低的，它只会
推荐那些热门的物品，这些物品在总物品中占的比例很小。一个好的推荐系统不仅需要有比较高
的用户满意度，也要有较高的覆盖率。
"""


#=================基于用户的协同过滤算法======================
import math
def UserSimilarity(train):
    """
    train={u1:{i1:1,i2:1},u2:{i1:1,i3:1}}
    item_user={i1:{u1,u2,u3},i2:{u3,u5}}
    N=={u1:2,u2:4,u3:8}
    C={{u1:{u2:1,u3:4,u4:5},u2:{u1:1,u3:5,u4:7},u3:{u1:4,u2:5,u4:9}}}
    W={{u1:{u2:0.2,u3:0.2,u4:0.3},u2:{u1:0.2,u3:0.5,u4:0.8}}}
    """
    #build inverse table for item_users
    item_users = dict()
    for u, items in train.items():
        for i in items.keys():
            if i not in item_users:
                item_users[i] = set()
            item_users[i].add(u)
    
    #calculate co-rated items between users
    C = dict() #物品-物品的共现矩阵 
    N = dict() #物品被多少个不同用户购买
    for i, users in item_users.items():
        for u in users:
            N[u] += 1
            for v in users:
                if u == v:
                    continue
                C[u][v] += 1
    #calculate finial similarity matrix W
    W = dict()
    for u, related_users in C.items():
        for v, cuv in related_users.items():
                W[u][v] = cuv / math.sqrt(N[u] * N[v])
    return W

import operator
def Recommend(user, train,K, W):
    """
    rank={i1:0.5,i2:0.7}
    """
    rank = dict()
    interacted_items = train[user]
    for v, wuv in sorted(W[user].items(), key=operator.itemgetter(1),reverse=True)[0:K]:
        for i, rvi in train[v].items:
            if i in interacted_items:
            #we should filter items user interacted before
                continue
            rank[i] += wuv * rvi
    return rank


#换句话说，两个用户对冷门物品采取过同样的行为更能说明他们兴趣的相似度

#将基于上述用户相似度公式的UserCF算法记为User-IIF算法
def UserSimilarity(train):
    # build inverse table for item_users
    item_users = dict()
    for u, items in train.items():
        for i in items.keys():
            if i not in item_users:
                item_users[i] = set()
            item_users[i].add(u)
    
    #calculate co-rated items between users
    C = dict()
    N = dict()
    for i, users in item_users.items():
        for u in users:
            N[u] += 1
            for v in users:
                if u == v:
                    continue
                C[u][v] += 1 / math.log(1 + len(users))
            #calculate finial similarity matrix W
    W = dict()
    for u, related_users in C.items():
        for v, cuv in related_users.items():
            W[u][v] = cuv / math.sqrt(N[u] * N[v])
    return W