def TrainData(LineSample):
    """
    return {line1:{dri1:cnt1,dri2:cnt2},line2:{dri:cnt1,dri:cnt2}}
    """
    train={}
    for t in LineSample:
        train.setdefault(t[0],{})
        train[t[0]][t[5]]=t[6]
    return train


def sim(vec1,vec2):
    s=0
    l=len(vec1)
    index=range(l)
    for i in index:
        if i!=l-1:
            if vec1[i]==vec2[i] :
                s+=1
        else:
            if vec1[i] is not None and vec2[i] is not None:
                if max(vec1[i],vec2[i])>0 and min(vec1[i],vec2[i])/max(vec1[i],vec2[i])<50:
                    s+=1
                if max(vec1[i],vec2[i])==0:
                    s+=1
    w=s/len(index)
    return w


def ItemSimilarity(DriSample):
    """
    return {'dir1':{'dri2':0.2,'dri3':0.5},'dir2':{'dri1':0.2,'dri3':0.5}}
    df=pd.DataFrame({'dri':['dri1','dri2','dri3','dri4'],'f1':['a','b','c','d'],'f2':[1,2,3,4]})
    """
#    DriSample=pd.DataFrame({'dri':['dri1','dri2','dri3','dri4'],'f1':[np.nan,'a',np.nan,'c'],'f2':[1,2,np.nan,np.nan]})
    dri_sim={}
    for i_1,t_1 in enumerate(DriSample):
        to_dri={}
        for i_2,t_2 in enumerate(DriSample):
            if i_1==i_2:
                continue
            try:
                w=sim(t_1[1:],t_2[1:])
            except:
                print(t_1,t_2)
            to_dri[t_2[0]]=w
        dri_sim[t_1[0]]=to_dri
    return dri_sim


import operator
def Recommendation(train,line, W, K):
    rank = dict()
    ru = train[line]
    print(ru)
    for i,pi in ru.items():
        print(W[i])
        for j, wj in sorted(W[i].items(),key=operator.itemgetter(1),reverse=True)[0:K]:
            print(ru)
            print(W[i])
            if j in ru:
                continue
            rank.setdefault(j,0)
            rank[j]= pi * wj
    return rank 

from mysql import mysql_operation

if __name__=='__main__':
    
    mysql=mysql_operation()
    LineSql="select * from rep_op_line_order_cnt where line is not null"
    # columns=['line','load_city','load_region_name','upload_city','upload_region_name','driver_code','order_cnt']
    LineSample=mysql.select(LineSql)

    DriSql="select * from rep_op_dri_features"
    #columns=['driver_code','most_load_city','most_load_region','most_upload_city','most_upload_region','most_line','recent_line','recent_gap_hours']
    DriSample=mysql.select(DriSql)

    mysql.close()
    train=TrainData(LineSample)

    W=ItemSimilarity(DriSample)
    
    
    line=" 图木舒克市_乌鲁木齐市 头屯河区"
    K=2
    rank=Recommendation(train,line, W, K)
    print(rank)

#
#    K=1
#    r=Recommendation(train,line, W, K)
#    print(r)