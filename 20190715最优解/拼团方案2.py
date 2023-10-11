# _*_coding:utf-8 _*_
# @Time     :2019-07-16 14:06:08
# @Author   :
# @File     :拼团方案二
# @Software :
# @Comment  :
"""
1 拼团方案2按照先递增再递减(递减按照越后加团递减越多的原则)
2 最后一个司机的价格不得超过市场价，第二阶段最后一个司机梯度价格的不得低于第一阶段最后司机的50%
3 货主利润不低于900 ，
"""

result_tuple_lst=[]
def getAllCombination(sums, result, count):
    """
    # 求和为sums的所有整数组合
    :param sums: sums正整数
    :param result: 存储组合的结果
    :param count: 记录组合中数字的个数
    :return:
    """
    if sums < 0:
        return
    # 数字的组合满足和为sums的条件，打印出所有组合
    if sums == 0:
#        print("满足条件的组合")
        i = 0
        result_tuple=[]
        while i < count:
            result_tuple.append(result[i])
            i += 1
        return result_tuple
#    print("———————————当前组合———————————")
    i = 0
    while i < count:
        i += 1
    # 确定组合中下一个取值
    i = (1 if count == 0 else result[count - 1])
    while i <= sums:
        result[count] = i
        count += 1
        aa=getAllCombination(sums - i, result, count)  # 求和为sums-i的组合
        if isinstance(aa,list):
            result_tuple_lst.append(aa)
        count -= 1
        i += 1  # 找下一个数字作为组合中的数字
# 方法功能：找出和为n的所有整数的组合
def showAllCombination(n):
    if n < 1:
        print("参数不满足要求")
        return
    result = [None]*n
    getAllCombination(n, result, 0)

def numberCombination(n,count):
    showAllCombination(n)
    return [a for a in result_tuple_lst if len(a)==count]



def groupby_statis(df,groupby_name,column_name):
    """
    return: 根据分组变量返回指定指标的统计值
    """
    grouped=df[column_name].groupby(df[groupby_name])
    return grouped.max(),grouped.min()

def standardize_df(origin_df,groupby_name,column_name_lst):
    df=origin_df.copy()
    for column_name in column_name_lst:
        aa=groupby_statis(df,groupby_name,column_name)
        print(aa)
        column_name_new=column_name+'_01'
        df[column_name_new]=''
        for lst_value in aa[0].index:
            df[column_name_new][df.lst==lst_value]=(df[column_name][df.lst==lst_value]-aa[1][lst_value])/(aa[0][lst_value]-aa[1][lst_value])
    return df


import numpy as np
import pandas as pd

def the_car_price_dict(total_car_cnt,step_1,delta_price_1,step_2,delta_price_2):
    """
    total_car_cnt:单个包车辆数
    step_1:第一个区间的下界
    delta__price_1:第一个区间递涨单价
    step_2:第二个区间的步长
    delta_price_2:第二个区间递涨单价

    return:price={the_car_number:price} the_car_number 该团中的司机及实际运价
    """
    price={}
    if total_car_cnt>=1 and total_car_cnt<=step_1:
        step_price_1=delta_price_1*total_car_cnt

        for i in range(1,total_car_cnt+1):
            price[i]=step_price_1

    elif total_car_cnt>=1+step_1 and total_car_cnt<=step_2+step_1:
        step_price_1=step_1*delta_price_1
        for i in range(1,step_1+1):
            price[i]=step_price_1
        for i in range(step_1+1,total_car_cnt+1):
            price[i]=step_price_1+(i-step_1)*delta_price_2
    else:
        pass
    return price

# 测试
print(the_car_price_dict(24,24,1.1,6,-0.3))

def anticipate_profit(actual_price_dict):
    """
    actual_price_dict:车辆实际运价字典
    return:返回货主预期盈利
    """
    actual_price_array=np.array(list(actual_price_dict.values()))
    profit=(anticipate_price-base_price-actual_price_array).sum()
    return profit

def unit_actual_sum_price(actual_price_dict):
    """
    actual_price_dict:车辆实际运价字典
    return:返回该团的平均运价
    """
    actual_price_array=np.array(list(actual_price_dict.values()))
    sum_price=(base_price+actual_price_array).sum()
    return sum_price


base_price_lst=[]
if __name__=='__main__':
    # 
    base_price=205
    market_price=225
    max_delta_price=market_price-base_price
    anticipate_price=250
    total_car_cnt=30
    unit_count=2

    split_unit_lst_tmp=numberCombination(total_car_cnt,unit_count)
    split_unit_lst=[l for l in split_unit_lst_tmp if min(l)>=6]

    actual_anticipate_profit_lst=[]
    # 第一个区间至少6辆车
    step_1_r=range(6,25)
    step_2_r=[0]+list(range(6,25))

    for step_1 in step_1_r:
        step_2=total_car_cnt-step_1
        for delta_price_1 in np.arange(0.5,1.1,0.1):
            for delta_price_2 in np.arange(-0.3,-1.1,-0.1):
                if abs(step_2*delta_price_2)>step_1*delta_price_1*0.5:
                    continue
                for lst in split_unit_lst:
                    # lst 其中一种拼团组合 
                    # break_flag=1 放弃该种拼团组合
                    break_flag=0
                    i=0
                    actual_anticipate_profit_tmp=[]
                    last_actual_market_price_tmp=[]
                    actual_mean_price_tmp=[]
                    actual_sum_price_tmp=[]
                    for total_car_lst in lst:
                        # 每个团的司机运价字典
                        actual_price_dict=the_car_price_dict(total_car_lst,step_1,delta_price_1,step_2,delta_price_2)
                        # 每个团的司机个数
                        l=len(actual_price_dict)
                        # 如果第一阶段最高价格高于市场价格 or 如果第二阶段最高价格高于市场价格 or 第二阶段最低价格低于第一阶段最高价格的50%则放弃该种拼团方式
                        last_car=max(actual_price_dict.keys())
                        last_actual_price=actual_price_dict[last_car]
                        if (len(actual_price_dict)<=step_1 and last_actual_price>max_delta_price) or (len(actual_price_dict)>step_1 and (last_actual_price<step_1*delta_price_1*0.5 or last_actual_price>max_delta_price)):
                            break_flag=1
                            break
                        else:
                            i=i+1
                            # 货主实际盈利
                            actual_anticipate_profit=round(anticipate_profit(actual_price_dict),2)
                            actual_anticipate_profit_tmp.append(actual_anticipate_profit)
                            # 该团最后一个司机的运价
                            last_actual_market_price=round(last_actual_price+base_price,2)
                            last_actual_market_price_tmp.append(last_actual_market_price)
                            # 该团的司机总运价
                            actual_sum_price=unit_actual_sum_price(actual_price_dict)
                            actual_sum_price_tmp.append(actual_sum_price)
                            # 该团的司机平均运价
                            actual_mean_price=round(actual_sum_price/l,2)
                            actual_mean_price_tmp.append(actual_mean_price)

                    actual_anticipate_profit_sum=sum(actual_anticipate_profit_tmp)
                    last_actual_market_price_avg_tmp=np.mean(last_actual_market_price_tmp)
                    unit_actual_mean_price_tmp=round(np.sum(actual_sum_price_tmp)/total_car_cnt,2)

                    if i==len(lst):
                        actual_anticipate_profit_lst.append((lst,step_1,delta_price_1,step_2,delta_price_2,total_car_cnt,last_actual_market_price_tmp,last_actual_market_price_avg_tmp,actual_mean_price_tmp,unit_actual_mean_price_tmp,actual_anticipate_profit_tmp,actual_anticipate_profit_sum))
                    if break_flag==1:
                        continue
    df=pd.DataFrame(actual_anticipate_profit_lst)

    df.columns=['lst','step_1','delta_price_1','step_2','delta_price_2','total_car_cnt','last_actual_market_price_tmp','last_actual_market_price_avg_tmp','actual_mean_price_tmp','unit_actual_mean_price_tmp','actual_anticipate_profit_tmp','actual_anticipate_profit_sum']
    df['lst']=df['lst'].astype(str)
    df.sort_values(by='actual_anticipate_profit_sum',ascending=False,inplace=True)
    df_final=df[df.actual_anticipate_profit_sum>=total_car_cnt*(anticipate_price-market_price)]
    df_final.reset_index(drop=True,inplace=True)
    
    origin_df=df_final
    groupby_name='lst'
    column_name_lst=['unit_actual_mean_price_tmp','actual_anticipate_profit_sum']
    df_final_1=standardize_df(origin_df,groupby_name,column_name_lst)

    df_final_1.to_excel(r"C:\Users\dell\Desktop\tmp_2.xlsx")
