# _*_coding:utf-8 _*_
# @Time　　 :2019/7/26   17:29
# @Author　 : Antipa
#@ File　　 :拼团方案1_V.py
#@Software  :PyCharm

# 引入递归模块
from group.split_to_numbers import numberCombination


def groupby_statis(df, groupby_name, column_name):
    """
    return: 根据分组变量返回指定指标的统计值
    """
    grouped = df[column_name].groupby(df[groupby_name])
    return grouped.max(), grouped.min()


def standardize_df(origin_df, groupby_name, column_name_lst):
    """
    :param origin_df:
    :param groupby_name: 分组变量
    :param column_name_lst: 标准化变量列表
    :return: 添加标准化变量后的数据框
    """
    df = origin_df.copy()
    for column_name in column_name_lst:
        aa = groupby_statis(df, groupby_name, column_name)
        print(aa)
        column_name_new = column_name + '_standard'
        df[column_name_new] = ''
        for lst_value in aa[0].index:
            df[column_name_new][df.lst == lst_value] = (df[column_name][df.lst == lst_value] - aa[1][lst_value]) / (
                        aa[0][lst_value] - aa[1][lst_value])
    return df

import numpy as np
import pandas as pd


def the_car_price_dict(total_car_cnt, step_1, delta_price_1, step_2, delta_price_2, step_3, delta_price_3):
    """
    total_car_cnt:单个包车辆数
    step_1:第一个区间的下界
    delta__price_1:第一个区间递涨单价
    step_2:第二个区间的步长
    delta_price_2:第二个区间递涨单价
    step_3:第三个区间的步长
    delta_price_3:第三个区间的递涨单价

    return:price={the_car_number:price}
    """
    price = {}
    if total_car_cnt >= 1 and total_car_cnt <= step_1:
        step_price_1 = delta_price_1 * total_car_cnt
        for i in range(1, total_car_cnt + 1):
            price[i] = step_price_1

    elif total_car_cnt >= 1 + step_1 and total_car_cnt <= step_2 + step_1:
        step_price_2 = step_1 * delta_price_1 + (total_car_cnt - step_1) * delta_price_2
        step_price_1 = step_price_2
        for i in range(1, total_car_cnt + 1):
            price[i] = step_price_1
    #        for i in range(total_car_cnt+1,step_1+step_2+step_3+1):
    #            price[i]=0

    elif total_car_cnt >= 1 + step_2 + step_1 and total_car_cnt <= step_1 + step_2 + step_3:
        step_price_3 = step_1 * delta_price_1 + step_2 * delta_price_2 + (
                    total_car_cnt - step_1 - step_2) * delta_price_3
        step_price_2, step_price_1 = step_price_3, step_price_3
        for i in range(1, total_car_cnt + 1):
            price[i] = step_price_1
    #        for i in range(total_car_cnt+1,step_1+step_2+step_3+1):
    #            price[i]=0
    else:
        step_price_3, step_price_2, step_price_1 = 0, 0, 0

    return price


def anticipate_profit(actual_price_dict):
    """
    actual_price_dict:车辆实际运价字典
    return:返回货主预期盈利
    """
    actual_price_array = np.array(list(actual_price_dict.values()))
    profit = (anticipate_price - base_price - actual_price_array).sum()
    return profit

def unit_actual_sum_price(actual_price_dict):
    """
    actual_price_dict:车辆实际运价字典
    return:返回该团的总运价
    """
    actual_price_array = np.array(list(actual_price_dict.values()))
    sum_price = (base_price + actual_price_array).sum()
    return sum_price


base_price_lst = []
if __name__ == '__main__':
    #
    base_price = 205
    market_price = 225
    max_delta_price = market_price - base_price
    anticipate_price = 250
    total_car_cnt = 30
    unit_count = 2

    split_unit_lst_tmp = numberCombination(total_car_cnt, unit_count)
    split_unit_lst = [l for l in split_unit_lst_tmp if min(l) >= 6]

    actual_anticipate_profit_lst = []
    # 第一个区间至少6辆车
    step_1_r = range(6, 25)
    step_2_r = [0] + list(range(6, 25))
    step_3_r = [0] + list(range(6, 25))

    for step_1 in step_1_r:
        for step_2 in step_2_r:
            for step_3 in step_3_r:
                if step_1 + step_2 + step_3 != 30 or 0 in (step_1, step_2, step_3):
                    continue
                else:
                    for delta_price_1 in np.arange(0.1, 1.1, 0.1):
                        for delta_price_2 in np.arange(delta_price_1 + 0.1, 1.1, 0.1):
                            for delta_price_3 in np.arange(delta_price_2, 1.1, 0.1):
                                # min(delta_price_1,delta_price_2,delta_price_3)==max(delta_price_1,delta_price_2,delta_price_3) or
                                if delta_price_1 * step_1 + delta_price_2 * step_2 + delta_price_3 * step_3 >= max_delta_price:
                                    continue
                                for lst in split_unit_lst:
                                    # lst 其中一种拼团组合
                                    actual_anticipate_profit_tmp = []
                                    last_actual_market_price_tmp = []
                                    actual_mean_price_tmp = []
                                    actual_sum_price_tmp = []
                                    for total_car_lst in lst:
                                        # 每个团的司机运价字典
                                        actual_price_dict = the_car_price_dict(total_car_lst, step_1, delta_price_1,
                                                                               step_2, delta_price_2, step_3,
                                                                               delta_price_3)
                                        # 每个团的司机个数
                                        l = len(actual_price_dict)
                                        # 货主实际盈利
                                        actual_anticipate_profit = round(anticipate_profit(actual_price_dict), 2)
                                        actual_anticipate_profit_tmp.append(actual_anticipate_profit)
                                        # 该团最后一个司机的运价
                                        last_actual_market_price = round(max(actual_price_dict.values()) + base_price,
                                                                         2)
                                        last_actual_market_price_tmp.append(last_actual_market_price)
                                        # 该团的司机总运价
                                        actual_sum_price = unit_actual_sum_price(actual_price_dict)
                                        actual_sum_price_tmp.append(actual_sum_price)
                                        # 该团的司机平均运价
                                        actual_mean_price = round(actual_sum_price / l, 2)
                                        actual_mean_price_tmp.append(actual_mean_price)

                                    actual_anticipate_profit_sum = sum(actual_anticipate_profit_tmp)
                                    last_actual_market_price_avg_tmp = np.mean(last_actual_market_price_tmp)
                                    unit_actual_mean_price_tmp = round(np.sum(actual_sum_price_tmp) / total_car_cnt, 2)
                                    actual_anticipate_profit_lst.append((lst, step_1, delta_price_1, step_2,
                                                                         delta_price_2, step_3, delta_price_3,
                                                                         total_car_cnt, last_actual_market_price_tmp,
                                                                         last_actual_market_price_avg_tmp,
                                                                         actual_mean_price_tmp,
                                                                         unit_actual_mean_price_tmp,
                                                                         actual_anticipate_profit_tmp,
                                                                         actual_anticipate_profit_sum))
    df = pd.DataFrame(actual_anticipate_profit_lst)

    df.columns = ['lst', 'step_1', 'delta_price_1', 'step_2', 'delta_price_2', 'step_3', 'delta_price_3',
                  'total_car_cnt', 'last_actual_market_price_tmp', 'last_actual_market_price_avg_tmp',
                  'actual_mean_price_tmp', 'unit_actual_mean_price_tmp', 'actual_anticipate_profit_tmp',
                  'actual_anticipate_profit_sum']

    df['lst'] = df['lst'].astype(str)
    df.sort_values(by='actual_anticipate_profit_sum', ascending=False, inplace=True)
    df_final = df[df.actual_anticipate_profit_sum >= total_car_cnt * (anticipate_price - market_price)]
    df_final.reset_index(drop=True, inplace=True)

    # 标准化每组的变量
    origin_df = df_final
    groupby_name = 'lst'
    column_name_lst = ['unit_actual_mean_price_tmp', 'actual_anticipate_profit_sum']
    df_final_1 = standardize_df(origin_df, groupby_name, column_name_lst)

    # 标准化 司机 货主利润后，计算绝对误差
    df_final_1['abs_error']=df_final_1.apply(lambda row:abs(row['unit_actual_mean_price_tmp_standard']-row['actual_anticipate_profit_sum_standard']),axis=1)

    # 分组 绝对误差最小的方案
    def top(df,by_column='abs_error',n=1):
        return df.sort_values(by=by_column,ascending=True).head(n)

    df_final_2=df_final_1.groupby('lst').apply(top)
    df_final_2.to_excel(r"C:\Users\dell\Desktop\tmp_1.xlsx")
