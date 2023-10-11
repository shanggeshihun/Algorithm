# _*_coding:utf-8 _*_
# @Time　　 :2019/7/29   11:09
# @Author　 :
#@ File　　 :split_to_numbers.py 分解整数成若干数值
#@Software  :PyCharm

result_tuple_lst = []
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
        result_tuple = []
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
        aa = getAllCombination(sums - i, result, count)  # 求和为sums-i的组合
        if isinstance(aa, list):
            result_tuple_lst.append(aa)
        count -= 1
        i += 1  # 找下一个数字作为组合中的数字


# 方法功能：找出和为n的所有整数的组合
def showAllCombination(n):
    if n < 1:
        print("参数不满足要求")
        return
    result = [None] * n
    getAllCombination(n, result, 0)


def numberCombination(n, count):
    showAllCombination(n)
    return [a for a in result_tuple_lst if len(a) == count]