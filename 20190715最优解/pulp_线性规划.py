# _*_coding:utf-8 _*_
# @Time    :${DATE}   ${TIME}
# @Author  : Antipa
# @File     :${NAME}.py
# @Comment  :${PRODUCT_NAME}

import pulp
Ingredients=['CHICKEN', 'BEEF', 'MUTTON', 'RICE', 'WHEAT', 'GEL']
costs={'CHICKEN': 0.013, 
         'BEEF': 0.008, 
         'MUTTON': 0.010, 
         'RICE': 0.002, 
         'WHEAT': 0.005, 
         'GEL': 0.001}

proteinPercent={'CHICKEN': 0.100, 
                  'BEEF': 0.200, 
                  'MUTTON': 0.150, 
                  'RICE': 0.000, 
                  'WHEAT': 0.040, 
                  'GEL': 0.000}
fatPercent = {'CHICKEN': 0.080, 
              'BEEF': 0.100, 
              'MUTTON': 0.110, 
              'RICE': 0.010, 
              'WHEAT': 0.010, 
              'GEL': 0.000}

fibrePercent = {'CHICKEN': 0.001, 
                'BEEF': 0.005, 
                'MUTTON': 0.003, 
                'RICE': 0.100, 
                'WHEAT': 0.150, 
                'GEL': 0.000}

saltPercent = {'CHICKEN': 0.002, 
               'BEEF': 0.005, 
               'MUTTON': 0.007, 
               'RICE': 0.002, 
               'WHEAT': 0.008, 
               'GEL': 0.000}

# 创建问题实例，求最小极值
prob=pulp.LpProblem(name="The Whiskas Problem",sense=pulp.LpMinimize)

# 构建Lp变量字典，变量名以Ingr开头，如Ingr_CHICKEN，下界是0
ingredient_vars=pulp.LpVariable.dict("Ingr",Ingredients,0)

# 添加目标方程
prob+=pulp.lpSum(costs[i]*ingredient_vars[i] for i in Ingredients)

# 添加约束条件
prob+=pulp.lpSum([ingredient_vars[i] for i in Ingredients])==100
prob+=pulp.lpSum([proteinPercent[i]*ingredient_vars[i] for i in Ingredients])>=8.0
prob+=pulp.lpSum([fatPercent[i]*ingredient_vars[i] for i in Ingredients])>=6.0
prob += pulp.lpSum([fibrePercent[i] * ingredient_vars[i] for i in Ingredients]) <= 2.0
prob += pulp.lpSum([saltPercent[i] * ingredient_vars[i] for i in Ingredients]) <= 0.4

# 求解
prob.solve()

# 查看解的状态
print("Status:",pulp.LpStatus[prob.status])

# 查看解
for v in prob.variables():
    print(v.name,"=",v.varValue)

# 另一种查看解的方式
for i in Ingredients:
    print(ingredient_vars[i],"=",ingredient_vars[i].value())




import pulp 
Paint=['Exterior','Interior','Theme']
Profit={'Exterior':1000,
        'Interior':2000,
        'Theme':3000}
M1={'Exterior':1,
    'Interior':2,
    'Theme':3}
M2={'Exterior':0,
    'Interior':1,
    'Theme':2}
# 求解Profit最大值
prob=pulp.LpProblem(name="The Max Profit Of Producing Paints",sense=pulp.LpMaximize)

# 构建LP字典
Paint_vars=pulp.LpVariable.dict(name='Patint',indexs=Paint,lowBound=0)

# 添加目标方程
prob+=pulp.lpSum([Profit[i]*Paint_vars[i] for i in Paint])

# 添加约束条件
prob+=pulp.lpSum([M1[i]*Paint_vars[i] for i in Paint])<=10
prob+=pulp.lpSum([M2[i]*Paint_vars[i] for i in Paint])<=5

# 求解
prob.solve()
print(pulp.LpStatus[prob.status])
for v in Paint:
    print(Paint_vars[v].value())