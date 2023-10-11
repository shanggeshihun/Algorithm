import matplotlib.pyplot as plt
import numpy as np
#根据列数显示箱型图数量
all_data = [np.random.normal(0, std, 100) for std in range(1, 4)]
fig = plt.figure(figsize=(8,6))
plt.boxplot(all_data,
            notch=False, # box instead of notch shape
            sym='rs',    # red squares for outliers
            vert=True)   # vertical box aligmnent
plt.xticks([y+1 for y in range(len(all_data))], ['x1', 'x2', 'x3'])
plt.xlabel('measurement x')
t = plt.title('Box plot')
plt.show()


#根据列内分组显示箱型图
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
df =pd.DataFrame(np.random.rand(10,1), columns=['Col1'] )
df['X']=pd.Series(['A','A','A','A','A','B','B','B','B','B'])
bp=df.boxplot(by='X',return_type='dict')
plt.ylabel('x')
plt.show()


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df =pd.DataFrame(np.random.rand(10,2), columns=['Col1', 'Col2'] )
df['X'] =pd.Series(['A','A','A','A','A','B','B','B','B','B'])
sns.boxplot(x='X',y='Col1',data=df)
plt.title('sns_bos_plot')
sns.boxplot()




#pip install pytesseract
from PIL import Image
import pytesseract
text=pytesseract.image_to_string(Image.open(r"C:\Users\dell\Desktop\img.png"),lang='chi_sim')
print(text)


lg_null_index=lg_df_temp.index[np.where(np.isnan(lg_df_temp))[0]].drop_duplicates()

dfnan=pd.DataFrame({'a':[1,np.NAN,np.NAN,2,5],'b':[6,np.NAN,9,np.NAN,10]})
n=dfnan.index[np.where(np.isnan(dfnan))[0]].drop_duplicates()
print(n)




python 日期、时间、字符串相互转换
在python中，日期类型date和日期时间类型dateTime是不能比较的。

（1）如果要比较，可以将dateTime转换为date，date不能直接转换为dateTime

import datetime
dateTime_p = datetime.datetime.now()  
date_p = dateTime_p.date() 
print(dateTime_p) #2019-01-30 15:17:46.573139
print(date_p) #2019-01-30
（2）日期类型date转换为字符串str

#!/usr/bin/env python3
import datetime
date_p = datetime.datetime.now().date()
str_p = str(date_p)
print(date_p,type(date_p)) #2019-01-30 <class 'datetime.date'>
print(str_p,type(str_p)) #2019-01-30 <class 'str'>
（3）字符串类型str转换为dateTime类型

import datetime
str_p = '2019-01-30 15:29:08'
dateTime_p = datetime.datetime.strptime(str_p,'%Y-%m-%d %H:%M:%S')
print(dateTime_p) # 2019-01-30 15:29:08
（4）dateTime类型转为str类型

　　这个地方我也不太理解，为什么指定格式无效

import datetime
dateTime_p = datetime.datetime.now()
str_p = datetime.datetime.strftime(dateTime_p,'%Y-%m-%d')
print(dateTime_p) # 2019-01-30 15:36:19.415157
（5）字符串类型str转换为date类型

#!/usr/bin/env python3
import datetime
str_p = '2019-01-30'
date_p = datetime.datetime.strptime(str_p,'%Y-%m-%d').date()
print(date_p,type(date_p)) # 2019-01-30 <class 'datetime.date'>
另外dateTime类型和date类型可以直接做加1减1这种操作

复制代码
#!/usr/bin/env python3
import datetime
# today = datetime.datetime.today()
today = datetime.datetime.today().date()
yestoday = today + datetime.timedelta(days=-1)
tomorrow = today + datetime.timedelta(days=1)
print(today) # 2019-01-30
print(yestoday)# 2019-01-29
print(tomorrow)# 2019-01-31
复制代码

