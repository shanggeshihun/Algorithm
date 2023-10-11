# -*- coding: utf-8 -*-
from collections import Counter
import numpy as np
import pandas as pd
arr=np.array([1,2,3,1,3,5])
lst=list('12243542')
df=pd.DataFrame(np.random.rand(10,2),columns=list('ab'))
arrCnt=Counter(arr)
lstCnt=Counter(lst)
dfCnt=Counter(df.loc[:,'a'])
print(arrCnt,type(arrCnt))
print(lstCnt)
print(dfCnt)


lst1=list('abcdefg')
lst2=list('A')
lst1.insert(0,lst2[0])
lst1.insert(0,'B')
print(lst1)