import pandas as pd
import numpy as np
df=pd.DataFrame({'a':[1,2,3,4,5,6,7]})
df_quantile=df.a.quantile(0.5)
print('df 50%分位点:{}'.format(df_quantile))

arr=np.array(df.a)
np_percentile=np.percentile(arr,50)
print('arr 50%分位点:{}'.format(np_percentile))