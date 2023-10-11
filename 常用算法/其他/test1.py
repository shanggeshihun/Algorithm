import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
df=pd.DataFrame({'x':[1,2,3,4,5,6],'y':[3,5,7,9,4,7]}) 
plt.figure(figsize=(10,10))
for i in range(6):
    x=df.iloc[i,1] 
    y=df.iloc[i,2]
    plt.scatter(x=x,y=y,label=i)
plt.scatter(x=df.iloc[:,1],y=df.iloc[:,1]*0.8+2) 
plt.show()