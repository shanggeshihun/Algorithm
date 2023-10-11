# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 10:18:54 2019

@author: dell
"""
#显示单个图像
import matplotlib.pyplot as plt
import numpy as np
x=np.linspace(-1,1,50)
y=2*x+1
plt.plot(x,y)
plt.show()


#显示多个图像
import matplotlib.pyplot as plt
import numpy as np 
x=np.linspace(-1,1,50)
y1=2*x+1
y2=2**x+1
#每次调用figure都会重新申请一个figure对象
plt.figure()
plt.plot(x,y1)
plt.figure()
plt.plot(x,y1)
plt.plot(x,y2,color='red',linewidth=1,linestyle='--')
plt.show()

#去除边框，指定轴的名称
import matplotlib.pyplot as plt
import numpy as np
x=np.linspace(-1,1,50)
y1=2*x+1
y2=2**x+1
plt.figure()
plt.plot(x,y1)
plt.xlabel('i am x')
plt.ylabel('i am y')
plt.show()


#同事绘制多条曲线
import matplotlib.pyplot as plt
import numpy as np
x=np.linspace(-1,1,50)
y1=2*x+1
y2=2**x+1
#图表长宽
plt.figure(num=3,figsize=(8,5))
plt.plot(x,y2)
plt.plot(x,y1,color='red',linewidth=1,linestyle='--')
#参数范围
plt.xlim((-1,2))
plt.ylim((1,3))
#点位置
new_ticks=np.linspace(-1,2,5)
plt.xticks(new_ticks)
#第一个参数是点的位置，第二个参数是点的文字提示
plt.yticks([-2,-1.8,-1,1.22,3],[r'$really\ bad$', r'$bad$', r'$normal$', r'$good$', r'$readly\ good$'])
#获取当前轴
ax=plt.gca()
#右边框和上边框颜色去掉
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
#绑定x和y
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
#定义x和y轴的位置
ax.spines['bottom'].set_position(('data',0))
ax.spines['left'].set_position(('data',0))
plt.show()



#多条曲线之曲线说明
import matplotlib.pyplot as plt
import numpy as np
x=np.linspace(-1,1,50)
y1=2*x+1
y2=2**x+1
plt.figure(num=3,figsize=(8,5))
plt.plot(x,y2)
plt.plot(x,y1,color='red',linewidth=1,linestyle='--')
plt.xlim((-1,2))
plt.ylim((1,3))
plt.xlabel('i am x')
plt.ylabel('i am y')
new_ticks=np.linspace(-1,2,5)
plt.xticks(new_ticks)
plt.yticks([-2, -1.8, -1, 1.22,3],
          [r'$really\ bad$', r'$bad$', r'$normal$', r'$good$', r'$readly\ good$'])
l1,=plt.plot(x,y2,label='aaa')
l2,=plt.plot(x,y1,color='red',linewidth=1,linestype='-',label='bbb')
plt.legend(handles=[l1,l2],
           labels=['aaa','bbb'],
           loc='best')
plt.show()



#多个figure并加上特殊点注释
import matplotlib.pyplot as plt
import numpy as np 
x=np.linspace(-1,1,50)
y1=2*x+1
y2=2**x+1
plt.figure(figsize=(12,8))
plt.plot(x,y2)
plt.plot(x,y1,color='red',linewidth=1,linestyle='--')
ax=plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.spines['bottom'].set_position(('data',0))
ax.spines['left'].set_position(('data',0))
#显示交叉点
x0=1
y0=2*x0+1
plt.scatter(x0,y0,s=66,color='b')
#定义线的范围，x的范围是定值，y的范围是y0到0
plt.plot([x0,x0],[y0,0],'k-',lw=2.5)

#设置关键位置的提示信息
plt.annotate(r'$2x+1=%s$' % y0,xy=(x0,y0),xycoords='data',
             xytext=(+30,-30),
             textcoords='offset points',
             fontsize=16,
             arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=.2'))
# 在figure中显示文字信息
# 可以使用\来输出特殊的字符\mu\ \sigma\ \alpha
plt.text(0, 3, 
         r'$This\ is\ a\ good\ idea.\ \mu\ \sigma_i\ \alpha_t$',
         fontdict={'size':16,'color':'r'})

plt.show()


import matplotlib.pyplot as plt
import numpy as np 
plt.scatter([1],[2])
plt.scatter([1,1,1,1],[3,2,3,4])
plt.show()