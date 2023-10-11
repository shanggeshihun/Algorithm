# _*_coding:utf-8 _*_
# @Time　　 : 2020/03/08   0:00
# @Author　 : zimo
# @File　   :
# @Software :PyCharm
# @Theme    :

# Opencv+Python制作人脸识别程序！

"""
Opencv+Python制作人脸识别程序！

# 机器学习在cv领域的应用
computer view

机器学习：
语音识别、语音搜索、照片、自动驾驶、地图、图像识别、语音识别



人脸解锁：
1 发现（照片不一定是标准人像提取面部图像）
2 抽取（提取特征并想量化）
3 评测（如何训练模型选择最优算法）
4 预测（如何呈现结果 预测 可视化）
"""

"""
通过pip安装scipy、scikit-learn等库的时候，可能会报上面的错误，国内通过翻墙手段，是可能解决该问题的。下面给个不用翻墙的办法。 

使用国内镜像下载python库的办法。
以下载pandas为例，终端输入命令（前提是python正确安装）：


pip  install --index https://pypi.mirrors.ustc.edu.cn/simple/ pandas
注：--index后面也可以换成别的镜像，比如http://mirrors.sohu.com/python/
"""